#!/usr/bin/env python3
"""
In-process llama.cpp bridge for the local AI robot assistant cognitive core.

This module provides :class:`LlamaCppBridge`, a drop-in replacement for the
``OllamaBridge`` HTTP client. It runs Moondream (or any compatible GGUF VLM)
directly inside the ROS2 process using ``llama-cpp-python`` with the CUDA
backend, eliminating the HTTP/JSON/base64 round trip and Ollama daemon
overhead.

Vision runs through llama.cpp's ``mtmd`` path (the ``MoondreamChatHandler``
subclasses ``Llava15ChatHandler``, which initializes the multimodal context
with ``use_gpu=True``), so the clip encoder is GPU-offloaded. Flash attention
is enabled on the LLM context by default, which is the decisive vision
end-to-end optimization on Orin (image prefill dominates the call).

The public surface deliberately mirrors ``OllamaBridge`` so the cognitive
client node only needs to swap which bridge it instantiates:

    is_available() -> bool
    generate(prompt, image_base64=None, num_ctx=512, num_predict=128,
             temperature=0.3, force_json=False) -> Dict[str, Any]

The returned dict follows the same shape the node already consumes:
``{"response": str, "total_duration": int_ns, "model": str}`` plus token
accounting keys (``eval_count``, ``prompt_eval_count``) and an ``error`` key
on failure.

Architecture Reference: docs/architecture.md §2.5, Plan.md Phase 2.
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, Optional

# GBNF grammar mirroring the {"action","target","explanation"} intent schema.
#
# NOTE: every rule must stay on a single line. This llama.cpp GBNF parser
# rejects multi-line rule continuations (the previous multi-line `root` rule
# failed at sampling time with "expecting name").
INTENT_GBNF = (
    'root ::= "{" ws "\\"action\\"" ws ":" ws string ws "," ws '
    '"\\"target\\"" ws ":" ws string ws "," ws '
    '"\\"explanation\\"" ws ":" ws string ws "}"\n'
    'string ::= "\\"" ( [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" '
    '( ["\\\\/bfnrt] | "u" hex hex hex hex ) )* "\\""\n'
    "hex ::= [0-9a-fA-F]\n"
    "ws ::= [ \\t\\n]*\n"
)

# Default Ollama registry manifest location for moondream:latest.
_OLLAMA_MANIFESTS = [
    "/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/moondream/latest",
    os.path.expanduser("~/.ollama/models/manifests/registry.ollama.ai/library/moondream/latest"),
]


def resolve_ollama_moondream_blobs(
    manifest_path: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve the GGUF model and projector blob paths from an Ollama manifest.

    Reusing the already-present Ollama blobs lets us A/B the *same weights* on
    two runtimes (Ollama HTTP vs in-process llama.cpp) without a second
    download — isolating the runtime as the only changed variable.

    Args:
        manifest_path: Explicit manifest path. If None, the standard
            ``moondream:latest`` manifest locations are tried.

    Returns:
        Dict with ``model_path`` and ``mmproj_path`` keys. ``mmproj_path`` is
        an empty string if no projector layer is present, or ``None`` values
        if the manifest could not be found.
    """
    candidates = [manifest_path] if manifest_path else _OLLAMA_MANIFESTS
    manifest_file = next((p for p in candidates if p and os.path.isfile(p)), None)
    if manifest_file is None:
        return {"model_path": None, "mmproj_path": None}

    with open(manifest_file) as fh:
        manifest = json.load(fh)

    model_path = ""
    mmproj_path = ""
    for layer in manifest.get("layers", []):
        media = layer.get("mediaType", "")
        digest = layer.get("digest", "")
        if not digest.startswith("sha256:"):
            continue
        blob = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(manifest_file))))
            ),
            "blobs",
            "sha256-" + digest.split(":", 1)[1],
        )
        if media == "application/vnd.ollama.image.model":
            model_path = blob
        elif media == "application/vnd.ollama.image.projector":
            mmproj_path = blob

    return {"model_path": model_path or None, "mmproj_path": mmproj_path or ""}


class LlamaCppBridge:
    """In-process llama.cpp client mirroring the ``OllamaBridge`` interface.

    The heavy ``llama_cpp.Llama`` object is created eagerly in ``__init__`` so
    model-load failures surface at node startup (matching the Ollama bridge's
    startup availability check) rather than on the first query.

    Args:
        model_path: Path to the GGUF language model.
        mmproj_path: Path to the multimodal projector GGUF (required for vision).
        n_ctx: Context window size (deterministic KV-cache sizing).
        n_gpu_layers: Layers to offload to GPU; -1 = all.
        n_threads: CPU threads for the non-offloaded work.
        flash_attn: Enable flash attention for the LLM context. This is a
            significant vision end-to-end win on Orin (~18%: 2.35s → 1.92s for
            a Moondream image query) because the image prefill dominates.
        chat_format: Optional explicit chat format (e.g. "moondream").
        verbose: Pass-through to llama.cpp logging.
        logger: ROS2 logger instance.
    """

    def __init__(
        self,
        model_path: str = "",
        mmproj_path: str = "",
        n_ctx: int = 512,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        flash_attn: bool = True,
        chat_format: Optional[str] = None,
        verbose: bool = False,
        logger=None,
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.flash_attn = flash_attn
        self.chat_format = chat_format
        self.verbose = verbose
        self.logger = logger
        self.llm = None
        self._chat_handler = None
        self._grammar = None
        self._available = False
        self._last_error: Optional[str] = None

        # If no explicit model was given, reuse the installed Ollama blobs so
        # the same weights are A/B'd across runtimes without a re-download.
        if not self.model_path:
            blobs = resolve_ollama_moondream_blobs()
            self.model_path = blobs.get("model_path") or ""
            if not self.mmproj_path:
                self.mmproj_path = blobs.get("mmproj_path") or ""

        self._load()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _log(self, level: str, message: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message)

    def _load(self) -> None:
        """Load the GGUF model (and projector) into memory."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - exercised on-device
            self._last_error = (
                "llama-cpp-python is not installed. Build it with "
                'CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87".'
            )
            self._log("error", f"{self._last_error} ({exc})")
            return

        if not self.model_path or not os.path.isfile(self.model_path):
            self._last_error = f"GGUF model not found: {self.model_path}"
            self._log("error", self._last_error)
            return

        kwargs: Dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "flash_attn": self.flash_attn,
            "verbose": self.verbose,
        }
        if self.n_threads:
            kwargs["n_threads"] = self.n_threads

        if self.mmproj_path and os.path.isfile(self.mmproj_path):
            handler = self._make_chat_handler()
            if handler is not None:
                self._chat_handler = handler
                kwargs["chat_handler"] = handler
        elif self.mmproj_path:
            self._log("warn", f"Projector GGUF not found, vision disabled: {self.mmproj_path}")

        try:
            self.llm = Llama(**kwargs)
            self._available = True
            self._log(
                "info",
                f"llama.cpp loaded '{os.path.basename(self.model_path)}' "
                f"(n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers}, "
                f"flash_attn={self.flash_attn})",
            )
        except Exception as exc:  # pragma: no cover - exercised on-device
            self._last_error = f"Failed to load GGUF model: {exc}"
            self._log("error", self._last_error)

    def _make_chat_handler(self):
        """Create a multimodal chat handler for the projector, if available."""
        try:
            from llama_cpp.llama_chat_format import MoondreamChatHandler

            return MoondreamChatHandler(clip_model_path=self.mmproj_path, verbose=self.verbose)
        except Exception:
            pass
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            return Llava15ChatHandler(clip_model_path=self.mmproj_path, verbose=self.verbose)
        except Exception as exc:  # pragma: no cover - exercised on-device
            self._log("warn", f"No multimodal chat handler available for vision: {exc}")
            return None

    def _get_intent_grammar(self):
        """Build and cache the intent-schema GBNF grammar.

        Returns:
            A ``LlamaGrammar`` instance, or ``None`` if llama-cpp-python is not
            importable. ``from_string`` does not parse eagerly in this version,
            so the grammar is built once per bridge and reused.
        """
        if self._grammar is not None:
            return self._grammar
        try:
            from llama_cpp import LlamaGrammar
        except ImportError:
            return None
        self._grammar = LlamaGrammar.from_string(INTENT_GBNF)
        return self._grammar

    @classmethod
    def from_ollama(
        cls,
        n_ctx: int = 512,
        n_gpu_layers: int = -1,
        logger=None,
        **kwargs,
    ) -> "LlamaCppBridge":
        """Build a bridge from the locally installed Ollama moondream blobs."""
        blobs = resolve_ollama_moondream_blobs()
        return cls(
            model_path=blobs.get("model_path") or "",
            mmproj_path=blobs.get("mmproj_path") or "",
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logger=logger,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public interface (mirrors OllamaBridge)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        return self._available and self.llm is not None

    def generate(
        self,
        prompt: str,
        image_base64: Optional[str] = None,
        num_ctx: int = 512,
        num_predict: int = 128,
        temperature: float = 0.3,
        force_json: bool = False,
    ) -> Dict[str, Any]:
        """Run a completion (optionally with an image) via llama.cpp.

        Args:
            prompt: Text prompt.
            image_base64: Optional base64 JPEG for vision queries.
            num_ctx: Context size requested for this call (the loaded model's
                ``n_ctx`` is fixed; this is logged for parity with Ollama).
            num_predict: Maximum tokens to generate.
            temperature: Sampling temperature.
            force_json: Constrain output to the intent JSON schema.

        Returns:
            Dict with ``response``, ``total_duration`` (ns), ``model`` and token
            counters, or an ``error`` key on failure.
        """
        if not self.is_available():
            return {"error": self._last_error or "llama.cpp model not available"}

        start = time.time()
        try:
            if image_base64:
                text, usage = self._generate_with_image(
                    prompt, image_base64, num_predict, temperature, force_json
                )
            else:
                text, usage = self._generate_text(prompt, num_predict, temperature, force_json)
        except Exception as exc:  # pragma: no cover - exercised on-device
            msg = f"llama.cpp inference failed: {exc}"
            self._log("error", msg)
            return {"error": msg}

        elapsed_ns = int((time.time() - start) * 1e9)
        return {
            "response": text,
            "total_duration": elapsed_ns,
            "model": os.path.basename(self.model_path),
            "eval_count": usage.get("completion_tokens", 0),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate_text(self, prompt: str, num_predict: int, temperature: float, force_json: bool):
        params: Dict[str, Any] = {
            "max_tokens": num_predict,
            "temperature": temperature,
        }
        if force_json:
            grammar = self._get_intent_grammar()
            if grammar is not None:
                params["grammar"] = grammar

        result = self.llm.create_completion(prompt, **params)
        text = result["choices"][0]["text"]
        usage = result.get("usage", {})
        return text, usage

    def _generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        num_predict: int,
        temperature: float,
        force_json: bool,
    ):
        if self._chat_handler is None:
            # Vision requested but no projector/handler; fall back to text.
            self._log("warn", "Vision requested but no multimodal handler; using text-only.")
            return self._generate_text(prompt, num_predict, temperature, force_json)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]
        params: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": num_predict,
            "temperature": temperature,
        }
        if force_json:
            grammar = self._get_intent_grammar()
            if grammar is not None:
                params["grammar"] = grammar

        result = self.llm.create_chat_completion(**params)
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return text, usage

    def close(self) -> None:
        """Release the underlying model."""
        self.llm = None
        self._available = False


def find_local_gguf(models_dir: str = "models/moondream2_gguf") -> Dict[str, str]:
    """Best-effort discovery of a downloaded Moondream GGUF + projector.

    Args:
        models_dir: Directory to scan for ``*text*.gguf`` / ``*mmproj*.gguf``.

    Returns:
        Dict with ``model_path`` and ``mmproj_path`` (empty strings if absent).
    """
    model_path = ""
    mmproj_path = ""
    if os.path.isdir(models_dir):
        for path in sorted(glob.glob(os.path.join(models_dir, "*.gguf"))):
            name = os.path.basename(path).lower()
            if "mmproj" in name or "projector" in name:
                mmproj_path = path
            elif "text" in name:
                model_path = path
    return {"model_path": model_path, "mmproj_path": mmproj_path}
