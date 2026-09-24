#!/usr/bin/env python3
"""
Minimal web server for health / status monitoring (Plan.md Phase 5 item 4).

This is intentionally small: a FastAPI app exposing ``/health`` and ``/status``
so the robot's liveness and a few key metrics can be scraped without building
the full dashboard (deferred until the rest of the system is stable).

The ROS2 node subscribes to existing status topics and the FastAPI app serves
the latest cached values. Uvicorn runs in a background thread while ``rclpy``
spins the node on the main thread.

Endpoints:
    GET /health   -> {"status": "ok"}
    GET /status   -> system + subsystem status JSON
"""

import threading
import time
from typing import Any, Dict, Optional

import psutil
import rclpy
import uvicorn
from fastapi import FastAPI
from rclpy.node import Node
from std_msgs.msg import String


class SystemState:
    """Thread-safe-ish cache of the latest subsystem status strings."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.cognitive_status: str = "unknown"
        self.verification_status: str = "unknown"

    def snapshot(self) -> Dict[str, Any]:
        mem = psutil.virtual_memory()
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory": {
                "total_mb": round(mem.total / 1024 / 1024, 1),
                "available_mb": round(mem.available / 1024 / 1024, 1),
                "percent": mem.percent,
            },
            "cognitive_status": self.cognitive_status,
            "verification_status": self.verification_status,
        }


def create_app(state: SystemState) -> FastAPI:
    """Build the FastAPI app around a shared :class:`SystemState`.

    Args:
        state: Shared status cache updated by the ROS2 node.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Local AI Robot Assistant", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> Dict[str, Any]:
        return state.snapshot()

    return app


class WebServerNode(Node):
    """ROS2 node that caches status topics and serves them over HTTP."""

    def __init__(self) -> None:
        super().__init__("web_server")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value

        self.state = SystemState()

        self.create_subscription(String, "/cognitive/status", self._on_cognitive_status, 10)
        self.create_subscription(String, "/verification/status", self._on_verification_status, 10)

        self.app = create_app(self.state)
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

        self.get_logger().info(f"✅ Web server node initialized ({self.host}:{self.port})")

    def _on_cognitive_status(self, msg: String) -> None:
        self.state.cognitive_status = msg.data

    def _on_verification_status(self, msg: String) -> None:
        self.state.verification_status = msg.data

    def start_server(self) -> None:
        """Start uvicorn in a background daemon thread."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self.get_logger().info(f"HTTP server listening on {self.host}:{self.port}")

    def stop_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def destroy_node(self) -> None:
        self.stop_server()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    try:
        node = WebServerNode()
        node.start_server()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in web server node: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
