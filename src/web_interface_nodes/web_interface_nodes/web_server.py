#!/usr/bin/env python3
"""
Monitoring web server for the robot (Plan.md Phase 9.1 / 9.3).

A small FastAPI app that caches lightweight ROS2 status topics and exposes them
alongside host resource metrics, plus a dependency-free HTML dashboard that
polls the JSON API. It builds on the original minimal ``/health`` + ``/status``
server.

Endpoints:
    GET /health            -> liveness probe
    GET /status            -> combined snapshot (subsystems + resources)
    GET /api/subsystems    -> per-subsystem status with staleness
    GET /api/resources     -> CPU / memory / disk / thermals / load
    GET /                  -> HTML dashboard
    GET /dashboard         -> HTML dashboard

The ROS2 node subscribes to existing status topics and the FastAPI app serves
the latest cached values. Uvicorn runs in a background thread while ``rclpy``
spins the node on the main thread.
"""

import glob
import os
import threading
import time
from typing import Any, Dict, List, Optional

import psutil
import rclpy
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from geometry_msgs.msg import PolygonStamped
from rclpy.node import Node
from std_msgs.msg import String

from robot_interfaces.msg import AudioEvent, ChassisState, PerceptionEvent

# Seconds after which a subsystem with no fresh message is flagged stale.
STALE_AFTER_S = 5.0

# Subsystems surfaced by the API (keys map to ``<name>_status`` attributes).
SUBSYSTEMS = ("cognitive", "verification", "chassis", "audio", "perception", "obstacles")

# Jetson exposes thermal zones here; also reachable via psutil on some boards.
_THERMAL_ZONE_DIR = "/sys/devices/virtual/thermal"

# Max characters kept for a single cached status line.
_STATUS_MAX_LEN = 120


def read_thermal_zones(base_dir: str = _THERMAL_ZONE_DIR) -> Dict[str, float]:
    """Best-effort read of ``thermal_zone*`` temperatures in Celsius.

    Args:
        base_dir: Directory containing the ``thermal_zone*`` entries.

    Returns:
        Mapping of zone type (or zone name) to temperature in Celsius. Empty if
        the platform exposes no thermal zones or they cannot be read.
    """
    temps: Dict[str, float] = {}
    for zone in sorted(glob.glob(os.path.join(base_dir, "thermal_zone*"))):
        try:
            with open(os.path.join(zone, "type"), "r", encoding="utf-8") as handle:
                name = handle.read().strip()
            with open(os.path.join(zone, "temp"), "r", encoding="utf-8") as handle:
                millidegrees = int(handle.read().strip())
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        temps[name or os.path.basename(zone)] = round(millidegrees / 1000.0, 1)
    return temps


class SystemState:
    """Thread-safe cache of subsystem statuses plus host resource metrics."""

    def __init__(self, stale_after_s: float = STALE_AFTER_S) -> None:
        self.started_at = time.time()
        self.stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._seen: Dict[str, float] = {}
        for name in SUBSYSTEMS:
            setattr(self, f"{name}_status", "unknown")

    def update(self, name: str, value: str) -> None:
        """Record a fresh status line for ``name`` and its timestamp.

        Args:
            name: Subsystem key (must be one of :data:`SUBSYSTEMS`).
            value: Human-readable status string.

        Raises:
            KeyError: If ``name`` is not a known subsystem.
        """
        attr = f"{name}_status"
        if not hasattr(self, attr):
            raise KeyError(f"unknown subsystem: {name}")
        with self._lock:
            setattr(self, attr, value)
            self._seen[name] = time.time()

    def subsystem_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return each subsystem's status, age, and staleness flag."""
        now = time.time()
        snapshot: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for name in SUBSYSTEMS:
                value = getattr(self, f"{name}_status")
                seen = self._seen.get(name)
                age = None if seen is None else round(now - seen, 1)
                snapshot[name] = {
                    "status": value,
                    "age_s": age,
                    "stale": age is None or age > self.stale_after_s,
                }
        return snapshot

    def resource_snapshot(self) -> Dict[str, Any]:
        """Return host resource metrics (CPU, memory, disk, load, thermals)."""
        mem = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage("/")
            disk_data: Dict[str, Any] = {
                "total_gb": round(disk.total / 1024**3, 1),
                "used_percent": disk.percent,
            }
        except (OSError, AttributeError):
            disk_data = {}
        try:
            load_avg: List[float] = [round(value, 2) for value in psutil.getloadavg()]
        except (AttributeError, OSError):
            load_avg = []
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count() or 0,
            "load_avg": load_avg,
            "memory": {
                "total_mb": round(mem.total / 1024 / 1024, 1),
                "available_mb": round(mem.available / 1024 / 1024, 1),
                "used_mb": round((mem.total - mem.available) / 1024 / 1024, 1),
                "percent": mem.percent,
            },
            "disk": disk_data,
            "temperatures_c": read_thermal_zones(),
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return the combined subsystem + resource snapshot."""
        data: Dict[str, Any] = {
            "status": "ok",
            "uptime_s": round(time.time() - self.started_at, 1),
            # Flat fields kept for backward compatibility with the original API.
            "cognitive_status": self.cognitive_status,
            "verification_status": self.verification_status,
            "subsystems": self.subsystem_snapshot(),
        }
        data.update(self.resource_snapshot())
        return data


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local AI Robot Assistant</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  header { padding: 16px 24px; background: #161a22; border-bottom: 1px solid #262c38; }
  h1 { font-size: 18px; margin: 0; }
  #updated { font-size: 12px; color: #8b95a5; margin-top: 4px; }
  main { padding: 16px 24px; display: grid; gap: 16px;
         grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
  .card { background: #161a22; border: 1px solid #262c38; border-radius: 10px; padding: 14px; }
  .card h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
             letter-spacing: .05em; color: #8b95a5; }
  .row { display: flex; justify-content: space-between; gap: 12px; padding: 3px 0; }
  .val { font-variant-numeric: tabular-nums; text-align: right; }
  .ok { color: #4ade80; } .stale { color: #fbbf24; } .unknown { color: #8b95a5; }
  .bar { height: 8px; background: #262c38; border-radius: 4px; overflow: hidden; margin-top: 6px; }
  .bar > span { display: block; height: 100%; background: #4ade80; }
  small { color: #8b95a5; }
</style>
</head>
<body>
<header>
  <h1>Local AI Robot Assistant</h1>
  <div id="updated">connecting&hellip;</div>
</header>
<main>
  <section class="card"><h2>System</h2><div id="system"></div></section>
  <section class="card"><h2>Subsystems</h2><div id="subsystems"></div></section>
  <section class="card"><h2>Thermals</h2><div id="thermals"></div></section>
</main>
<script>
function row(key, value, cls) {
  return '<div class="row"><span>' + key + '</span><span class="val ' + (cls || '') +
    '">' + value + '</span></div>';
}
function renderSystem(s) {
  var m = s.memory || {};
  var h = row('CPU', (s.cpu_percent || 0).toFixed(1) + '%');
  h += row('Memory', (m.percent || 0).toFixed(1) + '%');
  h += row('Used / Total', ((m.used_mb || 0) / 1024).toFixed(1) + ' / ' +
    ((m.total_mb || 0) / 1024).toFixed(1) + ' GB');
  h += row('Uptime', s.uptime_s + ' s');
  h += '<div class="bar"><span style="width:' + (m.percent || 0) + '%"></span></div>';
  document.getElementById('system').innerHTML = h;
}
function renderSubsystems(subs) {
  var h = '';
  Object.keys(subs).forEach(function (name) {
    var sub = subs[name];
    var cls = sub.stale ? 'stale' : (sub.status === 'unknown' ? 'unknown' : 'ok');
    var age = sub.age_s === null ? 'never' : sub.age_s + 's ago';
    h += row(name, String(sub.status).slice(0, 80) + ' <small>(' + age + ')</small>', cls);
  });
  document.getElementById('subsystems').innerHTML = h || row('no data', '');
}
function renderThermals(t) {
  var keys = Object.keys(t);
  document.getElementById('thermals').innerHTML = keys.length
    ? keys.map(function (k) { return row(k, t[k] + ' &deg;C'); }).join('')
    : row('none', '-', 'unknown');
}
async function refresh() {
  try {
    var resp = await fetch('/status');
    var s = await resp.json();
    renderSystem(s);
    renderSubsystems(s.subsystems || {});
    renderThermals(s.temperatures_c || {});
    document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch (err) {
    document.getElementById('updated').textContent = 'connection lost: ' + err;
  }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def create_app(state: SystemState) -> FastAPI:
    """Build the FastAPI app around a shared :class:`SystemState`.

    Args:
        state: Shared status cache updated by the ROS2 node.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="Local AI Robot Assistant", version="0.2.0")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> Dict[str, Any]:
        return state.snapshot()

    @app.get("/api/subsystems")
    def subsystems() -> Dict[str, Any]:
        return state.subsystem_snapshot()

    @app.get("/api/resources")
    def resources() -> Dict[str, Any]:
        return state.resource_snapshot()

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD_HTML

    return app


class WebServerNode(Node):
    """ROS2 node that caches status topics and serves them over HTTP."""

    def __init__(self) -> None:
        super().__init__("web_server")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("stale_after_s", STALE_AFTER_S)
        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value
        stale_after_s = self.get_parameter("stale_after_s").value

        self.state = SystemState(stale_after_s=stale_after_s)

        # Status topics (strings)
        self.create_subscription(String, "/cognitive/status", self._on_cognitive_status, 10)
        self.create_subscription(String, "/verification/status", self._on_verification_status, 10)
        # Lightweight typed telemetry
        self.create_subscription(ChassisState, "/chassis_state", self._on_chassis_state, 10)
        self.create_subscription(AudioEvent, "/audio/events", self._on_audio_event, 10)
        self.create_subscription(
            PerceptionEvent, "/perception/events", self._on_perception_event, 10
        )
        self.create_subscription(PolygonStamped, "/perception/obstacles", self._on_obstacles, 10)

        self.app = create_app(self.state)
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

        self.get_logger().info(f"✅ Web server node initialized ({self.host}:{self.port})")

    def _on_cognitive_status(self, msg: String) -> None:
        self.state.update("cognitive", msg.data[:_STATUS_MAX_LEN])

    def _on_verification_status(self, msg: String) -> None:
        self.state.update("verification", msg.data[:_STATUS_MAX_LEN])

    def _on_chassis_state(self, msg: ChassisState) -> None:
        self.state.update(
            "chassis",
            (
                f"battery={msg.battery_voltage:.2f}V "
                f"motors={'on' if msg.motors_enabled else 'off'} "
                f"estop={'on' if msg.emergency_stop_active else 'off'} "
                f"temp={msg.temperature:.1f}C"
            ),
        )

    def _on_audio_event(self, msg: AudioEvent) -> None:
        detail = f" {msg.data}" if msg.data else ""
        self.state.update("audio", f"{msg.event_type}{detail}"[:_STATUS_MAX_LEN])

    def _on_perception_event(self, msg: PerceptionEvent) -> None:
        self.state.update(
            "perception",
            f"{msg.class_name} id={msg.tracking_id} conf={msg.confidence:.2f}",
        )

    def _on_obstacles(self, msg: PolygonStamped) -> None:
        self.state.update("obstacles", f"count={len(msg.polygon.points)}")

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
