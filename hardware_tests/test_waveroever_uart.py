#!/usr/bin/env python3
"""Wave Rover UART test script.

Sends and receives JSON commands over serial. Designed to run on Jetson with
the Wave Rover connected to /dev/ttyTHS1 by default. Supports interactive mode
and an automated test sequence.

Usage:
  interactive:
    python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1

  automated:
    python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --auto

This script follows the project's guidelines: validate hardware connections,
handle errors, and keep outputs timestamped for logs.

For complete JSON command reference, see:
  docs/guides/wave_rover_json_commands.md
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from typing import Optional

import serial

logger = logging.getLogger("waverover_uart")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)


def read_serial(ser: serial.Serial, stop_event: threading.Event, out_q: queue.Queue) -> None:
    """Continuously read lines from serial, log them and push into out_q.

    out_q is a Queue used by test functions to assert responses.
    """
    while not stop_event.is_set():
        try:
            raw = ser.readline()
            if not raw:
                continue
            try:
                text = raw.decode("utf-8").strip()
            except Exception:
                logger.warning("Received non-UTF8 bytes: %s", raw)
                continue

            if not text:
                continue

            try:
                payload = json.loads(text)
                logger.info("Received JSON: %s", json.dumps(payload))
                # push json to queue for tests
                try:
                    out_q.put_nowait(payload)
                except queue.Full:
                    pass
            except json.JSONDecodeError:
                logger.info("Received: %s", text)
                try:
                    out_q.put_nowait(text)
                except queue.Full:
                    pass
        except serial.SerialException as e:
            logger.error("Serial read error: %s", e)
            break
        except Exception as e:  # pragma: no cover - robustness
            logger.exception("Unexpected error in read loop: %s", e)
            break


def send_command(ser: serial.Serial, cmd: dict) -> bool:
    """Send a JSON command over serial. Returns True on success."""
    try:
        line = json.dumps(cmd) + "\n"
        ser.write(line.encode("utf-8"))
        ser.flush()
        logger.info("Sent: %s", line.strip())
        return True
    except serial.SerialException as e:
        logger.error("Serial write error: %s", e)
        return False
    except Exception:  # pragma: no cover - robustness
        logger.exception("Unexpected error while sending command")
        return False


def wait_for_response(out_q: queue.Queue, predicate, timeout: float = 2.0):
    """Wait up to timeout seconds for an item in out_q matching predicate.

    predicate(item) -> bool
    Returns the matching item or None.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            item = out_q.get(timeout=deadline - time.time())
        except queue.Empty:
            break
        try:
            if predicate(item):
                return item
        finally:
            # continue consuming until timeout or match
            pass
    return None


def run_motor_tests(ser: serial.Serial, out_q: queue.Queue) -> bool:
    """Send forward/back/turn commands and check for any response.

    Returns True if all commands were sent successfully. Note: actual
    verification depends on rover responses; this performs a basic RTT check.
    """
    # Use device-documented CMD_SPEED_CTRL: {"T":1,"L":<left_speed>,"R":<right_speed>}
    # Speed range: -0.5 .. +0.5
    cmds = [
        {"T": 1, "L": 0.2, "R": 0.2},
        {"T": 1, "L": -0.2, "R": -0.2},
        # Turn in place: left forward, right backward
        {"T": 1, "L": 0.2, "R": -0.2},
    ]
    success = True
    for c in cmds:
        ok = send_command(ser, c)
        if not ok:
            success = False
        # wait briefly for any kind of echo/ack
        resp = wait_for_response(
            out_q, lambda it: isinstance(it, dict) or isinstance(it, str), timeout=1.0
        )
        logger.info("Motor test command %s -> resp=%s", c, resp)
    # Also test PWM input (debugging command)
    success = success and run_pwm_test(ser, out_q)
    # Test ROS-style control (if supported)
    success = success and run_ros_ctrl_test(ser, out_q)
    # Test PID setting command (UGV01 only; safe to send)
    success = success and run_pid_test(ser, out_q)
    return success


def run_pwm_test(ser: serial.Serial, out_q: queue.Queue) -> bool:
    """Send CMD_PWM_INPUT: {"T":11, "L":<pwm>, "R":<pwm>}"""
    # Use modest PWM values to avoid high-speed motion during tests
    req = {"T": 11, "L": 100, "R": 100}
    if not send_command(ser, req):
        return False
    item = wait_for_response(
        out_q, lambda it: isinstance(it, dict) or isinstance(it, str), timeout=2.0
    )
    logger.info("PWM test response: %s", item)
    return True


def run_ros_ctrl_test(ser: serial.Serial, out_q: queue.Queue) -> bool:
    """Send CMD_ROS_CTRL: {"T":13, "X":<linear m/s>, "Z":<angular rad/s>}.

    Note: only applicable if device supports ROS control (UGV01)."""
    req = {"T": 13, "X": 0.05, "Z": 0.1}
    if not send_command(ser, req):
        return False
    item = wait_for_response(
        out_q,
        lambda it: isinstance(it, dict) and (it.get("T") == 13 or "ros" in json.dumps(it).lower()),
        timeout=2.0,
    )
    logger.info("ROS ctrl test response: %s", item)
    return True


def run_pid_test(ser: serial.Serial, out_q: queue.Queue) -> bool:
    """Send Setting Motor PID: {"T":2,"P":...,"I":...,"D":...,"L":...}

    This is intended as a safe write; firmware should accept or ignore if unsupported.
    """
    req = {"T": 2, "P": 200, "I": 2500, "D": 0, "L": 255}
    if not send_command(ser, req):
        return False
    item = wait_for_response(
        out_q, lambda it: isinstance(it, dict) or isinstance(it, str), timeout=2.0
    )
    logger.info("PID set test response: %s", item)
    return True


def run_imu_test(ser: serial.Serial, out_q: queue.Queue) -> bool:
    """Request IMU data with {'T':126} and wait for JSON response containing 'imu' or similar."""
    req = {"T": 126}
    if not send_command(ser, req):
        return False

    def _imu_pred(it):
        return isinstance(it, dict) and ("imu" in it or "T" in it)

    item = wait_for_response(out_q, _imu_pred, timeout=3.0)
    logger.info("IMU test response: %s", item)
    return item is not None


def run_continuous_feedback_test(ser: serial.Serial, out_q: queue.Queue) -> bool:
    """Enable continuous feedback with {'T':131,'cmd':1}\
          and expect at least one periodic message."""
    req = {"T": 131, "cmd": 1}
    if not send_command(ser, req):
        return False
    # wait a bit longer for periodic feedback
    item = wait_for_response(
        out_q, lambda it: isinstance(it, dict) and it.get("T") == 131, timeout=5.0
    )
    logger.info("Continuous feedback test response: %s", item)
    return item is not None


def run_oled_test(ser: serial.Serial, out_q: queue.Queue) -> bool:
    """Send an OLED display command and look for ack/response."""
    # Use documented OLED Screen Control: {"T":3, "lineNum":0, "Text":"..."}
    req = {"T": 3, "lineNum": 0, "Text": "Hello from test"}
    if not send_command(ser, req):
        return False
    item = wait_for_response(
        out_q,
        lambda it: isinstance(it, dict) or (isinstance(it, str) and "OLED" in it.upper()),
        timeout=2.0,
    )
    logger.info("OLED test response: %s", item)
    return item is not None


def run_automated_tests(
    ser: serial.Serial, out_q: queue.Queue, which: Optional[str] = None
) -> None:
    """Run automated tests. which can be 'motor','imu','feedback','oled' or 'all'."""
    logger.info("Starting automated tests (%s)", which or "all")
    ok = True
    targets = [which] if which and which != "all" else ["motor", "imu", "feedback", "oled"]
    for t in targets:
        if t == "motor":
            ok = ok and run_motor_tests(ser, out_q)
        elif t == "imu":
            ok = ok and run_imu_test(ser, out_q)
        elif t == "feedback":
            ok = ok and run_continuous_feedback_test(ser, out_q)
        elif t == "oled":
            ok = ok and run_oled_test(ser, out_q)
        else:
            logger.warning("Unknown automated test: %s", t)

    logger.info("Automated tests complete. success=%s", ok)


def main() -> None:
    EXAMPLES = (
        "Examples:\n"
        "  # Interactive mode (type JSON or raw lines)\n"
        "  python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1\n\n"
        "  # Run the full automated sequence and exit\n"
        "  python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --auto\n\n"
        "  # Run only the motor test (single-shot)\n"
        "  python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test motor\n\n"
        "  # Run only the IMU test\n"
        "  python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test imu\n\n"
        "  # Turn on serial echo before running tests:\n"
        '  # (use device docs: {"T":143,"cmd":1} to enable echo)\n'
        '  # (use device docs: {"T":131,"cmd":1} to enable continuous feedback)\n'
    )

    parser = argparse.ArgumentParser(
        description="Wave Rover UART test (JSON)",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyTHS1",
        help="Serial port (default: /dev/ttyTHS1)",
    )
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--auto", action="store_true", help="Run automated test sequence and exit")
    parser.add_argument(
        "--test",
        type=str,
        choices=["motor", "imu", "feedback", "oled", "all"],
        help=(
            "Run only one automated test (motor, imu, feedback, oled, all). "
            "If omitted, the full sequence runs when --auto is set."
        ),
    )
    parser.add_argument("--timeout", type=float, default=1.0, help="Read timeout (seconds)")

    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, baudrate=args.baud, timeout=args.timeout)
    except serial.SerialException as e:
        logger.error("Cannot open serial port %s: %s", args.port, e)
        return

    # Recommended: disable hardware flow control lines unless needed
    try:
        ser.setRTS(False)
        ser.setDTR(False)
    except Exception:
        # Not all serial implementations support these methods
        pass

    # Queue used to deliver parsed serial responses to test routines
    out_q: queue.Queue = queue.Queue(maxsize=200)

    stop_event = threading.Event()
    reader = threading.Thread(target=read_serial, args=(ser, stop_event, out_q), daemon=True)
    reader.start()

    try:
        # Allow running all tests (--auto) or a single test (--test).
        if args.auto or args.test:
            run_automated_tests(ser, out_q, args.test)
        else:
            logger.info("Interactive mode. Type JSON commands or plain text. Ctrl-C to exit.")
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    continue
                # Try to parse JSON; if not valid, send as raw line
                try:
                    obj = json.loads(line)
                    send_command(ser, obj)
                except json.JSONDecodeError:
                    # send raw line
                    try:
                        ser.write(line.encode("utf-8") + b"\n")
                        ser.flush()
                        logger.info("Sent raw: %s", line)
                    except Exception as e:
                        logger.error("Failed to send raw line: %s", e)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_event.set()
        reader.join(timeout=1.0)
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
