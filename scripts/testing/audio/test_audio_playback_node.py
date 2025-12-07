#!/usr/bin/env python3
"""
Quick test script for the streamlined audio_playback_node.

Tests:
1. Node initialization
2. TTS synthesis
3. Notification sounds
4. Priority queue behavior
"""

import subprocess
import time


def run_command(cmd, description, wait=2):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True, capture_output=False)
    time.sleep(wait)

    return result.returncode == 0


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║   Audio Playback Node - Quick Test Script                   ║
╚══════════════════════════════════════════════════════════════╝

This script will test the streamlined audio_playback_node with:
  • TTS synthesis
  • Notification sounds
  • Priority queue behavior

Make sure:
  ✓ audio_playback_node is running
  ✓ USB speaker is connected
  ✓ ROS2 workspace is sourced

Press Enter to continue (or Ctrl+C to cancel)...
"""
    )

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nTest cancelled.")
        return

    # Test 1: Check node is running
    print("\n" + "=" * 60)
    print("Checking if audio_playback_node is running...")
    print("=" * 60)
    result = subprocess.run(["ros2", "node", "list"], capture_output=True, text=True)

    if "/audio_playback_node" not in result.stdout:
        print("\n❌ ERROR: audio_playback_node is not running!")
        print("\nStart it first:")
        print("  ros2 run audio_interface_nodes audio_playback_node")
        return
    else:
        print("✓ audio_playback_node is running")

    time.sleep(1)

    # Test 2: TTS Short Message
    run_command(
        """ros2 topic pub --once /audio/tts_request std_msgs/String \\
           "data: 'Hello, this is a short test message'"
        """,
        "TTS Test 1: Short message",
        wait=3,
    )

    # Test 3: TTS Long Message
    run_command(
        """ros2 topic pub --once /audio/tts_request std_msgs/String \\
           "data: 'This is a longer text to speech message to test the synthesis quality and \
            latency. The robot should speak clearly and naturally.'"
        """,
        "TTS Test 2: Long message",
        wait=5,
    )

    # Test 4: Wake Word Notification
    run_command(
        """ros2 topic pub --once /audio/events robot_interfaces/AudioEvent \\
           "header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}
            event_type: 'wake_word_detected'
            data: ''
            confidence: 0.9
            duration: 0.0
            device_id: ''"
        """,
        "Notification Test 1: Wake word detected (ascending tone)",
        wait=2,
    )

    # Test 5: Speech End Notification
    run_command(
        """ros2 topic pub --once /audio/events robot_interfaces/AudioEvent \\
           "header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}
            event_type: 'speech_ended'
            data: ''
            confidence: 0.0
            duration: 2.5
            device_id: ''"
        """,
        "Notification Test 2: Speech ended (descending tone)",
        wait=2,
    )

    # Test 6: Priority Interruption
    print(f"\n{'='*60}")
    print("TEST: Priority interruption (notification interrupts TTS)")
    print(f"{'='*60}")
    print("Starting long TTS message...")

    # Start long TTS in background
    subprocess.Popen(
        [
            "ros2",
            "topic",
            "pub",
            "--once",
            "/audio/tts_request",
            "std_msgs/String",
            "data: 'This is a very long text to speech message that will take several seconds to \
                complete. It is designed to test the priority queue interruption behavior when a \
                    high priority notification arrives during playback. The notification should \
                        interrupt this message.'",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(2)

    print("Sending wake word notification to interrupt...")
    subprocess.run(
        [
            "ros2",
            "topic",
            "pub",
            "--once",
            "/audio/events",
            "robot_interfaces/AudioEvent",
            "header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}",
            "event_type: 'wake_word_detected'",
            "data: ''",
            "confidence: 0.9",
            "duration: 0.0",
            "device_id: ''",
        ],
        stdout=subprocess.DEVNULL,
    )

    print("✓ Notification sent (should hear ascending tone interrupt TTS)")
    time.sleep(5)

    # Summary
    print(f"\n\n{'='*60}")
    print("TESTS COMPLETE")
    print(f"{'='*60}")
    print(
        """
Results:
  ✓ TTS synthesis tested
  ✓ Notification sounds tested
  ✓ Priority queue tested

Expected observations:
  1. TTS messages should be clear and natural
  2. Notification tones should play immediately
  3. Wake word notification should interrupt TTS

Check the node logs for detailed information:
  ros2 topic echo /rosout | grep audio_playback_node
"""
    )


if __name__ == "__main__":
    main()
