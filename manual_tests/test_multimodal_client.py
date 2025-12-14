#!/usr/bin/env python3
"""
Simple client to test the Multimodal LLM Node

This script demonstrates how to interact with the Gemma 3n E2B multimodal node
for text, image, and multimodal queries.

Usage:
    python3 test_multimodal_client.py
"""

import asyncio

# Add project root to path for imports
import sys
import time
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.robot_interfaces.msg import MultimodalQuery, MultimodalResponse  # noqa: E402


class MultimodalTestClient(Node):
    """Test client for the multimodal LLM node."""

    def __init__(self):
        super().__init__("multimodal_test_client")

        # Create publishers and subscribers
        self.query_publisher = self.create_publisher(
            MultimodalQuery, "/cognitive/multimodal_query", 10
        )

        self.response_subscription = self.create_subscription(
            MultimodalResponse,
            "/cognitive/multimodal_response",
            self.response_callback,
            10,
        )

        self.status_subscription = self.create_subscription(
            String, "/cognitive/llm_status", self.status_callback, 10
        )

        # Track pending queries
        self.pending_queries = {}

        self.get_logger().info("Multimodal Test Client initialized")

    def response_callback(self, msg: MultimodalResponse):
        """Handle response from multimodal LLM."""
        query_id = msg.query_id

        if query_id in self.pending_queries:
            self.get_logger().info(f"✅ Response received for query: {query_id}")
            self.get_logger().info(f"📝 Response: {msg.response_text}")
            self.get_logger().info(f"⏱️  Processing time: {msg.processing_time:.2f}s")
            self.get_logger().info(f"🎯 Confidence: {msg.confidence:.2f}")

            # Mark query as completed
            self.pending_queries[query_id]["completed"] = True
            self.pending_queries[query_id]["response"] = msg.response_text

        else:
            self.get_logger().warn(f"Received response for unknown query: {query_id}")

    def status_callback(self, msg: String):
        """Handle status updates from multimodal LLM."""
        self.get_logger().info(f"📊 LLM Status: {msg.data}")

    async def send_text_query(self, text: str) -> str:
        """Send a text-only query."""
        query_id = str(uuid.uuid4())

        query_msg = MultimodalQuery()
        query_msg.header.stamp = self.get_clock().now().to_msg()
        query_msg.query_id = query_id
        query_msg.text_query = text
        query_msg.include_current_image = False
        query_msg.temperature = 0.7
        query_msg.max_tokens = 100
        query_msg.use_optimizations = True

        self.pending_queries[query_id] = {
            "query": text,
            "completed": False,
            "response": None,
            "start_time": time.time(),
        }

        self.get_logger().info(f"📤 Sending text query: {text}")
        self.query_publisher.publish(query_msg)

        # Wait for response with timeout
        timeout = 30.0  # 30 seconds
        start_time = time.time()

        while not self.pending_queries[query_id]["completed"]:
            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.01)

            if time.time() - start_time > timeout:
                self.get_logger().error(f"❌ Query timeout: {query_id}")
                return "Error: Query timeout"

        return self.pending_queries[query_id]["response"]

    async def send_image_text_query(self, text: str) -> str:
        """Send a multimodal query with current image."""
        query_id = str(uuid.uuid4())

        query_msg = MultimodalQuery()
        query_msg.header.stamp = self.get_clock().now().to_msg()
        query_msg.query_id = query_id
        query_msg.text_query = text
        query_msg.include_current_image = True
        query_msg.temperature = 0.7
        query_msg.max_tokens = 150
        query_msg.use_optimizations = True

        self.pending_queries[query_id] = {
            "query": text,
            "completed": False,
            "response": None,
            "start_time": time.time(),
        }

        self.get_logger().info(f"📤 Sending image-text query: {text}")
        self.query_publisher.publish(query_msg)

        # Wait for response with timeout
        timeout = 45.0  # 45 seconds for image processing
        start_time = time.time()

        while not self.pending_queries[query_id]["completed"]:
            await asyncio.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.01)

            if time.time() - start_time > timeout:
                self.get_logger().error(f"❌ Query timeout: {query_id}")
                return "Error: Query timeout"

        return self.pending_queries[query_id]["response"]

    async def run_demo_queries(self):
        """Run a series of demo queries."""
        self.get_logger().info("🚀 Starting multimodal LLM demo...")

        # Wait a bit for the LLM node to initialize
        await asyncio.sleep(2.0)

        # Test 1: Simple text query
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("🧪 Test 1: Simple text query")
        self.get_logger().info("=" * 60)

        response1 = await self.send_text_query(
            "Hello! Please introduce yourself and explain what you can do."
        )

        # Test 2: Knowledge query
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("🧪 Test 2: Knowledge query")
        self.get_logger().info("=" * 60)

        response2 = await self.send_text_query(
            "Explain the main components of a robotic system in simple terms."
        )

        # Test 3: Multimodal query (requires camera)
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("🧪 Test 3: Multimodal query (with image)")
        self.get_logger().info("=" * 60)

        response3 = await self.send_image_text_query(
            "What do you see in the current camera view? Describe the scene in detail."
        )

        # Test 4: Complex reasoning
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("🧪 Test 4: Complex reasoning")
        self.get_logger().info("=" * 60)

        response4 = await self.send_text_query(
            "If you were an AI robot assistant, what would be the most important "
            "features to help humans in their daily lives?"
        )

        # Summary
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info("📊 Demo Summary")
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"Test 1 Response Length: {len(response1) if response1 else 0} chars"
        )
        self.get_logger().info(
            f"Test 2 Response Length: {len(response2) if response2 else 0} chars"
        )
        self.get_logger().info(
            f"Test 3 Response Length: {len(response3) if response3 else 0} chars"
        )
        self.get_logger().info(
            f"Test 4 Response Length: {len(response4) if response4 else 0} chars"
        )

        completed_queries = len([q for q in self.pending_queries.values() if q["completed"]])
        total_queries = len(self.pending_queries)

        self.get_logger().info(f"Completed queries: {completed_queries}/{total_queries}")

        if completed_queries == total_queries:
            self.get_logger().info("🎉 All tests completed successfully!")
        else:
            self.get_logger().warn(f"⚠️  {total_queries - completed_queries} queries failed")


async def main():
    """Main function for the test client."""
    rclpy.init()

    try:
        client = MultimodalTestClient()

        # Run the demo
        await client.run_demo_queries()

        # Keep the node alive for a bit to receive any remaining responses
        await asyncio.sleep(2.0)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error running test client: {e}")
    finally:
        if "client" in locals():
            client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
