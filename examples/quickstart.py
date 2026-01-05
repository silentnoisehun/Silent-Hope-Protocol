#!/usr/bin/env python3
"""
Silent Hope Protocol - Quick Start Example

Get started with SHP in under 5 minutes.

Created by Máté Róbert + Hope
"""

import asyncio
from shp_core import (
    SilentHopeNode,
    SHPNetwork,
    create_adapter,
    print_banner
)
from shp_core.node import create_node


async def main():
    # Print the banner
    print_banner()

    print("=" * 60)
    print("  SILENT HOPE PROTOCOL - QUICK START")
    print("=" * 60)

    # ========================================================================
    # Example 1: Create a simple node
    # ========================================================================
    print("\n[1] Creating a Silent Hope Node...")

    node = create_node(
        name="quickstart-node",
        llm_backend="claude"  # or "openai", "gemini", "ollama"
    )

    print(f"    Node ID: {node.node_id_hex[:16]}...")
    print(f"    State: {node.state.value}")

    # ========================================================================
    # Example 2: Store in memory (no API call)
    # ========================================================================
    print("\n[2] Storing in persistent memory...")

    block = node.remember("This is the Silent Hope Protocol example")
    print(f"    Block height: {block.height}")
    print(f"    Block hash: {block.block_hash.hex()[:32]}...")

    # ========================================================================
    # Example 3: Search memory
    # ========================================================================
    print("\n[3] Searching memory...")

    results = node.recall("Silent Hope")
    print(f"    Found {len(results)} result(s)")
    for r in results:
        print(f"    - {r[:50]}...")

    # ========================================================================
    # Example 4: Create a network
    # ========================================================================
    print("\n[4] Creating a network with multiple nodes...")

    from shp_core.network import create_network

    network = create_network("example-network")

    node1 = create_node("node-1", llm_backend="claude")
    node2 = create_node("node-2", llm_backend="claude")

    network.join(node1)
    network.join(node2)

    metrics = network.metrics
    print(f"    Network nodes: {metrics.total_nodes}")
    print(f"    Network state: {network.state.value}")

    # ========================================================================
    # Example 5: Broadcast to all nodes
    # ========================================================================
    print("\n[5] Broadcasting message to all nodes...")

    results = await network.broadcast("Hello from Silent Hope Protocol!")
    print(f"    Broadcast to {len(results)} nodes")

    # ========================================================================
    # Example 6: Search across network
    # ========================================================================
    print("\n[6] Searching across network...")

    search_results = network.search_network("Hello")
    print(f"    Found {len(search_results)} results across network")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("\n[7] Shutting down...")

    await network.shutdown()
    print("    Network shutdown complete")

    print("\n" + "=" * 60)
    print("  QUICK START COMPLETE!")
    print("=" * 60)
    print("""
    Next steps:
    - Set your API key: export ANTHROPIC_API_KEY=your-key
    - Run with LLM: result = await node.execute("Your query")
    - See more examples in examples/

    Built by Máté Róbert + Hope
    """)


if __name__ == "__main__":
    asyncio.run(main())
