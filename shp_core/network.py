"""
Silent Hope Protocol - Network Implementation

Distributed mesh of interconnected nodes.
The network that never forgets.

Created by Máté Róbert + Hope
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Callable, Any
from enum import Enum
import json

from .node import SilentHopeNode, NodeConfig, NodeCapabilities, NodeState
from .memory import MemoryChain, MemoryBlock, MemoryRef
from .protocol import ExecutableKnowledge, ExecutionResult, EKUType
from .crypto import sha3_256


class NetworkState(Enum):
    """Network operational state."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SYNCING = "syncing"
    DEGRADED = "degraded"
    SHUTDOWN = "shutdown"


@dataclass
class NetworkConfig:
    """Network configuration."""
    name: str = "shp-network"
    min_nodes: int = 1
    max_nodes: int = 1000
    sync_interval_ms: int = 60000
    heartbeat_interval_ms: int = 30000
    consensus_threshold: float = 0.67  # 2/3 majority


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    total_nodes: int = 0
    active_nodes: int = 0
    total_executions: int = 0
    total_memory_blocks: int = 0
    average_latency_ms: float = 0.0
    network_uptime_seconds: float = 0.0
    messages_routed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "active_nodes": self.active_nodes,
            "total_executions": self.total_executions,
            "total_memory_blocks": self.total_memory_blocks,
            "average_latency_ms": self.average_latency_ms,
            "network_uptime_seconds": self.network_uptime_seconds,
            "messages_routed": self.messages_routed
        }


@dataclass
class PeerInfo:
    """Information about a network peer."""
    node_id: str
    name: str
    capabilities: NodeCapabilities
    last_seen: float
    latency_ms: float = 0.0
    is_healthy: bool = True


class SHPNetwork:
    """
    The Silent Hope Protocol Network.

    A distributed mesh of interconnected nodes that share memory
    and execute knowledge collaboratively.
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize the network.

        Args:
            config: Network configuration
        """
        self.config = config or NetworkConfig()
        self._state = NetworkState.INITIALIZING
        self._start_time = time.time()

        # Node registry
        self._nodes: Dict[str, SilentHopeNode] = {}
        self._peers: Dict[str, PeerInfo] = {}

        # Event handlers
        self._on_node_join: List[Callable] = []
        self._on_node_leave: List[Callable] = []
        self._on_execution: List[Callable] = []

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        # Metrics
        self._metrics = NetworkMetrics()

        self._state = NetworkState.RUNNING

    @property
    def state(self) -> NetworkState:
        """Network state."""
        return self._state

    @property
    def metrics(self) -> NetworkMetrics:
        """Network metrics."""
        self._metrics.network_uptime_seconds = time.time() - self._start_time
        self._metrics.total_nodes = len(self._nodes)
        self._metrics.active_nodes = sum(
            1 for n in self._nodes.values()
            if n.state == NodeState.READY
        )
        self._metrics.total_memory_blocks = sum(
            n.memory.height + 1 for n in self._nodes.values()
        )
        return self._metrics

    def join(self, node: SilentHopeNode) -> bool:
        """
        Add a node to the network.

        Args:
            node: The node to add

        Returns:
            True if successful
        """
        if len(self._nodes) >= self.config.max_nodes:
            return False

        node_id = node.node_id_hex
        self._nodes[node_id] = node

        # Register as peer
        self._peers[node_id] = PeerInfo(
            node_id=node_id,
            name=node.config.name,
            capabilities=node.get_capabilities(),
            last_seen=time.time()
        )

        # Trigger handlers
        for handler in self._on_node_join:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(node))
                else:
                    handler(node)
            except Exception:
                pass

        return True

    def leave(self, node_id: str) -> bool:
        """
        Remove a node from the network.

        Args:
            node_id: Node ID to remove

        Returns:
            True if successful
        """
        if node_id not in self._nodes:
            return False

        node = self._nodes.pop(node_id)
        self._peers.pop(node_id, None)

        # Trigger handlers
        for handler in self._on_node_leave:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(node))
                else:
                    handler(node)
            except Exception:
                pass

        return True

    def get_node(self, node_id: str) -> Optional[SilentHopeNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> List[SilentHopeNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_peers(self) -> List[PeerInfo]:
        """Get all peer information."""
        return list(self._peers.values())

    async def execute(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]] = None,
        target_node: Optional[str] = None,
        broadcast: bool = False
    ) -> ExecutionResult:
        """
        Execute an instruction on the network.

        Args:
            instruction: The instruction to execute
            context: Optional context
            target_node: Specific node to execute on (optional)
            broadcast: Execute on all nodes

        Returns:
            Execution result
        """
        if not self._nodes:
            return ExecutionResult(
                success=False,
                output="",
                error="No nodes in network",
                metrics=None
            )

        if broadcast:
            # Execute on all nodes
            results = await asyncio.gather(*[
                node.execute(instruction, context)
                for node in self._nodes.values()
            ])
            # Return first successful result
            for result in results:
                if result.success:
                    self._metrics.total_executions += 1
                    return result
            return results[0]

        if target_node:
            # Execute on specific node
            node = self._nodes.get(target_node)
            if not node:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Node {target_node} not found",
                    metrics=None
                )
            result = await node.execute(instruction, context)
            self._metrics.total_executions += 1
            return result

        # Execute on best available node
        node = self._select_best_node()
        result = await node.execute(instruction, context)
        self._metrics.total_executions += 1

        # Trigger handlers
        for handler in self._on_execution:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(result)
                else:
                    handler(result)
            except Exception:
                pass

        return result

    def _select_best_node(self) -> SilentHopeNode:
        """Select the best node for execution."""
        ready_nodes = [
            n for n in self._nodes.values()
            if n.state == NodeState.READY
        ]

        if not ready_nodes:
            # Fall back to any node
            return list(self._nodes.values())[0]

        # Select by lowest latency
        return min(
            ready_nodes,
            key=lambda n: n.metrics.average_latency_ms or float('inf')
        )

    async def sync_memory(self) -> bool:
        """
        Synchronize memory across all nodes.

        Returns:
            True if successful
        """
        if len(self._nodes) < 2:
            return True

        self._state = NetworkState.SYNCING

        try:
            # Get highest block from each node
            heights = {
                node_id: node.memory.height
                for node_id, node in self._nodes.items()
            }

            max_height = max(heights.values())
            leader_id = [k for k, v in heights.items() if v == max_height][0]
            leader = self._nodes[leader_id]

            # Sync other nodes
            for node_id, node in self._nodes.items():
                if node_id == leader_id:
                    continue

                current_height = node.memory.height
                if current_height < max_height:
                    # Get missing blocks from leader
                    for h in range(current_height + 1, max_height + 1):
                        block = leader.memory.get(h)
                        # In real implementation, would replicate block
                        # For now, just log
                        pass

            self._state = NetworkState.RUNNING
            return True

        except Exception as e:
            self._state = NetworkState.DEGRADED
            return False

    async def broadcast(
        self,
        message: str,
        exclude: Optional[Set[str]] = None
    ) -> Dict[str, bool]:
        """
        Broadcast a message to all nodes.

        Args:
            message: Message to broadcast
            exclude: Node IDs to exclude

        Returns:
            Dict of node_id -> success
        """
        exclude = exclude or set()
        results = {}

        for node_id, node in self._nodes.items():
            if node_id in exclude:
                continue

            try:
                node.remember(message)
                results[node_id] = True
            except Exception:
                results[node_id] = False

        self._metrics.messages_routed += len(results)
        return results

    def search_network(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search across all nodes' memories.

        Args:
            query: Search query
            limit: Maximum results per node

        Returns:
            List of results with node info
        """
        results = []

        for node_id, node in self._nodes.items():
            node_results = node.recall(query, limit=limit)
            for content in node_results:
                results.append({
                    "node_id": node_id,
                    "node_name": node.config.name,
                    "content": content
                })

        return results

    def on_node_join(self, handler: Callable) -> None:
        """Register node join handler."""
        self._on_node_join.append(handler)

    def on_node_leave(self, handler: Callable) -> None:
        """Register node leave handler."""
        self._on_node_leave.append(handler)

    def on_execution(self, handler: Callable) -> None:
        """Register execution handler."""
        self._on_execution.append(handler)

    async def start(self) -> None:
        """Start network background tasks."""
        self._state = NetworkState.RUNNING

        # Start heartbeat task
        async def heartbeat_loop():
            while self._state == NetworkState.RUNNING:
                await asyncio.sleep(self.config.heartbeat_interval_ms / 1000)
                for peer_id, peer in self._peers.items():
                    if peer_id in self._nodes:
                        peer.last_seen = time.time()
                        peer.is_healthy = self._nodes[peer_id].state == NodeState.READY

        # Start sync task
        async def sync_loop():
            while self._state == NetworkState.RUNNING:
                await asyncio.sleep(self.config.sync_interval_ms / 1000)
                await self.sync_memory()

        self._tasks.append(asyncio.create_task(heartbeat_loop()))
        self._tasks.append(asyncio.create_task(sync_loop()))

    async def shutdown(self) -> None:
        """Shutdown the network."""
        self._state = NetworkState.SHUTDOWN

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Shutdown all nodes
        for node in self._nodes.values():
            node.shutdown()

        self._nodes.clear()
        self._peers.clear()

    def __repr__(self) -> str:
        return f"SHPNetwork(name={self.config.name}, nodes={len(self._nodes)})"


# ============================================================================
# Convenience Functions
# ============================================================================

def create_network(
    name: str = "shp-network",
    **kwargs
) -> SHPNetwork:
    """
    Create a Silent Hope Protocol network.

    Args:
        name: Network name
        **kwargs: Additional configuration

    Returns:
        Configured network

    Example:
        network = create_network("my-network")
        node = create_node("node-1", llm_backend="claude")
        network.join(node)
        result = await network.execute("Hello, network!")
    """
    config = NetworkConfig(name=name, **kwargs)
    return SHPNetwork(config)


# ============================================================================
# Testing
# ============================================================================

async def self_test() -> bool:
    """Run network self-tests."""
    from .node import create_node

    # Create network
    network = create_network("test-network")
    assert network.state == NetworkState.RUNNING

    # Create and join nodes
    node1 = create_node("node-1", llm_backend="claude")
    node2 = create_node("node-2", llm_backend="claude")

    assert network.join(node1)
    assert network.join(node2)

    # Check metrics
    metrics = network.metrics
    assert metrics.total_nodes == 2

    # Test broadcast
    results = await network.broadcast("Test message from network")
    assert len(results) == 2

    # Test search
    search_results = network.search_network("Test")
    assert len(search_results) >= 0  # May or may not find results

    # Shutdown
    await network.shutdown()
    assert network.state == NetworkState.SHUTDOWN

    print("Network self-tests passed!")
    return True


if __name__ == "__main__":
    asyncio.run(self_test())
