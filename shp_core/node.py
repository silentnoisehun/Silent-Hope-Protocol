"""
Silent Hope Protocol - Node Implementation

A SilentHopeNode is an individual participant in the SHP network.
Executes knowledge, maintains memory, communicates with peers.

Created by Máté Róbert + Hope
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .adapter import create_adapter
from .crypto import generate_node_identity
from .memory import InMemoryChain, MemoryBlock, MemoryChain, MemoryRef
from .protocol import ExecutionResult


class NodeState(Enum):
    """Node operational state."""
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    SYNCING = "syncing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class NodeConfig:
    """Node configuration."""
    name: str
    llm_backend: str = "claude"
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    storage_path: Optional[Path] = None
    max_memory_blocks: int = 10000
    auto_sync: bool = True
    heartbeat_interval_ms: int = 30000


@dataclass
class NodeCapabilities:
    """Node capabilities advertisement."""
    node_id: str
    name: str
    llm_backend: str
    llm_model: str
    max_tokens: int
    supports_vision: bool
    supports_tools: bool
    average_latency_ms: float
    memory_height: int
    state: str


@dataclass
class NodeMetrics:
    """Node performance metrics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    memory_writes: int = 0
    uptime_seconds: float = 0.0

    @property
    def average_latency_ms(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_latency_ms / self.total_executions

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "cache_hits": self.cache_hits,
            "memory_writes": self.memory_writes,
            "uptime_seconds": self.uptime_seconds
        }


class SilentHopeNode:
    """
    A node in the Silent Hope Protocol network.

    Executes knowledge, maintains persistent memory,
    communicates with peers using the SHP protocol.
    """

    def __init__(self, config: NodeConfig):
        """
        Initialize a Silent Hope Node.

        Args:
            config: Node configuration
        """
        self.config = config
        self._state = NodeState.INITIALIZING
        self._start_time = time.time()

        # Generate or load identity
        self._keypair = generate_node_identity()

        # Initialize memory chain
        if config.storage_path:
            self._memory = MemoryChain(
                storage_path=config.storage_path / "memory.db",
                keypair=self._keypair
            )
        else:
            self._memory = InMemoryChain(keypair=self._keypair)

        # Initialize LLM adapter
        self._adapter = create_adapter(
            provider=config.llm_backend,
            model=config.llm_model,
            api_key=config.api_key
        )

        # Metrics
        self._metrics = NodeMetrics()

        # Event handlers
        self._on_execute: list[Callable] = []
        self._on_memory_write: list[Callable] = []

        # Sequence counter
        self._sequence = 0

        self._state = NodeState.READY

    @property
    def node_id(self) -> bytes:
        """Node identifier."""
        return self._keypair.node_id

    @property
    def node_id_hex(self) -> str:
        """Node identifier as hex string."""
        return self._keypair.node_id.hex()

    @property
    def public_key(self) -> bytes:
        """Node public key."""
        return self._keypair.public_key

    @property
    def state(self) -> NodeState:
        """Current node state."""
        return self._state

    @property
    def metrics(self) -> NodeMetrics:
        """Node metrics."""
        self._metrics.uptime_seconds = time.time() - self._start_time
        return self._metrics

    @property
    def memory(self) -> MemoryChain:
        """Memory chain."""
        return self._memory

    def get_capabilities(self) -> NodeCapabilities:
        """Get node capabilities for advertisement."""
        adapter_caps = self._adapter.get_capabilities()
        return NodeCapabilities(
            node_id=self.node_id_hex,
            name=self.config.name,
            llm_backend=self.config.llm_backend,
            llm_model=adapter_caps.model,
            max_tokens=adapter_caps.max_tokens,
            supports_vision=adapter_caps.supports_vision,
            supports_tools=adapter_caps.supports_tools,
            average_latency_ms=self._metrics.average_latency_ms or adapter_caps.average_latency_ms,
            memory_height=self._memory.height,
            state=self._state.value
        )

    async def execute(
        self,
        instruction: str,
        context: Optional[dict[str, Any]] = None,
        memory_ref: Optional[str] = None,
        store_result: bool = True
    ) -> ExecutionResult:
        """
        Execute an instruction.

        This is the primary method for interacting with the node.
        The instruction is sent to the LLM adapter, and the result
        is optionally stored in the memory chain.

        Args:
            instruction: The instruction to execute
            context: Optional context dictionary
            memory_ref: Optional memory reference (e.g., "chain:latest")
            store_result: Whether to store the result in memory

        Returns:
            Execution result
        """
        self._state = NodeState.EXECUTING
        self._sequence += 1

        try:
            # Resolve memory reference if provided
            resolved_context = context or {}
            if memory_ref:
                ref = MemoryRef.parse(memory_ref)
                blocks = self._memory.resolve(ref)
                if blocks:
                    resolved_context["memory"] = [
                        block.content.decode('utf-8', errors='replace')
                        for block in blocks
                    ]

            # Execute via adapter
            result = await self._adapter.execute(
                instruction=instruction,
                context=resolved_context,
                memory_ref=memory_ref
            )

            # Update metrics
            self._metrics.total_executions += 1
            self._metrics.total_latency_ms += result.metrics.latency_ms

            if result.success:
                self._metrics.successful_executions += 1

                # Store in memory if requested
                if store_result:
                    content = json.dumps({
                        "instruction": instruction,
                        "response": result.output,
                        "timestamp": time.time()
                    }).encode('utf-8')

                    self._memory.append(content)
                    self._metrics.memory_writes += 1

                    # Trigger handlers
                    for handler in self._on_memory_write:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(self._memory.get_latest())
                            else:
                                handler(self._memory.get_latest())
                        except Exception:
                            pass

            else:
                self._metrics.failed_executions += 1

            if result.metrics.cache_hit:
                self._metrics.cache_hits += 1

            # Trigger handlers
            for handler in self._on_execute:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(result)
                    else:
                        handler(result)
                except Exception:
                    pass

            return result

        finally:
            self._state = NodeState.READY

    async def query(
        self,
        question: str,
        context: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Simple query interface.

        Args:
            question: Question to ask
            context: Optional context

        Returns:
            Answer string
        """
        result = await self.execute(
            instruction=question,
            context=context,
            store_result=False
        )
        return result.output if result.success else f"Error: {result.error}"

    def remember(self, content: str) -> MemoryBlock:
        """
        Store something in memory.

        Args:
            content: Content to remember

        Returns:
            The memory block
        """
        block = self._memory.append(content.encode('utf-8'))
        self._metrics.memory_writes += 1
        return block

    def recall(self, query: str, limit: int = 5) -> list[str]:
        """
        Search memory for content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching content strings
        """
        blocks = self._memory.search(query, limit=limit)
        return [
            block.content.decode('utf-8', errors='replace')
            for block in blocks
        ]

    def on_execute(self, handler: Callable) -> None:
        """Register execution event handler."""
        self._on_execute.append(handler)

    def on_memory_write(self, handler: Callable) -> None:
        """Register memory write event handler."""
        self._on_memory_write.append(handler)

    async def health_check(self) -> bool:
        """Check node health."""
        try:
            return await self._adapter.health_check()
        except Exception:
            return False

    def shutdown(self) -> None:
        """Shutdown the node."""
        self._state = NodeState.SHUTDOWN
        self._memory.close()

    def __repr__(self) -> str:
        return f"SilentHopeNode(name={self.config.name}, id={self.node_id_hex[:8]}...)"


# ============================================================================
# Convenience Functions
# ============================================================================

def create_node(
    name: str,
    llm_backend: str = "claude",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    storage_path: Optional[str] = None
) -> SilentHopeNode:
    """
    Create a Silent Hope Node with minimal configuration.

    Args:
        name: Node name
        llm_backend: LLM provider (claude, openai, gemini, ollama)
        llm_model: Model name (optional)
        api_key: API key (optional, uses environment variable)
        storage_path: Path for persistent storage (optional)

    Returns:
        Configured node

    Example:
        node = create_node("my-node", llm_backend="claude")
        result = await node.execute("What is the meaning of life?")
    """
    config = NodeConfig(
        name=name,
        llm_backend=llm_backend,
        llm_model=llm_model,
        api_key=api_key,
        storage_path=Path(storage_path) if storage_path else None
    )
    return SilentHopeNode(config)


# ============================================================================
# Testing
# ============================================================================

async def self_test() -> bool:
    """Run node self-tests."""

    # Create node config
    config = NodeConfig(
        name="test-node",
        llm_backend="claude"
    )

    # Just test initialization (no API calls)
    node = SilentHopeNode(config)

    assert node.state == NodeState.READY
    assert node.node_id is not None
    assert len(node.node_id) == 16

    # Test capabilities
    caps = node.get_capabilities()
    assert caps.name == "test-node"
    assert caps.llm_backend == "claude"

    # Test memory
    block = node.remember("Test memory entry")
    assert block.height == 0

    results = node.recall("Test")
    assert len(results) == 1

    # Test metrics
    metrics = node.metrics
    assert metrics.memory_writes == 1

    node.shutdown()
    assert node.state == NodeState.SHUTDOWN

    print("Node self-tests passed!")
    return True


if __name__ == "__main__":
    asyncio.run(self_test())
