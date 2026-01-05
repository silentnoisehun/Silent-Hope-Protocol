"""
Silent Hope Protocol - Protocol Tests

Comprehensive tests for ExecutableKnowledge and protocol messages.

Created by Máté Róbert + Hope
"""

import pytest
import time
from shp_core.protocol import (
    ExecutableKnowledge,
    EKUHeader,
    EKUType,
    EKUFlags,
    KnowledgeBlock,
    ExecutionResult,
    PROTOCOL_VERSION
)
from shp_core.crypto import generate_node_identity


class TestEKUHeader:
    """Test EKU header."""

    def test_create_header(self):
        """Test header creation."""
        header = EKUHeader(
            eku_type=EKUType.QUERY,
            sender_id=b"\x00" * 16
        )

        assert header.version == PROTOCOL_VERSION
        assert header.eku_type == EKUType.QUERY
        assert header.timestamp > 0

    def test_header_serialization(self):
        """Test header serialization."""
        header = EKUHeader(
            eku_type=EKUType.EXECUTE,
            flags=EKUFlags.PRIORITY,
            sender_id=b"\x01" * 16,
            sequence=42
        )

        data = header.to_bytes()
        assert len(data) == 64

        restored = EKUHeader.from_bytes(data)
        assert restored.eku_type == EKUType.EXECUTE
        assert restored.flags == EKUFlags.PRIORITY
        assert restored.sequence == 42


class TestKnowledgeBlock:
    """Test knowledge blocks."""

    def test_create_knowledge(self):
        """Test knowledge block creation."""
        knowledge = KnowledgeBlock(
            instruction="Analyze this data",
            context={"source": "test"},
            parameters={"depth": "full"},
            constraints=["no_hallucination"]
        )

        assert knowledge.instruction == "Analyze this data"
        assert knowledge.context["source"] == "test"

    def test_knowledge_serialization(self):
        """Test knowledge serialization."""
        knowledge = KnowledgeBlock(
            instruction="Test instruction",
            context={"key": "value"},
            parameters={"param": 123},
            constraints=["constraint1", "constraint2"]
        )

        data = knowledge.to_bytes()
        restored = KnowledgeBlock.from_bytes(data)

        assert restored.instruction == knowledge.instruction
        assert restored.context == knowledge.context
        assert restored.parameters == knowledge.parameters
        assert restored.constraints == knowledge.constraints


class TestExecutableKnowledge:
    """Test complete EKU."""

    def test_create_query(self):
        """Test creating query EKU."""
        keypair = generate_node_identity()

        eku = ExecutableKnowledge.create_query(
            instruction="What is the Silent Hope Protocol?",
            sender_id=keypair.node_id,
            context={"source": "test"}
        )

        assert eku.header.eku_type == EKUType.QUERY
        assert eku.knowledge.instruction == "What is the Silent Hope Protocol?"

    def test_create_execute(self):
        """Test creating execute EKU."""
        keypair = generate_node_identity()

        eku = ExecutableKnowledge.create_execute(
            instruction="Analyze market data",
            sender_id=keypair.node_id,
            parameters={"depth": "full"},
            constraints=["cite_sources"]
        )

        assert eku.header.eku_type == EKUType.EXECUTE
        assert eku.knowledge.parameters["depth"] == "full"
        assert "cite_sources" in eku.knowledge.constraints

    def test_create_response(self):
        """Test creating response EKU."""
        keypair = generate_node_identity()

        eku = ExecutableKnowledge.create_response(
            result="Analysis complete",
            sender_id=keypair.node_id,
            in_response_to=42
        )

        assert eku.header.eku_type == EKUType.RESPONSE
        assert eku.header.sequence == 42

    def test_sign_and_verify(self):
        """Test signing and verification."""
        keypair = generate_node_identity()

        eku = ExecutableKnowledge.create_query(
            instruction="Test query",
            sender_id=keypair.node_id
        )

        eku.sign(keypair.private_key)
        assert len(eku.signature) == 64

        is_valid = eku.verify(keypair.public_key)
        assert is_valid is True

    def test_reject_invalid_signature(self):
        """Test rejection of invalid signature."""
        keypair1 = generate_node_identity()
        keypair2 = generate_node_identity()

        eku = ExecutableKnowledge.create_query(
            instruction="Test query",
            sender_id=keypair1.node_id
        )

        eku.sign(keypair1.private_key)
        is_valid = eku.verify(keypair2.public_key)

        assert is_valid is False

    def test_serialization(self):
        """Test EKU serialization."""
        keypair = generate_node_identity()

        eku = ExecutableKnowledge.create_query(
            instruction="Test serialization",
            sender_id=keypair.node_id,
            context={"key": "value"}
        )
        eku.sign(keypair.private_key)

        data = eku.to_bytes()
        restored = ExecutableKnowledge.from_bytes(data)

        assert restored.header.eku_type == eku.header.eku_type
        assert restored.knowledge.instruction == eku.knowledge.instruction
        assert restored.verify(keypair.public_key)


class TestExecutionResult:
    """Test execution results."""

    def test_create_result(self):
        """Test result creation."""
        result = ExecutionResult(
            success=True,
            output="Test output",
            execution_time_ms=12.5,
            memory_refs=["chain:latest"],
            metadata={"tokens": 100}
        )

        assert result.success is True
        assert result.output == "Test output"
        assert result.execution_time_ms == 12.5

    def test_result_to_dict(self):
        """Test result dictionary conversion."""
        result = ExecutionResult(
            success=True,
            output="Test",
            execution_time_ms=10.0
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["execution_time_ms"] == 10.0


class TestPerformance:
    """Performance benchmarks."""

    def test_eku_creation_speed(self):
        """Benchmark EKU creation."""
        keypair = generate_node_identity()
        iterations = 10000

        start = time.perf_counter()

        for i in range(iterations):
            ExecutableKnowledge.create_query(
                instruction=f"Query {i}",
                sender_id=keypair.node_id
            )

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nEKU creation: {ops_per_sec:.0f} EKUs/sec")
        assert ops_per_sec > 10000  # At least 10k EKUs/sec

    def test_eku_sign_speed(self):
        """Benchmark EKU signing."""
        keypair = generate_node_identity()
        eku = ExecutableKnowledge.create_query(
            instruction="Test query",
            sender_id=keypair.node_id
        )
        iterations = 1000

        start = time.perf_counter()

        for _ in range(iterations):
            eku.sign(keypair.private_key)

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nEKU signing: {ops_per_sec:.0f} signs/sec")
        assert ops_per_sec > 100  # At least 100 signs/sec

    def test_eku_serialization_speed(self):
        """Benchmark EKU serialization."""
        keypair = generate_node_identity()
        eku = ExecutableKnowledge.create_query(
            instruction="Test query for serialization benchmark",
            sender_id=keypair.node_id,
            context={"key": "value", "nested": {"a": 1, "b": 2}}
        )
        eku.sign(keypair.private_key)
        iterations = 10000

        start = time.perf_counter()

        for _ in range(iterations):
            data = eku.to_bytes()
            ExecutableKnowledge.from_bytes(data)

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nEKU serialize+deserialize: {ops_per_sec:.0f} roundtrips/sec")
        assert ops_per_sec > 1000  # At least 1k roundtrips/sec
