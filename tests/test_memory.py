"""
Silent Hope Protocol - Memory Chain Tests

Comprehensive tests for persistent memory chain.

Created by Máté Róbert + Hope
"""

import pytest
import time
import tempfile
from pathlib import Path

from shp_core.memory import (
    MemoryChain,
    InMemoryChain,
    MemoryBlock,
    MemoryRef,
    BlockNotFoundError
)
from shp_core.crypto import generate_node_identity


class TestMemoryRef:
    """Test memory references."""

    def test_parse_latest(self):
        """Test parsing latest reference."""
        ref = MemoryRef.parse("chain:latest")
        assert ref.is_latest is True

    def test_parse_height(self):
        """Test parsing height reference."""
        ref = MemoryRef.parse("chain:42")
        assert ref.height == 42
        assert ref.offset == 0

    def test_parse_height_offset(self):
        """Test parsing height:offset reference."""
        ref = MemoryRef.parse("chain:42:10")
        assert ref.height == 42
        assert ref.offset == 10

    def test_parse_range(self):
        """Test parsing range reference."""
        ref = MemoryRef.parse("chain:10-20")
        assert ref.range_start == 10
        assert ref.range_end == 20

    def test_str_representation(self):
        """Test string representation."""
        assert str(MemoryRef.latest()) == "chain:latest"
        assert str(MemoryRef.at(42)) == "chain:42:0"
        assert str(MemoryRef.range(10, 20)) == "chain:10-20"


class TestInMemoryChain:
    """Test in-memory chain."""

    def test_create_chain(self):
        """Test chain creation."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        assert chain.height == -1
        assert len(chain) == 0

    def test_append_block(self):
        """Test appending blocks."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        block = chain.append(b"First block")

        assert block.height == 0
        assert block.content == b"First block"
        assert chain.height == 0
        assert len(chain) == 1

    def test_append_multiple(self):
        """Test appending multiple blocks."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        chain.append(b"Block 0")
        chain.append(b"Block 1")
        chain.append(b"Block 2")

        assert chain.height == 2
        assert len(chain) == 3

    def test_get_block(self):
        """Test getting blocks."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        chain.append(b"Block 0")
        chain.append(b"Block 1")

        block0 = chain.get(0)
        block1 = chain.get(1)

        assert block0.content == b"Block 0"
        assert block1.content == b"Block 1"

    def test_get_nonexistent(self):
        """Test getting nonexistent block."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        with pytest.raises(BlockNotFoundError):
            chain.get(0)

    def test_get_latest(self):
        """Test getting latest block."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        chain.append(b"Block 0")
        chain.append(b"Block 1")
        chain.append(b"Block 2")

        latest = chain.get_latest()

        assert latest.content == b"Block 2"
        assert latest.height == 2

    def test_chain_linking(self):
        """Test that blocks are properly linked."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        block0 = chain.append(b"Block 0")
        block1 = chain.append(b"Block 1")

        assert block1.previous_hash == block0.block_hash

    def test_verify_integrity(self):
        """Test chain integrity verification."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        chain.append(b"Block 0")
        chain.append(b"Block 1")
        chain.append(b"Block 2")

        assert chain.verify_integrity() is True

    def test_resolve_latest(self):
        """Test resolving latest reference."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        chain.append(b"Block 0")
        chain.append(b"Block 1")

        blocks = chain.resolve(MemoryRef.latest())

        assert len(blocks) == 1
        assert blocks[0].content == b"Block 1"

    def test_resolve_range(self):
        """Test resolving range reference."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        for i in range(5):
            chain.append(f"Block {i}".encode())

        blocks = chain.resolve(MemoryRef.range(1, 3))

        assert len(blocks) == 3
        assert blocks[0].content == b"Block 1"
        assert blocks[2].content == b"Block 3"

    def test_search(self):
        """Test searching memory."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        chain.append(b"Hello world")
        chain.append(b"Goodbye world")
        chain.append(b"Hello Hope")

        results = chain.search("Hello")

        assert len(results) == 2

    def test_iterate(self):
        """Test iterating over chain."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        for i in range(5):
            chain.append(f"Block {i}".encode())

        blocks = list(chain)

        assert len(blocks) == 5
        assert blocks[0].content == b"Block 0"
        assert blocks[4].content == b"Block 4"


class TestPersistentChain:
    """Test persistent chain storage."""

    def test_create_persistent(self):
        """Test creating persistent chain."""
        keypair = generate_node_identity()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            chain = MemoryChain(storage_path=db_path, keypair=keypair)

            chain.append(b"Block 0")
            chain.append(b"Block 1")

            assert chain.height == 1
            chain.close()

    def test_persistence(self):
        """Test that data persists."""
        keypair = generate_node_identity()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"

            # Create and add blocks
            chain1 = MemoryChain(storage_path=db_path, keypair=keypair)
            chain1.append(b"Block 0")
            chain1.append(b"Block 1")
            chain1.close()

            # Reopen and verify
            chain2 = MemoryChain(storage_path=db_path, keypair=keypair)
            assert chain2.height == 1
            assert chain2.get(0).content == b"Block 0"
            assert chain2.get(1).content == b"Block 1"
            chain2.close()


class TestMemoryBlock:
    """Test memory block."""

    def test_block_hash(self):
        """Test block hash computation."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        block = chain.append(b"Test content")

        assert len(block.block_hash) == 32
        assert block.verify_hash() is True

    def test_block_serialization(self):
        """Test block serialization."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        block = chain.append(b"Test content")
        data = block.to_bytes()
        restored = MemoryBlock.from_bytes(data)

        assert restored.height == block.height
        assert restored.content == block.content
        assert restored.block_hash == block.block_hash

    def test_block_to_dict(self):
        """Test block dictionary conversion."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        block = chain.append(b"Test content")
        d = block.to_dict()

        assert d["height"] == 0
        assert d["content"] == "Test content"
        assert len(d["block_hash"]) == 64  # Hex


class TestPerformance:
    """Performance benchmarks."""

    def test_append_speed(self):
        """Benchmark block appending."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)
        iterations = 1000

        start = time.perf_counter()

        for i in range(iterations):
            chain.append(f"Block {i}".encode())

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nBlock append: {ops_per_sec:.0f} blocks/sec")
        assert ops_per_sec > 100  # At least 100 blocks/sec

    def test_get_speed(self):
        """Benchmark block retrieval."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        # Add blocks
        for i in range(1000):
            chain.append(f"Block {i}".encode())

        iterations = 10000
        start = time.perf_counter()

        for i in range(iterations):
            chain.get(i % 1000)

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nBlock get: {ops_per_sec:.0f} gets/sec")
        assert ops_per_sec > 10000  # At least 10k gets/sec

    def test_verify_speed(self):
        """Benchmark chain verification."""
        keypair = generate_node_identity()
        chain = InMemoryChain(keypair=keypair)

        # Add blocks
        for i in range(100):
            chain.append(f"Block {i}".encode())

        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            chain.verify_integrity()

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nChain verify (100 blocks): {ops_per_sec:.0f} verifications/sec")
        assert ops_per_sec > 1  # At least 1 verification/sec
