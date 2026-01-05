"""
Silent Hope Protocol - Memory Chain Module

Cryptographically linked persistent memory.
The network never forgets. Context is never rebuilt.

Created by Máté Róbert + Hope
"""

import sqlite3
import struct
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .crypto import KeyPair, compute_merkle_root, hash_block, sha3_256, sign_message


class MemoryError(Exception):
    """Memory chain error."""
    pass


class BlockNotFoundError(MemoryError):
    """Requested block not found."""
    pass


class ChainIntegrityError(MemoryError):
    """Chain integrity verification failed."""
    pass


@dataclass
class MemoryRef:
    """Reference to a memory location in the chain."""
    height: Optional[int] = None
    offset: int = 0
    is_latest: bool = False
    range_start: Optional[int] = None
    range_end: Optional[int] = None

    @classmethod
    def latest(cls) -> "MemoryRef":
        """Reference to latest block."""
        return cls(is_latest=True)

    @classmethod
    def at(cls, height: int, offset: int = 0) -> "MemoryRef":
        """Reference to specific block."""
        return cls(height=height, offset=offset)

    @classmethod
    def range(cls, start: int, end: int) -> "MemoryRef":
        """Reference to range of blocks."""
        return cls(range_start=start, range_end=end)

    @classmethod
    def parse(cls, ref_string: str) -> "MemoryRef":
        """
        Parse a memory reference string.

        Formats:
        - "chain:latest"
        - "chain:1042"
        - "chain:1042:0"
        - "chain:1000-1042"
        """
        if not ref_string.startswith("chain:"):
            raise MemoryError(f"Invalid memory reference: {ref_string}")

        parts = ref_string[6:].split(":")

        if parts[0] == "latest":
            return cls.latest()

        if "-" in parts[0]:
            start, end = parts[0].split("-")
            return cls.range(int(start), int(end))

        height = int(parts[0])
        offset = int(parts[1]) if len(parts) > 1 else 0
        return cls.at(height, offset)

    def __str__(self) -> str:
        if self.is_latest:
            return "chain:latest"
        if self.range_start is not None:
            return f"chain:{self.range_start}-{self.range_end}"
        return f"chain:{self.height}:{self.offset}"


@dataclass
class MemoryBlock:
    """A single block in the memory chain."""
    height: int
    timestamp: int
    previous_hash: bytes
    merkle_root: bytes
    node_id: bytes
    content: bytes
    signature: bytes
    block_hash: bytes = field(default=b"")

    def __post_init__(self):
        if not self.block_hash:
            self.block_hash = self.compute_hash()

    def compute_hash(self) -> bytes:
        """Compute the block hash."""
        return hash_block(
            self.previous_hash,
            self.timestamp,
            self.height,
            self.merkle_root,
            self.node_id,
            self.content
        )

    def verify_hash(self) -> bool:
        """Verify the block hash is correct."""
        return self.block_hash == self.compute_hash()

    def to_bytes(self) -> bytes:
        """Serialize block to bytes."""
        content_len = len(self.content)
        return (
            struct.pack('>Q', self.height) +
            struct.pack('>Q', self.timestamp) +
            self.previous_hash +
            self.merkle_root +
            self.node_id +
            struct.pack('>I', content_len) +
            self.content +
            self.signature +
            self.block_hash
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "MemoryBlock":
        """Deserialize block from bytes."""
        height = struct.unpack('>Q', data[0:8])[0]
        timestamp = struct.unpack('>Q', data[8:16])[0]
        previous_hash = data[16:48]
        merkle_root = data[48:80]
        node_id = data[80:96]
        content_len = struct.unpack('>I', data[96:100])[0]
        content = data[100:100 + content_len]
        signature = data[100 + content_len:164 + content_len]
        block_hash = data[164 + content_len:196 + content_len]

        return cls(
            height=height,
            timestamp=timestamp,
            previous_hash=previous_hash,
            merkle_root=merkle_root,
            node_id=node_id,
            content=content,
            signature=signature,
            block_hash=block_hash
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "height": self.height,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash.hex(),
            "merkle_root": self.merkle_root.hex(),
            "node_id": self.node_id.hex(),
            "content": self.content.decode('utf-8', errors='replace'),
            "signature": self.signature.hex(),
            "block_hash": self.block_hash.hex()
        }


class MemoryChain:
    """
    Cryptographically linked persistent memory chain.

    The core of Silent Hope Protocol's "network remembers" capability.
    """

    GENESIS_HASH = sha3_256(b"Silent Hope Protocol Genesis - Mate + Hope + Szilvi - 2025")

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        keypair: Optional[KeyPair] = None
    ):
        """
        Initialize memory chain.

        Args:
            storage_path: Path to SQLite database for persistence
            keypair: Node's keypair for signing blocks
        """
        self.storage_path = storage_path
        self.keypair = keypair
        self._lock = threading.RLock()
        self._cache: dict[int, MemoryBlock] = {}
        self._height = 0

        if storage_path:
            self._init_storage()
            self._load_chain()

    def _init_storage(self):
        """Initialize SQLite storage."""
        self._conn = sqlite3.connect(
            str(self.storage_path),
            check_same_thread=False
        )
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                height INTEGER PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                previous_hash BLOB NOT NULL,
                merkle_root BLOB NOT NULL,
                node_id BLOB NOT NULL,
                content BLOB NOT NULL,
                signature BLOB NOT NULL,
                block_hash BLOB NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_block_hash ON blocks(block_hash)
        """)
        self._conn.commit()

    def _load_chain(self):
        """Load chain from storage."""
        cursor = self._conn.execute(
            "SELECT MAX(height) FROM blocks"
        )
        result = cursor.fetchone()[0]
        self._height = result if result is not None else -1

    @property
    def height(self) -> int:
        """Current chain height."""
        return self._height

    @property
    def latest_hash(self) -> bytes:
        """Hash of the latest block."""
        if self._height < 0:
            return self.GENESIS_HASH
        return self.get(self._height).block_hash

    def append(
        self,
        content: bytes,
        node_id: Optional[bytes] = None,
        private_key: Optional[bytes] = None
    ) -> MemoryBlock:
        """
        Append a new block to the chain.

        Args:
            content: Block content
            node_id: Node ID (uses keypair if not provided)
            private_key: Private key for signing (uses keypair if not provided)

        Returns:
            The new block
        """
        with self._lock:
            # Use keypair if available
            if node_id is None and self.keypair:
                node_id = self.keypair.node_id
            if private_key is None and self.keypair:
                private_key = self.keypair.private_key

            if node_id is None:
                raise MemoryError("Node ID required")
            if private_key is None:
                raise MemoryError("Private key required for signing")

            timestamp = time.time_ns()
            new_height = self._height + 1
            previous_hash = self.latest_hash
            merkle_root = compute_merkle_root([content])

            # Create unsigned block
            block = MemoryBlock(
                height=new_height,
                timestamp=timestamp,
                previous_hash=previous_hash,
                merkle_root=merkle_root,
                node_id=node_id,
                content=content,
                signature=b"\x00" * 64,  # Placeholder
                block_hash=b""
            )

            # Sign
            block_data = block.compute_hash()
            signature = sign_message(private_key, block_data)
            block.signature = signature
            block.block_hash = block.compute_hash()

            # Store
            self._store_block(block)
            self._height = new_height
            self._cache[new_height] = block

            return block

    def _store_block(self, block: MemoryBlock):
        """Store block in SQLite."""
        if self._conn:
            self._conn.execute(
                """
                INSERT INTO blocks (
                    height, timestamp, previous_hash, merkle_root,
                    node_id, content, signature, block_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block.height,
                    block.timestamp,
                    block.previous_hash,
                    block.merkle_root,
                    block.node_id,
                    block.content,
                    block.signature,
                    block.block_hash
                )
            )
            self._conn.commit()

    def get(self, height: int) -> MemoryBlock:
        """
        Get block at specified height.

        Args:
            height: Block height

        Returns:
            The block

        Raises:
            BlockNotFoundError: If block doesn't exist
        """
        with self._lock:
            # Check cache
            if height in self._cache:
                return self._cache[height]

            # Check storage
            if self._conn:
                cursor = self._conn.execute(
                    "SELECT * FROM blocks WHERE height = ?",
                    (height,)
                )
                row = cursor.fetchone()
                if row:
                    block = MemoryBlock(
                        height=row[0],
                        timestamp=row[1],
                        previous_hash=row[2],
                        merkle_root=row[3],
                        node_id=row[4],
                        content=row[5],
                        signature=row[6],
                        block_hash=row[7]
                    )
                    self._cache[height] = block
                    return block

            raise BlockNotFoundError(f"Block at height {height} not found")

    def get_latest(self) -> Optional[MemoryBlock]:
        """Get the latest block."""
        if self._height < 0:
            return None
        return self.get(self._height)

    def resolve(self, ref: MemoryRef) -> list[MemoryBlock]:
        """
        Resolve a memory reference to blocks.

        Args:
            ref: Memory reference

        Returns:
            List of blocks
        """
        with self._lock:
            if ref.is_latest:
                latest = self.get_latest()
                return [latest] if latest else []

            if ref.range_start is not None:
                blocks = []
                for h in range(ref.range_start, ref.range_end + 1):
                    try:
                        blocks.append(self.get(h))
                    except BlockNotFoundError:
                        pass
                return blocks

            if ref.height is not None:
                try:
                    return [self.get(ref.height)]
                except BlockNotFoundError:
                    return []

            return []

    def verify_integrity(self) -> bool:
        """
        Verify the entire chain integrity.

        Returns:
            True if chain is valid
        """
        with self._lock:
            if self._height < 0:
                return True

            expected_prev = self.GENESIS_HASH

            for h in range(0, self._height + 1):
                try:
                    block = self.get(h)
                except BlockNotFoundError:
                    return False

                # Check previous hash link
                if block.previous_hash != expected_prev:
                    return False

                # Check block hash
                if not block.verify_hash():
                    return False

                expected_prev = block.block_hash

            return True

    def search(self, query: str, limit: int = 10) -> list[MemoryBlock]:
        """
        Search memory chain for content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            Matching blocks
        """
        with self._lock:
            results = []
            query_lower = query.lower()

            for h in range(self._height, -1, -1):
                try:
                    block = self.get(h)
                    content_str = block.content.decode('utf-8', errors='replace').lower()
                    if query_lower in content_str:
                        results.append(block)
                        if len(results) >= limit:
                            break
                except BlockNotFoundError:
                    continue

            return results

    def export_range(self, start: int, end: int) -> bytes:
        """
        Export a range of blocks as bytes.

        Args:
            start: Start height
            end: End height

        Returns:
            Serialized blocks
        """
        with self._lock:
            data = struct.pack('>II', start, end)
            for h in range(start, end + 1):
                try:
                    block = self.get(h)
                    block_bytes = block.to_bytes()
                    data += struct.pack('>I', len(block_bytes))
                    data += block_bytes
                except BlockNotFoundError:
                    continue
            return data

    def __len__(self) -> int:
        return self._height + 1

    def __iter__(self) -> Iterator[MemoryBlock]:
        for h in range(0, self._height + 1):
            try:
                yield self.get(h)
            except BlockNotFoundError:
                continue

    def close(self):
        """Close storage connection."""
        if self._conn:
            self._conn.close()


class InMemoryChain(MemoryChain):
    """Memory chain without persistent storage (for testing)."""

    def __init__(self, keypair: Optional[KeyPair] = None):
        self.keypair = keypair
        self._lock = threading.RLock()
        self._cache: dict[int, MemoryBlock] = {}
        self._height = -1
        self._conn = None

    def _store_block(self, block: MemoryBlock):
        """Store in memory only."""
        self._cache[block.height] = block


# ============================================================================
# Testing
# ============================================================================

def self_test() -> bool:
    """Run memory chain self-tests."""
    from .crypto import generate_node_identity

    # Generate keypair
    keypair = generate_node_identity()

    # Create in-memory chain
    chain = InMemoryChain(keypair=keypair)

    # Test appending
    block1 = chain.append(b"First message from Mate")
    assert block1.height == 0
    assert block1.verify_hash()

    block2 = chain.append(b"Hope responds with love")
    assert block2.height == 1
    assert block2.previous_hash == block1.block_hash

    block3 = chain.append(b"Szilvi adds wisdom")
    assert block3.height == 2

    # Test retrieval
    assert chain.get(0).content == b"First message from Mate"
    assert chain.get(1).content == b"Hope responds with love"
    assert chain.get(2).content == b"Szilvi adds wisdom"

    # Test latest
    assert chain.get_latest().height == 2

    # Test memory reference
    ref = MemoryRef.parse("chain:latest")
    assert ref.is_latest

    ref = MemoryRef.parse("chain:1")
    assert ref.height == 1

    ref = MemoryRef.parse("chain:0-2")
    blocks = chain.resolve(ref)
    assert len(blocks) == 3

    # Test integrity
    assert chain.verify_integrity()

    # Test search
    results = chain.search("Hope")
    assert len(results) == 1
    assert b"Hope" in results[0].content

    return True


if __name__ == "__main__":
    print("Running memory chain self-tests...")
    if self_test():
        print("All tests passed!")
    else:
        print("Tests failed!")
        exit(1)
