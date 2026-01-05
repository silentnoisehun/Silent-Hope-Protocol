"""
Silent Hope Protocol - Cryptography Tests

Comprehensive tests for Ed25519, SHA3-256, and chain integrity.

Created by Máté Róbert + Hope
"""

import pytest
import time
from shp_core.crypto import (
    generate_node_identity,
    sign_message,
    verify_signature,
    sha3_256,
    sha3_512,
    hash_block,
    compute_merkle_root,
    KeyPair,
    CryptoError
)


class TestKeyGeneration:
    """Test key generation."""

    def test_generate_keypair(self):
        """Test keypair generation."""
        keypair = generate_node_identity()

        assert isinstance(keypair, KeyPair)
        assert len(keypair.public_key) == 32
        assert len(keypair.private_key) == 64
        assert len(keypair.node_id) == 16

    def test_unique_keypairs(self):
        """Test that keypairs are unique."""
        keypairs = [generate_node_identity() for _ in range(100)]
        node_ids = [kp.node_id for kp in keypairs]

        assert len(set(node_ids)) == 100

    def test_keypair_determinism(self):
        """Test that same seed produces different keys (randomness)."""
        kp1 = generate_node_identity()
        kp2 = generate_node_identity()

        assert kp1.public_key != kp2.public_key
        assert kp1.private_key != kp2.private_key


class TestSigning:
    """Test message signing."""

    def test_sign_message(self):
        """Test message signing."""
        keypair = generate_node_identity()
        message = b"Silent Hope Protocol test message"

        signature = sign_message(keypair.private_key, message)

        assert len(signature) == 64

    def test_verify_valid_signature(self):
        """Test verification of valid signature."""
        keypair = generate_node_identity()
        message = b"Test message"

        signature = sign_message(keypair.private_key, message)
        is_valid = verify_signature(keypair.public_key, message, signature)

        assert is_valid is True

    def test_reject_invalid_signature(self):
        """Test rejection of invalid signature."""
        keypair = generate_node_identity()
        message = b"Test message"
        wrong_message = b"Wrong message"

        signature = sign_message(keypair.private_key, message)
        is_valid = verify_signature(keypair.public_key, wrong_message, signature)

        assert is_valid is False

    def test_reject_wrong_key(self):
        """Test rejection with wrong public key."""
        keypair1 = generate_node_identity()
        keypair2 = generate_node_identity()
        message = b"Test message"

        signature = sign_message(keypair1.private_key, message)
        is_valid = verify_signature(keypair2.public_key, message, signature)

        assert is_valid is False

    def test_sign_empty_message(self):
        """Test signing empty message."""
        keypair = generate_node_identity()
        message = b""

        signature = sign_message(keypair.private_key, message)
        is_valid = verify_signature(keypair.public_key, message, signature)

        assert is_valid is True

    def test_sign_large_message(self):
        """Test signing large message."""
        keypair = generate_node_identity()
        message = b"x" * 1_000_000  # 1MB

        signature = sign_message(keypair.private_key, message)
        is_valid = verify_signature(keypair.public_key, message, signature)

        assert is_valid is True


class TestHashing:
    """Test hashing functions."""

    def test_sha3_256(self):
        """Test SHA3-256."""
        data = b"test"
        hash_result = sha3_256(data)

        assert len(hash_result) == 32

    def test_sha3_256_deterministic(self):
        """Test SHA3-256 is deterministic."""
        data = b"Silent Hope Protocol"

        hash1 = sha3_256(data)
        hash2 = sha3_256(data)

        assert hash1 == hash2

    def test_sha3_256_different_input(self):
        """Test SHA3-256 produces different output for different input."""
        hash1 = sha3_256(b"input1")
        hash2 = sha3_256(b"input2")

        assert hash1 != hash2

    def test_sha3_512(self):
        """Test SHA3-512."""
        data = b"test"
        hash_result = sha3_512(data)

        assert len(hash_result) == 64


class TestMerkleRoot:
    """Test Merkle root computation."""

    def test_merkle_root_single(self):
        """Test Merkle root with single block."""
        blocks = [b"block1"]
        root = compute_merkle_root(blocks)

        assert len(root) == 32

    def test_merkle_root_multiple(self):
        """Test Merkle root with multiple blocks."""
        blocks = [b"block1", b"block2", b"block3", b"block4"]
        root = compute_merkle_root(blocks)

        assert len(root) == 32

    def test_merkle_root_empty(self):
        """Test Merkle root with empty list."""
        root = compute_merkle_root([])

        assert len(root) == 32

    def test_merkle_root_deterministic(self):
        """Test Merkle root is deterministic."""
        blocks = [b"a", b"b", b"c"]

        root1 = compute_merkle_root(blocks)
        root2 = compute_merkle_root(blocks)

        assert root1 == root2

    def test_merkle_root_order_matters(self):
        """Test that block order matters."""
        blocks1 = [b"a", b"b"]
        blocks2 = [b"b", b"a"]

        root1 = compute_merkle_root(blocks1)
        root2 = compute_merkle_root(blocks2)

        assert root1 != root2


class TestBlockHashing:
    """Test block hash computation."""

    def test_hash_block(self):
        """Test block hashing."""
        previous_hash = b"\x00" * 32
        timestamp = 1234567890
        height = 0
        merkle_root = b"\x01" * 32
        node_id = b"\x02" * 16
        content = b"Block content"

        block_hash = hash_block(
            previous_hash, timestamp, height,
            merkle_root, node_id, content
        )

        assert len(block_hash) == 32

    def test_hash_block_deterministic(self):
        """Test block hashing is deterministic."""
        params = (
            b"\x00" * 32,  # previous_hash
            1234567890,    # timestamp
            0,             # height
            b"\x01" * 32,  # merkle_root
            b"\x02" * 16,  # node_id
            b"content"     # content
        )

        hash1 = hash_block(*params)
        hash2 = hash_block(*params)

        assert hash1 == hash2


class TestPerformance:
    """Performance benchmarks."""

    def test_key_generation_speed(self):
        """Benchmark key generation."""
        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            generate_node_identity()

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nKey generation: {ops_per_sec:.0f} ops/sec")
        assert ops_per_sec > 10  # At least 10 keys/sec

    def test_signing_speed(self):
        """Benchmark signing."""
        keypair = generate_node_identity()
        message = b"Test message for signing benchmark"
        iterations = 1000

        start = time.perf_counter()

        for _ in range(iterations):
            sign_message(keypair.private_key, message)

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nSigning: {ops_per_sec:.0f} ops/sec")
        assert ops_per_sec > 100  # At least 100 signs/sec

    def test_verification_speed(self):
        """Benchmark verification."""
        keypair = generate_node_identity()
        message = b"Test message for verification benchmark"
        signature = sign_message(keypair.private_key, message)
        iterations = 1000

        start = time.perf_counter()

        for _ in range(iterations):
            verify_signature(keypair.public_key, message, signature)

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nVerification: {ops_per_sec:.0f} ops/sec")
        assert ops_per_sec > 50  # At least 50 verifications/sec

    def test_hashing_speed(self):
        """Benchmark hashing."""
        data = b"x" * 1000  # 1KB
        iterations = 10000

        start = time.perf_counter()

        for _ in range(iterations):
            sha3_256(data)

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        print(f"\nSHA3-256 hashing: {ops_per_sec:.0f} ops/sec")
        assert ops_per_sec > 1000  # At least 1000 hashes/sec
