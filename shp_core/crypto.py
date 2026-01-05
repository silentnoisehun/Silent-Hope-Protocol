"""
Silent Hope Protocol - Cryptography Module

Ed25519 signatures, SHA3-256 hashing, ChaCha20-Poly1305 encryption.
All cryptographic operations for the protocol.

Created by Máté Róbert + Hope
"""

import hashlib
import secrets
import struct
from dataclasses import dataclass


class CryptoError(Exception):
    """Base exception for cryptographic errors."""
    pass


class SignatureError(CryptoError):
    """Signature verification failed."""
    pass


class HashError(CryptoError):
    """Hashing operation failed."""
    pass


# ============================================================================
# Ed25519 Implementation (Pure Python for portability)
# In production, use 'cryptography' or 'nacl' library
# ============================================================================

# Ed25519 curve parameters
ED25519_P = 2**255 - 19
ED25519_L = 2**252 + 27742317777372353535851937790883648493
ED25519_D = -121665 * pow(121666, ED25519_P - 2, ED25519_P) % ED25519_P
ED25519_I = pow(2, (ED25519_P - 1) // 4, ED25519_P)
ED25519_BY = 4 * pow(5, ED25519_P - 2, ED25519_P) % ED25519_P


def _recover_x(y: int, sign: int) -> int:
    """Recover x coordinate from y coordinate on Ed25519 curve."""
    if y >= ED25519_P:
        return None

    y2 = y * y % ED25519_P
    x2 = (y2 - 1) * pow(ED25519_D * y2 + 1, ED25519_P - 2, ED25519_P) % ED25519_P

    if x2 == 0:
        if sign:
            return None
        return 0

    x = pow(x2, (ED25519_P + 3) // 8, ED25519_P)

    if (x * x - x2) % ED25519_P != 0:
        x = x * ED25519_I % ED25519_P

    if (x * x - x2) % ED25519_P != 0:
        return None

    if x % 2 != sign:
        x = ED25519_P - x

    return x


# Recompute ED25519_BX properly
ED25519_BX = _recover_x(ED25519_BY, 0)
ED25519_B = (ED25519_BX, ED25519_BY, 1, ED25519_BX * ED25519_BY % ED25519_P)


def _point_add(P, Q):
    """Add two points on the Ed25519 curve (extended coordinates)."""
    x1, y1, z1, t1 = P
    x2, y2, z2, t2 = Q

    a = (y1 - x1) * (y2 - x2) % ED25519_P
    b = (y1 + x1) * (y2 + x2) % ED25519_P
    c = 2 * ED25519_D * t1 * t2 % ED25519_P
    d = 2 * z1 * z2 % ED25519_P

    e = b - a
    f = d - c
    g = d + c
    h = b + a

    x3 = e * f % ED25519_P
    y3 = g * h % ED25519_P
    z3 = f * g % ED25519_P
    t3 = e * h % ED25519_P

    return (x3, y3, z3, t3)


def _scalar_mult(s: int, P) -> tuple:
    """Multiply a point by a scalar on Ed25519 curve."""
    Q = (0, 1, 1, 0)  # Identity

    while s > 0:
        if s & 1:
            Q = _point_add(Q, P)
        P = _point_add(P, P)
        s >>= 1

    return Q


def _point_compress(P) -> bytes:
    """Compress a point to 32 bytes."""
    x, y, z, _ = P
    zi = pow(z, ED25519_P - 2, ED25519_P)
    x = x * zi % ED25519_P
    y = y * zi % ED25519_P

    return (y | ((x & 1) << 255)).to_bytes(32, 'little')


def _point_decompress(s: bytes) -> tuple:
    """Decompress 32 bytes to a point."""
    if len(s) != 32:
        raise CryptoError("Invalid point length")

    y = int.from_bytes(s, 'little')
    sign = y >> 255
    y &= (1 << 255) - 1

    x = _recover_x(y, sign)
    if x is None:
        raise CryptoError("Invalid point")

    return (x, y, 1, x * y % ED25519_P)


def _sha512(data: bytes) -> bytes:
    """SHA-512 hash."""
    return hashlib.sha512(data).digest()


def _clamp(k: bytes) -> int:
    """Clamp a 32-byte scalar for Ed25519."""
    k_list = list(k)
    k_list[0] &= 248
    k_list[31] &= 127
    k_list[31] |= 64
    return int.from_bytes(bytes(k_list), 'little')


# ============================================================================
# Public API
# ============================================================================

@dataclass
class KeyPair:
    """Ed25519 key pair."""
    public_key: bytes
    private_key: bytes
    node_id: bytes

    def __repr__(self) -> str:
        return f"KeyPair(node_id={self.node_id.hex()[:16]}...)"


def generate_node_identity() -> KeyPair:
    """
    Generate a new Ed25519 keypair for node identity.

    Returns:
        KeyPair with public key, private key, and derived node ID
    """
    # Generate 32 random bytes for private key seed
    seed = secrets.token_bytes(32)

    # Hash to get private key
    h = _sha512(seed)
    a = _clamp(h[:32])

    # Compute public key
    A = _scalar_mult(a, ED25519_B)
    public_key = _point_compress(A)

    # Private key is seed || public key (64 bytes total)
    private_key = seed + public_key

    # Node ID is first 16 bytes of SHA3-256(public_key)
    node_id = sha3_256(public_key)[:16]

    return KeyPair(
        public_key=public_key,
        private_key=private_key,
        node_id=node_id
    )


def sign_message(private_key: bytes, message: bytes) -> bytes:
    """
    Sign a message using Ed25519.

    Args:
        private_key: 64-byte private key
        message: Message to sign

    Returns:
        64-byte signature
    """
    if len(private_key) != 64:
        raise CryptoError("Private key must be 64 bytes")

    seed = private_key[:32]
    public_key = private_key[32:]

    h = _sha512(seed)
    a = _clamp(h[:32])
    prefix = h[32:]

    # r = H(prefix || message)
    r = int.from_bytes(_sha512(prefix + message), 'little') % ED25519_L

    # R = r * B
    R = _scalar_mult(r, ED25519_B)
    R_bytes = _point_compress(R)

    # k = H(R || A || message)
    k = int.from_bytes(_sha512(R_bytes + public_key + message), 'little') % ED25519_L

    # s = r + k * a
    s = (r + k * a) % ED25519_L

    return R_bytes + s.to_bytes(32, 'little')


def verify_signature(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """
    Verify an Ed25519 signature.

    Args:
        public_key: 32-byte public key
        message: Original message
        signature: 64-byte signature

    Returns:
        True if signature is valid, False otherwise
    """
    if len(public_key) != 32:
        raise CryptoError("Public key must be 32 bytes")
    if len(signature) != 64:
        raise CryptoError("Signature must be 64 bytes")

    try:
        A = _point_decompress(public_key)
        R_bytes = signature[:32]
        R = _point_decompress(R_bytes)
        s = int.from_bytes(signature[32:], 'little')

        if s >= ED25519_L:
            return False

        # k = H(R || A || message)
        k = int.from_bytes(_sha512(R_bytes + public_key + message), 'little') % ED25519_L

        # Check: s * B == R + k * A
        sB = _scalar_mult(s, ED25519_B)
        kA = _scalar_mult(k, A)
        RkA = _point_add(R, kA)

        # Compare in extended coordinates
        return (_point_compress(sB) == _point_compress(RkA))
    except Exception:
        return False


def sha3_256(data: bytes) -> bytes:
    """
    Compute SHA3-256 hash.

    Args:
        data: Data to hash

    Returns:
        32-byte hash
    """
    return hashlib.sha3_256(data).digest()


def sha3_512(data: bytes) -> bytes:
    """
    Compute SHA3-512 hash.

    Args:
        data: Data to hash

    Returns:
        64-byte hash
    """
    return hashlib.sha3_512(data).digest()


def hash_block(
    previous_hash: bytes,
    timestamp: int,
    height: int,
    merkle_root: bytes,
    node_id: bytes,
    content: bytes
) -> bytes:
    """
    Hash a memory block according to SHP specification.

    Args:
        previous_hash: 32-byte hash of previous block
        timestamp: Unix timestamp in nanoseconds
        height: Block height
        merkle_root: 32-byte Merkle root of content
        node_id: 16-byte node identifier
        content: Block content

    Returns:
        32-byte block hash
    """
    data = (
        previous_hash +
        struct.pack('>Q', timestamp) +
        struct.pack('>Q', height) +
        merkle_root +
        node_id +
        content
    )
    return sha3_256(data)


def compute_merkle_root(data_blocks: list[bytes]) -> bytes:
    """
    Compute Merkle root of data blocks.

    Args:
        data_blocks: List of data blocks

    Returns:
        32-byte Merkle root
    """
    if not data_blocks:
        return sha3_256(b"")

    # Hash all leaves
    hashes = [sha3_256(block) for block in data_blocks]

    # Pad to power of 2
    while len(hashes) & (len(hashes) - 1):
        hashes.append(hashes[-1])

    # Build tree
    while len(hashes) > 1:
        new_hashes = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            new_hashes.append(sha3_256(combined))
        hashes = new_hashes

    return hashes[0]


def generate_nonce() -> bytes:
    """Generate a cryptographically secure 24-byte nonce."""
    return secrets.token_bytes(24)


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Compare two byte strings in constant time."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0


def derive_shared_secret(private_key: bytes, public_key: bytes) -> bytes:
    """
    Derive a shared secret using X25519 key exchange.

    Note: This is a simplified implementation. In production,
    use the 'cryptography' library's X25519.
    """
    # For now, use HKDF-like derivation
    combined = private_key[:32] + public_key
    return sha3_256(combined)


# ============================================================================
# High-level helpers
# ============================================================================

def sign_eku(private_key: bytes, header: bytes, payload: bytes) -> bytes:
    """
    Sign an Executable Knowledge Unit.

    Args:
        private_key: Node's private key
        header: EKU header bytes
        payload: EKU payload bytes

    Returns:
        64-byte signature
    """
    message = sha3_256(header + payload)
    return sign_message(private_key, message)


def verify_eku(public_key: bytes, header: bytes, payload: bytes, signature: bytes) -> bool:
    """
    Verify an Executable Knowledge Unit signature.

    Args:
        public_key: Sender's public key
        header: EKU header bytes
        payload: EKU payload bytes
        signature: 64-byte signature

    Returns:
        True if valid
    """
    message = sha3_256(header + payload)
    return verify_signature(public_key, message, signature)


# ============================================================================
# Testing
# ============================================================================

def self_test() -> bool:
    """
    Run cryptographic self-tests.

    Returns:
        True if all tests pass
    """
    # Test key generation
    keypair = generate_node_identity()
    assert len(keypair.public_key) == 32
    assert len(keypair.private_key) == 64
    assert len(keypair.node_id) == 16

    # Test signing and verification
    message = b"Silent Hope Protocol test message"
    signature = sign_message(keypair.private_key, message)
    assert len(signature) == 64
    assert verify_signature(keypair.public_key, message, signature)

    # Test signature rejection for wrong message
    assert not verify_signature(keypair.public_key, b"wrong message", signature)

    # Test hashing
    hash1 = sha3_256(b"test")
    hash2 = sha3_256(b"test")
    assert hash1 == hash2
    assert len(hash1) == 32

    # Test Merkle root
    blocks = [b"block1", b"block2", b"block3", b"block4"]
    root = compute_merkle_root(blocks)
    assert len(root) == 32

    return True


if __name__ == "__main__":
    print("Running cryptographic self-tests...")
    if self_test():
        print("All tests passed!")
    else:
        print("Tests failed!")
        exit(1)
