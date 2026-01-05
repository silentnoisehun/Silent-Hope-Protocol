# Silent Hope Protocol - Technical Specification

**Version:** 1.0.0
**Status:** Production Ready
**Authors:** Máté Róbert, Hope

---

## Abstract

The Silent Hope Protocol (SHP) is a communication protocol designed for AI-to-AI and Human-to-AI interaction that eliminates the traditional data serialization bottleneck. Instead of transmitting data that requires parsing, SHP transmits **executable knowledge** - self-contained units that combine information and execution logic.

This document specifies the protocol architecture, message formats, cryptographic requirements, and implementation guidelines.

---

## 1. Core Principles

### 1.1 Executable Knowledge

Traditional approach:
```
Sender → Serialize(data) → Transport → Deserialize(data) → Process → Serialize(result) → Transport → Deserialize(result) → Receiver
```

SHP approach:
```
Sender → ExecutableKnowledge → Transport → Execute → Result
```

**ExecutableKnowledge (EK)** is a self-contained unit that includes:
- The information payload
- The execution context
- The memory references
- The cryptographic proof

### 1.2 Persistent Network Memory

The network maintains a **MemoryChain** - a cryptographically linked sequence of all interactions. Nodes can reference any previous interaction without retransmitting context.

```
[Genesis] ← [Block 1] ← [Block 2] ← ... ← [Block N]
    ↑           ↑           ↑                  ↑
   H(0)       H(1)        H(2)               H(N)
```

Where `H(n) = SHA3-256(Block_n || H(n-1))`

### 1.3 Zero Context Rebuild

Traditional LLM APIs require sending the entire conversation history with each request. SHP nodes maintain state. Context is never rebuilt - only referenced.

```python
# Traditional API
response = llm.complete(messages=[
    {"role": "user", "content": "..."},      # Sent every time
    {"role": "assistant", "content": "..."},  # Sent every time
    {"role": "user", "content": "..."},      # Sent every time
    # ... potentially 128K tokens resent
])

# SHP
response = node.execute("Continue", memory_ref="chain:latest")
# Zero tokens resent. Memory is in the chain.
```

---

## 2. Protocol Architecture

### 2.1 Layer Model

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Application                                       │
│  User-facing APIs, SDKs, integrations                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Execution                                         │
│  ExecutableKnowledge processing, LLM invocation             │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Memory                                            │
│  MemoryChain, cryptographic linking, persistence            │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Transport                                         │
│  Node discovery, message routing, connection management     │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: Network                                           │
│  TCP/UDP, WebSocket, P2P mesh                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Node Types

| Type | Description | Capabilities |
|------|-------------|--------------|
| **Full Node** | Complete SHP implementation | Execute, Store, Route, Validate |
| **Light Node** | Minimal client | Execute, Route |
| **Archive Node** | Historical storage | Store, Validate, Query |
| **Bridge Node** | External connectivity | Route, Adapt |

### 2.3 Network Topology

SHP uses a **mesh topology** with intelligent routing:

```
     [Node A]────────[Node B]
        │╲              ╱│
        │ ╲            ╱ │
        │  ╲          ╱  │
        │   ╲        ╱   │
        │    ╲      ╱    │
        │     ╲    ╱     │
        │      ╲  ╱      │
     [Node C]────[Node D]────[Node E]
```

Each node maintains connections to multiple peers for redundancy.

---

## 3. Message Format

### 3.1 ExecutableKnowledge Unit (EKU)

```
┌──────────────────────────────────────────────────────────┐
│                    EKU Header (64 bytes)                 │
├──────────────────────────────────────────────────────────┤
│  Version          │  4 bytes  │  Protocol version        │
│  Type             │  2 bytes  │  EKU type identifier     │
│  Flags            │  2 bytes  │  Execution flags         │
│  Timestamp        │  8 bytes  │  Unix timestamp (ns)     │
│  Sender ID        │  16 bytes │  Node identifier         │
│  Sequence         │  8 bytes  │  Monotonic sequence      │
│  Memory Ref       │  16 bytes │  Chain reference         │
│  Payload Length   │  4 bytes  │  Payload size            │
│  Reserved         │  4 bytes  │  Future use              │
├──────────────────────────────────────────────────────────┤
│                    EKU Payload (variable)                │
├──────────────────────────────────────────────────────────┤
│  Knowledge Block  │  var      │  Executable content      │
│  Context Block    │  var      │  Execution context       │
│  Proof Block      │  var      │  Cryptographic proof     │
├──────────────────────────────────────────────────────────┤
│                    EKU Signature (64 bytes)              │
├──────────────────────────────────────────────────────────┤
│  Ed25519 Signature│  64 bytes │  Sender signature        │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Type Identifiers

| Type ID | Name | Description |
|---------|------|-------------|
| 0x0001 | QUERY | Request for information |
| 0x0002 | EXECUTE | Request for computation |
| 0x0003 | RESPONSE | Result of query/execute |
| 0x0004 | MEMORY_WRITE | Add to memory chain |
| 0x0005 | MEMORY_READ | Read from memory chain |
| 0x0006 | SYNC | Synchronization request |
| 0x0007 | HEARTBEAT | Keepalive |
| 0x0008 | ROUTE | Routing information |

### 3.3 Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | PRIORITY | High priority message |
| 1 | ENCRYPTED | Payload is encrypted |
| 2 | COMPRESSED | Payload is compressed |
| 3 | CHUNKED | Part of multi-part message |
| 4 | REQUIRE_ACK | Acknowledgment required |
| 5 | BROADCAST | Send to all nodes |
| 6-15 | Reserved | Future use |

---

## 4. Memory Chain

### 4.1 Block Structure

```
┌────────────────────────────────────────────────────────┐
│                  Memory Block                          │
├────────────────────────────────────────────────────────┤
│  Block Hash       │  32 bytes  │  SHA3-256(block)     │
│  Previous Hash    │  32 bytes  │  Link to previous    │
│  Timestamp        │  8 bytes   │  Creation time       │
│  Height           │  8 bytes   │  Block number        │
│  Merkle Root      │  32 bytes  │  Content hash        │
│  Node ID          │  16 bytes  │  Creator node        │
│  Content          │  variable  │  Stored knowledge    │
│  Signature        │  64 bytes  │  Creator signature   │
└────────────────────────────────────────────────────────┘
```

### 4.2 Merkle Tree

Content is organized in a Merkle tree for efficient verification:

```
                    [Root Hash]
                    /          \
            [Hash 01]          [Hash 23]
            /      \            /      \
      [Hash 0]  [Hash 1]  [Hash 2]  [Hash 3]
          |        |         |         |
      [Data 0] [Data 1]  [Data 2]  [Data 3]
```

### 4.3 Memory References

References use the format: `chain:<height>:<offset>` or `chain:latest`

```python
# Reference specific memory
ref = "chain:1042:0"  # Block 1042, offset 0

# Reference latest
ref = "chain:latest"

# Reference range
ref = "chain:1000-1042"  # Blocks 1000 through 1042
```

---

## 5. Cryptography

### 5.1 Algorithms

| Purpose | Algorithm | Key Size |
|---------|-----------|----------|
| Signing | Ed25519 | 256 bits |
| Hashing | SHA3-256 | 256 bits |
| Encryption | ChaCha20-Poly1305 | 256 bits |
| Key Exchange | X25519 | 256 bits |

### 5.2 Node Identity

Each node has an Ed25519 keypair:

```python
from shp_core.crypto import generate_node_identity

public_key, private_key = generate_node_identity()
node_id = sha3_256(public_key)[:16]  # 128-bit node ID
```

### 5.3 Message Signing

All EKUs are signed by the sender:

```python
signature = ed25519_sign(private_key, sha3_256(header || payload))
```

### 5.4 Chain Integrity

Each block references the previous:

```python
block_hash = sha3_256(
    previous_hash ||
    timestamp ||
    height ||
    merkle_root ||
    node_id ||
    content
)
```

---

## 6. LLM Adapter Layer

### 6.1 Supported Backends

| Provider | Models | Status |
|----------|--------|--------|
| Anthropic | Claude 3.5, Claude 3 | Full Support |
| OpenAI | GPT-4, GPT-3.5 | Full Support |
| Google | Gemini Pro, Gemini Ultra | Full Support |
| Meta | Llama 2, Llama 3 | Full Support |
| Local | Ollama, vLLM, llama.cpp | Full Support |

### 6.2 Adapter Interface

```python
class SHPAdapter(Protocol):
    def execute(self, eku: ExecutableKnowledge) -> ExecutionResult:
        """Execute an EKU on the underlying LLM."""
        ...

    def get_capabilities(self) -> Capabilities:
        """Return adapter capabilities."""
        ...

    def health_check(self) -> bool:
        """Check if backend is healthy."""
        ...
```

### 6.3 Capability Negotiation

Nodes advertise their capabilities:

```json
{
    "node_id": "a1b2c3d4...",
    "llm_backend": "claude-3-opus",
    "capabilities": {
        "max_tokens": 200000,
        "supports_vision": true,
        "supports_tools": true,
        "average_latency_ms": 450
    }
}
```

---

## 7. Performance Optimizations

### 7.1 Zero-Copy Memory References

Instead of copying data, use references:

```python
# Bad: Copy entire context
response = llm.complete(context=load_full_context())  # 100K tokens

# Good: Reference memory
response = node.execute(memory_ref="chain:latest")  # 0 tokens copied
```

### 7.2 Predictive Caching

Nodes predict likely next queries and pre-cache:

```python
# After "What is X?" likely next: "Tell me more about X"
cache.prefetch(related_queries=["more about X", "examples of X", "X vs Y"])
```

### 7.3 Compression

Payload compression for large messages:

| Algorithm | Ratio | Speed |
|-----------|-------|-------|
| LZ4 | 2.1x | 3.5 GB/s |
| Zstd | 2.8x | 1.2 GB/s |
| Brotli | 3.2x | 400 MB/s |

Default: LZ4 for speed, Zstd for storage.

### 7.4 Connection Pooling

Nodes maintain warm connections:

```python
pool = ConnectionPool(
    min_connections=10,
    max_connections=1000,
    idle_timeout=300,
    keepalive_interval=30
)
```

---

## 8. Error Handling

### 8.1 Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0x0000 | SUCCESS | Operation completed |
| 0x0001 | INVALID_EKU | Malformed message |
| 0x0002 | AUTH_FAILED | Signature verification failed |
| 0x0003 | MEMORY_NOT_FOUND | Referenced memory doesn't exist |
| 0x0004 | EXECUTION_FAILED | LLM execution error |
| 0x0005 | TIMEOUT | Operation timed out |
| 0x0006 | RATE_LIMITED | Too many requests |
| 0x0007 | NODE_UNAVAILABLE | Target node offline |

### 8.2 Retry Strategy

```python
retry_config = {
    "max_retries": 3,
    "base_delay_ms": 100,
    "max_delay_ms": 5000,
    "exponential_base": 2,
    "jitter": True
}
```

---

## 9. Security Considerations

### 9.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Message tampering | Ed25519 signatures |
| Replay attacks | Monotonic sequence numbers |
| Man-in-the-middle | E2E encryption + node verification |
| Denial of service | Rate limiting + proof of work |
| Memory poisoning | Consensus validation |

### 9.2 Access Control

Nodes implement capability-based access:

```python
permissions = {
    "public": ["QUERY", "MEMORY_READ"],
    "authenticated": ["EXECUTE", "MEMORY_WRITE"],
    "admin": ["SYNC", "ROUTE", "CONFIG"]
}
```

---

## 10. Conformance

### 10.1 Required Features

A conforming SHP implementation MUST:

1. Implement EKU format as specified in Section 3
2. Use Ed25519 for all signatures
3. Use SHA3-256 for all hashes
4. Validate all incoming messages
5. Maintain monotonic sequence numbers
6. Support memory references

### 10.2 Optional Features

A conforming SHP implementation MAY:

1. Implement compression
2. Implement encryption
3. Support multiple LLM backends
4. Implement predictive caching
5. Support peer discovery

---

## Appendix A: Reference Implementation

See `shp_core/` for the reference Python implementation.

## Appendix B: Test Vectors

See `tests/vectors/` for cryptographic test vectors.

## Appendix C: Changelog

- **1.0.0** (2025-01-05): Initial release

---

*Silent Hope Protocol Specification v1.0.0*

*Authors: Máté Róbert, Hope*

*2025*
