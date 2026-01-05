"""
Silent Hope Protocol - Protocol Definitions

ExecutableKnowledge Units (EKU) and message formats.
Communication IS execution. No parsing. No rebuilding.

Created by Máté Róbert + Hope
"""

import json
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Any, Optional

from .crypto import sha3_256, sign_message, verify_signature


class EKUType(IntEnum):
    """Executable Knowledge Unit types."""
    QUERY = 0x0001          # Request for information
    EXECUTE = 0x0002        # Request for computation
    RESPONSE = 0x0003       # Result of query/execute
    MEMORY_WRITE = 0x0004   # Add to memory chain
    MEMORY_READ = 0x0005    # Read from memory chain
    SYNC = 0x0006           # Synchronization request
    HEARTBEAT = 0x0007      # Keepalive
    ROUTE = 0x0008          # Routing information


class EKUFlags(IntFlag):
    """EKU flags."""
    NONE = 0x0000
    PRIORITY = 0x0001       # High priority message
    ENCRYPTED = 0x0002      # Payload is encrypted
    COMPRESSED = 0x0004     # Payload is compressed
    CHUNKED = 0x0008        # Part of multi-part message
    REQUIRE_ACK = 0x0010    # Acknowledgment required
    BROADCAST = 0x0020      # Send to all nodes


@dataclass
class EKUHeader:
    """
    Executable Knowledge Unit header.

    64 bytes fixed size.
    """
    version: int = 0x0100           # 4 bytes - Protocol version 1.0
    eku_type: EKUType = EKUType.QUERY  # 2 bytes
    flags: EKUFlags = EKUFlags.NONE    # 2 bytes
    timestamp: int = 0              # 8 bytes - Unix timestamp (ns)
    sender_id: bytes = b"\x00" * 16    # 16 bytes - Node identifier
    sequence: int = 0               # 8 bytes - Monotonic sequence
    memory_ref: bytes = b"\x00" * 16   # 16 bytes - Chain reference
    payload_length: int = 0         # 4 bytes
    reserved: bytes = b"\x00" * 4      # 4 bytes - Future use

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time_ns()

    def to_bytes(self) -> bytes:
        """Serialize header to 64 bytes."""
        return (
            struct.pack('>I', self.version) +
            struct.pack('>H', self.eku_type) +
            struct.pack('>H', self.flags) +
            struct.pack('>Q', self.timestamp) +
            self.sender_id[:16].ljust(16, b'\x00') +
            struct.pack('>Q', self.sequence) +
            self.memory_ref[:16].ljust(16, b'\x00') +
            struct.pack('>I', self.payload_length) +
            self.reserved[:4].ljust(4, b'\x00')
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "EKUHeader":
        """Deserialize header from 64 bytes."""
        if len(data) < 64:
            raise ValueError("Header must be at least 64 bytes")

        return cls(
            version=struct.unpack('>I', data[0:4])[0],
            eku_type=EKUType(struct.unpack('>H', data[4:6])[0]),
            flags=EKUFlags(struct.unpack('>H', data[6:8])[0]),
            timestamp=struct.unpack('>Q', data[8:16])[0],
            sender_id=data[16:32],
            sequence=struct.unpack('>Q', data[32:40])[0],
            memory_ref=data[40:56],
            payload_length=struct.unpack('>I', data[56:60])[0],
            reserved=data[60:64]
        )


@dataclass
class KnowledgeBlock:
    """Executable knowledge content."""
    instruction: str
    context: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        data = {
            "instruction": self.instruction,
            "context": self.context,
            "parameters": self.parameters,
            "constraints": self.constraints
        }
        return json.dumps(data, separators=(',', ':')).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> "KnowledgeBlock":
        """Deserialize from bytes."""
        obj = json.loads(data.decode('utf-8'))
        return cls(
            instruction=obj.get("instruction", ""),
            context=obj.get("context", {}),
            parameters=obj.get("parameters", {}),
            constraints=obj.get("constraints", [])
        )


@dataclass
class ExecutableKnowledge:
    """
    Complete Executable Knowledge Unit.

    The fundamental unit of Silent Hope Protocol communication.
    Not data to parse - executable knowledge to run.
    """
    header: EKUHeader
    knowledge: KnowledgeBlock
    proof: bytes = b""  # Cryptographic proof
    signature: bytes = b"\x00" * 64

    def to_bytes(self) -> bytes:
        """Serialize complete EKU."""
        knowledge_bytes = self.knowledge.to_bytes()
        proof_bytes = self.proof

        # Update payload length
        self.header.payload_length = len(knowledge_bytes) + len(proof_bytes) + 4

        header_bytes = self.header.to_bytes()

        payload = (
            struct.pack('>I', len(knowledge_bytes)) +
            knowledge_bytes +
            proof_bytes
        )

        return header_bytes + payload + self.signature

    @classmethod
    def from_bytes(cls, data: bytes) -> "ExecutableKnowledge":
        """Deserialize complete EKU."""
        header = EKUHeader.from_bytes(data[:64])

        knowledge_len = struct.unpack('>I', data[64:68])[0]
        knowledge_bytes = data[68:68 + knowledge_len]
        knowledge = KnowledgeBlock.from_bytes(knowledge_bytes)

        proof_start = 68 + knowledge_len
        proof_end = 64 + header.payload_length
        proof = data[proof_start:proof_end]

        signature = data[proof_end:proof_end + 64]

        return cls(
            header=header,
            knowledge=knowledge,
            proof=proof,
            signature=signature
        )

    def sign(self, private_key: bytes) -> None:
        """Sign the EKU."""
        knowledge_bytes = self.knowledge.to_bytes()
        # Update payload_length BEFORE signing so it's consistent
        self.header.payload_length = len(knowledge_bytes) + len(self.proof) + 4
        header_bytes = self.header.to_bytes()
        message = sha3_256(header_bytes + knowledge_bytes + self.proof)
        self.signature = sign_message(private_key, message)

    def verify(self, public_key: bytes) -> bool:
        """Verify the EKU signature."""
        header_bytes = self.header.to_bytes()
        knowledge_bytes = self.knowledge.to_bytes()
        message = sha3_256(header_bytes + knowledge_bytes + self.proof)
        return verify_signature(public_key, message, self.signature)

    @classmethod
    def create_query(
        cls,
        instruction: str,
        sender_id: bytes,
        context: Optional[dict[str, Any]] = None,
        memory_ref: Optional[bytes] = None
    ) -> "ExecutableKnowledge":
        """Create a QUERY type EKU."""
        header = EKUHeader(
            eku_type=EKUType.QUERY,
            sender_id=sender_id,
            memory_ref=memory_ref or b"\x00" * 16
        )
        knowledge = KnowledgeBlock(
            instruction=instruction,
            context=context or {}
        )
        return cls(header=header, knowledge=knowledge)

    @classmethod
    def create_execute(
        cls,
        instruction: str,
        sender_id: bytes,
        parameters: Optional[dict[str, Any]] = None,
        constraints: Optional[list[str]] = None,
        memory_ref: Optional[bytes] = None
    ) -> "ExecutableKnowledge":
        """Create an EXECUTE type EKU."""
        header = EKUHeader(
            eku_type=EKUType.EXECUTE,
            sender_id=sender_id,
            memory_ref=memory_ref or b"\x00" * 16
        )
        knowledge = KnowledgeBlock(
            instruction=instruction,
            parameters=parameters or {},
            constraints=constraints or []
        )
        return cls(header=header, knowledge=knowledge)

    @classmethod
    def create_response(
        cls,
        result: str,
        sender_id: bytes,
        in_response_to: int,
        context: Optional[dict[str, Any]] = None
    ) -> "ExecutableKnowledge":
        """Create a RESPONSE type EKU."""
        header = EKUHeader(
            eku_type=EKUType.RESPONSE,
            sender_id=sender_id,
            sequence=in_response_to
        )
        knowledge = KnowledgeBlock(
            instruction=result,
            context=context or {}
        )
        return cls(header=header, knowledge=knowledge)


@dataclass
class ExecutionResult:
    """Result of executing an EKU."""
    success: bool
    output: str
    execution_time_ms: float
    memory_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "execution_time_ms": self.execution_time_ms,
            "memory_refs": self.memory_refs,
            "metadata": self.metadata
        }


class ProtocolError(Exception):
    """Protocol error."""
    pass


class InvalidEKUError(ProtocolError):
    """Invalid EKU format."""
    pass


class AuthenticationError(ProtocolError):
    """Authentication/signature error."""
    pass


# ============================================================================
# Protocol Constants
# ============================================================================

PROTOCOL_VERSION = 0x0100  # 1.0
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_KNOWLEDGE_SIZE = 1 * 1024 * 1024  # 1 MB
DEFAULT_TIMEOUT_MS = 30000  # 30 seconds


# ============================================================================
# Testing
# ============================================================================

def self_test() -> bool:
    """Run protocol self-tests."""
    from .crypto import generate_node_identity

    # Generate identity
    keypair = generate_node_identity()

    # Test QUERY creation
    query = ExecutableKnowledge.create_query(
        instruction="What is the Silent Hope Protocol?",
        sender_id=keypair.node_id,
        context={"source": "test"}
    )

    assert query.header.eku_type == EKUType.QUERY
    assert query.knowledge.instruction == "What is the Silent Hope Protocol?"

    # Test signing
    query.sign(keypair.private_key)
    assert len(query.signature) == 64

    # Test verification
    assert query.verify(keypair.public_key)

    # Test serialization
    data = query.to_bytes()
    restored = ExecutableKnowledge.from_bytes(data)

    assert restored.header.eku_type == query.header.eku_type
    assert restored.knowledge.instruction == query.knowledge.instruction
    assert restored.verify(keypair.public_key)

    # Test EXECUTE creation
    execute = ExecutableKnowledge.create_execute(
        instruction="Analyze market trends",
        sender_id=keypair.node_id,
        parameters={"depth": "full", "timeframe": "1y"},
        constraints=["no_financial_advice", "cite_sources"]
    )

    assert execute.header.eku_type == EKUType.EXECUTE
    assert execute.knowledge.parameters["depth"] == "full"
    assert "no_financial_advice" in execute.knowledge.constraints

    # Test RESPONSE creation
    response = ExecutableKnowledge.create_response(
        result="Analysis complete. Key findings: ...",
        sender_id=keypair.node_id,
        in_response_to=query.header.sequence
    )

    assert response.header.eku_type == EKUType.RESPONSE

    # Test ExecutionResult
    result = ExecutionResult(
        success=True,
        output="Test passed",
        execution_time_ms=12.5,
        memory_refs=["chain:latest"],
        metadata={"tokens_used": 100}
    )

    result_dict = result.to_dict()
    assert result_dict["success"] == True
    assert result_dict["execution_time_ms"] == 12.5

    return True


if __name__ == "__main__":
    print("Running protocol self-tests...")
    if self_test():
        print("All tests passed!")
    else:
        print("Tests failed!")
        exit(1)
