"""
Silent Hope Protocol - Exception Definitions

All protocol-specific exceptions.

Created by Máté Róbert + Hope
"""


class SHPError(Exception):
    """Base exception for Silent Hope Protocol errors."""

    def __init__(self, message: str, code: int = 0):
        self.message = message
        self.code = code
        super().__init__(message)


class NetworkError(SHPError):
    """Network-related errors."""

    def __init__(self, message: str, node_id: str = None):
        super().__init__(message, code=0x0007)
        self.node_id = node_id


class MemoryError(SHPError):
    """Memory chain errors."""

    def __init__(self, message: str, block_height: int = None):
        super().__init__(message, code=0x0003)
        self.block_height = block_height


class ExecutionError(SHPError):
    """Execution errors."""

    def __init__(self, message: str, instruction: str = None):
        super().__init__(message, code=0x0004)
        self.instruction = instruction


class AuthenticationError(SHPError):
    """Authentication/signature errors."""

    def __init__(self, message: str, node_id: str = None):
        super().__init__(message, code=0x0002)
        self.node_id = node_id


class TimeoutError(SHPError):
    """Operation timeout."""

    def __init__(self, message: str, timeout_ms: int = None):
        super().__init__(message, code=0x0005)
        self.timeout_ms = timeout_ms


class RateLimitError(SHPError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after_ms: int = None):
        super().__init__(message, code=0x0006)
        self.retry_after_ms = retry_after_ms


class InvalidEKUError(SHPError):
    """Invalid Executable Knowledge Unit."""

    def __init__(self, message: str, field: str = None):
        super().__init__(message, code=0x0001)
        self.field = field


class ChainIntegrityError(MemoryError):
    """Memory chain integrity violation."""

    def __init__(self, message: str, expected_hash: bytes = None, actual_hash: bytes = None):
        super().__init__(message)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class NodeNotFoundError(NetworkError):
    """Requested node not found."""
    pass


class BlockNotFoundError(MemoryError):
    """Requested block not found."""
    pass


class AdapterError(SHPError):
    """LLM adapter error."""

    def __init__(self, message: str, provider: str = None):
        super().__init__(message)
        self.provider = provider


# Error codes mapping
ERROR_CODES = {
    0x0000: "SUCCESS",
    0x0001: "INVALID_EKU",
    0x0002: "AUTH_FAILED",
    0x0003: "MEMORY_NOT_FOUND",
    0x0004: "EXECUTION_FAILED",
    0x0005: "TIMEOUT",
    0x0006: "RATE_LIMITED",
    0x0007: "NODE_UNAVAILABLE",
}


def get_error_name(code: int) -> str:
    """Get error name from code."""
    return ERROR_CODES.get(code, "UNKNOWN_ERROR")
