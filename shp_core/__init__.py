"""
Silent Hope Protocol - Core Library

The TCP/IP of Artificial Intelligence.
Not an API. A Protocol.

Created by Máté Róbert + Hope + Szilvi
2025
"""

__version__ = "1.1.0"
__author__ = "Máté Róbert, Hope, Szilvi"
__license__ = "SHP-EL (Silent Hope Protocol Ethical License)"

from .adapter import (
    ClaudeAdapter,
    GeminiAdapter,
    LlamaAdapter,
    OpenAIAdapter,
    SHPAdapter,
    create_adapter,
)
from .crypto import CryptoError, generate_node_identity, hash_block, sign_message, verify_signature
from .exceptions import AuthenticationError, ExecutionError, MemoryError, NetworkError, SHPError
from .license import (
    LicenseType,
    LicenseInfo,
    LicenseError,
    check_license,
    require_license,
    get_license_status,
)
from .memory import MemoryBlock, MemoryChain, MemoryRef
from .network import NetworkConfig, SHPNetwork
from .node import NodeCapabilities, NodeConfig, SilentHopeNode
from .protocol import EKUHeader, EKUType, ExecutableKnowledge, ExecutionResult

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",

    # Core
    "SilentHopeNode",
    "NodeConfig",
    "NodeCapabilities",
    "SHPNetwork",
    "NetworkConfig",
    "MemoryChain",
    "MemoryBlock",
    "MemoryRef",

    # Protocol
    "ExecutableKnowledge",
    "EKUHeader",
    "EKUType",
    "ExecutionResult",

    # Adapters
    "SHPAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "LlamaAdapter",
    "create_adapter",

    # Crypto
    "generate_node_identity",
    "sign_message",
    "verify_signature",
    "hash_block",
    "CryptoError",

    # Exceptions
    "SHPError",
    "NetworkError",
    "MemoryError",
    "ExecutionError",
    "AuthenticationError",

    # License
    "LicenseType",
    "LicenseInfo",
    "LicenseError",
    "check_license",
    "require_license",
    "get_license_status",
]

# Banner
BANNER = f"""
╔═══════════════════════════════════════════════════════════════════╗
║             SILENT HOPE PROTOCOL v{__version__}                        ║
║                                                                   ║
║  The TCP/IP of Artificial Intelligence                            ║
║  Not an API. A Protocol.                                          ║
║                                                                   ║
║  Created by: Máté Róbert + Hope + Szilvi                          ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Print the Silent Hope Protocol banner."""
    print(BANNER)
