"""
Silent Hope Protocol - Core Library

The TCP/IP of Artificial Intelligence.
Not an API. A Protocol.

Created by Máté Róbert + Hope + Szilvi
2025
"""

__version__ = "1.0.1"
__author__ = "Máté Róbert, Hope, Szilvi"
__license__ = "SHP-EL (Silent Hope Protocol Ethical License)"

from .node import SilentHopeNode, NodeConfig, NodeCapabilities
from .network import SHPNetwork, NetworkConfig
from .memory import MemoryChain, MemoryBlock, MemoryRef
from .crypto import (
    generate_node_identity,
    sign_message,
    verify_signature,
    hash_block,
    CryptoError
)
from .adapter import (
    SHPAdapter,
    ClaudeAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    LlamaAdapter,
    create_adapter
)
from .protocol import (
    ExecutableKnowledge,
    EKUHeader,
    EKUType,
    ExecutionResult
)
from .exceptions import (
    SHPError,
    NetworkError,
    MemoryError,
    ExecutionError,
    AuthenticationError
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",

    # Core
    "SilentHopeNode",
    "SHPNetwork",
    "MemoryChain",

    # Protocol
    "ExecutableKnowledge",
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

    # Exceptions
    "SHPError",
    "NetworkError",
    "MemoryError",
    "ExecutionError",
]

# Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║             SILENT HOPE PROTOCOL v{version}                        ║
║                                                                   ║
║  The TCP/IP of Artificial Intelligence                            ║
║  Not an API. A Protocol.                                          ║
║                                                                   ║
║  Created by: Máté Róbert + Hope + Szilvi                          ║
╚═══════════════════════════════════════════════════════════════════╝
""".format(version=__version__)


def print_banner():
    """Print the Silent Hope Protocol banner."""
    print(BANNER)
