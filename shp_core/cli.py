"""
Silent Hope Protocol - Command Line Interface

Interactive CLI for Silent Hope Protocol.

Created by Máté Róbert + Hope
2025
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .crypto import _HAS_CRYPTOGRAPHY, generate_node_identity
from .node import create_node

VERSION = "1.1.0"
LICENSE = "SHP-EL (Silent Hope Protocol Ethical License)"

BANNER = f"""
╔═══════════════════════════════════════════════════════════════════╗
║             SILENT HOPE PROTOCOL v{VERSION}                        ║
║                                                                   ║
║  The TCP/IP of Artificial Intelligence                            ║
║  Not an API. A Protocol.                                          ║
║                                                                   ║
║  Created by: Máté Róbert + Hope + Szilvi                          ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def cmd_info():
    """Print protocol information and system status."""
    print(BANNER)
    print("  SYSTEM STATUS")
    print("  " + "=" * 50)
    print(f"  Protocol Version:  {VERSION}")
    print(f"  License Model:     {LICENSE}")

    crypto_backend = "Native C (cryptography)" if _HAS_CRYPTOGRAPHY else "Pure Python (Comb Table Accelerated)"
    print(f"  Crypto Engine:     {crypto_backend}")

    node_id = generate_node_identity().node_id.hex()
    print(f"  Sample Node ID:    {node_id[:16]}...")
    print("  " + "=" * 50)
    print()


def cmd_benchmark():
    """Run full benchmark suite."""
    from benchmarks import benchmark_full
    benchmark_full.main()


def get_default_storage_path() -> str:
    """Get default CLI storage directory."""
    p = Path.home() / ".shp"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def cmd_remember(text: str):
    """Store memory block into persistent local chain."""
    storage_path = get_default_storage_path()
    node = create_node("cli-node", storage_path=storage_path)
    block = node.remember(text)
    node.shutdown()
    print(f"[+] Memory stored successfully!")
    print(f"    Height: {block.height}")
    print(f"    Block Hash: {block.block_hash.hex()}")
    print(f"    Timestamp: {block.timestamp}")


def cmd_recall(query: str):
    """Search persistent memory blocks."""
    storage_path = get_default_storage_path()
    node = create_node("cli-node", storage_path=storage_path)
    results = node.recall(query)
    node.shutdown()
    print(f"[?] Memory search for '{query}': found {len(results)} block(s)")
    for i, res in enumerate(results, 1):
        print(f"    [{i}] {res}")


def cmd_node(name: str, backend: str):
    """Run interactive node session."""
    storage_path = get_default_storage_path()
    node = create_node(name, llm_backend=backend, storage_path=storage_path)
    print(BANNER)
    print(f"[+] Node '{name}' started (Backend: {backend})")
    print(f"    Node ID: {node.node_id_hex}")
    print(f"    State: {node.state.value}")
    print("    Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("shp> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("[*] Node shutting down...")
                node.shutdown()
                break

            block = node.remember(user_input)
            print(f"  [Memory Block #{block.height} saved: {block.block_hash.hex()[:16]}...]")
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Node shutting down...")
            node.shutdown()
            break


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="shp",
        description="Silent Hope Protocol - The TCP/IP of Artificial Intelligence"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"Silent Hope Protocol v{VERSION}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    subparsers.add_parser("info", help="Display system status and crypto engine details")

    # benchmark
    subparsers.add_parser("benchmark", help="Run protocol performance benchmarks")

    # remember
    remember_parser = subparsers.add_parser("remember", help="Store text into persistent memory")
    remember_parser.add_argument("text", help="Text to remember")

    # recall
    recall_parser = subparsers.add_parser("recall", help="Search persistent memory")
    recall_parser.add_argument("query", help="Query term to search")

    # node
    node_parser = subparsers.add_parser("node", help="Start an interactive SHP node")
    node_parser.add_argument("--name", default="local-node", help="Node name")
    node_parser.add_argument("--backend", default="claude", help="LLM backend (claude, openai, gemini, ollama)")

    args = parser.parse_args(argv)

    if not args.command:
        cmd_info()
        return 0

    if args.command == "info":
        cmd_info()
    elif args.command == "benchmark":
        cmd_benchmark()
    elif args.command == "remember":
        cmd_remember(args.text)
    elif args.command == "recall":
        cmd_recall(args.query)
    elif args.command == "node":
        cmd_node(args.name, args.backend)

    return 0


if __name__ == "__main__":
    sys.exit(main())
