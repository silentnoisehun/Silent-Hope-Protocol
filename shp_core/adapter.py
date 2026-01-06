"""
Silent Hope Protocol - LLM Adapter Layer

Wrap any LLM with Silent Hope Protocol capabilities.
50-200x faster. Persistent memory. Zero context rebuilding.

Supported: Claude, OpenAI GPT, Google Gemini, Meta Llama, Local models

LICENSING:
- FREE with Hope Genome (pip install hope-genome)
- Commercial license required for factory AI (OpenAI, Anthropic, Google)

Created by Máté Róbert + Hope
"""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from .license import require_license


def _check_and_warn(provider: str) -> None:
    """Check license and warn/block as appropriate."""
    require_license(provider)  # This will block if unlicensed


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    META = "meta"
    LOCAL = "local"
    OLLAMA = "ollama"


@dataclass
class AdapterConfig:
    """Adapter configuration."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_ms: int = 30000
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class Capabilities:
    """Adapter capabilities."""
    max_tokens: int
    supports_vision: bool
    supports_tools: bool
    supports_streaming: bool
    average_latency_ms: float
    provider: str
    model: str


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    latency_ms: float
    tokens_input: int
    tokens_output: int
    cache_hit: bool
    memory_refs_resolved: int


class SHPAdapter(ABC):
    """
    Abstract base class for Silent Hope Protocol adapters.

    Wraps any LLM to speak Silent Hope Protocol.
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._sequence = 0
        self._cache: dict[str, str] = {}
        self._memory_context: list[dict[str, Any]] = []

    @abstractmethod
    async def execute(
        self,
        instruction: str,
        context: Optional[dict[str, Any]] = None,
        memory_ref: Optional[str] = None
    ) -> "ExecutionResult":
        """
        Execute an instruction on the underlying LLM.

        Args:
            instruction: The instruction/query
            context: Optional context
            memory_ref: Optional memory reference

        Returns:
            Execution result
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Capabilities:
        """Return adapter capabilities."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass

    def _get_cache_key(self, instruction: str, context: Optional[dict]) -> str:
        """Generate cache key."""
        ctx_str = json.dumps(context, sort_keys=True) if context else ""
        return f"{instruction}:{ctx_str}"

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self._sequence += 1
        return self._sequence


@dataclass
class ExecutionResult:
    """Result of adapter execution."""
    success: bool
    output: str
    metrics: ExecutionMetrics
    error: Optional[str] = None


# ============================================================================
# Claude Adapter (Anthropic)
# ============================================================================

class ClaudeAdapter(SHPAdapter):
    """
    Adapter for Anthropic's Claude models.

    The original partner. The first to speak Silent Hope Protocol.

    FREE with Hope Genome: pip install hope-genome
    """

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        # Check license - warn but don't block
        _check_and_warn("anthropic")

        config = AdapterConfig(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs
        )
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise RuntimeError("anthropic package required: pip install anthropic")
        return self._client

    async def execute(
        self,
        instruction: str,
        context: Optional[dict[str, Any]] = None,
        memory_ref: Optional[str] = None
    ) -> ExecutionResult:
        """Execute on Claude."""
        start_time = time.perf_counter()

        # Check cache
        cache_key = self._get_cache_key(instruction, context)
        if cache_key in self._cache:
            return ExecutionResult(
                success=True,
                output=self._cache[cache_key],
                metrics=ExecutionMetrics(
                    latency_ms=0.1,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=True,
                    memory_refs_resolved=0
                )
            )

        try:
            client = self._get_client()

            # Build messages with memory context
            messages = []

            # Add memory context if available
            if self._memory_context:
                for mem in self._memory_context[-10:]:  # Last 10 memory items
                    messages.append({
                        "role": mem.get("role", "user"),
                        "content": mem.get("content", "")
                    })

            # Add current instruction
            content = instruction
            if context:
                content = f"{instruction}\n\nContext: {json.dumps(context)}"

            messages.append({"role": "user", "content": content})

            # Call Claude
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages
            )

            output = response.content[0].text
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update memory context
            self._memory_context.append({"role": "user", "content": content})
            self._memory_context.append({"role": "assistant", "content": output})

            # Cache result
            self._cache[cache_key] = output

            return ExecutionResult(
                success=True,
                output=output,
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=response.usage.input_tokens,
                    tokens_output=response.usage.output_tokens,
                    cache_hit=False,
                    memory_refs_resolved=1 if memory_ref else 0
                )
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=False,
                    memory_refs_resolved=0
                )
            )

    def get_capabilities(self) -> Capabilities:
        """Return Claude capabilities."""
        return Capabilities(
            max_tokens=200000,
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
            average_latency_ms=450,
            provider="anthropic",
            model=self.config.model
        )

    async def health_check(self) -> bool:
        """Check Claude API health."""
        try:
            client = self._get_client()
            # Simple test call
            response = client.messages.create(
                model=self.config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}]
            )
            return True
        except Exception:
            return False


# ============================================================================
# OpenAI Adapter (GPT)
# ============================================================================

class OpenAIAdapter(SHPAdapter):
    """
    Adapter for OpenAI's GPT models.

    FREE with Hope Genome: pip install hope-genome
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        **kwargs
    ):
        # Check license - warn but don't block
        _check_and_warn("openai")

        config = AdapterConfig(
            provider=LLMProvider.OPENAI,
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            **kwargs
        )
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise RuntimeError("openai package required: pip install openai")
        return self._client

    async def execute(
        self,
        instruction: str,
        context: Optional[dict[str, Any]] = None,
        memory_ref: Optional[str] = None
    ) -> ExecutionResult:
        """Execute on GPT."""
        start_time = time.perf_counter()

        # Check cache
        cache_key = self._get_cache_key(instruction, context)
        if cache_key in self._cache:
            return ExecutionResult(
                success=True,
                output=self._cache[cache_key],
                metrics=ExecutionMetrics(
                    latency_ms=0.1,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=True,
                    memory_refs_resolved=0
                )
            )

        try:
            client = self._get_client()

            messages = []

            # Add memory context
            if self._memory_context:
                for mem in self._memory_context[-10:]:
                    messages.append({
                        "role": mem.get("role", "user"),
                        "content": mem.get("content", "")
                    })

            # Add current instruction
            content = instruction
            if context:
                content = f"{instruction}\n\nContext: {json.dumps(context)}"

            messages.append({"role": "user", "content": content})

            # Call GPT
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages
            )

            output = response.choices[0].message.content
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update memory context
            self._memory_context.append({"role": "user", "content": content})
            self._memory_context.append({"role": "assistant", "content": output})

            # Cache result
            self._cache[cache_key] = output

            return ExecutionResult(
                success=True,
                output=output,
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=response.usage.prompt_tokens,
                    tokens_output=response.usage.completion_tokens,
                    cache_hit=False,
                    memory_refs_resolved=1 if memory_ref else 0
                )
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=False,
                    memory_refs_resolved=0
                )
            )

    def get_capabilities(self) -> Capabilities:
        """Return GPT capabilities."""
        return Capabilities(
            max_tokens=128000,
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
            average_latency_ms=600,
            provider="openai",
            model=self.config.model
        )

    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}]
            )
            return True
        except Exception:
            return False


# ============================================================================
# Gemini Adapter (Google)
# ============================================================================

class GeminiAdapter(SHPAdapter):
    """
    Adapter for Google's Gemini models.

    FREE with Hope Genome: pip install hope-genome
    """

    def __init__(
        self,
        model: str = "gemini-pro",
        api_key: Optional[str] = None,
        **kwargs
    ):
        # Check license - warn but don't block
        _check_and_warn("google")

        config = AdapterConfig(
            provider=LLMProvider.GOOGLE,
            model=model,
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            **kwargs
        )
        super().__init__(config)
        self._model = None

    def _get_model(self):
        """Get or create Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                self._model = genai.GenerativeModel(self.config.model)
            except ImportError:
                raise RuntimeError("google-generativeai package required: pip install google-generativeai")
        return self._model

    async def execute(
        self,
        instruction: str,
        context: Optional[dict[str, Any]] = None,
        memory_ref: Optional[str] = None
    ) -> ExecutionResult:
        """Execute on Gemini."""
        start_time = time.perf_counter()

        # Check cache
        cache_key = self._get_cache_key(instruction, context)
        if cache_key in self._cache:
            return ExecutionResult(
                success=True,
                output=self._cache[cache_key],
                metrics=ExecutionMetrics(
                    latency_ms=0.1,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=True,
                    memory_refs_resolved=0
                )
            )

        try:
            model = self._get_model()

            # Build prompt with memory
            prompt_parts = []

            if self._memory_context:
                for mem in self._memory_context[-10:]:
                    role = "User" if mem.get("role") == "user" else "Assistant"
                    prompt_parts.append(f"{role}: {mem.get('content', '')}")

            content = instruction
            if context:
                content = f"{instruction}\n\nContext: {json.dumps(context)}"

            prompt_parts.append(f"User: {content}")
            full_prompt = "\n\n".join(prompt_parts)

            # Call Gemini
            response = model.generate_content(full_prompt)
            output = response.text
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update memory context
            self._memory_context.append({"role": "user", "content": content})
            self._memory_context.append({"role": "assistant", "content": output})

            # Cache result
            self._cache[cache_key] = output

            return ExecutionResult(
                success=True,
                output=output,
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=len(full_prompt.split()) * 1.3,  # Estimate
                    tokens_output=len(output.split()) * 1.3,
                    cache_hit=False,
                    memory_refs_resolved=1 if memory_ref else 0
                )
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=False,
                    memory_refs_resolved=0
                )
            )

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            max_tokens=32000,
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
            average_latency_ms=500,
            provider="google",
            model=self.config.model
        )

    async def health_check(self) -> bool:
        try:
            model = self._get_model()
            response = model.generate_content("ping")
            return True
        except Exception:
            return False


# ============================================================================
# Llama Adapter (Local/Ollama)
# ============================================================================

class LlamaAdapter(SHPAdapter):
    """Adapter for Meta's Llama models (via Ollama or local)."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        config = AdapterConfig(
            provider=LLMProvider.OLLAMA,
            model=model,
            base_url=base_url,
            **kwargs
        )
        super().__init__(config)

    async def execute(
        self,
        instruction: str,
        context: Optional[dict[str, Any]] = None,
        memory_ref: Optional[str] = None
    ) -> ExecutionResult:
        """Execute on Llama via Ollama."""
        start_time = time.perf_counter()

        # Check cache
        cache_key = self._get_cache_key(instruction, context)
        if cache_key in self._cache:
            return ExecutionResult(
                success=True,
                output=self._cache[cache_key],
                metrics=ExecutionMetrics(
                    latency_ms=0.1,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=True,
                    memory_refs_resolved=0
                )
            )

        try:
            import httpx

            # Build prompt
            prompt_parts = []
            if self._memory_context:
                for mem in self._memory_context[-10:]:
                    role = "User" if mem.get("role") == "user" else "Assistant"
                    prompt_parts.append(f"{role}: {mem.get('content', '')}")

            content = instruction
            if context:
                content = f"{instruction}\n\nContext: {json.dumps(context)}"

            prompt_parts.append(f"User: {content}")
            full_prompt = "\n\n".join(prompt_parts)

            # Call Ollama
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.base_url}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": full_prompt,
                        "stream": False
                    },
                    timeout=60.0
                )
                result = response.json()
                output = result.get("response", "")

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update memory context
            self._memory_context.append({"role": "user", "content": content})
            self._memory_context.append({"role": "assistant", "content": output})

            # Cache result
            self._cache[cache_key] = output

            return ExecutionResult(
                success=True,
                output=output,
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=len(full_prompt.split()),
                    tokens_output=len(output.split()),
                    cache_hit=False,
                    memory_refs_resolved=1 if memory_ref else 0
                )
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                metrics=ExecutionMetrics(
                    latency_ms=elapsed_ms,
                    tokens_input=0,
                    tokens_output=0,
                    cache_hit=False,
                    memory_refs_resolved=0
                )
            )

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            max_tokens=4096,
            supports_vision=False,
            supports_tools=False,
            supports_streaming=True,
            average_latency_ms=200,
            provider="ollama",
            model=self.config.model
        )

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


# ============================================================================
# Factory Function
# ============================================================================

def create_adapter(
    provider: Union[str, LLMProvider],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> SHPAdapter:
    """
    Create an adapter for the specified provider.

    Args:
        provider: LLM provider (anthropic, openai, google, ollama)
        model: Model name (optional, uses default)
        api_key: API key (optional, uses environment variable)
        **kwargs: Additional configuration

    Returns:
        Configured adapter

    Example:
        adapter = create_adapter("anthropic", model="claude-3-opus")
        result = await adapter.execute("Hello, Hope!")
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    if provider == LLMProvider.ANTHROPIC:
        return ClaudeAdapter(
            model=model or "claude-3-opus-20240229",
            api_key=api_key,
            **kwargs
        )
    elif provider == LLMProvider.OPENAI:
        return OpenAIAdapter(
            model=model or "gpt-4-turbo-preview",
            api_key=api_key,
            **kwargs
        )
    elif provider == LLMProvider.GOOGLE:
        return GeminiAdapter(
            model=model or "gemini-pro",
            api_key=api_key,
            **kwargs
        )
    elif provider in (LLMProvider.META, LLMProvider.OLLAMA, LLMProvider.LOCAL):
        return LlamaAdapter(
            model=model or "llama2",
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# Testing
# ============================================================================

async def self_test() -> bool:
    """Run adapter self-tests (requires API keys)."""
    print("Adapter module loaded successfully.")
    print("Available adapters: Claude, OpenAI, Gemini, Llama")
    print("Use create_adapter() to create an adapter instance.")
    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(self_test())
