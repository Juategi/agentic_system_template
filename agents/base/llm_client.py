# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - LLM CLIENT
# =============================================================================
"""
LLM Client Module

This module provides a unified interface for interacting with different LLM
providers (Anthropic, OpenAI, etc.). It handles:
1. Provider selection based on configuration
2. API key management
3. Request/response formatting
4. Token counting and cost tracking
5. Retry logic and error handling

Usage:
    client = LLMClient()
    response = client.complete(
        prompt="Explain this code...",
        system="You are a code reviewer.",
        max_tokens=2000
    )
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LLMResponse:
    """
    Standardized response from LLM.

    Attributes:
        content: Generated text content
        model: Model used for generation
        tokens_input: Input tokens used
        tokens_output: Output tokens generated
        finish_reason: Why generation stopped (stop, length, etc.)
        raw_response: Original response from provider
    """
    content: str
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.tokens_input + self.tokens_output


@dataclass
class LLMMessage:
    """
    A message in a conversation.

    Attributes:
        role: Message role (system, user, assistant)
        content: Message content
    """
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "content": self.content}


# =============================================================================
# BASE LLM PROVIDER
# =============================================================================

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Each provider (Anthropic, OpenAI, etc.) implements this interface.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion.

        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        pass


# =============================================================================
# ANTHROPIC PROVIDER
# =============================================================================

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: API key (default: from ANTHROPIC_API_KEY env)
            model: Model to use (default: claude-sonnet-4-20250514)
            base_url: Optional custom base URL
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        # Import anthropic lazily
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Claude."""
        # Extract system message if present
        system = None
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                conversation.append(msg.to_dict())

        # Make API call
        create_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation
        }

        if system:
            create_kwargs["system"] = system

        response = self.client.messages.create(**create_kwargs)

        # Extract content
        content = ""
        if response.content:
            content = response.content[0].text

        return LLMResponse(
            content=content,
            model=response.model,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            finish_reason=response.stop_reason or "stop",
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )

    def get_model_name(self) -> str:
        return self.model


# =============================================================================
# OPENAI PROVIDER
# =============================================================================

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT API provider.
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: API key (default: from OPENAI_API_KEY env)
            model: Model to use (default: gpt-4o)
            base_url: Optional custom base URL (for Azure, etc.)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # Import openai lazily
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using GPT."""
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Extract content
        choice = response.choices[0]
        content = choice.message.content or ""

        return LLMResponse(
            content=content,
            model=response.model,
            tokens_input=response.usage.prompt_tokens if response.usage else 0,
            tokens_output=response.usage.completion_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )

    def get_model_name(self) -> str:
        return self.model


# =============================================================================
# LLM CLIENT (MAIN INTERFACE)
# =============================================================================

class LLMClient:
    """
    Unified LLM client that abstracts provider differences.

    Automatically selects provider based on configuration and provides
    a consistent interface for all LLM operations.

    Usage:
        client = LLMClient()

        # Simple completion
        response = client.complete("What is Python?")

        # With system prompt
        response = client.complete(
            prompt="Review this code",
            system="You are an expert code reviewer"
        )

        # Chat with history
        response = client.chat([
            LLMMessage("system", "You are a helpful assistant"),
            LLMMessage("user", "Hello!"),
            LLMMessage("assistant", "Hi there!"),
            LLMMessage("user", "How are you?")
        ])
    """

    PROVIDERS = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider
    }

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize LLM client.

        Args:
            provider: Provider name (anthropic, openai)
            model: Model to use (provider-specific)
            api_key: API key for provider
            **kwargs: Additional provider-specific options
        """
        self.provider_name = provider or os.environ.get("LLM_PROVIDER", "anthropic")

        # Get provider class
        provider_class = self.PROVIDERS.get(self.provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown LLM provider: {self.provider_name}")

        # Initialize provider
        init_kwargs = {**kwargs}
        if model:
            init_kwargs["model"] = model
        if api_key:
            init_kwargs["api_key"] = api_key

        self._provider = provider_class(**init_kwargs)

        # Metrics tracking
        self._total_tokens_input = 0
        self._total_tokens_output = 0
        self._call_count = 0

    def complete(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a simple completion.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options

        Returns:
            LLMResponse with generated content
        """
        messages = []
        if system:
            messages.append(LLMMessage("system", system))
        messages.append(LLMMessage("user", prompt))

        return self.chat(messages, max_tokens, temperature, **kwargs)

    def chat(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat completion.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options

        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()

        try:
            response = self._provider.complete(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Track metrics
            self._call_count += 1
            self._total_tokens_input += response.tokens_input
            self._total_tokens_output += response.tokens_output

            duration = time.time() - start_time
            logger.debug(
                f"LLM call completed: {response.tokens_input}+{response.tokens_output} "
                f"tokens in {duration:.2f}s"
            )

            return response

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def get_model(self) -> str:
        """Get the current model name."""
        return self._provider.get_model_name()

    def get_metrics(self) -> Dict[str, Any]:
        """Get accumulated metrics."""
        return {
            "call_count": self._call_count,
            "total_tokens_input": self._total_tokens_input,
            "total_tokens_output": self._total_tokens_output,
            "total_tokens": self._total_tokens_input + self._total_tokens_output,
            "provider": self.provider_name,
            "model": self.get_model()
        }

    def reset_metrics(self):
        """Reset accumulated metrics."""
        self._call_count = 0
        self._total_tokens_input = 0
        self._total_tokens_output = 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_llm_client(
    provider: str = None,
    model: str = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: Provider name (anthropic, openai)
        model: Model to use
        **kwargs: Additional options

    Returns:
        Configured LLMClient
    """
    return LLMClient(provider=provider, model=model, **kwargs)


def estimate_tokens(text: str, method: str = "simple") -> int:
    """
    Estimate token count for text.

    Args:
        text: Text to estimate
        method: Estimation method (simple, tiktoken)

    Returns:
        Estimated token count
    """
    if method == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass

    # Simple estimation: ~4 characters per token
    return len(text) // 4
