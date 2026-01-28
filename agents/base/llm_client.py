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
from typing import Optional, Dict, Any, List, Union
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
class TextBlock:
    """A text content block within a multimodal message."""
    text: str
    type: str = "text"


@dataclass
class ImageBlock:
    """An image content block within a multimodal message."""
    media_type: str  # e.g. "image/png"
    base64_data: str  # base64-encoded image bytes
    alt_text: str = ""
    type: str = "image"


@dataclass
class LLMMessage:
    """
    A message in a conversation.

    Supports both plain text and multimodal content (text + images).
    When ``content`` is a string, the message behaves exactly as before.
    When ``content`` is a list of TextBlock/ImageBlock, it represents
    multimodal content.

    Attributes:
        role: Message role (system, user, assistant)
        content: Plain text string or list of content blocks
    """
    role: str
    content: Union[str, List[Union[TextBlock, ImageBlock]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for provider APIs."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": self._serialize_blocks()}

    def _serialize_blocks(self) -> List[Dict[str, Any]]:
        """Serialize content blocks to a provider-agnostic format."""
        blocks = []
        for block in self.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                blocks.append({
                    "type": "image",
                    "media_type": block.media_type,
                    "data": block.base64_data,
                })
        return blocks

    @property
    def is_multimodal(self) -> bool:
        """Check if this message contains images."""
        if isinstance(self.content, str):
            return False
        return any(isinstance(b, ImageBlock) for b in self.content)

    @property
    def text_content(self) -> str:
        """Extract only the text portions (for logging, token estimation)."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            b.text for b in self.content if isinstance(b, TextBlock)
        )


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

    @staticmethod
    def _format_message(msg: LLMMessage) -> Dict[str, Any]:
        """Format an LLMMessage for the Anthropic API."""
        if isinstance(msg.content, str):
            return {"role": msg.role, "content": msg.content}

        content_blocks: List[Dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.media_type,
                        "data": block.base64_data,
                    }
                })
        return {"role": msg.role, "content": content_blocks}

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
                system = msg.text_content if msg.is_multimodal else msg.content
            else:
                conversation.append(self._format_message(msg))

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

    @staticmethod
    def _format_message(msg: LLMMessage) -> Dict[str, Any]:
        """Format an LLMMessage for the OpenAI API."""
        if isinstance(msg.content, str):
            return {"role": msg.role, "content": msg.content}

        content_blocks: List[Dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                data_url = f"data:{block.media_type};base64,{block.base64_data}"
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
        return {"role": msg.role, "content": content_blocks}

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using GPT."""
        # Convert messages to OpenAI format
        openai_messages = [self._format_message(msg) for msg in messages]

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
# OLLAMA PROVIDER
# =============================================================================

class OllamaProvider(BaseLLMProvider):
    """
    Ollama local LLM provider.

    Ollama runs locally and provides an API for running open-source models
    like LLaMA, Mistral, CodeLlama, etc. No API key required.

    Requires Ollama to be running: https://ollama.ai
    """

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,  # Ignored, kept for interface compatibility
        **kwargs
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model to use (default: llama3.2)
            base_url: Ollama server URL (default: http://localhost:11434)
            api_key: Ignored (Ollama doesn't require authentication)
        """
        self.model = (
            model
            or os.environ.get("OLLAMA_MODEL")
            or os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)
        )
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        )
        # Remove trailing slash if present
        self.base_url = self.base_url.rstrip("/")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Ollama models.

        Ollama doesn't always report exact token counts, so we estimate.
        Most LLaMA-based models use ~4 chars per token.
        """
        return len(text) // 4

    def _check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Ollama."""
        import requests

        # Convert messages to Ollama chat format
        ollama_messages = []
        for msg in messages:
            ollama_msg: Dict[str, Any] = {"role": msg.role}

            if isinstance(msg.content, str):
                ollama_msg["content"] = msg.content
            else:
                # Multimodal: extract text and images separately
                text_parts = [
                    b.text for b in msg.content if isinstance(b, TextBlock)
                ]
                image_parts = [
                    b.base64_data for b in msg.content if isinstance(b, ImageBlock)
                ]
                ollama_msg["content"] = "\n".join(text_parts)
                if image_parts:
                    ollama_msg["images"] = image_parts

            ollama_messages.append(ollama_msg)

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # Calculate input tokens estimate before request
        input_text = " ".join(m.text_content for m in messages)
        tokens_input_estimate = self._estimate_tokens(input_text)

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=kwargs.get("timeout", 300)
            )

            # Handle specific error cases
            if response.status_code == 404:
                raise RuntimeError(
                    f"Model '{self.model}' not found. "
                    f"Run: ollama pull {self.model}"
                )

            response.raise_for_status()
            result = response.json()

            # Extract content
            content = result.get("message", {}).get("content", "")

            # Token handling: Ollama may return token info in some versions
            # Use estimates as fallback
            tokens_output = result.get("eval_count", self._estimate_tokens(content))
            tokens_input = result.get("prompt_eval_count", tokens_input_estimate)

            # Determine finish reason
            finish_reason = "stop"
            if result.get("done_reason"):
                finish_reason = result["done_reason"]
            elif not result.get("done", True):
                finish_reason = "length"

            return LLMResponse(
                content=content,
                model=result.get("model", self.model),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                finish_reason=finish_reason,
                raw_response=result
            )

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {kwargs.get('timeout', 300)}s. "
                "Consider using a smaller model or increasing timeout."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

    def get_model_name(self) -> str:
        return self.model

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m.get("name") for m in models]
        except Exception:
            return []


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
        "openai": OpenAIProvider,
        "ollama": OllamaProvider
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
        images: Optional[List[Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a simple completion.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            images: Optional list of ``ImageContent`` objects to include
            **kwargs: Provider-specific options

        Returns:
            LLMResponse with generated content
        """
        messages = []
        if system:
            messages.append(LLMMessage("system", system))
        messages.append(create_multimodal_message("user", prompt, images))

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

def create_multimodal_message(
    role: str,
    text: str,
    images: Optional[List[Any]] = None,
) -> LLMMessage:
    """
    Create an LLMMessage with text and optional images.

    When *images* is ``None`` or empty, returns a plain text message
    (fully backward-compatible).  Otherwise, returns a multimodal
    message with image blocks followed by a text block.

    Args:
        role: Message role (``"user"``, ``"assistant"``).
        text: Text content.
        images: Optional list of ``ImageContent`` objects.

    Returns:
        An ``LLMMessage`` ready for provider consumption.
    """
    if not images:
        return LLMMessage(role=role, content=text)

    blocks: List[Union[TextBlock, ImageBlock]] = []

    # Images first (common practice for vision models)
    for img in images:
        blocks.append(ImageBlock(
            media_type=img.media_type,
            base64_data=img.base64_data,
            alt_text=getattr(img, "alt_text", ""),
        ))

    # Text after images
    blocks.append(TextBlock(text=text))

    return LLMMessage(role=role, content=blocks)


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
