# =============================================================================
# AI AGENT DEVELOPMENT SYSTEM - LANGCHAIN SETUP
# =============================================================================
"""
LangChain Setup Module

This module configures and initializes LangChain components for the orchestrator.
It provides:
- LLM client initialization (Anthropic, OpenAI, Azure, Ollama)
- Tool definitions for orchestrator operations
- Prompt template management
- Callback handlers for tracing and metrics

Supported LLM Providers:
    - Anthropic (Claude models)
    - OpenAI (GPT models)
    - Azure OpenAI
    - Local models via Ollama
"""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str = "ollama"
    model: str = "llava"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3

    # Provider-specific settings
    api_base: Optional[str] = None
    api_version: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'LLMConfig':
        """Create config from dictionary."""
        llm_config = config.get("llm", {})
        return cls(
            provider=llm_config.get("provider", "ollama"),
            model=llm_config.get("model", "llava"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 4096),
            timeout=llm_config.get("timeout", 120),
            max_retries=llm_config.get("max_retries", 3),
            api_base=llm_config.get("api_base"),
            api_version=llm_config.get("api_version"),
        )


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_call_id: str
    output: Any
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class LLMResponse:
    """Response from LLM invocation."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class LLMMetrics:
    """Collected metrics from LLM operations."""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0

    def record(self, response: LLMResponse):
        """Record metrics from a response."""
        self.total_requests += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_latency_ms += response.latency_ms

    def record_error(self):
        """Record an error."""
        self.errors += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.total_latency_ms / max(self.total_requests, 1),
            "errors": self.errors,
        }


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LangChainError(Exception):
    """Base exception for LangChain errors."""
    pass


class LLMProviderError(LangChainError):
    """Error related to LLM provider."""
    pass


class ToolExecutionError(LangChainError):
    """Error during tool execution."""
    pass


class ConfigurationError(LangChainError):
    """Error in configuration."""
    pass


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOL_SCHEMAS = {
    "read_issue": {
        "name": "read_issue",
        "description": "Read a GitHub issue by its number. Returns the issue title, body, labels, and current state.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_number": {
                    "type": "integer",
                    "description": "The GitHub issue number"
                }
            },
            "required": ["issue_number"]
        }
    },

    "update_issue": {
        "name": "update_issue",
        "description": "Update a GitHub issue's labels, state, or add a comment.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_number": {
                    "type": "integer",
                    "description": "The GitHub issue number"
                },
                "add_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to add"
                },
                "remove_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to remove"
                },
                "comment": {
                    "type": "string",
                    "description": "Comment to add to the issue"
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed"],
                    "description": "Issue state to set"
                }
            },
            "required": ["issue_number"]
        }
    },

    "launch_agent": {
        "name": "launch_agent",
        "description": "Launch an agent container to work on a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": ["planner", "developer", "qa", "reviewer", "doc"],
                    "description": "Type of agent to launch"
                },
                "issue_number": {
                    "type": "integer",
                    "description": "Issue for the agent to work on"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for the agent"
                }
            },
            "required": ["agent_type", "issue_number"]
        }
    },

    "read_memory": {
        "name": "read_memory",
        "description": "Read a project memory file (PROJECT.md, ARCHITECTURE.md, etc).",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to memory file relative to memory directory"
                }
            },
            "required": ["file_path"]
        }
    },

    "check_agent_status": {
        "name": "check_agent_status",
        "description": "Check the status of a running agent container.",
        "parameters": {
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Docker container ID"
                }
            },
            "required": ["container_id"]
        }
    },

    "list_issues": {
        "name": "list_issues",
        "description": "List GitHub issues with optional filtering.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by issue state"
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by labels"
                }
            }
        }
    },

    "create_subtask": {
        "name": "create_subtask",
        "description": "Create a subtask issue linked to a parent feature.",
        "parameters": {
            "type": "object",
            "properties": {
                "parent_issue": {
                    "type": "integer",
                    "description": "Parent feature issue number"
                },
                "title": {
                    "type": "string",
                    "description": "Subtask title"
                },
                "body": {
                    "type": "string",
                    "description": "Subtask description"
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels for the subtask"
                }
            },
            "required": ["parent_issue", "title", "body"]
        }
    }
}


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

TRIAGE_PROMPT = """You are the orchestrator for an AI agent development system.

Analyze this GitHub issue and determine the appropriate action:

Issue #{issue_number}
Title: {title}
Body:
{body}

Labels: {labels}

Current project context:
{project_context}

Determine:
1. Issue Type: Is this a "feature" (needs decomposition into subtasks) or a "task" (ready for direct development)?
2. Priority: What is the priority (high/medium/low)?
3. Dependencies: Are there any dependencies on other issues?
4. Validity: Is the issue well-formed with sufficient information?

If the issue is a feature:
- It should be decomposed by the Planner agent
- Typically features describe high-level functionality
- Features often mention multiple components or areas

If the issue is a task:
- It should go directly to development
- Tasks are specific, actionable items
- Tasks have clear acceptance criteria

Respond in JSON format:
{{
    "issue_type": "feature" | "task" | "invalid",
    "priority": "high" | "medium" | "low",
    "dependencies": [<issue numbers>],
    "is_valid": true | false,
    "invalid_reason": "<reason if invalid>",
    "recommended_labels": [<labels to add>],
    "analysis": "<brief analysis>"
}}
"""

TRANSITION_PROMPT = """You are managing the workflow for issue #{issue_number}.

Current state: {current_state}
Issue type: {issue_type}
Iteration count: {iteration_count} / {max_iterations}

Last agent output:
{agent_output}

QA Feedback (if any):
{qa_feedback}

Based on the agent output, determine the next state and actions:

State transitions:
- TRIAGE → PLANNING (if feature) or DEVELOPMENT (if task)
- PLANNING → AWAIT_SUBTASKS (subtasks created)
- AWAIT_SUBTASKS → DONE (when all subtasks complete)
- DEVELOPMENT → QA (code complete)
- QA → REVIEW (if passed) or QA_FAILED (if failed)
- QA_FAILED → DEVELOPMENT (if iterations < max) or BLOCKED (if max reached)
- REVIEW → DOCUMENTATION (if approved) or QA (if changes requested)
- DOCUMENTATION → DONE (docs complete)

Respond in JSON format:
{{
    "next_state": "<state>",
    "trigger": "<what triggered this transition>",
    "actions": [
        {{
            "type": "add_label" | "remove_label" | "add_comment" | "launch_agent",
            "params": {{...}}
        }}
    ],
    "reason": "<explanation>"
}}
"""

PLANNING_PROMPT = """You are analyzing a feature issue to create subtasks.

Feature Issue #{issue_number}
Title: {title}
Body:
{body}

Project Architecture:
{architecture}

Conventions:
{conventions}

Create a list of subtasks that fully implement this feature.
Each subtask should be:
- Specific and actionable
- Independently testable
- Properly scoped (not too large)

Respond in JSON format:
{{
    "subtasks": [
        {{
            "title": "<subtask title>",
            "description": "<detailed description>",
            "acceptance_criteria": ["<criterion 1>", "<criterion 2>"],
            "dependencies": [<subtask indices that must complete first>],
            "estimated_complexity": "low" | "medium" | "high"
        }}
    ],
    "implementation_order": [<subtask indices in recommended order>],
    "notes": "<any additional notes>"
}}
"""


# =============================================================================
# LLM CLIENT INTERFACE
# =============================================================================

class LLMClientInterface(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke the LLM with messages and optional tools."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


# =============================================================================
# ANTHROPIC CLIENT
# =============================================================================

class AnthropicClient(LLMClientInterface):
    """Client for Anthropic Claude models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("orchestrator.llm.anthropic")
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ConfigurationError("ANTHROPIC_API_KEY not set")

                self._client = anthropic.AsyncAnthropic(
                    api_key=api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            except ImportError:
                raise LLMProviderError("anthropic package not installed. Install with: pip install anthropic")
        return self._client

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"]
            })
        return anthropic_tools

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke Claude with messages."""
        client = self._get_client()
        start_time = time.time()

        system_message = None
        api_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                api_messages.append(msg)

        request_kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": api_messages,
        }

        if system_message:
            request_kwargs["system"] = system_message

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await client.messages.create(**request_kwargs)

            latency = (time.time() - start_time) * 1000

            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input
                    ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=response.model,
                latency_ms=latency
            )

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise LLMProviderError(f"Anthropic API error: {e}")

    def get_model_name(self) -> str:
        return self.config.model


# =============================================================================
# OPENAI CLIENT
# =============================================================================

class OpenAIClient(LLMClientInterface):
    """Client for OpenAI GPT models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("orchestrator.llm.openai")
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ConfigurationError("OPENAI_API_KEY not set")

                kwargs = {
                    "api_key": api_key,
                    "timeout": self.config.timeout,
                    "max_retries": self.config.max_retries,
                }

                if self.config.api_base:
                    kwargs["base_url"] = self.config.api_base

                self._client = openai.AsyncOpenAI(**kwargs)
            except ImportError:
                raise LLMProviderError("openai package not installed. Install with: pip install openai")
        return self._client

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function format."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return openai_tools

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke OpenAI with messages."""
        client = self._get_client()
        start_time = time.time()

        request_kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await client.chat.completions.create(**request_kwargs)

            latency = (time.time() - start_time) * 1000

            message = response.choices[0].message
            content = message.content or ""
            tool_calls = []

            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                model=response.model,
                latency_ms=latency
            )

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise LLMProviderError(f"OpenAI API error: {e}")

    def get_model_name(self) -> str:
        return self.config.model


# =============================================================================
# AZURE OPENAI CLIENT
# =============================================================================

class AzureOpenAIClient(LLMClientInterface):
    """Client for Azure OpenAI."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("orchestrator.llm.azure")
        self._client = None

    def _get_client(self):
        """Get or create Azure OpenAI client."""
        if self._client is None:
            try:
                import openai

                api_key = os.environ.get("AZURE_OPENAI_API_KEY")
                endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or self.config.api_base
                api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or self.config.api_version or "2024-02-15-preview"

                if not api_key or not endpoint:
                    raise ConfigurationError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")

                self._client = openai.AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version=api_version,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            except ImportError:
                raise LLMProviderError("openai package not installed. Install with: pip install openai")
        return self._client

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function format."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return openai_tools

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke Azure OpenAI with messages."""
        client = self._get_client()
        start_time = time.time()

        request_kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await client.chat.completions.create(**request_kwargs)

            latency = (time.time() - start_time) * 1000

            message = response.choices[0].message
            content = message.content or ""
            tool_calls = []

            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                model=response.model,
                latency_ms=latency
            )

        except Exception as e:
            self.logger.error(f"Azure OpenAI API error: {e}")
            raise LLMProviderError(f"Azure OpenAI API error: {e}")

    def get_model_name(self) -> str:
        return self.config.model


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient(LLMClientInterface):
    """Client for local Ollama models."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("orchestrator.llm.ollama")
        self.base_url = config.api_base or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Ollama format."""
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return ollama_tools

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Invoke Ollama with messages."""
        session = await self._get_session()
        start_time = time.time()

        request_data = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        if tools:
            request_data["tools"] = self._convert_tools(tools)

        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=request_data
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise LLMProviderError(f"Ollama error {response.status}: {text}")

                data = await response.json()

                latency = (time.time() - start_time) * 1000

                message = data.get("message", {})
                content = message.get("content", "")
                tool_calls = []

                if message.get("tool_calls"):
                    for tc in message["tool_calls"]:
                        func = tc.get("function", {})
                        tool_calls.append(ToolCall(
                            id=tc.get("id", f"call_{len(tool_calls)}"),
                            name=func.get("name", ""),
                            arguments=func.get("arguments", {})
                        ))

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    input_tokens=data.get("prompt_eval_count", 0),
                    output_tokens=data.get("eval_count", 0),
                    model=self.config.model,
                    latency_ms=latency
                )

        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            raise LLMProviderError(f"Ollama API error: {e}")

    def get_model_name(self) -> str:
        return self.config.model

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """Registry for orchestrator tools."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger("orchestrator.tools")

        for name, schema in TOOL_SCHEMAS.items():
            self.register_schema(name, schema)

    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register a tool schema."""
        self.tools[name] = schema

    def register_handler(self, name: str, handler: Callable):
        """Register a tool handler function."""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        self.handlers[name] = handler
        self.logger.debug(f"Registered handler for tool: {name}")

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool schema by name."""
        return self.tools.get(name)

    def get_tools(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get tool schemas, optionally filtered by names."""
        if names is None:
            return list(self.tools.values())
        return [self.tools[name] for name in names if name in self.tools]

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        handler = self.handlers.get(tool_call.name)

        if handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=f"No handler registered for tool: {tool_call.name}"
            )

        try:
            import asyncio
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_call.arguments)
            else:
                result = handler(**tool_call.arguments)

            return ToolResult(
                tool_call_id=tool_call.id,
                output=result
            )

        except Exception as e:
            self.logger.error(f"Tool execution error ({tool_call.name}): {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=str(e)
            )


# =============================================================================
# LANGCHAIN ENGINE
# =============================================================================

class LangChainEngine:
    """
    Manages LLM interactions for the orchestrator.

    This class handles:
    1. LLM client initialization based on provider
    2. Tool registration and execution
    3. Message formatting and chain execution
    4. Metrics collection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LangChain engine with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_config = LLMConfig.from_dict(config)
        self.logger = logging.getLogger("orchestrator.langchain")

        self._client: Optional[LLMClientInterface] = None
        self.tool_registry = ToolRegistry()
        self.metrics = LLMMetrics()

        self._init_client()

    def _init_client(self):
        """Initialize the LLM client based on provider."""
        provider = self.llm_config.provider.lower()

        if provider == "anthropic":
            self._client = AnthropicClient(self.llm_config)
        elif provider == "openai":
            self._client = OpenAIClient(self.llm_config)
        elif provider == "azure":
            self._client = AzureOpenAIClient(self.llm_config)
        elif provider == "ollama":
            self._client = OllamaClient(self.llm_config)
        else:
            raise ConfigurationError(f"Unknown LLM provider: {provider}")

        self.logger.info(f"Initialized {provider} client with model {self.llm_config.model}")

    def register_tool_handler(self, name: str, handler: Callable):
        """Register a handler for a tool."""
        self.tool_registry.register_handler(name, handler)

    def register_tool_handlers(self, handlers: Dict[str, Callable]):
        """Register multiple tool handlers."""
        for name, handler in handlers.items():
            self.register_tool_handler(name, handler)

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[str]] = None,
        execute_tools: bool = True,
        max_tool_iterations: int = 10
    ) -> LLMResponse:
        """
        Invoke the LLM with messages and optional tools.

        Args:
            messages: List of message dicts (role, content)
            tools: List of tool names to make available
            execute_tools: Whether to execute tool calls automatically
            max_tool_iterations: Maximum tool execution iterations

        Returns:
            LLMResponse with content and/or tool results
        """
        tool_schemas = None
        if tools:
            tool_schemas = self.tool_registry.get_tools(tools)

        try:
            response = await self._client.invoke(messages, tool_schemas)
            self.metrics.record(response)

            if execute_tools and response.has_tool_calls:
                response = await self._execute_tool_loop(
                    messages, response, tool_schemas, max_tool_iterations
                )

            return response

        except Exception as e:
            self.metrics.record_error()
            raise

    async def _execute_tool_loop(
        self,
        messages: List[Dict[str, Any]],
        response: LLMResponse,
        tool_schemas: List[Dict[str, Any]],
        max_iterations: int
    ) -> LLMResponse:
        """Execute tools in a loop until no more tool calls."""
        conversation = messages.copy()
        current_response = response

        for iteration in range(max_iterations):
            if not current_response.has_tool_calls:
                break

            conversation.append({
                "role": "assistant",
                "content": current_response.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in current_response.tool_calls
                ]
            })

            for tool_call in current_response.tool_calls:
                result = await self.tool_registry.execute(tool_call)

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.output) if result.output else result.error
                })

            current_response = await self._client.invoke(conversation, tool_schemas)
            self.metrics.record(current_response)

        return current_response

    async def triage_issue(
        self,
        issue_number: int,
        title: str,
        body: str,
        labels: List[str],
        project_context: str = ""
    ) -> Dict[str, Any]:
        """
        Triage a GitHub issue to determine its type and priority.

        Args:
            issue_number: Issue number
            title: Issue title
            body: Issue body
            labels: Current labels
            project_context: Project context from memory

        Returns:
            Triage result dictionary
        """
        prompt = TRIAGE_PROMPT.format(
            issue_number=issue_number,
            title=title,
            body=body,
            labels=", ".join(labels) if labels else "none",
            project_context=project_context or "No context available"
        )

        messages = [
            {"role": "system", "content": "You are a workflow orchestrator. Always respond in valid JSON."},
            {"role": "user", "content": prompt}
        ]

        response = await self.invoke(messages, execute_tools=False)

        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse triage response: {response.content}")
            return {
                "issue_type": "task",
                "priority": "medium",
                "dependencies": [],
                "is_valid": True,
                "analysis": "Failed to parse LLM response, defaulting to task"
            }

    async def determine_transition(
        self,
        issue_number: int,
        current_state: str,
        issue_type: str,
        iteration_count: int,
        max_iterations: int,
        agent_output: Optional[Dict[str, Any]] = None,
        qa_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determine the next state transition for an issue.

        Args:
            issue_number: Issue number
            current_state: Current workflow state
            issue_type: Type of issue (feature/task)
            iteration_count: Current iteration count
            max_iterations: Maximum iterations allowed
            agent_output: Output from last agent
            qa_feedback: Feedback from QA if applicable

        Returns:
            Transition decision dictionary
        """
        prompt = TRANSITION_PROMPT.format(
            issue_number=issue_number,
            current_state=current_state,
            issue_type=issue_type,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            agent_output=json.dumps(agent_output, indent=2) if agent_output else "None",
            qa_feedback=qa_feedback or "None"
        )

        messages = [
            {"role": "system", "content": "You are a workflow orchestrator. Always respond in valid JSON."},
            {"role": "user", "content": prompt}
        ]

        response = await self.invoke(messages, execute_tools=False)

        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse transition response: {response.content}")
            return {
                "next_state": "BLOCKED",
                "trigger": "parse_error",
                "actions": [],
                "reason": "Failed to parse LLM response"
            }

    async def plan_subtasks(
        self,
        issue_number: int,
        title: str,
        body: str,
        architecture: str = "",
        conventions: str = ""
    ) -> Dict[str, Any]:
        """
        Plan subtasks for a feature issue.

        Args:
            issue_number: Feature issue number
            title: Feature title
            body: Feature description
            architecture: Project architecture from memory
            conventions: Project conventions from memory

        Returns:
            Planning result with subtasks
        """
        prompt = PLANNING_PROMPT.format(
            issue_number=issue_number,
            title=title,
            body=body,
            architecture=architecture or "No architecture documentation available",
            conventions=conventions or "No conventions documentation available"
        )

        messages = [
            {"role": "system", "content": "You are a software architect. Always respond in valid JSON."},
            {"role": "user", "content": prompt}
        ]

        response = await self.invoke(messages, execute_tools=False)

        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse planning response: {response.content}")
            return {
                "subtasks": [],
                "implementation_order": [],
                "notes": "Failed to parse LLM response"
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.to_dict()

    def get_model_name(self) -> str:
        """Get the configured model name."""
        return self._client.get_model_name()

    async def close(self):
        """Close any open connections."""
        if hasattr(self._client, 'close'):
            await self._client.close()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

async def create_langchain_engine(config: Dict[str, Any]) -> LangChainEngine:
    """
    Factory function to create and configure a LangChainEngine.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LangChainEngine instance
    """
    engine = LangChainEngine(config)
    return engine


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "LangChainEngine",
    "create_langchain_engine",
    # Configuration
    "LLMConfig",
    "LLMProvider",
    # Data structures
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "LLMMetrics",
    # Tools
    "ToolRegistry",
    "TOOL_SCHEMAS",
    # Prompts
    "TRIAGE_PROMPT",
    "TRANSITION_PROMPT",
    "PLANNING_PROMPT",
    # Clients
    "LLMClientInterface",
    "AnthropicClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "OllamaClient",
    # Exceptions
    "LangChainError",
    "LLMProviderError",
    "ToolExecutionError",
    "ConfigurationError",
]