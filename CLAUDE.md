# Claude Agent SDK Python - Development Guide

## Overview

The Claude Agent SDK for Python provides a comprehensive interface for building AI agents that interact with Claude Code. This is a modern async-first Python library that supports both simple one-shot queries and complex interactive sessions with custom tools and hooks.

## High-Level Architecture

### Core Components

The SDK follows a layered architecture:

1. **Public API Layer** (`src/claude_agent_sdk/`)
   - `query.py` - Simple one-shot query function for stateless interactions
   - `client.py` - ClaudeSDKClient for interactive, stateful conversations
   - `types.py` - Comprehensive type definitions and data structures
   - `__init__.py` - Main exports and MCP server creation utilities

2. **Internal Implementation** (`src/claude_agent_sdk/_internal/`)
   - `client.py` - InternalClient that handles query processing logic
   - `query.py` - Query class managing the control protocol and message flow
   - `message_parser.py` - Message parsing and type conversion
   - `transport/` - Transport layer for CLI communication

3. **Transport Layer** (`src/claude_agent_sdk/_internal/transport/`)
   - `subprocess_cli.py` - Subprocess management for Claude Code CLI
   - `__init__.py` - Transport interface definition

### Key Architectural Patterns

- **Async-First Design**: All operations are async using anyio for compatibility with asyncio and trio
- **Transport Abstraction**: Clean separation between protocol logic and communication transport
- **Type Safety**: Comprehensive type hints with mypy enforcement
- **Message-Based Protocol**: JSON-based message protocol with structured parsing
- **MCP Integration**: Built-in support for both external and in-process MCP servers

## Development Workflow

### Prerequisites

- Python 3.10+
- Node.js (for Claude Code CLI)
- Claude Code installed globally: `npm install -g @anthropic-ai/claude-code`

### Environment Setup

```bash
# Clone and setup
git clone <repository>
cd claude-agent-sdk-python

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import claude_agent_sdk; print(claude_agent_sdk.__version__)"
```

### Code Quality Tools

The project uses modern Python tooling configured in `pyproject.toml`:

```bash
# Lint and auto-fix issues
python -m ruff check src/ tests/ --fix

# Format code
python -m ruff format src/ tests/

# Type checking (strict mode)
python -m mypy src/

# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=src/claude_agent_sdk

# Run specific test file
python -m pytest tests/test_client.py

# Run with verbose output
python -m pytest tests/ -v
```

### Testing Strategy

- **Unit Tests**: `tests/test_*.py` - Test individual components in isolation
- **Integration Tests**: `tests/integration/` - Test end-to-end workflows
- **Type Checking**: Enforced with mypy in strict mode
- **Code Coverage**: Configured with pytest-cov

## Build and Distribution

### Building the Package

```bash
# Build wheel and source distribution
python -m build

# Build wheel only
python -m build --wheel

# Build source distribution only
python -m build --sdist
```

### Version Management

- Version is defined in `src/claude_agent_sdk/_version.py`
- Build system uses hatchling (configured in `pyproject.toml`)
- Package name: `kev-claude-agent-sdk` (custom name in pyproject.toml)

## Core Usage Patterns

### 1. Simple Queries (One-Shot)

```python
import anyio
from claude_agent_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

### 2. Interactive Sessions

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful Python developer",
        permission_mode="acceptEdits"
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Create a hello.py file")
        async for response in client.receive_response():
            print(response)

anyio.run(main)
```

### 3. Custom Tools (SDK MCP Servers)

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeSDKClient

@tool("calculate", "Perform calculations", {"expression": str})
async def calculate(args):
    try:
        result = eval(args["expression"])  # Safe eval in production
        return {"content": [{"type": "text", "text": f"Result: {result}"}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}

# Create SDK MCP server
calc_server = create_sdk_mcp_server("calculator", tools=[calculate])

# Use with Claude
options = ClaudeAgentOptions(
    mcp_servers={"calc": calc_server},
    allowed_tools=["mcp__calc__calculate"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Calculate 2 + 2 * 3")
    # ... handle response
```

### 4. Hooks for Permission Control

```python
from claude_agent_sdk import ClaudeSDKClient, HookMatcher, ClaudeAgentOptions

async def block_dangerous_commands(input_data, tool_use_id, context):
    if input_data.get("tool_name") == "Bash":
        command = input_data["tool_input"].get("command", "")
        if "rm -rf" in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Dangerous command blocked"
                }
            }
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[block_dangerous_commands])
        ]
    }
)
```

## Internal Architecture Deep Dive

### Message Flow

1. **Query Initiation**: User calls `query()` or `ClaudeSDKClient.query()`
2. **Transport Creation**: SubprocessCLITransport spawns Claude Code CLI process
3. **Protocol Handshake**: SDK and CLI establish JSON-based communication protocol
4. **Message Exchange**: Messages flow through Query class for processing
5. **Response Parsing**: Raw JSON responses parsed into typed Message objects
6. **Tool Execution**: MCP servers handle tool invocations (external or in-process)

### Transport Layer

The transport layer handles communication with the Claude Code CLI:

- **Subprocess Management**: Spawns and monitors CLI process
- **Stream Handling**: Manages stdin/stdout/stderr streams asynchronously
- **Buffer Management**: Implements bounded buffering to prevent memory issues
- **Error Handling**: Comprehensive error detection and reporting

### Control Protocol

The SDK uses a control protocol for advanced features:

- **Streaming Mode**: Enables bidirectional communication
- **Permission Control**: Dynamic permission management during sessions
- **Session Management**: Support for multiple concurrent sessions
- **Interrupt Handling**: Ability to interrupt long-running operations

### Type System

Comprehensive type system defined in `types.py`:

- **Message Types**: UserMessage, AssistantMessage, SystemMessage, ResultMessage
- **Content Blocks**: TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
- **Configuration**: ClaudeAgentOptions with extensive customization
- **Permission System**: Fine-grained permission control types
- **Hook System**: Types for implementing custom hooks

## Development Guidelines

### Code Style

- Follow PEP 8 with ruff formatting (88-character line length)
- Use type hints everywhere (enforced by mypy strict mode)
- Prefer async/await over callback patterns
- Use dataclasses for structured data
- Document all public APIs with comprehensive docstrings

### Testing Guidelines

- Write tests for all public APIs
- Use pytest-asyncio for async test support
- Mock external dependencies (CLI process) in unit tests
- Test error conditions and edge cases
- Maintain high code coverage (>90%)

### Adding New Features

1. **Types First**: Define types in `types.py`
2. **Internal Implementation**: Add logic in `_internal/`
3. **Public API**: Expose through main module
4. **Tests**: Comprehensive unit and integration tests
5. **Documentation**: Update docstrings and examples

## Debugging and Troubleshooting

### Common Issues

1. **CLI Not Found**: Ensure Claude Code is installed and in PATH
2. **Permission Issues**: Check file permissions and working directory
3. **Process Errors**: Examine stderr output from CLI process
4. **JSON Parsing**: Verify message format matches expected protocol

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# SDK will emit detailed debug information
```

### Common Debug Commands

```bash
# Check CLI installation
claude --version

# Test basic connectivity
python -c "
import asyncio
from claude_agent_sdk import query
async for msg in query(prompt='test'):
    print(msg)
"

# Run with specific Python path
CLAUDE_CODE_ENTRYPOINT=sdk-py python your_script.py
```

## Examples and Reference Implementations

See the `examples/` directory for comprehensive examples:

- `quick_start.py` - Basic usage examples
- `streaming_mode.py` - Interactive session examples
- `mcp_calculator.py` - Custom tools implementation
- `hooks.py` - Permission control examples
- `agents.py` - Agent definition examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with full test coverage
4. Run code quality checks
5. Submit a pull request with description

### Pre-commit Checklist

- [ ] Code passes `ruff check` and `ruff format`
- [ ] Code passes `mypy src/` with no errors
- [ ] All tests pass: `pytest tests/`
- [ ] Documentation updated
- [ ] Examples tested (if applicable)