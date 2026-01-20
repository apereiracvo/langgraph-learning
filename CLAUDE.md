# CLAUDE.md

## Project Overview

LangGraph learning repository with samples and patterns for agent development using LangChain and LangGraph. Uses UV workspace monorepo structure with pyenv for Python version management.

## Commands

All commands use Task (taskfile.dev). Run `task --list` for all available commands.

### Essential Commands
```bash
task setup          # Full project setup (creates pyenv virtualenv + syncs deps)
task uv-sync        # Sync dependencies after modifying pyproject.toml
task ci             # Run all CI checks (format, lint, test)
```

### Code Quality
```bash
task format         # Format code with Ruff
task lint           # Run all linters (Ruff + MyPy)
task ruff-fix       # Auto-fix linting issues
```

### Testing
```bash
task test           # Run all tests except integration
task test-unit      # Run unit tests only
task test-integration  # Run integration tests (requires API keys)
task test-cov       # Run tests with coverage report
```

### Running Patterns
```bash
task run -- p01-basic-graph         # Run a pattern by name
task run-module -- p01_basic_graph  # Run as module
task list-patterns                  # List all available patterns
```

## Architecture

### Workspace Structure
- **Root**: UV workspace configuration (`pyproject.toml`) with shared dependencies
- **packages/shared/**: Common utilities (env setup, response formatting)
- **packages/p##_name/**: Individual learning patterns, each self-contained
- **tests/**: Pytest suite with `unit/` and `integration/` directories

### Pattern Convention
- Naming: `p##_snake_case_name` (e.g., `p01_basic_graph`)
- Each pattern has its own `pyproject.toml` with entry point
- Entry point format: `package-name = "module_name.main:main"`
- Patterns depend on shared package via workspace configuration

### Testing Strategy
- Unit tests: `tests/unit/` - no external API calls
- Integration tests: `tests/integration/` - requires API keys, marked with `@pytest.mark.integration`
- Fixtures defined in `tests/conftest.py`

## Code Quality Standards

- **Python**: 3.12.12 (strict version via pyenv)
- **Type Checking**: MyPy strict mode enabled
- **Linting**: Comprehensive Ruff rules (F, E, W, I, B, C4, SIM, ANN, D, S, UP, ASYNC, TRY, PT, N)
- **Line Length**: 88 characters
- **Docstrings**: Google convention
- **Quotes**: Double quotes

## Environment Setup

Requires `.env` file with API keys (see `.env.example`):
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` for LLM calls
- `LANGCHAIN_API_KEY` for LangSmith tracing (optional)
