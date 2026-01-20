# LangGraph Learning

LangGraph samples and patterns for learning agent development with LangChain and LangGraph.

## Prerequisites

- **Python 3.12+** via [pyenv](https://github.com/pyenv/pyenv)
- **[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)** plugin
- **[UV](https://docs.astral.sh/uv/)** package manager
- **[Task](https://taskfile.dev/)** task runner

### Installing Prerequisites

```bash
# Install pyenv (macOS)
brew install pyenv pyenv-virtualenv

# Install Python 3.12.12
pyenv install 3.12.12

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Task
brew install go-task
```

## Setup

```bash
# Clone the repository
git clone https://github.com/apereiracv/langgraph-learning.git
cd langgraph-learning

# Full setup (creates pyenv virtualenv + syncs dependencies)
task setup
```

Or step by step:

```bash
# Create pyenv virtualenv
task pyenv-create

# Set local Python version
task pyenv-local

# Sync all packages with dev dependencies
task uv-sync
```

## Environment Variables

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## Available Commands

### Environment Management

| Command | Description |
|---------|-------------|
| `task setup` | Full project setup (pyenv + UV sync) |
| `task pyenv-create` | Create pyenv virtualenv |
| `task pyenv-local` | Set local Python version |
| `task pyenv-remove` | Remove pyenv virtualenv |

### Package Management (UV)

| Command | Description |
|---------|-------------|
| `task uv-sync` | Sync all workspace packages with dev dependencies |
| `task uv-lock` | Update the lockfile |
| `task uv-add -- <pkg>` | Add a dependency to root |
| `task uv-add-dev -- <pkg>` | Add a dev dependency |

### Code Quality

| Command | Description |
|---------|-------------|
| `task format` | Format code with Ruff |
| `task format-check` | Check code formatting (no changes) |
| `task lint` | Run all linters (Ruff + MyPy) |
| `task ruff` | Run Ruff linter |
| `task ruff-fix` | Run Ruff with auto-fix |
| `task mypy` | Run MyPy type checker |

### Testing

| Command | Description |
|---------|-------------|
| `task test` | Run all tests except integration |
| `task test-unit` | Run unit tests only |
| `task test-integration` | Run integration tests (requires API keys) |
| `task test-cov` | Run tests with coverage report |

### Running Patterns

| Command | Description |
|---------|-------------|
| `task run -- <pattern>` | Run a pattern (e.g., `task run -- p01-basic-graph`) |
| `task run-module -- <module>` | Run as module (e.g., `task run-module -- p01_basic_graph`) |
| `task list-patterns` | List all available patterns |

### CI/CD

| Command | Description |
|---------|-------------|
| `task ci` | Run all CI checks (format, lint, test) |
| `task clean` | Clean build artifacts and caches |

## Project Structure

```
langgraph-learning/
├── packages/                    # UV workspace packages
│   ├── shared/                  # Shared utilities
│   │   ├── pyproject.toml
│   │   └── src/shared/
│   └── p01_basic_graph/         # Pattern 01: Basic Graph
│       ├── pyproject.toml
│       └── src/p01_basic_graph/
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml               # Root workspace config
├── Taskfile.yml                 # Task automation
├── uv.lock                      # Dependency lockfile
└── .python-version              # Python version for pyenv
```

## Patterns

| Pattern | Description |
|---------|-------------|
| `p01-basic-graph` | Basic StateGraph with single node and linear flow |

## Adding a New Pattern

1. Create the package structure:
   ```bash
   mkdir -p packages/p02_new_pattern/src/p02_new_pattern
   ```

2. Create `packages/p02_new_pattern/pyproject.toml`:
   ```toml
   [project]
   name = "p02-new-pattern"
   version = "0.1.0"
   requires-python = ">=3.12"
   dependencies = ["shared", "langgraph>=0.2"]

   [tool.uv.sources]
   shared = { workspace = true }

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src/p02_new_pattern"]

   [project.scripts]
   p02-new-pattern = "p02_new_pattern.main:main"
   ```

3. Create `packages/p02_new_pattern/src/p02_new_pattern/__init__.py`:
   ```python
   """Pattern 02: New Pattern."""
   ```

4. Create `packages/p02_new_pattern/src/p02_new_pattern/main.py`:
   ```python
   def main() -> None:
       """Run the pattern."""
       print("Hello from Pattern 02!")

   if __name__ == "__main__":
       main()
   ```

5. Sync and run:
   ```bash
   task uv-sync
   task run -- p02-new-pattern
   ```

## License

MIT
