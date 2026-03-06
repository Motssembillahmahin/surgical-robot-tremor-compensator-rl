# Contributing

## Getting Started

1. Fork the repository
2. Create a feature branch from `dev`: `git checkout -b feature/your-feature`
3. Install dependencies: `uv sync`
4. Make your changes
5. Run tests: `uv run pytest`
6. Run linting: `uv run ruff check . && uv run ruff format --check .`
7. Submit a pull request to `dev`

## Code Standards

- Python code must pass `ruff check` and `ruff format` with no errors
- All new functions require type hints and docstrings
- Reward function changes must include clinical rationale in comments
- Safety-critical code changes require two reviewer approvals
- Never hardcode hyperparameters — use config.yaml

## Pull Request Requirements

- All CI checks must pass
- At least one approving review
- Safety-critical changes (`safety/`, reward function) require two reviews
- Update CHANGELOG.md with your changes
- Update `docs/` if your change affects architecture or configuration

## Reporting Issues

- Use GitHub Issues with the appropriate label (`bug`, `feature`, `safety`)
- Security vulnerabilities: report privately, do not open a public issue
