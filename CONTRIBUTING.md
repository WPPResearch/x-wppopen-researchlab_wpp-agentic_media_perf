# Contributing to Data Science Multi-Agent

Thank you for your interest in contributing to the Data Science Multi-Agent project! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [Issues](../../issues) section
2. If not, create a new issue with:
   - A clear, descriptive title
   - Detailed description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment details (OS, Python version, etc.)

### Submitting Changes

1. **Fork the repository** and create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following these guidelines:
   - Write clear, concise commit messages
   - Follow the existing code style (we use `ruff` for linting)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run linting
   uv run ruff check .

   # Run tests (if available)
   uv run pytest tests/
   ```

4. **Submit a pull request**:
   - Provide a clear description of your changes
   - Reference any related issues
   - Ensure all tests pass

## Development Setup

1. **Install dependencies**:
   ```bash
   uv sync --python 3.13
   ```

2. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

3. **Configure environment**:
   - Copy `.env.example` to `.env`
   - Fill in required values for your GCP project

## Code Style

- We use **Ruff** for linting and code formatting
- Run `uv run ruff check .` before committing
- Follow PEP 8 naming conventions
- Add docstrings to new functions and classes
- Keep functions focused and modular

## Testing Guidelines
- Ensure all tests pass before submitting PR
- Aim for meaningful test coverage, not just high percentages

## Documentation

- Update README.md if you add new features or change setup steps
- Add docstrings to new code
- Update type hints for better code clarity
- Add examples for new functionality

## Pull Request Process

1. Update documentation and tests
2. Ensure the PR description clearly describes the problem and solution
3. Link to any relevant issues
4. Wait for review from maintainers
5. Address any requested changes
6. Once approved, your PR will be merged

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the spirit of open source collaboration

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the `question` label
- Reach out to the maintainers

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for helping make this project better!