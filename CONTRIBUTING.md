# Contributing to NIH Chest X-Ray Disease Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the NIH Chest X-Ray Disease Detection project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project is part of the Code Institute Data Analytics & AI Bootcamp capstone project. All contributors are expected to:

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CapStone.git
   cd CapStone
   ```
3. **Set up the development environment** (see [Development Setup](#development-setup))
4. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **Feature enhancements**: Add new features to the dashboard or models
- **Documentation improvements**: Enhance README, docstrings, or notebooks
- **Code quality**: Refactoring, optimization, or better practices
- **Testing**: Add or improve test coverage
- **Visualization**: Improve charts, graphs, or UI elements

### Areas for Contribution

- **Model Development**: Improve baseline models or add new deep learning architectures
- **Data Pipeline**: Enhance ETL, preprocessing, or feature engineering
- **Dashboard**: Add new tabs, visualizations, or interactive features
- **Documentation**: Improve clarity, add examples, or fix typos
- **Testing**: Add notebook tests, unit tests, or integration tests
- **Performance**: Optimize data loading, caching, or model inference

## Development Setup

### Prerequisites

- Python 3.11
- Git
- pyenv (recommended for Python version management)

### Installation

1. **Install pyenv** (if not already installed):
   ```bash
   # macOS
   brew install pyenv

   # Linux
   curl https://pyenv.run | bash
   ```

2. **Install Python 3.11**:
   ```bash
   pyenv install 3.11.13
   pyenv local 3.11.13
   ```

3. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   pip install -e .
   ```

5. **Download dataset** (if working on data-related features):
   ```bash
   # Follow instructions in jupyter_notebooks/01_data_collection_and_setup.ipynb
   # Note: Dataset is ~42GB, may take time to download
   ```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use **Black** for code formatting: `make format`
- Use **Ruff** for linting: `make lint`
- Use **Pyright** for type checking: `make typecheck`

### Pre-commit Checks

Before committing, run:
```bash
make pre-commit
```

This runs linting and formatting checks automatically.

### Code Organization

- **Notebooks**: Place in `jupyter_notebooks/` with descriptive names (e.g., `05_baseline_models.ipynb`)
- **Source code**: Place in `src/` directory
- **Tests**: Place in `tests/` directory
- **Dashboard tabs**: Place in `src/tabs/` directory

### Documentation

- Add docstrings to all functions and classes (Google style)
- Update README.md if adding new features
- Add inline comments for complex logic
- Keep notebook markdown cells descriptive and clear

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(dashboard): add interactive disease heatmap

Add new visualization showing disease co-occurrence patterns
with interactive hover tooltips and drill-down capabilities.

Closes #42
```

```
fix(preprocessing): correct image normalization range

Images were normalized to [0, 255] instead of [0, 1].
Updated normalization to use correct range for model input.
```

## Testing Guidelines

### Running Tests

```bash
# Run all notebook tests (excluding slow tests)
make test

# Run fast tests only
make test-fast

# Run all tests including slow ones
make test-all
```

### Adding Tests

- Use `pytest` for Python tests
- Use `nbmake` for notebook execution tests
- Add test markers for slow tests:
  ```python
  @pytest.mark.slow
  def test_large_dataset():
      ...
  ```

### Test Coverage

- Aim for >80% code coverage for new features
- Test edge cases and error conditions
- Test with sample data (don't require full 42GB dataset)

## Pull Request Process

1. **Update your fork** with the latest changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure all tests pass**:
   ```bash
   make test
   make pre-commit
   ```

3. **Update documentation** if needed:
   - README.md for new features
   - DEPLOYMENT.md for deployment changes
   - Docstrings for new functions

4. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Reference any related issues (e.g., "Fixes #123")
   - Provide detailed description of changes
   - Include screenshots for UI changes
   - List any breaking changes

5. **Respond to review feedback**:
   - Address all comments
   - Push updates to the same branch
   - Re-request review when ready

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #(issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests for this feature
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No console warnings or errors
```

## Reporting Issues

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if issue is already fixed
3. **Reproduce the issue** with minimal steps

### Creating a Good Issue

Include:
- **Clear title** describing the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Environment details**: OS, Python version, browser (for dashboard issues)
- **Screenshots or logs** if applicable
- **Relevant code snippets** (use code blocks)

### Issue Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g., macOS 14.0]
 - Python: [e.g., 3.11.13]
 - Browser: [e.g., Chrome 120] (for dashboard issues)

**Additional context**
Any other relevant information.
```

## Development Workflow

### Typical Workflow

1. Pick an issue or create one
2. Create a feature branch
3. Make changes iteratively
4. Test locally
5. Commit with conventional commit messages
6. Push to your fork
7. Create a pull request
8. Address review feedback
9. Merge (after approval)

### Branch Naming

- `feat/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions

## Questions or Need Help?

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check README.md, DEPLOYMENT.md, and notebook documentation

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to NIH Chest X-Ray Disease Detection!** 🎉
