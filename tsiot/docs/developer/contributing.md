# Contributing to TSIoT

Welcome to the TSIoT (Time Series IoT) project! We're excited to have you contribute to our open-source synthetic data generation platform. This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)
- [Recognition](#recognition)

## Getting Started

### Prerequisites

Before contributing to TSIoT, make sure you have:

- **Go 1.21 or later** installed
- **Git** for version control
- **Docker** for containerized development (optional)
- **Make** for build automation
- A **GitHub account** for submitting contributions

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tsiot.git
   cd tsiot
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/inferloop/tsiot.git
   ```
4. **Set up your development environment** (see [Development Setup](#development-setup))

## Development Setup

### Local Development Environment

```bash
# Install dependencies
make deps

# Install development tools
make install-dev-tools

# Set up pre-commit hooks
make install-hooks

# Run tests to ensure everything works
make test
```

### Docker Development Environment

```bash
# Build development containers
make docker-dev

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access development container
docker-compose -f docker-compose.dev.yml exec tsiot-dev bash
```

### IDE Configuration

#### VS Code
Recommended extensions and settings:

```json
// .vscode/settings.json
{
    "go.useLanguageServer": true,
    "go.formatTool": "goimports",
    "go.lintTool": "golangci-lint",
    "go.testFlags": ["-v"],
    "go.coverOnSave": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

```json
// .vscode/extensions.json
{
    "recommendations": [
        "golang.go",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml",
        "ms-vscode.docker"
    ]
}
```

#### GoLand/IntelliJ
- Enable "Go Modules" integration
- Configure "File Watchers" for goimports and golangci-lint
- Set up "Run Configurations" for tests and main applications

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

#### = Bug Fixes
- Fix existing bugs in the codebase
- Improve error handling and edge cases
- Performance optimizations

#### ( New Features
- New data generation algorithms
- Additional storage backends
- Enhanced validation metrics
- Improved observability features

#### =Ú Documentation
- API documentation improvements
- Tutorial and guide enhancements
- Code comments and examples
- Architecture documentation

#### >ê Testing
- Unit test improvements
- Integration test additions
- Benchmark tests
- End-to-end test scenarios

#### =' Infrastructure
- CI/CD pipeline improvements
- Docker and Kubernetes enhancements
- Build system optimizations
- Monitoring and alerting

### Contribution Workflow

1. **Check existing issues** to avoid duplicate work
2. **Create or comment on an issue** to discuss your proposed changes
3. **Fork and clone** the repository
4. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** following our code standards
6. **Test your changes** thoroughly
7. **Commit your changes** with clear messages
8. **Push to your fork** and create a pull request
9. **Address feedback** from code review
10. **Celebrate** when your PR is merged! <‰

### Branch Naming Convention

```bash
# Feature branches
feature/add-timegan-generator
feature/improve-validation-metrics

# Bug fix branches
bugfix/fix-memory-leak
bugfix/correct-arima-parameters

# Documentation branches
docs/update-api-reference
docs/add-deployment-guide

# Infrastructure branches
infra/add-kubernetes-manifests
infra/improve-docker-build
```

## Code Standards

### Code Style

We follow the guidelines outlined in our [Code Style Guide](code-style.md). Key points:

- Use `gofmt` and `goimports` for formatting
- Follow Go naming conventions
- Write comprehensive tests
- Include documentation for public APIs
- Handle errors appropriately

### Pre-commit Checks

Before committing, ensure your code passes:

```bash
# Format code
make fmt

# Run linters
make lint

# Run tests
make test

# Run security checks
make security-check

# Generate documentation
make docs
```

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
# Format
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

# Examples
feat(generators): add TimeGAN algorithm implementation
fix(validation): correct statistical test calculations
docs(api): update generation endpoint documentation
test(arima): add comprehensive unit tests
refactor(storage): improve connection pooling
```

#### Commit Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

## Pull Request Process

### Creating a Pull Request

1. **Ensure your branch is up-to-date**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request** with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - References to related issues
   - Screenshots/examples if applicable

### Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance benchmarks run (if applicable)

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is properly commented
- [ ] Corresponding documentation updated
- [ ] No new warnings introduced
- [ ] Tests added/updated for the changes
- [ ] All CI checks pass

## Related Issues
- Fixes #123
- Relates to #456

## Screenshots (if applicable)
<!-- Add screenshots or examples here -->

## Additional Notes
<!-- Any additional information or context -->
```

### Code Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Peer Review**: At least one maintainer reviews the code
3. **Discussion**: Address any feedback or questions
4. **Approval**: Maintainer approves the changes
5. **Merge**: Changes are merged into the main branch

### Review Criteria

Reviewers will check for:
- **Functionality**: Does the code work as intended?
- **Code Quality**: Is the code clean, readable, and maintainable?
- **Testing**: Are there adequate tests for the changes?
- **Documentation**: Is the code properly documented?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?
- **Compatibility**: Do the changes maintain backward compatibility?

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

```markdown
## Bug Description
A clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Go version: [e.g., 1.21.0]
- TSIoT version: [e.g., v1.2.0]
- Docker version: [if applicable]

## Logs
```
[Include relevant log output]
```

## Additional Context
Any other context about the problem.
```

### Feature Requests

For feature requests, please provide:

```markdown
## Feature Description
A clear and concise description of the feature.

## Use Case
Describe the problem this feature would solve.

## Proposed Solution
Describe the solution you'd like to see.

## Alternatives Considered
Describe any alternative solutions or features you've considered.

## Additional Context
Any other context or screenshots about the feature request.
```

### Issue Labels

We use labels to categorize issues:

- **Priority**: `priority/critical`, `priority/high`, `priority/medium`, `priority/low`
- **Type**: `bug`, `enhancement`, `feature`, `documentation`, `question`
- **Component**: `generators`, `validation`, `storage`, `api`, `cli`, `ui`
- **Status**: `needs-investigation`, `in-progress`, `blocked`, `ready-for-review`
- **Difficulty**: `good-first-issue`, `help-wanted`, `expert-needed`

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and follow these principles:

- **Be respectful** and considerate in all interactions
- **Be collaborative** and help others learn
- **Be patient** with newcomers and questions
- **Be constructive** in feedback and criticism
- **Be inclusive** and welcome diverse perspectives

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code reviews and technical discussions
- **Discord**: Real-time chat and community support
- **Email**: security@inferloop.com for security issues

### Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Ask in GitHub Discussions** for general questions
4. **Create an issue** for bugs or feature requests
5. **Join our Discord** for real-time help

## Recognition

### Contributor Recognition

We value all contributions and recognize contributors through:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Contributions mentioned in release notes
- **Social media**: Highlighting significant contributions
- **Swag**: Stickers and swag for regular contributors
- **Mentorship**: Opportunities to mentor new contributors

### Becoming a Maintainer

Active contributors may be invited to become maintainers. Maintainers:

- Review and merge pull requests
- Help with issue triage and project planning
- Mentor new contributors
- Represent the project in the community

### Maintainer Responsibilities

- **Code Review**: Thoroughly review pull requests
- **Issue Management**: Triage and label issues appropriately
- **Documentation**: Keep documentation up-to-date
- **Testing**: Ensure comprehensive test coverage
- **Security**: Address security concerns promptly
- **Communication**: Maintain clear communication with contributors

## Development Tips

### Debugging

```bash
# Enable debug logging
export TSIOT_LOG_LEVEL=debug

# Run with profiling
go run -race cmd/server/main.go --profile

# Use delve debugger
dlv debug cmd/server/main.go
```

### Testing

```bash
# Run specific tests
go test -v ./internal/generators/timegan

# Run tests with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./internal/generators/

# Run integration tests
make test-integration
```

### Performance Testing

```bash
# CPU profiling
go test -cpuprofile=cpu.prof -bench=.

# Memory profiling
go test -memprofile=mem.prof -bench=.

# Load testing
k6 run tests/load/basic-load-test.js
```

## Frequently Asked Questions

### Q: How do I add a new data generation algorithm?

A: Create a new package under `internal/generators/`, implement the `Generator` interface, and add it to the factory. See the [Architecture Guide](../architecture/overview.md) for details.

### Q: How do I add support for a new storage backend?

A: Implement the storage interfaces in `internal/storage/interfaces/` and add your implementation to the factory. Look at existing implementations like InfluxDB for reference.

### Q: How do I run the project locally?

A: Use `make dev-server` to start the development server, or follow the [Quick Start Guide](../user-guide/getting-started.md).

### Q: How do I add new metrics for data validation?

A: Add your metric implementation to `internal/validation/metrics/` and register it in the validation engine.

### Q: What's the difference between unit and integration tests?

A: Unit tests test individual components in isolation, while integration tests test the interaction between components. Both are important for ensuring code quality.

### Q: How do I propose a breaking change?

A: Create an issue first to discuss the change with maintainers and the community. Breaking changes require careful consideration and planning.

## Thank You!

Thank you for considering contributing to TSIoT! Your contributions help make synthetic time series data generation more accessible and powerful for everyone. We look forward to working with you! =€

---

For more information, check out our:
- [Code Style Guide](code-style.md)
- [Testing Guide](testing-guide.md)
- [Performance Tuning Guide](performance-tuning.md)
- [API Documentation](../api/openapi.yaml)
- [Architecture Overview](../architecture/overview.md)