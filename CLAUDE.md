# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Python project standards and configuration template** designed for development with Claude Code. It establishes best practices and development workflows for Python projects using the `uv` package manager.

## Python Project Standards

All Python development in this repository must follow these standards (from `.clauderc`):

- **Path Resolution**: Never use absolute paths. Use relative paths or path resolution functions.
- **Package Structure**: Every component/module must be a proper Python package with `__init__.py` files.
- **Logging**: Implement log tracing in every component for observability.
- **Package Management**: Use `uv` for all dependency management.
- **Dependency Tracking**: Always update both `uv.lock` and `pyproject.toml` when adding/modifying dependencies.
- **Documentation**: Maintain README.md, PRD.md, TASKS.md, and `__init__.py` files for all packages.
- **Output Handling**: Save all graph outputs and generated files to the `output/` folder.

## Development Workflows

The `/commands` directory contains workflow documentation:

- **test.md**: Guidelines for adding comprehensive tests (unit tests, edge cases, error handling)
- **refactor.md**: Code refactoring guidelines (readability, duplication, SOLID principles, comments)
- **review.md**: Code review checklist (quality issues, test coverage, documentation, potential bugs)

## Development Commands

When developing, ensure you:

1. **Testing**: Add comprehensive unit tests, handle edge cases, and verify error handling.
2. **Code Quality**: Follow refactoring guidelines to maintain readability and apply SOLID principles.
3. **Code Review**: Validate changes against the review checklist before committing.

## Architecture Notes

This is a **configuration and standards repository** with minimal actual implementation code. It serves as:
- A template for new Python projects
- A reference for development practices
- A configuration baseline for Claude Code workflows

When extending this repository with actual implementation code, ensure all code follows the Python project standards outlined above.
