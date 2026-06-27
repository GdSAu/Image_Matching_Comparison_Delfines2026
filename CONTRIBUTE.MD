# Contributing

This document describes the recommended workflow, coding standards, and contribution process to ensure the repository remains consistent, maintainable, and reproducible.

---

# Development Setup

## 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

## 3. Install dependencies

Check INSTALL.MD to install the appropiate version of pytorch, and proceed with the installation there.

## 4. Install pre-commit hooks

```bash
pre-commit install
```

The hooks automatically run formatting and linting before every commit.

---

# Branching Strategy

Never commit directly to the `main` branch.

Create a feature branch from the latest version of `main`.

```bash
git checkout main
git pull
git checkout -b feature/my-feature
```

Recommended branch prefixes:

```
feature/<name>
fix/<name>
docs/<name>
refactor/<name>
test/<name>
```

Examples:

```
feature/lightglue

feature/gradio-interface

fix/ransac-timeout

docs/readme

refactor/dataset-loader
```

---

# Commit Guidelines

Keep commits focused on a single logical change.

Good commit messages describe **what** changed.

Examples:

```
Add HPatches dataset loader

Implement LightGlue matcher

Refactor benchmark pipeline

Fix descriptor normalization

Update installation guide
```

Avoid vague commit messages such as:

```
changes

fix

update

stuff
```

If a change is large, prefer several small commits over one large commit.

---

# Keeping Your Branch Updated

Before opening a Pull Request, synchronize your branch with `main`.

```bash
git checkout main
git pull

git checkout <your-branch>
git merge main
```

Resolve any conflicts locally before submitting the Pull Request.

---

# Local Checks

Before opening a Pull Request, run all quality checks locally.

## Lint

```bash
ruff check .
```

## Formatting

```bash
ruff format .
```

## Tests

```bash
pytest
```

All checks should complete successfully before creating a Pull Request.

---

# Pull Request Process

Every contribution should be submitted through a Pull Request.

The Pull Request should include:

* a clear description of the changes
* the motivation behind the changes
* any relevant benchmark results or screenshots
* testing information

Small Pull Requests are encouraged.

If your work depends on another Pull Request, reference it in the description.

---

# Code Style

The project uses:

* Ruff for linting
* Ruff Formatter for formatting
* Pytest for testing

Formatting should never be done manually if it can be handled automatically.

Run:

```bash
ruff format .
```

before committing.

---

# Project Structure

New code should follow the existing repository organization.

Example:

```
src/
    image_matching/
        extractors/
        matchers/
        datasets/
        benchmarks/
        evaluation/
        geometry/
        visualization/
        utils/
```

Avoid creating new top-level directories unless there is a clear architectural reason.

---

# Testing

Whenever practical:

* add tests for new functionality
* update tests affected by code changes
* ensure existing tests continue to pass

Unit tests should be independent and deterministic.

Avoid relying on large external datasets or GPU hardware unless the test explicitly targets those features.

---

# Documentation

Documentation is part of the project.

Please update documentation when:

* adding new modules
* changing public APIs
* modifying benchmark procedures
* introducing new datasets
* changing installation steps

---

# Dependencies

Only add dependencies when they are necessary.

If a new runtime dependency is introduced:

* update `requirements.txt`
* update any relevant installation documentation

Development tools should be added to `requirements-dev.txt`.

---

# Code Review

Pull Requests should be reviewed before merging.

Review focuses on:

* correctness
* readability
* maintainability
* consistency with the project architecture

Requested changes should be addressed before merging.

---

# Questions

If you're unsure where a new feature belongs or how it should be implemented, open an issue or start a discussion before writing a large amount of code.

Early discussion helps avoid duplicated work and large refactoring later.
