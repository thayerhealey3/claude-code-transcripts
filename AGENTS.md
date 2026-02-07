Uses uv. Run tests like this:

    uv run pytest

Run the development version of the tool like this:

    uv run claude-code-transcripts --help

Always practice TDD: write a faliing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

Run Black to format code before you commit:

    uv run black .

## Pull Requests

This repository is a fork. **Always create PRs targeting this fork's `main` branch, NOT the upstream/parent repository.**

When creating a PR, use:

    gh pr create --repo thayerhealey3/claude-code-transcripts --base main

Do NOT use `gh pr create` without the `--repo` flag, as it will default to the upstream repository.

The target for all PRs is: `thayerhealey3/claude-code-transcripts:main`
