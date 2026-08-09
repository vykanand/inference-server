# Agents Ã¢ Inference Server

## Overview

This project uses opencode agents for code generation, review, and debugging.

## Available Tools

- **Bash (pwsh)**: Shell commands, git, npm, docker
- **Edit**: File edits with exact string replacement
- **Glob/Grep/Read/Write**: File operations
- **Task/Plan/Research**: Subagent orchestration
- **GitNexus CLI**: Index, analyze, wiki generation

## Skills

- `plan` Ã¢ Break down tasks into implementation steps
- `code-review` Ã¢ Review PRs and diffs
- `gitnexus-cli` Ã¢ GitNexus commands
- `gitnexus-impact-analysis` Ã¢ Safety analysis before changes
- `zen-review` Ã¢ Multi-model code review (expensive)

## Usage

1. **Plan a task**: `/plan "Implement feature X"`
2. **Review code**: `/code-review <PR_URL>` or paste diff
3. **Index repo**: `gitnexus-cli index .`
4. **Generate wiki**: `gitnexus-cli generate-wiki`

## Contributing

See `CONTRIBUTING.md` for guidelines.