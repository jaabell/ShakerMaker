# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- Subagents are in `.claude/agents/`; delegate kernel work to `fk-core`,
  Python/MPI/HDF5 work to `drm-pipeline`, verification runs to
  `example-runner`, and documentation to `docs-writer`.
- Sign every commit per the "Signing commits" section above: an `Agent:`
  trailer plus your `Co-Authored-By:` line.
