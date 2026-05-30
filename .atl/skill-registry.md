# Skill Registry

**Delegator use only.** Any agent that launches sub-agents reads this registry to resolve compact rules, then injects them directly into sub-agent prompts. Sub-agents do NOT read this registry or individual SKILL.md files.

See `_shared/skill-resolver.md` for the full resolution protocol.

## User Skills

| Trigger | Skill | Path |
|---------|-------|------|
| browser automation, web interactions, screenshots, QA, Electron/Slack automation | `agent-browser` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/.agents/skills/agent-browser/SKILL.md` |
| discover/install skills for new capabilities | `find-skills` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/.agents/skills/find-skills/SKILL.md` |
| Playwright web testing and browser automation | `playwright-skill` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/.agents/skills/playwright-skill/SKILL.md` |
| creating pull requests / preparing branch for review | `branch-pr` | `/home/rodriggod/.cursor/skills/branch-pr/SKILL.md` |
| writing Go tests / Bubbletea tests / coverage | `go-testing` | `/home/rodriggod/.cursor/skills/go-testing/SKILL.md` |
| creating GitHub issues / bug or feature templates | `issue-creation` | `/home/rodriggod/.cursor/skills/issue-creation/SKILL.md` |
| adversarial dual-agent review ("judgment day") | `judgment-day` | `/home/rodriggod/.cursor/skills/judgment-day/SKILL.md` |
| create new agent skills and SKILL.md specs | `skill-creator` | `/home/rodriggod/.cursor/skills/skill-creator/SKILL.md` |

## Compact Rules

Pre-digested rules per skill. Delegators copy matching blocks into sub-agent prompts as `## Project Standards (auto-resolved)`.

### agent-browser
- Use `agent-browser` instead of built-in browser tools for browser/Electron/Slack tasks.
- Before automation, load core instructions: `agent-browser skills get core`.
- Keep interactions on accessibility refs (`@eN`) and CLI workflows.
- Use specialized modules (`electron`, `slack`, `dogfood`, `vercel-sandbox`, `agentcore`) when applicable.
- Prefer persistent sessions/vault features when auth or long flows are involved.

### find-skills
- Use `npx skills find <query>` to search installable ecosystem skills.
- Check [skills.sh](https://skills.sh/) leaderboard before broad search.
- Validate quality before recommending: installs, source reputation, repository credibility.
- Provide install command and source link for each recommendation.
- If no match exists, proceed directly with native capabilities and optionally suggest creating a skill.

### playwright-skill
- Detect running dev servers first before localhost automation.
- Write scripts to `/tmp/playwright-test-*.js`, never into project or skill directory.
- Parameterize target URL with a top-level constant/env.
- Default to `headless: false`; only use headless when user explicitly asks.
- Execute via the skill wrapper from skill directory: `node run.js /tmp/script.js`.
- Use robust waits (`waitForURL`, `waitForSelector`) and explicit error handling.

### branch-pr
- Every PR MUST link an approved issue (`status:approved`) and include exactly one `type:*` label.
- Use branch naming `type/description` with allowed lowercase pattern.
- Follow conventional commits and ensure commit type maps to PR label.
- Include required PR template sections: issue link, type, summary, changes table, test plan.
- Run shellcheck on touched scripts before opening PR.

### go-testing
- Prefer table-driven tests for functions and branching behavior.
- For Bubbletea, test `Model.Update()` transitions directly, then cover flows with `teatest`.
- Use golden files for TUI render snapshots where stability matters.
- Test both success and failure paths (including error returns).
- Use `t.TempDir()` for filesystem isolation in tests.

### issue-creation
- Always use issue templates; blank issues are disallowed.
- New issues start as `status:needs-review`; PRs require `status:approved`.
- Route questions to Discussions rather than Issues.
- Verify duplicates before creating a new issue.
- Include required reproduction/problem fields and pre-flight checkboxes.

### judgment-day
- Launch two independent blind judges in parallel and synthesize findings centrally.
- Resolve and inject project standards from registry before launching judges/fixers.
- Treat findings as confirmed only when both judges agree; triage one-sided findings separately.
- Re-judge after fixes when required by the protocol and stop only at APPROVED or ESCALATED state.
- Do not commit/push between fix and required re-judgment checkpoints.

### skill-creator
- Create skills only for reusable patterns, not one-off tasks.
- Use canonical structure: `skills/<name>/SKILL.md` (+ optional `assets/`, `references/`).
- Keep frontmatter complete with explicit trigger text in description.
- Prioritize critical actionable patterns and concise examples over long explanations.
- Add the new skill to project guidance index (`AGENTS.md`) after creation.

## Project Conventions

| File | Path | Notes |
|------|------|-------|
| `AGENTS.md` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/AGENTS.md` | Index - references files below |
| `main.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/main.py` | Referenced by `AGENTS.md` |
| `src/operator/entrypoint.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/src/operator/entrypoint.py` | Referenced by `AGENTS.md` |
| `src/operator/api/server.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/src/operator/api/server.py` | Referenced by `AGENTS.md` |
| `qa/check_repo.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/qa/check_repo.py` | Referenced by `AGENTS.md` |
| `qa/subnet_compat.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/qa/subnet_compat.py` | Referenced by `AGENTS.md` |
| `scripts/deploy_check.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/scripts/deploy_check.py` | Referenced by `AGENTS.md` |
| `src/eval/runner.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/src/eval/runner.py` | Referenced by `AGENTS.md` |
| `deploy/pm2/operator.config.cjs` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/deploy/pm2/operator.config.cjs` | Referenced by `AGENTS.md` |
| `deploy/pm2/mcp.config.cjs` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/deploy/pm2/mcp.config.cjs` | Referenced by `AGENTS.md` |
| `training/pipeline.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/training/pipeline.py` | Referenced by `AGENTS.md` |
| `training/iwap_client.py` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/training/iwap_client.py` | Referenced by `AGENTS.md` |
| `README.md` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/README.md` | Referenced by `AGENTS.md` |
| `copilot-instructions.md` | `/home/rodriggod/AUTOPPIA/Autoppia_repos/autoppia_operator/.github/copilot-instructions.md` | Security-focused review policy and remediation expectations |

Read the convention files listed above for project-specific patterns and rules. All referenced paths have been extracted — no need to read index files to discover more.
