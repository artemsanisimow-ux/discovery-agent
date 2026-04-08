# 🔍 Discovery Agent

AI-powered product discovery agent built with LangGraph + Claude. Analyzes data from multiple sources, synthesizes insights, validates hypotheses, and stops when it needs a human.

## What it does

Most AI tools work as "ask → get answer." Product discovery doesn't. You need to synthesize data from interviews, analytics, reviews, and stakeholder requests — then check your reasoning, iterate, and only then commit to a direction.

This agent automates that cycle:

1. **Synthesize** — analyzes raw data from all sources, structures insights by category
2. **Critique** — a second model call reviews the draft for weak evidence, contradictions, and gaps
3. **Refine** — improves the document based on critique, reruns critique
4. **Human checkpoint** — shows the final document, audit snapshot, and asks for approval

If you say **n** — give feedback, agent runs another iteration.
If you say **y** — report is saved, session ends.

## How it works

```
Raw data (interviews, analytics, reviews, stakeholder requests)
        ↓
   Synthesize insights
        ↓
   Critique draft ──→ Refine ──→ Critique again
        ↓ (approved)
   Human checkpoint
        ↓ (y)
   Save report + audit log
```

Every step is logged with timestamps — audit log shows exactly what went into context and when.

## Quick start

```bash
# 1. Clone
git clone https://github.com/artemsanisimow-ux/discovery-agent.git
cd discovery-agent

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Dependencies
pip install langgraph langchain-anthropic python-dotenv

# 4. Create .env
touch .env
```

Add to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
LANGUAGE=en
```

```bash
# 5. Run
python3 discovery_agent_v2.py
```

## Language support

The agent is fully bilingual — all output, prompts, and reports follow the selected language.

```bash
# English
python3 discovery_agent_v2.py --lang en

# Russian
python3 discovery_agent_v2.py --lang ru
```

Priority: `--lang` flag → `LANGUAGE` in `.env` → default `ru`

## Input data

Pass your raw data directly in the `run_discovery()` call:

```python
run_discovery(
    product_context="B2B SaaS for HR teams. Metric: activation rate (28%, target 45%).",
    interviews="User 1: couldn't figure out where to start...",
    analytics="Day-7 retention: 34%. Drop-off at step 3: 60%.",
    reviews="App Store: 'setup is too complicated' (×12)",
    call_recordings="...",
    stakeholder_requests="CEO: why isn't activation rate growing despite new features?"
)
```

## Output

Each session saves two files:

- `discovery_report_SESSION_TIMESTAMP.md` — structured insights document with problems, patterns, signals, hypotheses, contradictions
- `discovery_audit_SESSION_TIMESTAMP.json` — full step-by-step log with timestamps and what went into context at each step

## Resume a session

```python
run_discovery(
    session_id="20260402_091036",
    resume=True,
    product_context="..."
)
```

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration and state persistence
- [Claude](https://anthropic.com) — language model (claude-opus-4-5)
- SQLite — checkpointing between sessions

## Part of a larger system

| Agent | Repo | Description |
|-------|------|-------------|
| Discovery | this repo | Raw data → insights → hypotheses |
| Grooming | [grooming-agent](https://github.com/artemsanisimow-ux/grooming-agent) | Jira + Linear → estimate → acceptance criteria → prioritize |
| Planning | coming soon | Groomed tasks → sprint plan with Monte Carlo + pre-mortem |
