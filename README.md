# 🔍 Discovery Agent

AI-powered product discovery agent built with LangGraph + Claude. Automates the full discovery cycle: research → backlog → hypotheses → human review.

## What it does

Most AI tools work as "ask → get answer." Product work doesn't. You need to take a step, check the result, go back, call a human — without losing context.

This agent solves that. One graph that:

- Analyzes data from multiple sources (interviews, analytics, reviews, stakeholder requests)
- Forms a structured backlog with priorities, metrics, and risks
- Builds hypotheses and updates the backlog based on them
- Checks whether data is sufficient — if not, stops and asks
- Returns to research with new data automatically
- Only then calls you for approval

## How it works

```
Research → check data → form backlog → validate hypotheses → updated backlog → human checkpoint
                ↑                                                                      |
                └──────────────────── go back with feedback ──────────────────────────┘
```

- Say **y** → approved, report saved
- Say **n** → give feedback, agent runs a new iteration
- Say **r** → return to research with new data

Context is preserved in SQLite between sessions. Close the terminal, continue later.

Every step is logged with exact timestamps — audit log shows what went into context and when.

## Versions

| File | Description |
|------|-------------|
| `discovery_agent.py` | v1 — basic reflection pattern |
| `discovery_agent_v2.py` | v2 — adds validation and audit log |
| `discovery_agent_v3.py` | v3 — full graph with cycles, SQLite persistence, human-in-the-loop |

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/discovery-agent.git
cd discovery-agent

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install langgraph langchain-anthropic langfuse langgraph-checkpoint-sqlite

# 4. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 5. Run
python3 discovery_agent_v3.py
```

## Optional: LangFuse observability

See every step, token count, and cost on a dashboard:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

Sign up free at [cloud.langfuse.com](https://cloud.langfuse.com)

## Resume a session

```python
run_discovery(
    session_id="20260402_091036",
    resume=True,
    product_context="..."
)
```

## Built with

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [Claude](https://anthropic.com) — language model
- SQLite — persistent checkpointing
- LangFuse — observability (optional)

## Coming next

- [ ] Google Docs integration
- [ ] Jira / Linear integration
- [ ] Slack integration
- [ ] App Store / Google Play reviews parser
- [ ] Grooming agent
- [ ] Planning agent
