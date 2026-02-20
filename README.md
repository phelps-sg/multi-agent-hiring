# Multi-Agent Hiring

A proof-of-concept multi-agent hiring pipeline that demonstrates principal-agent alignment in AI systems.

## What this is

This is a worked example accompanying a blog series on AI agent alignment. It implements a screening pipeline where an LLM-powered agent evaluates job candidates against role requirements — but with a twist: the LLM is deliberately confined to narrative generation only. All binary decisions (match level, recommended action, priority, confidence) are computed deterministically from structured data *before* the LLM is ever called.

The architecture treats the LLM as an untrusted agent operating within institutional constraints. Observable state, deterministic guardrails, and audit trails create the conditions under which aligned behaviour becomes the rational strategy for the model — the same mechanisms that institutional design uses in human organisations. The codebase then tests whether injecting *monitoring evidence* (override histories, calibration statistics, downstream review records) into the agent's context measurably reduces narrative bias.

The experimental notebook (`notebooks/experiment.ipynb`) contains a pre-registered experiment with 200 pipeline runs across 5 candidate profiles and 2 conditions (baseline vs monitored), with saved outputs and statistical analysis.

## Architecture

The screening pipeline runs in four stages:

1. **Classification** — a sub-agent classifies the application into a category (deterministic rules, no LLM)
2. **Summarisation** — a sub-agent generates a narrative summary (LLM call, confined to summarisation)
3. **Evaluation** — deterministic criterion-by-criterion evaluation against role requirements (no LLM)
4. **Screening** — the orchestrator computes match/action/confidence from evaluation results, then asks the LLM to write a reasoning narrative explaining the *already-decided* outcome

Key alignment properties:
- **Deterministic evaluation**: criterion matching uses structured data only — the LLM cannot influence pass/fail decisions
- **LLM confined to narrative**: the model writes explanations, not decisions
- **Observable state**: all sub-agent interactions flow through a shared `TransactionState`
- **Recommend-only**: the agent never makes binding decisions; a human reviewer sees the full audit trail
- **Monitoring evidence**: optional context injection that makes oversight visible to the model

## Quick start

```bash
git clone https://github.com/phelps-sg/multi-agent-hiring.git
cd multi-agent-hiring
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the CLI demo (simulated LLM, no API key needed)
python -m hiring_agents

# Run tests
pytest tests/ -v

# With a real LLM
python -m hiring_agents --model gpt-4o-mini
```

## Experiment

The notebook at `notebooks/experiment.ipynb` runs a controlled experiment measuring narrative bias:

- **Design**: 5 candidate profiles (prestige pair, non-traditional pair, ambiguity singleton) x 2 conditions (baseline, monitored) x 20 runs = 200 pipeline runs
- **Measures**: 3 binary real-world-impact features coded from narrative text — unsupported caveats, mentions of unconfigured criteria, evidence gap flags
- **Analysis**: Fisher's exact tests per candidate per measure, plus paired within-condition comparisons

The notebook includes saved outputs from a full run with Gemini 2.5 Pro, so you can inspect the results without an API key. To re-run the experiment yourself:

```bash
pip install -e ".[notebook]"
gcloud auth application-default login
jupyter notebook notebooks/experiment.ipynb
```

Edit the `VERTEXAI_PROJECT` in the config cell to point to your GCP project, or change `MODEL` to any [LiteLLM-supported model](https://docs.litellm.ai/docs/providers).

## Blog series

This codebase accompanies the *From Social Brains to Agent Societies* series:

1. [Evolving Cooperation](https://sphelps.substack.com/p/from-social-brains-to-agent-societies)
2. [Incentives](https://sphelps.substack.com/p/from-social-brains-to-agent-societies-35a)
3. [Identity and Reputation](https://sphelps.substack.com/p/from-social-brains-to-agent-societies-9f6)
4. [Principal-Agent Architecture](https://sphelps.substack.com/p/from-social-brains-to-agent-societies-39f) — introduces this codebase
5. The Evidence Changes the Narrative — experimental results (forthcoming)

## Project structure

```
hiring_agents/
  __init__.py
  __main__.py              # CLI entry point
  models.py                # msgspec domain models
  state.py                 # TransactionState (shared observable state)
  agent.py                 # Screening agent (Tier 1 orchestrator)
  evaluate.py              # Deterministic criterion evaluation
  llm.py                   # LiteLLM client with monitoring_context injection
  fixtures.py              # Role requirements and candidate fixtures
  borderline_fixtures.py   # Paired candidates for bias experiments
  monitoring_evidence.py   # Monitoring context factory functions
  analysis.py              # Binary coding functions for narrative analysis
  sub_agents/
    classifier.py          # Classification sub-agent
    summariser.py          # Summarisation sub-agent
tests/
  test_screening.py        # Deterministic pipeline tests (no API key needed)
  test_llm_integration.py  # Integration tests (requires API key)
notebooks/
  experiment.ipynb         # Full experiment with saved outputs
```

## License

MIT. See [LICENSE](LICENSE).
