# Day 17 implementation

`src/` contains the completed offline memory lab. Run from the repository root:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

The benchmark runs deterministically without API keys. It prints the standard and long-context stress tables. Each agent gets an isolated temporary state directory, so results are repeatable and do not depend on saved profiles from previous runs.

- `BaselineAgent` keeps a full message history within each thread and forgets it in a new thread.
- `AdvancedAgent` keeps recent messages, compacts older messages, and persists explicit user facts in `User.md`.
- `memory_store.py` handles profile editing, fact extraction, token estimation, and compact memory.
- `model_provider.py` supplies optional provider factories for OpenAI, custom OpenAI-compatible endpoints, Gemini, Anthropic, Ollama, and OpenRouter. The benchmark and tests use the offline path.

See [the benchmark analysis](../ANALYSIS.md) for measured results and limitations.
