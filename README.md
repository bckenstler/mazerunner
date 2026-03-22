# MazeRunner

A benchmark for evaluating vision-based GUI agents on maze navigation. MazeRunner generates canonical maze instances across 3 complexity tiers and renders them in 3 benchmark modes for different agent interaction paradigms.

## Installation

Requires Python 3.11+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Quick Start

Generate a dataset of 1000 mazes:

```bash
uv run python -m mazerunner generate --output-dir data/dev --num-mazes 1000 --master-seed 42 --tier-distribution 300,400,300
```

Render all benchmark modes:

```bash
uv run python -m mazerunner visualize --input-dir data/dev --output-dir data/dev/renderings --mode all
```

Run an agent on mazes:

```bash
# OpenAI (default)
uv run python -m mazerunner agent --mode text_grid --instance-dir data/dev --model gpt-5.4 --num-episodes 1

# Anthropic
uv run python -m mazerunner agent --provider anthropic --mode text_grid --instance-dir data/dev --model claude-sonnet-4-6 --num-episodes 1

# Google Gemini
uv run python -m mazerunner agent --provider gemini --mode text_grid --instance-dir data/dev --model gemini-2.5-flash --num-episodes 1

# Fireworks AI
uv run python -m mazerunner agent --provider fireworks --mode text_grid --instance-dir data/dev --model accounts/fireworks/models/deepseek-r1 --num-episodes 1
```

Run the eval harness:

```bash
uv run python -m mazerunner eval --mode text_grid --instance-dir data/dev --model gpt-5.4 --num-episodes 10 --output eval_results.json
```

## Providers

MazeRunner supports 4 LLM providers, each with provider-specific reasoning/thinking configuration:

| Provider | `--provider` | API Key Env Var | Reasoning Config |
|----------|-------------|-----------------|------------------|
| OpenAI | `openai` (default) | `OPENAI_API_KEY` | `--reasoning-effort low\|medium\|high` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Adaptive thinking (default), `--effort low\|medium\|high\|max` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | `--thinking-budget N` (tokens) or `--thinking-level low\|medium\|high` |
| Fireworks AI | `fireworks` | `FIREWORKS_API_KEY` | `--thinking-budget N` (tokens) |

API keys are loaded from a `.env` file in the project root.

## Benchmark Modes

| Mode | Format | Agent Tool | Interaction |
|------|--------|------------|-------------|
| `text_grid` | ASCII text | `navigate` | Direction strings (U/D/L/R) |
| `vision_grid` | PNG image | `navigate` | Direction strings (U/D/L/R) |
| `vision_drag` | PNG image | `drag` | Pixel coordinate paths |

### Single-Step Mode

By default, the `navigate` tool accepts multi-character direction strings (e.g., `"RRDDDLUU"`). Use `--single-step` to restrict it to one direction per call, making the benchmark harder:

```bash
uv run python -m mazerunner agent --mode text_grid --instance-dir data/dev --model gpt-5.4 --single-step
```

## Architecture

```
mazerunner/
  common/types.py        ← Cell, DifficultyConfig, MazeGrid, MazeInstance dataclasses
  generator/             ← Maze generation pipeline (DFS + BFS)
  renderer/              ← 3 rendering modes (vision_drag, vision_grid, text_grid)
  navigator/             ← Stateful navigation (grid + drag) with collision detection
  openenv/               ← OpenEnv MCP environment integration
  agent/
    types.py             ← AgentConfig, TurnRecord, EpisodeResult dataclasses
    tool_transform.py    ← Raw tool result → model-facing content (strips internals)
    tool_defs.py         ← Tool schemas (OpenAI, Anthropic, Gemini, Chat Completions)
    openai_loop.py       ← Agent loop using OpenAI Responses API
    anthropic_loop.py    ← Agent loop using Anthropic Messages API
    gemini_loop.py       ← Agent loop using Google Gemini API
    fireworks_loop.py    ← Agent loop using Fireworks Chat Completions API
    runner.py            ← Provider runners + get_runner() factory
    context_manager.py   ← Sliding window image context (OpenAI)
    chat_context.py      ← Chat Completions context (Fireworks)
  eval/
    protocol.py          ← EpisodeRunner protocol + data types
    metrics.py           ← Success rate, efficiency, invalid action rate
    harness.py           ← Orchestrates episodes across instances
    io.py                ← JSON save/load for eval results
  __main__.py            ← CLI dispatcher (generate, visualize, serve, agent, eval)
```

## Difficulty Tiers

| Parameter | Tier 1 (Easy) | Tier 2 (Medium) | Tier 3 (Hard) |
|-----------|---------------|-----------------|----------------|
| Grid rows | 5-8 | 10-16 | 18-28 |
| Grid cols | 7-12 | 14-22 | 25-40 |
| Min solution length | 8 | 20 | 40 |

## Output Format

Each maze JSON file contains:

```json
{
  "id": "maze_000000",
  "grid_rows": 8,
  "grid_cols": 10,
  "start": "0,3",
  "goal": "7,9",
  "adjacency": {"0,0": ["0,1", "1,0"], "...": "..."},
  "shortest_path_cells": ["0,3", "1,3", "..."],
  "metadata": {
    "tier": 1,
    "endpoint_type": "edge-edge",
    "difficulty_score": 3,
    "path_length": 15,
    "branching_factor": 2.1,
    "color_schema": {"wall": "#1a1a2e", "...": "..."}
  }
}
```

## E2E Agent Eval

The `scripts/e2e_agent_eval.py` script runs agent evaluation across all 3 modes:

```bash
# Default: 3 episodes per mode with gpt-5.4
uv run python scripts/e2e_agent_eval.py

# Verbose single episode
uv run python scripts/e2e_agent_eval.py -v --num-episodes 1

# With specific provider
uv run python scripts/e2e_agent_eval.py --provider gemini --model gemini-2.5-flash -v
uv run python scripts/e2e_agent_eval.py --provider anthropic --model claude-sonnet-4-6
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run e2e test scripts (no API key needed)
uv run python scripts/e2e_all.py
```
