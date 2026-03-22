"""CLI dispatcher for mazerunner."""

import sys


def _run_agent():
    """Run the agent loop on maze instances."""
    import argparse

    parser = argparse.ArgumentParser(description="Run agent on maze instances")
    parser.add_argument("--mode", required=True, choices=["text_grid", "vision_grid", "vision_drag"])
    parser.add_argument("--instance-dir", required=True, help="Directory containing maze instances")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-turns", type=int, default=100, help="Max turns per episode")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini", "fireworks"])
    parser.add_argument("--thinking-budget", type=int, default=None, help="Gemini/Fireworks thinking budget tokens")
    parser.add_argument("--thinking-level", default=None, choices=["low", "medium", "high"], help="Gemini 3 thinking level")
    parser.add_argument("--single-step", action="store_true", help="Restrict navigate to one direction per call")
    args = parser.parse_args()

    from mazerunner.agent.types import AgentConfig
    from mazerunner.openenv.server.maze_environment import MazeEnvironment

    config = AgentConfig(
        model=args.model,
        mode=args.mode,
        max_turns=args.max_turns,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        provider=args.provider,
        thinking_budget=args.thinking_budget,
        thinking_level=args.thinking_level,
        single_step=args.single_step,
    )

    env = MazeEnvironment(mode=args.mode, instance_dir=args.instance_dir, single_step=args.single_step)

    if config.provider == "anthropic":
        from mazerunner.agent.anthropic_loop import run_anthropic_episode
        run_episode = run_anthropic_episode
    elif config.provider == "gemini":
        from mazerunner.agent.gemini_loop import run_gemini_episode
        run_episode = run_gemini_episode
    elif config.provider == "fireworks":
        from mazerunner.agent.fireworks_loop import run_fireworks_episode
        run_episode = run_fireworks_episode
    else:
        from mazerunner.agent.openai_loop import run_openai_episode
        run_episode = run_openai_episode

    for i in range(args.num_episodes):
        result = run_episode(config, env)
        status = "SUCCESS" if result.success else "FAIL"
        print(f"Episode {i + 1}: {status} | maze={result.maze_id} turns={result.total_turns} reward={result.total_reward:.2f}")


def _run_eval():
    """Run evaluation harness across maze instances."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--mode", required=True, choices=["text_grid", "vision_grid", "vision_drag"])
    parser.add_argument("--instance-dir", required=True, help="Directory containing maze instances")
    parser.add_argument("--model", default="gpt-5.4", help="Model name")
    parser.add_argument("--num-episodes", type=int, default=None, help="Limit number of episodes")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--reward-mode", default="sparse", choices=["sparse", "shaped", "efficiency"])
    parser.add_argument("--output", required=True, help="Output JSON path for results")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini", "fireworks"])
    parser.add_argument("--thinking-budget", type=int, default=None, help="Gemini/Fireworks thinking budget tokens")
    parser.add_argument("--thinking-level", default=None, choices=["low", "medium", "high"], help="Gemini 3 thinking level")
    parser.add_argument("--single-step", action="store_true", help="Restrict navigate to one direction per call")
    args = parser.parse_args()

    from pathlib import Path

    from mazerunner.agent.runner import get_runner
    from mazerunner.agent.types import AgentConfig
    from mazerunner.eval.harness import run_eval
    from mazerunner.eval.io import save_eval_result

    # Discover instance paths
    inst_dir = Path(args.instance_dir)
    instances_subdir = inst_dir / "instances"
    if instances_subdir.is_dir():
        paths = sorted(str(p) for p in instances_subdir.glob("*.json"))
    else:
        paths = sorted(str(p) for p in inst_dir.glob("*.json"))

    config = AgentConfig(
        model=args.model,
        mode=args.mode,
        max_turns=args.max_turns,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        provider=args.provider,
        thinking_budget=args.thinking_budget,
        thinking_level=args.thinking_level,
        single_step=args.single_step,
    )
    runner = get_runner(config)

    result = run_eval(
        runner=runner,
        instance_paths=paths,
        mode=args.mode,
        max_steps=args.max_steps,
        reward_mode=args.reward_mode,
        num_episodes=args.num_episodes,
        model=args.model,
        single_step=args.single_step,
    )

    save_eval_result(result, args.output)
    print(f"Eval complete: {result.num_episodes} episodes")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Results saved to {args.output}")


def main():
    """Dispatch CLI subcommands."""
    if len(sys.argv) < 2:
        print("Usage: python -m mazerunner <command> [options]", file=sys.stderr)
        print("Commands: generate, visualize, serve, agent, eval", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "generate":
        from mazerunner.generate import main as generate_main
        generate_main()
    elif command == "visualize":
        from mazerunner.visualize import main as visualize_main
        visualize_main()
    elif command == "serve":
        from mazerunner.openenv.server.app import main as serve_main
        serve_main()
    elif command == "agent":
        _run_agent()
    elif command == "eval":
        _run_eval()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Commands: generate, visualize, serve, agent, eval", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
