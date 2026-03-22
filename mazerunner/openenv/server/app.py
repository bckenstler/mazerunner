"""FastAPI entry point for MazeEnvironment server."""

import os

from openenv.core.env_server import Action, Observation, create_app

from mazerunner.openenv.server.maze_environment import MazeEnvironment


def _env_factory() -> MazeEnvironment:
    mode = os.environ.get("MAZE_MODE", "text_grid")
    instance_dir = os.environ.get("MAZE_INSTANCE_DIR", "data/dev")
    reward_mode = os.environ.get("MAZE_REWARD_MODE", "sparse")
    max_steps = int(os.environ.get("MAZE_MAX_STEPS", "100"))
    seed = int(os.environ.get("MAZE_SEED", "42"))
    return MazeEnvironment(
        mode=mode,
        instance_dir=instance_dir,
        reward_mode=reward_mode,
        max_steps=max_steps,
        seed=seed,
    )


app = create_app(
    env=_env_factory,
    action_cls=Action,
    observation_cls=Observation,
    env_name="MazeRunner",
)


def main() -> None:
    import uvicorn

    host = os.environ.get("MAZE_HOST", "0.0.0.0")
    port = int(os.environ.get("MAZE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
