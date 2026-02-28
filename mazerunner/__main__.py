"""CLI dispatcher for MazeRunner benchmark."""

import sys


if len(sys.argv) < 2:
    print("Usage: python -m mazerunner {generate,evaluate,ui}")
    sys.exit(1)

command = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]  # Strip subcommand for argparse

if command == "generate":
    from mazerunner.generate import main
    main()
elif command == "evaluate":
    from mazerunner.evaluate_cli import main
    main()
elif command == "ui":
    from mazerunner.ui.cli import main
    main()
else:
    print(f"Unknown command: {command}")
    sys.exit(1)
