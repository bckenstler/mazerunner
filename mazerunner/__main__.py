"""CLI dispatcher for mazerunner."""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m mazerunner <command> [options]", file=sys.stderr)
        print("Commands: generate, visualize", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "generate":
        from mazerunner.generate import main as generate_main
        generate_main()
    elif command == "visualize":
        from mazerunner.visualize import main as visualize_main
        visualize_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Commands: generate, visualize", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
