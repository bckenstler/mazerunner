"""CLI entry point for MazeRunner Attempt Viewer."""

import argparse

from dotenv import load_dotenv
import uvicorn

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description="MazeRunner Attempt Viewer")
    parser.add_argument("--image-dir", default="data/dev/images",
                        help="Directory with maze PNG images")
    parser.add_argument("--gt-dir", default="data/dev/gt",
                        help="Directory with GT JSON files")
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory for saved run logs")
    parser.add_argument("--results-dir", default="results",
                        help="Directory with JSONL result files")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind (default: 8000)")
    args = parser.parse_args()

    # Configure app directories before starting
    from mazerunner.ui import app as app_module
    app_module.IMAGE_DIR = args.image_dir
    app_module.GT_DIR = args.gt_dir
    app_module.RUNS_DIR = args.runs_dir
    app_module.RESULTS_DIR = args.results_dir

    uvicorn.run(app_module.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
