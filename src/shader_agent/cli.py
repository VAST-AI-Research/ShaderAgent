"""Installed single-item ShaderAgent CLI."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ShaderAgent for one material.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        from .pipeline import ShaderPipeline
        result = ShaderPipeline.from_config_path(args.config).run(
            text_description=args.text, image_path=args.image, tag=args.output,
        )
    except Exception as exc:
        parser.error(str(exc))
    if result is None:
        parser.error("pipeline failed without producing an output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
