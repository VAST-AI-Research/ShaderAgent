"""Stateless DSL validation and rendering, shared by the CLI and MCP server."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from .render.blender import RENDER_MODE_CONFIGS, RenderConfig, SceneRenderer
from .dsl.graph import ShaderGraph, ShaderGraphParser

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VALID_MODES = tuple(RENDER_MODE_CONFIGS)


def validate_dsl(dsl: str) -> dict[str, Any]:
    """Parse shader DSL and return ``{ok, errors, node_count}``."""
    graph = ShaderGraph.from_dsl(dsl)
    errors = list(graph.errors or [])
    return {
        "ok": not errors,
        "errors": errors,
        "node_count": 0 if errors else len(graph.nodes),
    }


def _resolve_blend(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    rooted = _PROJECT_ROOT / path
    if rooted.is_file():
        return str(rooted.resolve())
    raise FileNotFoundError(
        f"Blend file not found: {path} (looked next to cwd and {_PROJECT_ROOT})"
    )


def render_dsl(
    dsl: str,
    output_path: str,
    *,
    render_mode: str = "ball",
    blender: str = "blender",
    samples: int = 128,
    resolution: int = 512,
    device: str = "GPU",
    compute_device_type: str = "CUDA",
) -> str:
    """Validate DSL, compile it to bpy, and render through a Blender subprocess.

    Returns ``output_path``. The caller process does not import ``bpy``.
    """
    if render_mode not in _VALID_MODES:
        raise ValueError(f"render_mode must be one of {_VALID_MODES}, got {render_mode!r}")
    check = validate_dsl(dsl)
    if not check["ok"]:
        raise ValueError("DSL has structural errors:\n- " + "\n- ".join(check["errors"]))
    graph = ShaderGraph.from_dsl(dsl)
    bpy_text = ShaderGraphParser.to_bpy(graph)

    mode_cfg = RENDER_MODE_CONFIGS[render_mode]
    blend_file = _resolve_blend(mode_cfg["blend_file"])
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    render_config = RenderConfig(
        engine="CYCLES",
        samples=samples,
        use_denoising=True,
        film_transparent=True,
        resolution=(resolution, resolution),
        device=device,
        compute_device_type=compute_device_type,
    )
    renderer = SceneRenderer(executable_path=blender, render_config=render_config)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(bpy_text)
        script_path = handle.name
    try:
        return renderer.render(
            bpy_script=script_path,
            output_path=str(out),
            blend_file=blend_file,
            part_name=mode_cfg.get("part_name") or "SolidModel",
            render_mode=render_mode,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
