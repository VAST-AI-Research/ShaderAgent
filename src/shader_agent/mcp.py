"""stdio MCP server exposing DSL validation and rendering."""
from __future__ import annotations


def main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "shader-agent-mcp is installed with the package; check that mcp is importable"
        ) from exc

    from . import tools

    mcp = FastMCP("shader-agent")

    @mcp.tool()
    def validate_dsl(dsl: str) -> dict:
        """Parse ShaderAgent DSL and return ok / errors / node_count."""
        return tools.validate_dsl(dsl)

    @mcp.tool()
    def render_dsl(
        dsl: str,
        output_path: str,
        render_mode: str = "ball",
        samples: int = 128,
        resolution: int = 512,
        device: str = "GPU",
        compute_device_type: str = "CUDA",
        blender: str = "blender",
    ) -> str:
        """Render DSL onto the bundled material ball or plane. Returns the PNG path."""
        return tools.render_dsl(
            dsl,
            output_path,
            render_mode=render_mode,
            blender=blender,
            samples=samples,
            resolution=resolution,
            device=device,
            compute_device_type=compute_device_type,
        )

    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
