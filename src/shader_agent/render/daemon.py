"""Persistent Blender daemon for Monte Carlo param tuning.

Each ``SceneRenderer.render`` call pays ~7s of cold start (Python + bpy import
+ scene open + GPU device init) before any rendering. Negligible for one-shot
steps, but the MC tuner fires 3 renders x N iterations per refinement step, so
startup dominates.

A daemon opens its blend file once, sets up device / camera / render config once,
then reads JSON render commands from stdin and writes JSON responses to stdout,
each prefixed with a magic sentinel so they survive Blender's own stdout noise.
Used ONLY by ``ParamTuner``; other render paths keep using
``SceneRenderer.render``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _bootstrap_import_path() -> None:
    """Make ``shader_agent`` importable under Blender's ``-P``, which runs this
    file as a top-level script so package-relative imports fail. Mirrors blender.py.
    """
    here = Path(__file__).resolve().parent       # .../shader_agent/render
    pkg_root = here.parent                        # .../shader_agent
    for p in (here, pkg_root, pkg_root.parent):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_bootstrap_import_path()

_SENTINEL = "###DAEMON###"


# Client side: runs in the main pipeline process.
class BlenderDaemon:
    """Long-lived Blender subprocess that renders on demand.

    Not thread-safe by itself: a single daemon serializes render commands over
    one stdin. Use one daemon per blend file you want to render from, and
    parallelize across daemons (see ``ParallelViewRenderer``).
    """

    _READY_TIMEOUT_S: float = 60.0
    _RENDER_TIMEOUT_S: float = 300.0

    def __init__(
        self,
        executable_path: str,
        blend_file: str,
        render_config,            # RenderConfig
        camera_config,            # CameraConfig
        scene_config,             # SceneConfig (for env strength / hdri)
        log_prefix: str = "blender_daemon",
    ):
        from ..io import get_logger
        self._logger = get_logger(f"{__name__}.{log_prefix}")

        res = render_config.resolution
        cmd = [
            executable_path, "-b", "-noaudio", blend_file,
            "-P", __file__, "--",
            "--worker",
            "--samples", str(render_config.samples),
            "--resolution", f"{res[0]},{res[1]}",
            "--device", render_config.device,
            "--engine", render_config.engine,
            "--compute_device_type", render_config.compute_device_type,
            "--use_denoising", "1" if render_config.use_denoising else "0",
            "--film_transparent", "1" if render_config.film_transparent else "0",
            "--environment_strength", str(scene_config.environment_strength),
        ]
        if scene_config.hdri_path:
            cmd.extend(["--hdri", os.path.abspath(scene_config.hdri_path)])

        self._logger.info("Launching daemon: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )
        # Drain stderr in a background thread so it never fills the pipe.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name=f"{log_prefix}-stderr",
        )
        self._stderr_thread.start()

        self._wait_for("ready", timeout=self._READY_TIMEOUT_S)
        self._logger.info("Daemon ready.")

    def render(
        self,
        bpy_script: str,
        output_path: str,
        material_name: str,
        part_name: str,
        hidden_objects: Optional[List[str]] = None,
    ) -> float:
        """Render one image; returns wall-clock seconds per daemon report."""
        req = {
            "cmd": "render",
            "bpy_script": os.path.abspath(bpy_script),
            "output_path": os.path.abspath(output_path),
            "material_name": material_name,
            "part_name": part_name,
            "hidden_objects": list(hidden_objects or []),
        }
        self._send(req)
        reply = self._wait_for("ok", timeout=self._RENDER_TIMEOUT_S)
        return float(reply.get("time", 0.0))

    def close(self, timeout: float = 5.0) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._logger.warning("Daemon did not quit cleanly; killing.")
            self._proc.kill()
            self._proc.wait(timeout=5)

    def __enter__(self) -> "BlenderDaemon":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _send(self, msg: dict) -> None:
        line = json.dumps(msg) + "\n"
        assert self._proc.stdin is not None
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _wait_for(self, expected_status: str, timeout: float) -> dict:
        assert self._proc.stdout is not None
        deadline = time.time() + timeout
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Daemon died (exit={self._proc.returncode}) "
                    f"waiting for {expected_status!r}"
                )
            line = self._proc.stdout.readline()
            if not line:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Daemon wait_for({expected_status!r}) timed out"
                    )
                continue
            line = line.rstrip("\n")
            if not line.startswith(_SENTINEL):
                # Plain Blender stdout (render progress, etc.) — ignore.
                continue
            payload = line[len(_SENTINEL):]
            try:
                msg = json.loads(payload)
            except json.JSONDecodeError:
                self._logger.warning("Unparseable daemon msg: %s", payload)
                continue
            status = msg.get("status")
            if status == expected_status:
                return msg
            if status == "error":
                raise RuntimeError(f"Daemon error: {msg.get('message', '?')}")
            self._logger.warning("Unexpected daemon status %r: %s", status, msg)

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            line = line.rstrip()
            if line:
                self._logger.debug("[stderr] %s", line)


class ParallelViewRenderer:
    """Owns one ``BlenderDaemon`` per blend file and renders views concurrently
    across them.

    Life-cycle scoped to a single ``ParamTuner.tune()`` call: daemons spin up in
    ``__enter__`` and are killed in ``__exit__``.
    """

    def __init__(
        self,
        executable_path: str,
        render_config,            # RenderConfig
        camera_config,            # CameraConfig
        scene_config,             # SceneConfig (template; per-view overrides applied below)
        views: List[str],
    ) -> None:
        from .blender import RENDER_MODE_CONFIGS, SceneConfig

        self.executable_path = executable_path
        self.render_config = render_config
        self.camera_config = camera_config
        self.scene_config_base = scene_config
        self.views = list(views)

        # One daemon per blend file: all ball variants share the ball blend,
        # plane has its own.
        self._view_to_daemon_key: Dict[str, str] = {}
        self._blend_per_key: Dict[str, str] = {}
        self._hidden_per_view: Dict[str, List[str]] = {}
        for view in views:
            cfg = RENDER_MODE_CONFIGS.get(view)
            if cfg is None:
                raise ValueError(f"Unknown render_mode: {view}")
            blend = cfg.get("blend_file") or scene_config.blend_file
            if not blend:
                raise ValueError(f"No blend_file resolvable for view {view}")
            self._view_to_daemon_key[view] = blend
            self._blend_per_key[blend] = blend  # dedup
            self._hidden_per_view[view] = list(cfg.get("hidden_objects", []))

        self._daemons: Dict[str, BlenderDaemon] = {}
        self._daemon_locks: Dict[str, threading.Lock] = {}
        self._executor: Optional[ThreadPoolExecutor] = None

    def __enter__(self) -> "ParallelViewRenderer":
        for blend in self._blend_per_key.values():
            tag = Path(blend).stem
            self._daemons[blend] = BlenderDaemon(
                executable_path=self.executable_path,
                blend_file=blend,
                render_config=self.render_config,
                camera_config=self.camera_config,
                scene_config=self.scene_config_base,
                log_prefix=f"daemon[{tag}]",
            )
            self._daemon_locks[blend] = threading.Lock()
        # One worker per daemon is plenty: within a daemon, stdin is serial.
        self._executor = ThreadPoolExecutor(max_workers=len(self._daemons))
        return self

    def __exit__(self, *exc) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for d in self._daemons.values():
            d.close()
        self._daemons.clear()
        self._daemon_locks.clear()

    def render_views(
        self,
        view_specs: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        """Render the given view jobs concurrently across daemons.

        Each spec holds view / bpy_script / output_path / material_name / part_name.
        Returns (view, output_path) pairs in input order.
        """
        assert self._executor is not None, "use as context manager"

        def _run(spec: Dict[str, Any]) -> Tuple[str, str]:
            view = spec["view"]
            blend = self._view_to_daemon_key[view]
            daemon = self._daemons[blend]
            # Lock the daemon: multiple view_specs might map to the same blend
            # (bright_ball + dark_ball → ball daemon) and daemon stdin is
            # single-consumer.
            with self._daemon_locks[blend]:
                daemon.render(
                    bpy_script=spec["bpy_script"],
                    output_path=spec["output_path"],
                    material_name=spec["material_name"],
                    part_name=spec["part_name"],
                    hidden_objects=self._hidden_per_view[view],
                )
            return view, spec["output_path"]

        futures = [self._executor.submit(_run, s) for s in view_specs]
        return [f.result() for f in futures]


# Worker side: runs INSIDE Blender via ``-P daemon.py -- --worker``.
def _emit(msg: dict) -> None:
    """Worker → client: JSON line with sentinel prefix, flushed."""
    sys.stdout.write(_SENTINEL + json.dumps(msg) + "\n")
    sys.stdout.flush()


def _worker_main(worker_argv: List[str]) -> None:
    """Entry point when this script runs inside Blender."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--resolution", type=str, required=True)
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--engine", type=str, default="CYCLES")
    parser.add_argument("--compute_device_type", type=str, default="CUDA")
    parser.add_argument("--use_denoising", type=str, default="0")
    parser.add_argument("--film_transparent", type=str, default="1")
    parser.add_argument("--environment_strength", type=float, default=0.0)
    parser.add_argument("--hdri", type=str, default=None)
    args = parser.parse_args(worker_argv)

    # Import blender.py helpers — it handles sys.path bootstrap itself.
    import bpy  # type: ignore
    from blender import RenderConfig, SceneConfig, CameraConfig, SceneRenderer, LoopException

    w, h = (int(x) for x in args.resolution.split(","))
    render_cfg = RenderConfig(
        engine=args.engine,
        samples=args.samples,
        use_denoising=(args.use_denoising == "1"),
        resolution=(w, h),
        device=args.device,
        compute_device_type=args.compute_device_type,
        film_transparent=(args.film_transparent == "1"),
    )
    scene_cfg = SceneConfig(environment_strength=args.environment_strength)
    camera_cfg = CameraConfig()
    renderer = SceneRenderer(
        render_config=render_cfg,
        scene_config=scene_cfg,
        camera_config=camera_cfg,
    )

    # One-time setup (the expensive parts we refuse to pay per render).
    renderer._ensure_camera()
    renderer._setup_renderer(render_cfg)
    if args.hdri:
        renderer._setup_world_background(
            hdri_path=args.hdri,
            strength=args.environment_strength,
        )

    _emit({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"status": "error", "message": f"bad json: {e}"})
            continue

        op = cmd.get("cmd")
        if op == "quit":
            _emit({"status": "bye"})
            return
        if op != "render":
            _emit({"status": "error", "message": f"unknown cmd: {op}"})
            continue

        try:
            dt = _render_once(renderer, cmd)
            _emit({"status": "ok", "time": dt})
        except LoopException as e:
            _emit({"status": "error", "message": f"LoopException: {e}"})
        except Exception as e:
            _emit({"status": "error", "message": f"{type(e).__name__}: {e}"})


def _render_once(renderer, cmd: dict) -> float:
    """Daemon-side per-render dispatch.

    The hide → build material → assign → render sequence lives on
    ``SceneRenderer.apply_material_and_render`` so the one-shot and daemon paths
    share one implementation; the daemon-specific part is only the pre-reset that
    clears state leaked from the previous render.
    """
    renderer.reset_state_for_new_render(cmd["material_name"])
    return renderer.apply_material_and_render(
        bpy_script=cmd["bpy_script"],
        part_name=cmd["part_name"],
        output_path=cmd["output_path"],
        hidden_objects=list(cmd.get("hidden_objects", [])),
    )


def _extract_worker_argv() -> Optional[List[str]]:
    """Return the args after ``--`` when Blender ran this file as a daemon worker,
    else None.
    """
    if "--" not in sys.argv:
        return None
    idx = sys.argv.index("--")
    tail = sys.argv[idx + 1:]
    if "--worker" in tail:
        return tail
    return None


_worker_argv = _extract_worker_argv()
if _worker_argv is not None:
    # We're inside Blender, invoked as a daemon worker.
    _worker_main(_worker_argv)
