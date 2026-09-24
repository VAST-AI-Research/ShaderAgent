"""Monte Carlo parameter tuning for shader graphs.

An LLM inspector picks the few parameters worth searching; those are then
perturbed log-uniformly with momentum, rendered at low quality, and scored
against the reference. Uphill moves are always taken and downhill ones
sometimes (``accept_prob``), so the walker can escape local optima.

Inspired by VLMaterial (https://github.com/mit-gfx/VLMaterial).
"""

import colorsys
import copy
import csv
import json
import math
import os
import re
import shutil
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import ModelServiceConfig, ParamTuningConfig
from ..io import get_logger, load_image, save_to_file
from ..prompt import INSPECTOR_TEMPLATE
from ..dsl.graph import ParamRef, ShaderGraph, ShaderGraphParser
from ..render.daemon import ParallelViewRenderer
from ..llm.clients import VisionToTextModelClient

if TYPE_CHECKING:
    from ..scoring.metrics import VLMScorer

logger = get_logger(__name__)

# Each round spins up a pool of warm Blender daemons, so gating at context
# entry bounds daemon count and VRAM footprint when dataset_workers > 1.
_TUNING_SEMAPHORE = threading.Semaphore(
    int(os.environ.get("SHADER_PIPELINE_TUNING_MAX_CONCURRENT", "3"))
)


class ParamTuner:
    """Monte Carlo parameter tuner. Instantiated once; :meth:`tune` is called
    once per refinement step.
    """

    FLOAT_CLIP_MIN: float = -1e4
    FLOAT_CLIP_MAX: float = 1e4

    def __init__(
        self,
        inspector_config: ModelServiceConfig,
        renderer,           # SceneRenderer (low-quality / fast)
        metric_fn: "VLMScorer",
        config: ParamTuningConfig,
    ):
        self.inspector_client = VisionToTextModelClient(inspector_config)
        self.renderer = renderer
        self.metric_fn = metric_fn
        self.config = config

    def tune(
        self,
        graph: ShaderGraph,
        selected_params: List[ParamRef],
        reference_image_paths: List[str],
        save_dir: str,
        step_id: str,
        texture_map_path: Optional[str] = None,
        subgraphs_for_bpy: Optional[Dict[str, ShaderGraph]] = None,
        target_bbox: Optional[List[int]] = None,
    ) -> Tuple[ShaderGraph, Optional[str], Optional[float]]:
        """Run MC parameter search over the caller-provided ``selected_params``.

        Returns ``(best_graph, best_render_path, best_score)``, the render being
        the best iteration's ``bright_ball`` view (closest to the main
        pipeline's default ``render_mode=ball``). With no selected params the
        search is skipped but the baseline is still rendered and scored, so the
        caller always gets a comparable ``best_score``.

        ``target_bbox`` is ``[x1, y1, x2, y2]`` in [0, 1000] scale marking the
        target material in the first reference image; ``None`` scores the full
        image.
        """

        selected = list(selected_params)
        if selected:
            logger.info("Starting MC param tuning (%d iterations, %d params)",
                         self.config.max_iter, len(selected))
        else:
            logger.info(
                "No selected params; tuning will score baseline only."
            )

        # A fixed seed made every step walk an identical trajectory whenever
        # the graph and selected params matched. XOR-ing in the step_id hash
        # decouples the steps while keeping the run reproducible.
        rng = np.random.default_rng(
            self.config.seed ^ (hash(step_id) & 0xFFFFFFFF)
        )
        tuning_dir = os.path.join(save_dir, f"{step_id}_tuning")
        os.makedirs(tuning_dir, exist_ok=True)
        ref_image = load_image(reference_image_paths[0])
        material_name = f"mat_tuning_{step_id.replace(' ', '_')}"
        scorer_bbox = list(target_bbox) if target_bbox else [0, 0, 1000, 1000]
        if not target_bbox:
            logger.info("No target_bbox provided; scoring against the full image.")

        # Blender daemons live only inside this ``with`` block.
        part_name = self.renderer.scene_config.part_name
        with _TUNING_SEMAPHORE, ParallelViewRenderer(
            executable_path=self.renderer.executable_path,
            render_config=self.renderer.render_config,
            camera_config=self.renderer.camera_config,
            scene_config=self.renderer.scene_config,
            views=list(self._SCORER_VIEWS),
        ) as pvr:
            # VLMScorer returns similarity in [0, 1], higher = better.
            best_graph = graph
            best_tag = "baseline"
            best_score = self._render_and_score(
                best_graph, ref_image, scorer_bbox, tuning_dir, best_tag,
                material_name, part_name, texture_map_path, subgraphs_for_bpy,
                pvr=pvr,
            )
            logger.info("Baseline score: %.6f (higher is better)", best_score)

            # One row per iter (plus baseline), recording the walker state so
            # score curves can be plotted without replaying logs.
            trajectory_path = os.path.join(tuning_dir, "trajectory.csv")
            self._init_trajectory(trajectory_path)
            self._append_trajectory(
                trajectory_path,
                iter_tag="baseline", decision="baseline", new_best=True,
                cand_score=best_score,
                cur_tag=best_tag, cur_score=best_score,
                best_tag=best_tag, best_score=best_score,
            )

            if not selected:
                baseline_artifacts = self._save_best_artifacts(
                    best_graph, best_tag, tuning_dir
                )
                baseline_render_path = baseline_artifacts.get("bright_ball")
                logger.info(
                    "MC param tuning skipped (no selected params). "
                    "Baseline score=%.6f (tag=%s) — artifacts: %s",
                    best_score, best_tag,
                    ", ".join(f"{k}={v}" for k, v in baseline_artifacts.items()),
                )
                logger.info("ABLATION_TUNER %s", json.dumps({
                    "step_id": step_id,
                    "baseline_score": float(best_score),
                    "final_score": float(best_score),
                    "improvements": 0,
                    "total_iters": 0,
                    "n_selected_params": 0,
                }))
                return best_graph, baseline_render_path, best_score

            cur_graph = best_graph
            cur_score = best_score
            cur_tag = best_tag
            prob_rng = np.random.default_rng(rng.integers(0x7FFFFFFF))
            baseline_score = best_score
            improvements_count = 0

            # Per-param momentum, in log-space units (see ``_perturb_scalar``).
            velocity: Dict[Tuple[str, str], float] = {
                (ref.node_id, ref.param_name): 0.0 for ref in selected
            }

            for i in range(self.config.max_iter):
                iter_tag = f"iter_{i:03d}"
                candidate = self.perturb_params(
                    cur_graph, selected, rng, velocity,
                    self.config.momentum, self.config.log_range
                )

                try:
                    cand_score = self._render_and_score(
                        candidate, ref_image, scorer_bbox, tuning_dir, iter_tag,
                        material_name, part_name, texture_map_path, subgraphs_for_bpy,
                        pvr=pvr,
                    )
                except Exception as e:
                    logger.warning("Tuning iter %d render failed: %s", i, e)
                    continue

                # Downhill moves are accepted with fixed probability rather
                # than a cooling schedule: no temperature to calibrate, and the
                # walker keeps a constant chance of leaving a local optimum.
                better = cand_score > cur_score
                rand_accepted = not better and prob_rng.random() < self.config.accept_prob
                accepted = better or rand_accepted

                if accepted:
                    self._log_param_changes(iter_tag, cur_graph, candidate, selected,
                                            cur_score, cand_score)
                    self._update_velocity(
                        velocity, cur_graph, candidate, selected,
                        self.config.momentum
                    )
                    cur_graph = candidate
                    cur_score = cand_score
                    cur_tag = iter_tag
                else:
                    # A rejected direction loses confidence, so let momentum
                    # decay towards zero instead of pushing the same way again.
                    for key in velocity:
                        velocity[key] *= self.config.momentum

                new_best = cur_score > best_score
                if new_best:
                    best_score = cur_score
                    best_graph = cur_graph
                    best_tag = cur_tag
                    improvements_count += 1

                tag = (
                    "better" if better
                    else "rand_accept" if rand_accepted
                    else "reject"
                )
                # Tags are the on-disk filenames (iter_NNN, 0-indexed), so a
                # log line maps straight to its PNG/bpy files.
                logger.info(
                    "Tuning %s/%d [%s%s]: cand=%.6f | cur=%.6f (%s) | best=%.6f (%s)",
                    iter_tag, self.config.max_iter, tag,
                    " *NEW_BEST*" if new_best else "",
                    cand_score,
                    cur_score, cur_tag,
                    best_score, best_tag,
                )
                self._append_trajectory(
                    trajectory_path,
                    iter_tag=iter_tag, decision=tag, new_best=new_best,
                    cand_score=cand_score,
                    cur_tag=cur_tag, cur_score=cur_score,
                    best_tag=best_tag, best_score=best_score,
                )

            best_artifacts = self._save_best_artifacts(
                best_graph, best_tag, tuning_dir
            )
            best_render_path = best_artifacts.get("bright_ball")
            if not best_render_path:
                logger.warning(
                    "Expected best render missing for tag=%s in %s",
                    best_tag, tuning_dir,
                )
            logger.info(
                "MC param tuning done. Best score=%.6f (tag=%s) — artifacts: %s, trajectory=%s",
                best_score, best_tag,
                ", ".join(f"{k}={v}" for k, v in best_artifacts.items()),
                trajectory_path,
            )
            logger.info("ABLATION_TUNER %s", json.dumps({
                "step_id": step_id,
                "baseline_score": float(baseline_score),
                "final_score": float(best_score),
                "improvements": improvements_count,
                "total_iters": self.config.max_iter,
                "n_selected_params": len(selected),
            }))
            return best_graph, best_render_path, best_score

    @staticmethod
    def _perturb_scalar(
        val: float,
        rng: np.random.Generator,
        velocity: float = 0.0,
        momentum: float = 0.9,
        log_range: float = 2.3,
    ) -> float:
        """Perturb a non-colour scalar log-uniformly, biased by momentum.

        Shader scalars span orders of magnitude, so the step is multiplicative:
        ``val * exp(noise)`` with ``noise`` drawn from ``±log_range``
        (2.3 ≈ ln(10), i.e. 0.1x–10x). ``momentum`` weights the accumulated
        ``velocity`` against fresh noise, so a direction that has been paying
        off keeps being explored (0.9 = strong memory).
        """
        if abs(val) < 1e-6:
            # A multiplicative step can never leave zero, so perturb additively.
            noise = rng.uniform(-1.0, 1.0)
            new_val = val + momentum * velocity + (1.0 - momentum) * noise
        else:
            log_noise = rng.uniform(-log_range, log_range)
            biased_noise = momentum * velocity + (1.0 - momentum) * log_noise
            new_val = val * math.exp(biased_noise)

        return float(np.clip(new_val,
                             ParamTuner.FLOAT_CLIP_MIN, ParamTuner.FLOAT_CLIP_MAX))

    @staticmethod
    def _is_color_tuple(val: tuple) -> bool:
        """Heuristic: a 3- or 4-tuple confined to [0, 1] is treated as RGB(A),
        so it gets HSV steps instead of log-uniform ones.
        """
        return len(val) in (3, 4) and all(
            isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in val
        )

    def _perturb_color_hsv(self, color: tuple, rng: np.random.Generator) -> tuple:
        """Perturb an RGB(A) colour in HSV space, where the axes match how a
        material reads: hue held by default (``hue_perturb == 0``) since the
        designer's hue choice is usually deliberate, S and V searched within
        ±``color_sv_perturb`` and clamped to [0, 1]. Alpha is never touched.
        """
        r, g, b = float(color[0]), float(color[1]), float(color[2])
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        hue_range = self.config.hue_perturb
        if hue_range > 0:
            h = (h + rng.uniform(-hue_range, hue_range)) % 1.0

        pr = self.config.color_sv_perturb
        s = float(rng.uniform(max(s - pr, 0.0), min(s + pr, 1.0)))
        v = float(rng.uniform(max(v - pr, 0.0), min(v + pr, 1.0)))

        r_new, g_new, b_new = colorsys.hsv_to_rgb(h, s, v)

        if len(color) == 4:
            return (r_new, g_new, b_new, float(color[3]))
        return (r_new, g_new, b_new)

    def _perturb_value(
        self,
        val,
        rng: np.random.Generator,
        velocity: float = 0.0,
        momentum: float = 0.9,
        log_range: float = 2.3,
    ):
        """Dispatch on value type: bools flip with ``bool_flip_prob``, colour
        tuples move in HSV, everything numeric moves log-uniformly.
        """
        if isinstance(val, bool):
            return (not val) if rng.random() < self.config.bool_flip_prob else val

        if isinstance(val, (int, float)):
            new_val = ParamTuner._perturb_scalar(
                float(val), rng, velocity, momentum, log_range
            )
            return int(round(new_val)) if isinstance(val, int) else new_val

        if isinstance(val, tuple):
            if ParamTuner._is_color_tuple(val):
                return self._perturb_color_hsv(val, rng)
            return tuple(
                ParamTuner._perturb_scalar(
                    float(v), rng, velocity, momentum, log_range
                )
                if isinstance(v, (int, float)) else v
                for v in val
            )
        return val

    def perturb_params(
        self,
        graph: ShaderGraph,
        selected: List[ParamRef],
        rng: np.random.Generator,
        velocity: Optional[Dict[Tuple[str, str], float]] = None,
        momentum: float = 0.9,
        log_range: float = 2.3,
    ) -> ShaderGraph:
        """Return a deep copy of *graph* with *selected* params perturbed."""
        if velocity is None:
            velocity = {}
        new_graph = copy.deepcopy(graph)

        for ref in selected:
            node = new_graph.nodes.get(ref.node_id)
            if node is None:
                continue

            key = (ref.node_id, ref.param_name)
            vel = velocity.get(key, 0.0)

            if ref.kind == "input":
                old = node.inputs.get(ref.param_name)
                if old is not None and old.default is not None:
                    node.set_input_value(
                        ref.param_name,
                        self._perturb_value(
                            old.default, rng, vel, momentum, log_range
                        ),
                    )
            elif ref.kind == "prop":
                if ref.param_name in node.props:
                    node.set_prop_value(
                        ref.param_name,
                        self._perturb_value(
                            node.props[ref.param_name], rng, vel, momentum, log_range
                        ),
                    )

        return new_graph

    @staticmethod
    def _update_velocity(
        velocity: Dict[Tuple[str, str], float],
        old_graph: ShaderGraph,
        new_graph: ShaderGraph,
        selected: List[ParamRef],
        momentum: float,
    ) -> None:
        """Fold an accepted move into ``velocity`` in place.

        The delta is measured in log space (``log(new / old)``) to match the
        multiplicative steps ``_perturb_scalar`` takes, falling back to a linear
        difference near zero where the ratio is meaningless.
        """
        for ref in selected:
            old_node = old_graph.nodes.get(ref.node_id)
            new_node = new_graph.nodes.get(ref.node_id)
            if old_node is None or new_node is None:
                continue

            key = (ref.node_id, ref.param_name)

            if ref.kind == "input":
                old_inp = old_node.inputs.get(ref.param_name)
                new_inp = new_node.inputs.get(ref.param_name)
                old_val = old_inp.default if old_inp else None
                new_val = new_inp.default if new_inp else None
            else:  # prop
                old_val = old_node.props.get(ref.param_name)
                new_val = new_node.props.get(ref.param_name)

            if old_val is None or new_val is None:
                continue

            delta = 0.0
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                old_f = float(old_val)
                new_f = float(new_val)
                if abs(old_f) > 1e-6 and abs(new_f) > 1e-6:
                    delta = math.log(new_f / old_f)
                else:
                    delta = new_f - old_f
            elif isinstance(old_val, tuple) and isinstance(new_val, tuple):
                # One velocity per param, so collapse a tuple to the mean delta.
                deltas = []
                for ov, nv in zip(old_val, new_val):
                    if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                        of = float(ov)
                        nf = float(nv)
                        if abs(of) > 1e-6 and abs(nf) > 1e-6:
                            deltas.append(math.log(nf / of))
                        else:
                            deltas.append(nf - of)
                if deltas:
                    delta = sum(deltas) / len(deltas)
            # Bools have no meaningful direction, so they keep delta = 0.

            velocity[key] = momentum * velocity.get(key, 0.0) + (1.0 - momentum) * delta

    def inspect_params(
        self,
        graph: ShaderGraph,
        description: str,
        render_path: str,
        reference_image_path: Optional[str] = None,
    ) -> Tuple[List[ParamRef], str]:
        """Pick at most ``max_params`` tunable params from the visual gap
        between reference and current render. Without ``reference_image_path``
        the inspector sees only the render and reasons one-sidedly.

        Returns ``(selected_params, reasoning)``.
        """
        tunable = graph.collect_tunable_params()
        if not tunable:
            logger.info("No tunable params in graph; inspection skipped.")
            return [], ""

        param_lines = "\n".join(
            f"  {p.node_id}.{p.param_name} = {p.value}" for p in tunable
        )
        prompt = INSPECTOR_TEMPLATE.format(
            description=description or "(no description provided)",
            graph_dsl=graph.to_dsl(),
            param_list=param_lines,
            max_params=self.config.max_params,
        )

        images: List[Image.Image] = []
        if reference_image_path:
            images.append(Image.open(reference_image_path).convert("RGB"))
        else:
            logger.warning(
                "Inspector called without a reference image; visual gap "
                "comparison will be one-sided."
            )
        images.append(Image.open(render_path).convert("RGB"))

        response = self.inspector_client.generate_text(prompt, images=images)
        logger.info("Inspector response: %s", response)

        reasoning, raw_params = self._parse_inspector_response(response)

        lookup = {(p.node_id, p.param_name): p for p in tunable}
        result: List[ParamRef] = []
        for item in raw_params[: self.config.max_params]:
            key = (item.get("node", ""), item.get("param", ""))
            if key in lookup:
                result.append(lookup[key])
            else:
                logger.warning("Inspector selected unknown param: %s", item)

        if reasoning:
            logger.info("Inspector reasoning: %s", reasoning)
        logger.info(
            "Inspector selected %d params: %s",
            len(result),
            [(p.node_id, p.param_name) for p in result],
        )
        return result, reasoning

    @staticmethod
    def _parse_inspector_response(text: str) -> Tuple[str, list]:
        """Extract ``(reasoning, params)`` from possibly noisy LLM output.

        Accepts ``{"reasoning": ..., "params": [...]}`` and the legacy bare
        array ``[{...}, ...]``, which yields empty reasoning.
        """
        text = text.strip()
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    reasoning = str(obj.get("reasoning", "") or "").strip()
                    params = obj.get("params", []) or []
                    if isinstance(params, list):
                        return reasoning, params
                    return reasoning, []
            elif ch == "[":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, list):
                    return "", obj
        return "", []

    # VLMScorer was trained with three views per material: plane + bright_ball +
    # dark_ball. Rendering all three keeps tuning-time scoring in-distribution.
    _SCORER_VIEWS: tuple = ("plane", "bright_ball", "dark_ball")

    def _render_and_score(
        self,
        graph: ShaderGraph,
        ref_image: Image.Image,
        bbox: List[int],
        tuning_dir: str,
        tag: str,
        material_name: str,
        part_name: str,
        texture_map_path: Optional[str],
        subgraphs_for_bpy: Optional[Dict[str, ShaderGraph]],
        pvr: ParallelViewRenderer,
    ) -> float:
        """Render the 3 scorer views via the daemon pool and return the
        VLMScorer score (higher = better). ``bbox`` is in the [0, 1000] integer
        scale the scorer was trained on.
        """
        bpy_script = ShaderGraphParser.to_bpy(
            graph,
            material_name=material_name,
            texture_map_path=texture_map_path,
            subgraphs=subgraphs_for_bpy,
        )
        bpy_path = os.path.join(tuning_dir, f"{tag}.py")
        save_to_file(bpy_script, bpy_path)

        view_specs = [
            {
                "view": view,
                "bpy_script": bpy_path,
                "output_path": os.path.join(tuning_dir, f"{tag}_{view}.png"),
                "material_name": material_name,
                "part_name": part_name,
            }
            for view in self._SCORER_VIEWS
        ]
        rendered = pvr.render_views(view_specs)
        # The scorer is order-sensitive, so index by view rather than trusting
        # completion order.
        out_by_view = dict(rendered)
        candidate_images = [
            Image.open(out_by_view[view]) for view in self._SCORER_VIEWS
        ]

        return self.metric_fn.score(
            ref_image=ref_image,
            bbox=bbox,
            candidate_images=candidate_images,
        )

    _TRAJECTORY_COLS: Tuple[str, ...] = (
        "iter_tag", "decision", "new_best",
        "cand_score", "cur_tag", "cur_score", "best_tag", "best_score",
    )

    @classmethod
    def _init_trajectory(cls, path: str) -> None:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(cls._TRAJECTORY_COLS)

    @classmethod
    def _append_trajectory(cls, path: str, **row) -> None:
        """Append one row; floats formatted to 6 decimals, bools lowercased."""
        def _fmt(v):
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, float):
                return f"{v:.6f}"
            return v
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([_fmt(row[c]) for c in cls._TRAJECTORY_COLS])

    def _save_best_artifacts(
        self,
        best_graph: ShaderGraph,
        best_tag: str,
        tuning_dir: str,
    ) -> Dict[str, str]:
        """Copy the winning iteration's renders and bpy script to stable
        ``best_*`` names in ``tuning_dir`` and dump its DSL, so callers need not
        know which ``iter_NNN`` won. Returns view/bpy/dsl paths by key.
        """
        paths: Dict[str, str] = {}

        for view in self._SCORER_VIEWS:
            src = os.path.join(tuning_dir, f"{best_tag}_{view}.png")
            dst = os.path.join(tuning_dir, f"best_{view}.png")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                paths[view] = dst

        src_py = os.path.join(tuning_dir, f"{best_tag}.py")
        dst_py = os.path.join(tuning_dir, "best.py")
        if os.path.exists(src_py):
            shutil.copy2(src_py, dst_py)
            paths["bpy"] = dst_py

        dsl_text = getattr(best_graph, "_raw_dsl", None) or best_graph.to_dsl()
        dsl_path = os.path.join(tuning_dir, "best.dsl")
        save_to_file(dsl_text, dsl_path)
        paths["dsl"] = dsl_path

        return paths

    @staticmethod
    def _log_param_changes(
        iter_tag: str,
        old_graph: ShaderGraph,
        new_graph: ShaderGraph,
        selected: List[ParamRef],
        old_score: float,
        new_score: float,
    ) -> None:
        lines = [
            f"Tuning {iter_tag}: accepted "
            f"(score {old_score:.6f} → {new_score:.6f})"
        ]
        for ref in selected:
            old_node = old_graph.nodes.get(ref.node_id)
            new_node = new_graph.nodes.get(ref.node_id)
            if old_node is None or new_node is None:
                continue

            if ref.kind == "input":
                old_val = old_node.inputs.get(ref.param_name, None)
                new_val = new_node.inputs.get(ref.param_name, None)
                old_v = old_val.default if old_val else "?"
                new_v = new_val.default if new_val else "?"
            else:
                old_v = old_node.props.get(ref.param_name, "?")
                new_v = new_node.props.get(ref.param_name, "?")

            if old_v != new_v:
                lines.append(f"  {ref.node_id}.{ref.param_name}: {old_v} → {new_v}")

        logger.info("\n".join(lines))
