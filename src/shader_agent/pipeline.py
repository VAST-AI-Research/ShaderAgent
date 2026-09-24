import json
import os
import random
import shutil
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .agents.designer import SubMaterialAgent
from .render.blender import LoopException, RepairExhausted, SceneRenderer
from .config import Config
from .io import get_logger, load_image, load_images, save_to_file, set_log_dir
from .render.daemon import ParallelViewRenderer
from .agents.classifier import MaterialClassifierService, MaterialType
from .agents.critic import CriticService
from .llm.clients import (
    ImageGeneratorService,
    PromptEnricherService,
    TextureGeneratorService,
)
from .dsl.graph import ShaderGraph, ShaderGraphParser
from .usage import get_active, usage_session

logger = get_logger(__name__)

class ShaderPipeline:
    def __init__(
        self,
        config: Config,
        subagent_factory: Optional[Callable] = None,
        classifier: Optional[MaterialClassifierService] = None,
        critic_factory: Optional[Callable[[], CriticService]] = None,
        texture_generator: Optional[TextureGeneratorService] = None,
        prompt_enricher: Optional[PromptEnricherService] = None,
        image_generator: Optional[ImageGeneratorService] = None,
        renderer: Optional[SceneRenderer] = None,
        debug_mode: bool = False,
    ):
        self.config = config
        self.debug_mode = debug_mode

        self.subagent_factory = subagent_factory or (lambda: SubMaterialAgent(self.config.agents["designer"]))
        
        self.classifier = classifier or MaterialClassifierService.from_config(self.config.services["classifier"])
        self._texture_generator = texture_generator
        self._prompt_enricher = prompt_enricher
        self._image_generator = image_generator
        
        self.image_metric = None
        self.renderer = renderer or SceneRenderer(
            executable_path=self.config.blender_scene.executable_path,
            render_config=self.config.blender_scene.render,
            scene_config=self.config.blender_scene.scene,
            camera_config=self.config.blender_scene.camera,
        )

        # Low-quality renderer for the param tuner's inner loop and for
        # ``_render_views_standalone`` when tuning is disabled.
        from .render.blender import RenderConfig
        pt_cfg = self.config.param_tuning
        fast_render_cfg = RenderConfig(
            engine=self.config.blender_scene.render.engine,
            samples=pt_cfg.render_samples,
            use_denoising=True,
            film_transparent=self.config.blender_scene.render.film_transparent,
            resolution=(pt_cfg.render_resolution, pt_cfg.render_resolution),
            device=self.config.blender_scene.render.device,
            compute_device_type=self.config.blender_scene.render.compute_device_type,
        )
        self.fast_renderer = SceneRenderer(
            executable_path=self.config.blender_scene.executable_path,
            render_config=fast_render_cfg,
            scene_config=self.config.blender_scene.scene,
            camera_config=self.config.blender_scene.camera,
        )
        if self.config.pipeline.param_tuning_enabled:
            from .scoring.metrics import get_image_similarity_metric
            from .agents.tuner import ParamTuner

            self.image_metric = get_image_similarity_metric(
                self.config.image_metrics, services=self.config.services
            )
            self.param_tuner = ParamTuner(
                inspector_config=self.config.services["inspector"],
                renderer=self.fast_renderer,
                metric_fn=self.image_metric,
                config=self.config.param_tuning,
            )
        else:
            self.param_tuner = None

        self.critic_factory = critic_factory or (
            lambda: CriticService.from_config(
                self.config.services["critic"],
                self.config.critic,
            )
        )

    @property
    def texture_generator(self) -> TextureGeneratorService:
        if self._texture_generator is None:
            self._texture_generator = TextureGeneratorService.from_config(
                self.config.services["texture_generator"]
            )
        return self._texture_generator

    @property
    def prompt_enricher(self) -> PromptEnricherService:
        if self._prompt_enricher is None:
            self._prompt_enricher = PromptEnricherService.from_config(
                self.config.services["prompt_enricher"]
            )
        return self._prompt_enricher

    @property
    def image_generator(self) -> ImageGeneratorService:
        if self._image_generator is None:
            self._image_generator = ImageGeneratorService.from_config(
                self.config.services["image_generator"]
            )
        return self._image_generator

    @classmethod
    def from_config_path(cls, config_path: str, debug_mode: bool = False) -> "ShaderPipeline":
        return cls(Config(config_path), debug_mode=debug_mode)
        
    def run(
        self,
        text_description: str,
        tag: str,
        image_path: Optional[str] = None,
        additional_image_paths: Optional[List[str]] = None,
        run_name: Optional[str] = None,
    ) -> Optional[str]:
        """Entry point. Wraps :meth:`_run` in a :func:`usage_session` so
        ``usage.json`` (calls/tokens/wall-clock) is written whether or not the
        run succeeds.
        """
        save_dir = os.path.join(tag, run_name) if run_name else tag
        os.makedirs(save_dir, exist_ok=True)
        with usage_session(save_dir):
            return self._run(
                text_description,
                tag,
                save_dir,
                image_path=image_path,
                additional_image_paths=additional_image_paths,
                run_name=run_name,
            )

    def _run(
        self,
        text_description: str,
        tag: str,
        save_dir: str,
        image_path: Optional[str] = None,
        additional_image_paths: Optional[List[str]] = None,
        run_name: Optional[str] = None,
    ) -> Optional[str]:
        """Returns the final render path, or None if every trial failed.

        ``tag`` is the experiment dir (``<tag>/<run_name>/`` in batch mode,
        ``<tag>/`` otherwise); re-running with the same tag repopulates it, so
        failed items can be retried alongside prior results.
        """
        set_log_dir(save_dir)
        logger.info("Starting pipeline for: %s", text_description)
        logger.info("Saving run outputs to: %s", save_dir)

        image_paths = []
        if not image_path:
            prompt = self.prompt_enricher.enrich(text_description)
            image_path = self.image_generator.generate(prompt, save_dir)
        image_paths.append(image_path)

        if additional_image_paths:
            image_paths.extend(additional_image_paths)

        ref_images = load_images(image_paths)

        classification = self.classifier.classify(ref_images, text_description)
        material_type = classification.material_type
        target_bbox = classification.bbox  # [x1,y1,x2,y2] in [0,1000] or None

        if self.debug_mode:
            logger.info("DEBUG: Entering ipdb after classification.")
            import ipdb; ipdb.set_trace()

        logger.info("Material classified as: %s (bbox=%s)", material_type, target_bbox)

        if target_bbox is not None:
            try:
                self._save_bbox_overlay(
                    ref_image=ref_images[0],
                    bbox_1000=target_bbox,
                    output_path=os.path.join(save_dir, "grounding_bbox.png"),
                )
            except Exception as e:
                logger.warning("Could not save grounding bbox overlay: %s", e)

        max_trials = self.config.pipeline.max_trials
        max_steps = self.config.pipeline.max_steps

        for trial in range(max_trials):
            logger.info("=== Trial %s/%s ===", trial + 1, max_trials)
            try:
                if material_type == MaterialType.TEXTURE:
                    result_render = self._run_texture_flow(
                        text_description, ref_images, image_paths, save_dir, trial, max_steps,
                        target_bbox=target_bbox,
                    )
                else:
                    result_render = self._run_procedural_flow(
                        text_description, ref_images, image_paths, save_dir, trial, max_steps,
                        target_bbox=target_bbox,
                    )

                if result_render:
                    logger.info("Trial %s succeeded with render: %s", trial + 1, result_render)
                    collector = get_active()
                    if collector is not None:
                        collector.note("outcome", "success")
                        collector.note("trials_used", trial + 1)
                    return result_render
            except RepairExhausted as e:
                logger.warning("Trial %s aborted (repair exhausted): %s", trial + 1, e)
                continue
            except Exception as e:
                logger.exception("Trial %s failed with error: %s", trial + 1, e)
                continue

        logger.warning("All trials failed or reached max steps.")
        collector = get_active()
        if collector is not None:
            collector.note("outcome", "failed")
            collector.note("trials_used", max_trials)
        return None

    def _run_texture_flow(
        self,
        text_description: str,
        ref_images: List,
        image_paths: List[str],
        save_dir: str,
        trial_idx: int,
        max_steps: int,
        target_bbox: Optional[List[int]] = None,
    ) -> Optional[str]:
        logger.info("Material is TEXTURE. Generating base color texture...")
        
        texture_map_path = self.texture_generator.generate(text_description, save_dir, reference_images=ref_images)

        if self.debug_mode:
            logger.info(f"DEBUG: Entering ipdb after texture generation. Texture map is saved in {texture_map_path}")
            import ipdb; ipdb.set_trace()

        if not texture_map_path:
             logger.error("Failed to generate texture map.")
             return None
        
        texture_map = load_image(texture_map_path)
        logger.info("Generated base color texture at: %s", texture_map_path)
        
        gen_fn = partial(
            self.subagent_factory().derive_pbr_graph, 
            description=text_description, 
            reference_images=ref_images, 
            texture_map=texture_map
        )
        
        # The base-color texture goes in as one more reference image so the
        # critic can judge the PBR derivations against the fixed albedo.
        _, render_path = self._run_refinement_loop(
            description=text_description,
            reference_image_paths=image_paths + [texture_map_path],
            step_name=f"trial{trial_idx}_texture_pbr",
            generation_func=gen_fn,
            save_dir=save_dir,
            max_steps=max_steps,
            texture_map_path=texture_map_path,
            target_bbox=target_bbox,
        )
        return render_path
    
    def _run_procedural_flow(
        self,
        text_description: str,
        ref_images: List,
        image_paths: List[str],
        save_dir: str,
        trial_idx: int,
        max_steps: int,
        target_bbox: Optional[List[int]] = None,
    ) -> Optional[str]:
        gen_fn = partial(
            self.subagent_factory().generate_procedural_graph,
            description=text_description,
            reference_images=ref_images,
        )

        _, render_path = self._run_refinement_loop(
            description=text_description,
            reference_image_paths=image_paths,
            step_name=f"trial{trial_idx}_procedural",
            generation_func=gen_fn,
            save_dir=save_dir,
            max_steps=max_steps,
            target_bbox=target_bbox,
        )
        return render_path

    # Extra calls to the agent's repair branch per step when ``to_bpy`` /
    # render fails; exhausting it raises ``RepairExhausted`` and kills the trial.
    REPAIR_BUDGET: int = 2

    # 3-view set the param tuner / MatScorer renders. The critic only sees
    # bright_ball (see ``_critic_views``).
    _VIEWS: Tuple[str, ...] = ("bright_ball", "dark_ball", "plane")

    @staticmethod
    def _critic_views(paths: List[str]) -> List[str]:
        """One bright_ball PNG for the critic; fall back to the first path."""
        for path in paths:
            if os.path.basename(path) in (
                "bright_ball.png",
                "baseline_bright_ball.png",
                "best_bright_ball.png",
            ):
                return [path]
        return paths[:1]

    def _run_refinement_loop(
        self,
        description: str,
        reference_image_paths: List[str],
        step_name: str,
        generation_func: Callable,
        save_dir: str,
        max_steps: int,
        texture_map_path: Optional[str] = None,
        allow_debug: bool = True,
        target_bbox: Optional[List[int]] = None,
    ) -> Tuple[Optional[ShaderGraph], str]:
        """Each step generates / refines a graph, repairs render failures in
        place, tunes parameters, then asks the pairwise critic two questions:
        pretune vs tuned picks the step winner, and winner vs running best
        decides promotion. Feedback for the next step is the latest
        comparison's ``winner_remaining_issues``.

        Raises ``RepairExhausted`` if any step exhausts its repair budget.
        Returns this trial's ``(best_graph, best_render)``.
        """
        best_graph: Optional[ShaderGraph] = None
        best_render: str = ""
        best_step_id: Optional[str] = None
        best_critic_feedback: str = ""
        best_bpy_path: str = ""
        best_view_paths: List[str] = []

        scene_cfg = self.config.blender_scene.scene

        material_name = f"mat_{step_name.replace(' ', '_')}"

        def _main_render(bpy_path: str, output_path: str) -> str:
            return self.renderer.render(
                bpy_script=bpy_path,
                output_path=output_path,
                blend_file=scene_cfg.blend_file,
                part_name=scene_cfg.part_name,
                model_path=None,
                render_mode=None,
                focus_target="all",
            )

        for step in range(max_steps):
            step_id = f"{step_name}_step_{step}"
            logger.info("--- Refinement Step: %s ---", step_id)

            try:
                graph = generation_func(
                    critic_feedback=best_critic_feedback or None,
                    previous_graph=best_graph,
                )
            except Exception as e:
                logger.exception(
                    "Generation failed at %s; aborting trial: %s", step_id, e,
                )
                # Persist the rejected DSL: on failure the raw text never
                # reaches a graph object, so without this the run leaves no
                # artifact explaining why it failed.
                agent = getattr(getattr(generation_func, "func", None), "__self__", None)
                rejected = getattr(agent, "last_raw_dsl", None)
                if rejected:
                    save_to_file(rejected, os.path.join(save_dir, f"{step_id}_rejected.dsl"))
                break

            try:
                graph, untuned_bpy_path, pretune_render = self._render_with_repair(
                    graph=graph,
                    step_id=step_id,
                    save_dir=save_dir,
                    material_name=material_name,
                    texture_map_path=texture_map_path,
                    main_render=_main_render,
                    generation_func=generation_func,
                )
            except RepairExhausted:
                # Propagate: the trial loop in ``run`` skips to the next trial.
                raise

            # A "candidate" is a dict with: label, graph, views, bpy_path, main_render.
            candidates: List[Dict] = []
            if self.param_tuner is not None:
                if getattr(self.config.param_tuning, "random_inspector", False):
                    tunables = graph.collect_tunable_params()
                    n = min(len(tunables), self.config.param_tuning.max_params)
                    selected_params = random.Random(hash(step_id) & 0xFFFFFFFF).sample(tunables, n)
                    logger.info(
                        "ABLATION: random_inspector active — sampled %d/%d params: %s",
                        n, len(tunables),
                        [(p.node_id, p.param_name) for p in selected_params],
                    )
                else:
                    selected_params, _ = self.param_tuner.inspect_params(
                        graph=graph,
                        description=description,
                        render_path=pretune_render,
                        reference_image_path=reference_image_paths[0] if reference_image_paths else None,
                    )
                tuned, _, _ = self.param_tuner.tune(
                    graph=graph,
                    selected_params=selected_params,
                    reference_image_paths=reference_image_paths,
                    save_dir=save_dir,
                    step_id=step_id,
                    texture_map_path=texture_map_path,
                    subgraphs_for_bpy=None,
                    target_bbox=target_bbox,
                )

                tuned_bpy = ShaderGraphParser.to_bpy(
                    tuned,
                    material_name=material_name,
                    texture_map_path=texture_map_path,
                )
                tuned_bpy_path = os.path.join(save_dir, f"{step_id}.py")
                save_to_file(tuned_bpy, tuned_bpy_path)

                tuning_dir = os.path.join(save_dir, f"{step_id}_tuning")
                pretune_views = [
                    os.path.join(tuning_dir, f"baseline_{v}.png") for v in self._VIEWS
                ]
                tuned_views = [
                    os.path.join(tuning_dir, f"best_{v}.png") for v in self._VIEWS
                ]
                # A missing view must not starve the critic: fall back to
                # whatever rendered.
                pretune_views = [p for p in pretune_views if os.path.exists(p)] or [pretune_render]
                tuned_views = [p for p in tuned_views if os.path.exists(p)] or [pretune_render]

                candidates = [
                    {
                        "label": "pretune",
                        "graph": graph,
                        "views": pretune_views,
                        "bpy_path": untuned_bpy_path,
                        "main_render": pretune_views[0],
                    },
                    {
                        "label": "tuned",
                        "graph": tuned,
                        "views": tuned_views,
                        "bpy_path": tuned_bpy_path,
                        "main_render": tuned_views[0],
                    },
                ]
            else:
                view_paths = self._render_views_standalone(
                    graph=graph,
                    save_dir=save_dir,
                    step_id=step_id,
                    material_name=material_name,
                    texture_map_path=texture_map_path,
                )
                # Mirror the canonical step render path.
                final_render_path = os.path.join(save_dir, f"{step_id}.png")
                try:
                    shutil.copy2(pretune_render, final_render_path)
                except Exception:
                    final_render_path = pretune_render
                candidates = [
                    {
                        "label": "main",
                        "graph": graph,
                        "views": view_paths or [pretune_render],
                        "bpy_path": untuned_bpy_path,
                        "main_render": final_render_path,
                    },
                ]

            r1 = None
            if len(candidates) == 2:
                pretune_c, tuned_c = candidates
                r1 = self.critic_factory().evaluate(
                    description=description,
                    candidate_a_views=self._critic_views(pretune_c["views"]),
                    candidate_b_views=self._critic_views(tuned_c["views"]),
                    reference_image_paths=reference_image_paths,
                    a_label="pretune",
                    b_label="tuned",
                )
                step_winner = pretune_c if r1.winner == "A" else tuned_c
                logger.info(
                    "Step %s pretune-vs-tuned: winner=%s (scores A=%.3f B=%.3f) %s",
                    step_id, step_winner["label"], r1.score_a, r1.score_b, r1.reason,
                )
                latest_feedback = r1.winner_remaining_issues
            else:
                pretune_c = tuned_c = None
                step_winner = candidates[0]
                latest_feedback = ""

            if best_graph is None:
                promote = True
                final_pair = None
            else:
                final_pair = self.critic_factory().evaluate(
                    description=description,
                    candidate_a_views=self._critic_views(best_view_paths),
                    candidate_b_views=self._critic_views(step_winner["views"]),
                    reference_image_paths=reference_image_paths,
                    a_label="incumbent",
                    b_label="challenger",
                )
                promote = final_pair.winner == "B"
                logger.info(
                    "Step %s challenger-vs-incumbent: %s (scores incumbent=%.3f challenger=%.3f) %s",
                    step_id,
                    "promote" if promote else "keep incumbent",
                    final_pair.score_a, final_pair.score_b, final_pair.reason,
                )

            if self.debug_mode and allow_debug:
                logger.info("DEBUG: Entering ipdb after critic evaluation step %s.", step_id)
                import ipdb; ipdb.set_trace()

            if promote:
                best_graph = step_winner["graph"]
                best_render = step_winner["main_render"]
                best_step_id = step_id
                best_bpy_path = step_winner["bpy_path"]
                best_view_paths = list(step_winner["views"])
                logger.info(
                    "New trial best at %s [%s] → %s",
                    step_id, step_winner["label"], best_render,
                )

            # Feedback describes what the current best still lacks, so the
            # winner-vs-best verdict supersedes the within-step one.
            if final_pair is not None:
                best_critic_feedback = final_pair.winner_remaining_issues
            elif latest_feedback:
                best_critic_feedback = latest_feedback

            # One JSON line per step: enough to reconstruct the "w/o tuning"
            # trajectory and pipeline statistics post hoc.
            good_enough = bool(
                final_pair.good_enough if final_pair is not None
                else (r1.good_enough if r1 is not None else False)
            )
            logger.info("ABLATION_STEP %s", json.dumps({
                "case_id": step_name,
                "step": step,
                "step_id": step_id,
                "pretune_score": r1.score_a if r1 is not None else None,
                "tuned_score":   r1.score_b if r1 is not None else None,
                "step_winner_label": step_winner["label"],
                "promote": promote,
                "good_enough": good_enough,
                "incumbent_score":  final_pair.score_a if final_pair is not None else None,
                "challenger_score": final_pair.score_b if final_pair is not None else None,
                "critic_rubric": final_pair.rubric if final_pair is not None else (r1.rubric if r1 is not None else []),
                "pretune_bpy":  pretune_c["bpy_path"]  if pretune_c is not None else None,
                "pretune_view": pretune_c["views"][0]  if pretune_c is not None and pretune_c["views"] else None,
                "tuned_bpy":    tuned_c["bpy_path"]    if tuned_c is not None else None,
                "tuned_view":   tuned_c["views"][0]    if tuned_c is not None and tuned_c["views"] else None,
                "best_step_id": best_step_id,
            }))

            if good_enough:
                logger.info(
                    "Step %s critic marked winner good_enough; stopping. Best at %s.",
                    step_id, best_step_id,
                )
                break

        if best_graph is not None and best_step_id is not None:
            self._save_refinement_best(
                best_graph=best_graph,
                best_render=best_render,
                best_bpy_path=best_bpy_path,
                best_view_paths=best_view_paths,
                best_step_id=best_step_id,
                save_dir=save_dir,
                step_name=step_name,
            )
            self._render_preset_views(save_dir, step_name)
            logger.info(
                "Refinement loop %s done. Trial best at %s → %s",
                step_name, best_step_id, best_render,
            )
        else:
            logger.warning("Refinement loop %s produced no usable graph.", step_name)
        return best_graph, best_render

    def _render_with_repair(
        self,
        graph: ShaderGraph,
        step_id: str,
        save_dir: str,
        material_name: str,
        texture_map_path: Optional[str],
        main_render: Callable[[str, str], str],
        generation_func: Callable,
    ) -> Tuple[ShaderGraph, str, str]:
        """Render ``graph``, feeding any error back to the agent's repair
        branch for up to ``REPAIR_BUDGET`` extra attempts.

        Returns ``(repaired_graph, bpy_path, render_path)`` of the first attempt
        that renders; raises ``RepairExhausted`` once the budget is spent.
        """
        bpy_path = os.path.join(save_dir, f"{step_id}_pretune.py")
        render_path = os.path.join(save_dir, f"{step_id}_pretune.png")
        last_error: Optional[BaseException] = None
        parse_errors = 0

        for attempt in range(self.REPAIR_BUDGET + 1):
            try:
                raw_dsl = getattr(graph, "_raw_dsl", None) or graph.to_dsl()
                logger.info("Step %s DSL (attempt %d):\n%s", step_id, attempt, raw_dsl)
                save_to_file(raw_dsl, os.path.join(save_dir, f"{step_id}.dsl"))

                bpy_text = ShaderGraphParser.to_bpy(
                    graph,
                    material_name=material_name,
                    texture_map_path=texture_map_path,
                )
                save_to_file(bpy_text, bpy_path)
                produced = main_render(bpy_path, render_path)
                logger.info("ABLATION_REPAIR %s", json.dumps({
                    "step_id": step_id,
                    "attempts": attempt,           # extra repair attempts beyond the first
                    "parse_errors": parse_errors,
                    "succeeded": True,
                }))
                return graph, bpy_path, produced
            except (LoopException, ValueError, RuntimeError) as e:
                last_error = e
                # Structural DSL / to_bpy errors surface as ValueError; render
                # failures as LoopException/RuntimeError.
                if isinstance(e, ValueError):
                    parse_errors += 1
                if attempt >= self.REPAIR_BUDGET:
                    logger.error(
                        "Repair budget (%d) exhausted at %s: %s",
                        self.REPAIR_BUDGET, step_id, e,
                    )
                    logger.info("ABLATION_REPAIR %s", json.dumps({
                        "step_id": step_id,
                        "attempts": attempt,
                        "parse_errors": parse_errors,
                        "succeeded": False,
                    }))
                    raise RepairExhausted(
                        f"{step_id}: repair budget {self.REPAIR_BUDGET} exhausted; "
                        f"last error: {e}"
                    ) from e
                logger.warning(
                    "Render failed at %s (attempt %d/%d): %s — invoking repair.",
                    step_id, attempt + 1, self.REPAIR_BUDGET + 1, e,
                )
                try:
                    graph = generation_func(
                        error_feedback=str(e),
                        previous_graph=graph,
                    )
                except Exception as repair_e:
                    logger.exception(
                        "Repair attempt failed at %s: %s", step_id, repair_e,
                    )
                    raise RepairExhausted(
                        f"{step_id}: repair agent crashed: {repair_e}"
                    ) from repair_e

        # Unreachable, but keep the type checker happy.
        raise RepairExhausted(f"{step_id}: unexpected repair-loop exit ({last_error})")

    def _render_views_standalone(
        self,
        graph: ShaderGraph,
        save_dir: str,
        step_id: str,
        material_name: str,
        texture_map_path: Optional[str],
    ) -> List[str]:
        """Render the 3 preset views for the tuner-disabled path, returning
        PNG paths in ``self._VIEWS`` order.
        """
        scorer_dir = os.path.join(save_dir, f"{step_id}_scoring")
        os.makedirs(scorer_dir, exist_ok=True)
        bpy_text = ShaderGraphParser.to_bpy(
            graph,
            material_name=material_name,
            texture_map_path=texture_map_path,
        )
        bpy_path = os.path.join(scorer_dir, "standalone.py")
        save_to_file(bpy_text, bpy_path)

        part_name = self.fast_renderer.scene_config.part_name
        with ParallelViewRenderer(
            executable_path=self.fast_renderer.executable_path,
            render_config=self.fast_renderer.render_config,
            camera_config=self.fast_renderer.camera_config,
            scene_config=self.fast_renderer.scene_config,
            views=list(self._VIEWS),
        ) as pvr:
            view_specs = [
                {
                    "view": view,
                    "bpy_script": bpy_path,
                    "output_path": os.path.join(scorer_dir, f"{view}.png"),
                    "material_name": material_name,
                    "part_name": part_name,
                }
                for view in self._VIEWS
            ]
            rendered = pvr.render_views(view_specs)
        return [path for _, path in rendered]

    def _render_preset_views(self, save_dir: str, step_name: str) -> None:
        """Re-render ``<step_name>_best.py`` in the three preset scenes so this
        run's output is directly comparable to baselines publishing the same
        views. Existing ``<prefix>.png`` / ``<prefix>_plane.png`` from the
        critic-driven pass are left untouched.
        """
        prefix = os.path.join(save_dir, f"{step_name}_best")
        bpy_path = f"{prefix}.py"
        if not os.path.exists(bpy_path):
            logger.warning("No best.py at %s; skipping preset renders.", bpy_path)
            return
        for mode in ("bright_ball", "dark_ball", "plane"):
            out_path = f"{prefix}_{mode}.png"
            try:
                self.renderer.render(
                    bpy_script=bpy_path,
                    output_path=out_path,
                    blend_file=None,  # auto-resolved via RENDER_MODE_CONFIGS
                    part_name="SolidModel",
                    model_path=None,
                    render_mode=mode,
                    focus_target="all",
                )
            except Exception as e:
                logger.warning("Preset render (%s) failed: %s", mode, e)

    @staticmethod
    def _save_refinement_best(
        best_graph,
        best_render: str,
        best_bpy_path: str,
        best_view_paths: List[str],
        best_step_id: str,
        save_dir: str,
        step_name: str,
    ) -> None:
        """Copy the winning step's artifacts to ``<step_name>_best.*`` at the
        run-dir root, so callers need not know whether pretune or tuned won.
        The DSL is re-emitted from the in-memory graph; the rest is copied.
        """
        prefix = os.path.join(save_dir, f"{step_name}_best")

        try:
            dsl_text = getattr(best_graph, "_raw_dsl", None) or best_graph.to_dsl()
            save_to_file(dsl_text, f"{prefix}.dsl")
        except Exception as e:
            logger.warning("Could not write %s.dsl: %s", prefix, e)

        if best_render and os.path.exists(best_render):
            try:
                shutil.copy2(best_render, f"{prefix}.png")
            except Exception as e:
                logger.warning("Could not copy render to %s.png: %s", prefix, e)

        if best_bpy_path and os.path.exists(best_bpy_path):
            try:
                shutil.copy2(best_bpy_path, f"{prefix}.py")
            except Exception as e:
                logger.warning("Could not copy bpy script to %s.py: %s", prefix, e)

        plane_src = next(
            (p for p in best_view_paths if p.endswith("_plane.png")), None,
        )
        if plane_src and os.path.exists(plane_src):
            try:
                shutil.copy2(plane_src, f"{prefix}_plane.png")
            except Exception as e:
                logger.warning("Could not copy plane render to %s_plane.png: %s", prefix, e)

        logger.info(
            "Saved refinement best for %s: prefix=%s, source=%s",
            step_name, prefix, best_step_id,
        )

    @staticmethod
    def _save_bbox_overlay(
        ref_image: Image.Image,
        bbox_1000: List[int],
        output_path: str,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        """Draw the classifier's [0, 1000]-normalized bbox on a copy of
        ``ref_image``, so its grounding can be audited by eye.
        """
        img = ref_image.convert("RGB").copy()
        w, h = img.size
        x1, y1, x2, y2 = bbox_1000
        px = [
            max(0, int(round(x1 * w / 1000))),
            max(0, int(round(y1 * h / 1000))),
            min(w, int(round(x2 * w / 1000))),
            min(h, int(round(y2 * h / 1000))),
        ]
        draw = ImageDraw.Draw(img)
        thickness = max(2, min(w, h) // 200)
        draw.rectangle(px, outline=color, width=thickness)
        img.save(output_path)
        logger.info("Saved grounding bbox overlay → %s (px=%s)", output_path, px)
