import json
from typing import Optional, Sequence

from PIL import Image

from ..config import AgentConfig
from ..io import get_logger
from ..llm.runtime import create_client
from ..usage import get_active
from ..render.blender import LoopException
from ..prompt import (
    DERIVE_PBR_TEMPLATE,
    SHADER_GENERATION_TEMPLATE,
    SHADER_REFINE_FULL_TEMPLATE,
    COMMON_DSL_BLOCK,
)
from ..dsl.graph import ReplaceShaderNodesTool, ShaderGraph

logger = get_logger(__name__)

class SubMaterialAgent:
    def __init__(self, model_config: AgentConfig):
        self.model = create_client(model_config)
        # Diagnostics for the caller. ``last_raw_dsl`` is kept even when the
        # DSL failed validation, since it is discarded everywhere else.
        self.last_dsl_attempts = 0
        self.last_raw_dsl: Optional[str] = None

    DSL_RETRY_BUDGET: int = 2

    def _generate_graph(self, prompt: str, images: Sequence[Image.Image]) -> ShaderGraph:
        return self._generate_graph_lightweight(prompt, images)

    def _generate_graph_lightweight(
        self, prompt: str, images: Sequence[Image.Image],
    ) -> ShaderGraph:
        """Request a DSL, validating locally and re-prompting on failure.

        Leaves ``last_dsl_attempts`` / ``last_raw_dsl`` set for the caller to
        log, including on the raise.
        """
        base = (
            prompt
            + "\n\nOutput ONLY the complete DSL. Do not use Python, tools, markdown fences, or commentary."
        )
        attempt_prompt = base
        last_errors: list[str] = []
        self.last_dsl_attempts = 0

        for attempt in range(self.DSL_RETRY_BUDGET + 1):
            raw_dsl = self.model.complete(attempt_prompt, images)
            self.last_dsl_attempts = attempt + 1
            self.last_raw_dsl = raw_dsl
            if attempt:
                collector = get_active()
                if collector is not None:
                    collector.bump("dsl_validation_retries")
            graph = ShaderGraph.from_dsl(raw_dsl)
            if not graph.errors:
                graph._raw_dsl = raw_dsl
                return graph

            last_errors = list(graph.errors)
            if attempt == self.DSL_RETRY_BUDGET:
                break
            logger.warning(
                "DSL validation failed (attempt %d/%d): %s — re-prompting.",
                attempt + 1, self.DSL_RETRY_BUDGET + 1, "; ".join(last_errors[:3]),
            )
            attempt_prompt = (
                base
                + "\n\nYour previous DSL was rejected by the local validator.\n"
                + "Previous DSL:\n" + raw_dsl
                + "\n\nValidator errors:\n- " + "\n- ".join(last_errors)
                + "\n\nEmit a corrected, complete DSL. Every node must reach an "
                  "output node; remove or wire up any dangling node."
            )

        raise LoopException(
            f"DSL has structural errors after {self.DSL_RETRY_BUDGET + 1} attempts:\n- "
            + "\n- ".join(last_errors)
        )

    def generate_procedural_graph(
        self,
        description: str,
        reference_images: Sequence[Image.Image],
        error_feedback: Optional[str] = None,
        critic_feedback: Optional[str] = None,
        previous_graph: Optional[ShaderGraph] = None,
    ) -> ShaderGraph:
        # Only the in-step repair sub-loop in pipeline.py passes error_feedback.
        if error_feedback and previous_graph is not None:
            return self._repair_graph(
                description, reference_images, previous_graph,
                error_feedback=error_feedback,
            )

        if critic_feedback and previous_graph is not None:
            return self._refine_graph_full(
                description, reference_images, previous_graph,
                critic_feedback=critic_feedback,
            )

        prompt = self._build_procedural_prompt(description)
        logger.info(f"Shader graph generating...\n\nDescription: {description}")
        res = self._generate_graph(prompt, reference_images)
        return res

    def _refine_graph_full(
        self,
        description: str,
        reference_images: Sequence[Image.Image],
        previous_graph: ShaderGraph,
        critic_feedback: str,
        role_context: str = "",
    ) -> ShaderGraph:
        """Whole-DSL rewrite seeded by the previous graph's DSL."""
        prompt = SHADER_REFINE_FULL_TEMPLATE.format(
            description=description,
            previous_dsl=previous_graph.to_dsl(),
            feedback=critic_feedback,
            role_context=role_context,
        )
        logger.info(
            "Shader graph refining (full rewrite)...\nDescription: %s\nFeedback: %s",
            description[:80], critic_feedback[:200],
        )
        res = self._generate_graph(prompt, reference_images)
        return res

    def _repair_graph(
        self,
        description: str,
        reference_images: Sequence[Image.Image],
        previous_graph: ShaderGraph,
        error_feedback: str,
        role_context: str = "",
    ) -> ShaderGraph:
        """Request a minimal JSON patch and validate it transactionally."""
        schema = {
            "type": "object",
            "properties": {
                "node_ids_to_remove": {
                    "type": "array", "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
                "replacement_dsl": {"type": "string"},
            },
            "required": ["node_ids_to_remove", "replacement_dsl"],
            "additionalProperties": False,
        }
        prompt = (
            "Repair only the broken shader nodes with a minimal local patch. "
            "Keep working nodes intact. Return ONLY a JSON object matching the schema.\n"
            + role_context + "\n" + COMMON_DSL_BLOCK
            + "\nTask:\n" + description
            + "\nPrevious graph (DSL):\n" + previous_graph.to_dsl()
            + "\nBlender/validator error:\n" + error_feedback
            + "\nReplacement DSL contains only fixed/new nodes and links. Links may "
              "reference surviving nodes from the previous graph. To redefine an ID, "
              "include it in node_ids_to_remove. Removal IDs must exist and be unique. "
              "Valid old links are restored when both endpoints survive and the target "
              "has capacity; explicit replacement links take priority. "
              "Return JSON data, with DSL in replacement_dsl, rather than Python or a complete rewrite.\n"
            + "JSON schema:\n" + json.dumps(schema)
        )

        logger.info(
            "Shader graph repairing...\nDescription: %s\nError: %s",
            description[:80], error_feedback[:200],
        )

        attempt_prompt = prompt
        last_error = ""
        self.last_dsl_attempts = 0
        for attempt in range(self.DSL_RETRY_BUDGET + 1):
            raw_patch = self.model.complete(attempt_prompt, reference_images)
            self.last_dsl_attempts = attempt + 1
            self.last_raw_dsl = raw_patch
            if attempt:
                collector = get_active()
                if collector is not None:
                    collector.bump("dsl_validation_retries")
            try:
                payload = json.loads(raw_patch)
                if not isinstance(payload, dict) or set(payload) != set(schema["required"]):
                    raise ValueError("Patch must contain exactly node_ids_to_remove and replacement_dsl.")
                return ReplaceShaderNodesTool().forward(previous_graph, **payload)
            except (ValueError, TypeError) as error:
                last_error = str(error)
            attempt_prompt = (
                prompt + "\nPrevious JSON patch rejected by the local validator:\n"
                + raw_patch + "\nValidator errors:\n" + last_error
                + "\nReturn a corrected JSON patch against the unchanged previous graph."
            )
        raise LoopException(
            f"Graph repair failed after {self.DSL_RETRY_BUDGET + 1} attempts:\n{last_error}"
        )

    def _build_procedural_prompt(self, description: str) -> str:
        return SHADER_GENERATION_TEMPLATE.format(description=description)

    _PBR_ROLE_CONTEXT = (
        "You are working on a PBR texture-derivation graph that derives "
        "Roughness, Metallic, Normal, etc. from a Base Color texture. "
        "The ShaderNodeTexImage for Base Color is correct; focus on "
        "the PBR derivation nodes."
    )

    def derive_pbr_graph(
        self,
        description: str,
        reference_images: Sequence[Image.Image],
        texture_map: Image.Image,
        error_feedback: Optional[str] = None,
        critic_feedback: Optional[str] = None,
        previous_graph: Optional[ShaderGraph] = None,
    ) -> ShaderGraph:
        if error_feedback and previous_graph is not None:
            return self._repair_graph(
                description=description,
                reference_images=reference_images,
                previous_graph=previous_graph,
                error_feedback=error_feedback,
                role_context=self._PBR_ROLE_CONTEXT,
            )

        if critic_feedback and previous_graph is not None:
            return self._refine_graph_full(
                description=description,
                reference_images=reference_images,
                previous_graph=previous_graph,
                critic_feedback=critic_feedback,
                role_context=self._PBR_ROLE_CONTEXT,
            )

        prompt = self._build_texture_prompt(description)
        logger.info("PBR attributes aligning...")
        res = self._generate_graph(prompt, list(reference_images) + [texture_map])
        return res

    def _build_texture_prompt(
        self,
        description: str,
    ) -> str:
        return DERIVE_PBR_TEMPLATE.format(
            description=description,
        )
