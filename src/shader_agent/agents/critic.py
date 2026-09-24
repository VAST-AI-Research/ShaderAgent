import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import CriticConfig, ModelServiceConfig
from ..io import get_logger, load_images
from ..llm.runtime import create_client
from ..prompt import CRITIC_PAIRWISE_PROMPT

logger = get_logger(__name__)


@dataclass
class PairwiseResult:
    winner: str                   # "A" or "B"
    score_a: float                # critic's absolute score for A in [0, 1]
    score_b: float                # critic's absolute score for B in [0, 1]
    reason: str
    good_enough: bool
    winner_remaining_issues: str
    rubric: List[Dict[str, Any]] = field(default_factory=list)


_VIEW_LABELS = ("bright_ball", "dark_ball", "plane")


def _format_view_indices(start: int, count: int) -> str:
    descs = []
    for i in range(count):
        label = _VIEW_LABELS[i] if i < len(_VIEW_LABELS) else f"view{i+1}"
        descs.append(f"Image {start + i} ({label})")
    return ", ".join(descs)


class CriticService:
    """Pairwise visual critic: given two shader-render candidates and the
    reference image(s), picks the better one and returns structural feedback
    for refining the winner. The pipeline sends one bright_ball view each.
    """

    def __init__(self, model, critic_config: CriticConfig | None = None):
        self.model = model
        self.critic_config = critic_config or CriticConfig()

    @classmethod
    def from_config(
        cls,
        model_config: ModelServiceConfig,
        critic_config: CriticConfig | None = None,
    ) -> "CriticService":
        return cls(create_client(model_config), critic_config)

    def evaluate(
        self,
        description: str,
        candidate_a_views: List[str],
        candidate_b_views: List[str],
        reference_image_paths: List[str],
        a_label: str,
        b_label: str,
    ) -> PairwiseResult:
        if not candidate_a_views or not candidate_b_views:
            raise ValueError("evaluate requires non-empty view lists for both candidates.")
        if not reference_image_paths:
            raise ValueError("evaluate requires at least one reference image.")

        num_refs = len(reference_image_paths)
        num_a = len(candidate_a_views)
        num_b = len(candidate_b_views)

        ref_indices = ", ".join(f"Image {i+1}" for i in range(num_refs))
        a_view_indices = _format_view_indices(num_refs + 1, num_a)
        b_view_indices = _format_view_indices(num_refs + 1 + num_a, num_b)

        prompt = CRITIC_PAIRWISE_PROMPT.format(
            description=description,
            ref_indices=ref_indices,
            a_label=a_label,
            b_label=b_label,
            a_view_indices=a_view_indices,
            b_view_indices=b_view_indices,
            good_enough_min=self.critic_config.good_enough_min,
        )

        image_paths = list(reference_image_paths) + list(candidate_a_views) + list(candidate_b_views)
        images = load_images(image_paths)

        logger.info("Calling Pairwise Critic (%s vs %s)...", a_label, b_label)
        # A blank or unparseable verdict would abort the trial and throw away
        # its whole tuning search, so retry before giving up.
        instruction = "\n\nReturn ONLY the requested JSON object; no final_answer wrapper or markdown."
        data = None
        last_exc: Exception | None = None
        for attempt in range(1, self.PARSE_ATTEMPTS + 1):
            raw = self.model.complete(prompt + instruction, images)
            logger.info("Critic raw result (attempt %d/%d): %s",
                        attempt, self.PARSE_ATTEMPTS, raw)
            if isinstance(raw, dict):
                data = raw
                break
            try:
                data = self._parse_json(str(raw))
                break
            except ValueError as exc:
                last_exc = exc
                logger.warning("Critic response unusable on attempt %d/%d: %s",
                               attempt, self.PARSE_ATTEMPTS, exc)
        if data is None:
            raise last_exc if last_exc else ValueError("Critic returned no usable verdict")
        normalized_rubric = self._validate_rubric(data.get("rubric"))
        score_a = sum(item["weight"] * item["score_a"] for item in normalized_rubric)
        score_b = sum(item["weight"] * item["score_b"] for item in normalized_rubric)
        winner = "B" if score_b > score_a else "A"
        return PairwiseResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            reason=str(data.get("reason", "")),
            # Every rubric dimension must clear the bar, not just the weighted
            # total, and it is recomputed rather than trusting the model's flag.
            good_enough=all(
                (item["score_b"] if winner == "B" else item["score_a"])
                >= self.critic_config.good_enough_min
                for item in normalized_rubric
            ),
            winner_remaining_issues=str(data.get("winner_remaining_issues", "")),
            rubric=normalized_rubric,
        )

    PARSE_ATTEMPTS = 3

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Critic returned an empty response")
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n", "", text)
            text = re.sub(r"\n```$", "", text)
        text = text.strip()
        # Models sometimes ignore the "no final_answer wrapper" instruction;
        # unwrap it rather than losing the verdict.
        wrapped = re.match(r"^final_answer\s*\(\s*(.*?)\s*\)\s*$", text, re.DOTALL)
        if wrapped:
            text = wrapped.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Critic returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Critic JSON must be an object")
        return data

    @staticmethod
    def _validate_rubric(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list) or not 3 <= len(raw) <= 6:
            raise ValueError("Critic rubric must contain 3 to 6 dimensions")
        result = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Critic rubric entries must be objects")
            try:
                weight = float(item["weight"])
                score_a = float(item["score_a"])
                score_b = float(item["score_b"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Critic rubric fields are invalid") from exc
            if not all(math.isfinite(value) for value in (weight, score_a, score_b)):
                raise ValueError("Critic rubric cannot contain NaN or infinity")
            if weight <= 0 or not 0 <= score_a <= 1 or not 0 <= score_b <= 1:
                raise ValueError("Critic rubric weights and scores are out of range")
            result.append({"dimension": str(item.get("dimension", "")), "weight": weight, "score_a": score_a, "score_b": score_b, "evidence": str(item.get("evidence", ""))})
        total = sum(item["weight"] for item in result)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("Critic rubric weights must sum to 1")
        for item in result:
            item["weight"] /= total
        return result
