from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from ..config import ModelServiceConfig
from ..io import get_logger
from ..prompt import CLASSIFICATION_TEMPLATE
from ..llm.clients import VisionToTextModelClient

logger = get_logger(__name__)

class MaterialType(Enum):
    PROCEDURAL = auto()
    TEXTURE = auto()


@dataclass
class ClassificationResult:
    """Classifier output: material label plus target-region bbox.

    ``bbox`` is ``[x1, y1, x2, y2]`` (xyxy) with width and height both mapped
    to [0, 1000], the format the fine-tuned VLMScorer expects. The VLM answers
    in its native yxyx convention; ``_validate_bbox_1000`` swaps it once, so
    every caller sees xyxy. ``None`` means no usable bbox came back and callers
    should fall back to the full image.
    """
    material_type: MaterialType
    bbox: Optional[List[int]] = None


class MaterialClassifierService:
    """Classifies materials with a single vision-language call.

    Grounding the target material before labelling it acts as chain-of-thought
    and yields a bbox the VLMScorer reuses at param-tuning time.
    """

    def __init__(self, model_client: VisionToTextModelClient):
        self.model_client = model_client

    @classmethod
    def from_config(cls, model_config: ModelServiceConfig) -> "MaterialClassifierService":
        return cls(VisionToTextModelClient(model_config))

    # Gemini 3.x drops or mangles the bbox on a well-formed prompt often
    # enough to matter, and an identical retry usually recovers within 1-2
    # tries. Retrying here lets the rest of the pipeline assume a bbox exists.
    _MAX_ATTEMPTS = 3

    def classify(
        self, images: Sequence[Image.Image], text_description: str,
    ) -> ClassificationResult:
        prompt = self._build_prompt(text_description)

        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            response = self.model_client.generate_text(prompt, images=images)
            logger.info(
                "Material classification response (attempt %d/%d): %s",
                attempt, self._MAX_ATTEMPTS, response,
            )
            material_type, bbox = self._parse_response(response)
            if bbox is not None:
                return ClassificationResult(material_type=material_type, bbox=bbox)
            logger.warning(
                "Classifier: bbox missing on attempt %d/%d; retrying.",
                attempt, self._MAX_ATTEMPTS,
            )

        raise RuntimeError(
            f"Classifier failed to produce a valid bbox after {self._MAX_ATTEMPTS} attempts."
        )

    @staticmethod
    def _build_prompt(text_description: str) -> str:
        description = (text_description or "").strip()
        return (
            f"{CLASSIFICATION_TEMPLATE}\n\n"
            "Text description (may be incomplete; prioritize the image when conflicting):\n"
            f"{description}\n\n"
            "Return the JSON now."
        )

    @classmethod
    def _parse_response(
        cls, response: str,
    ) -> Tuple[MaterialType, Optional[List[int]]]:
        """Extract (material_type, bbox_1000) from VLM output."""
        if not response:
            return MaterialType.PROCEDURAL, None

        obj = _extract_json_object(response)
        if obj is not None:
            label_raw = str(obj.get("label", "")).strip().upper()
            material_type = cls._match_label(label_raw) or cls._fallback_label(response)
            bbox = _validate_bbox_1000(obj.get("bbox"))
            return material_type, bbox

        logger.warning("Classifier: could not parse JSON; falling back to label-only parse.")
        return cls._fallback_label(response), None

    @staticmethod
    def _match_label(text: str) -> Optional[MaterialType]:
        if text in {"PROCEDURAL", "TEXTURE"}:
            return MaterialType[text]
        return None

    @staticmethod
    def _fallback_label(response: str) -> MaterialType:
        """Loose recovery when the JSON is malformed: prefer the last explicit
        label mention, else the latest TEXTURE-vs-PROCEDURAL keyword, else
        PROCEDURAL as the broader, safer bucket.
        """
        normalized = response.strip().upper()
        matches = re.findall(r"\b(PROCEDURAL|TEXTURE)\b", normalized)
        if matches:
            return MaterialType[matches[-1]]

        candidates: List[Tuple[int, MaterialType]] = []
        if m := list(re.finditer(r"\b(TEXTURE|TILE|TILING)\b", normalized)):
            candidates.append((m[-1].end(), MaterialType.TEXTURE))
        if m := list(re.finditer(r"\bPROCEDURAL\b", normalized)):
            candidates.append((m[-1].end(), MaterialType.PROCEDURAL))
        if candidates:
            return max(candidates, key=lambda x: x[0])[1]
        return MaterialType.PROCEDURAL


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the first top-level JSON object from possibly noisy VLM output."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(stripped[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _validate_bbox_1000(raw) -> Optional[List[int]]:
    """Convert a Gemini-native ``[ymin, xmin, ymax, xmax]`` bbox in the
    [0, 1000] scale to the internal ``[x1, y1, x2, y2]``.

    The prompt asks for Gemini's own yxyx convention rather than fighting it,
    and this is the single place the swap happens. Floats are rounded and
    legacy [0, 1] values rescaled, to absorb VLM output drift.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if max(vals) <= 1.5:
        vals = [v * 1000.0 for v in vals]
    ymin, xmin, ymax, xmax = (int(round(v)) for v in vals)
    if not (0 <= xmin < xmax <= 1000 and 0 <= ymin < ymax <= 1000):
        logger.warning("Classifier: ignoring out-of-range bbox=%s", raw)
        return None
    return [xmin, ymin, xmax, ymax]
