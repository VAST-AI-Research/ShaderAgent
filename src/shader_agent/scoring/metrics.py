"""Image similarity metrics: VLM scorer used during parameter search."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..config import ImageSimilarityMetricConfig, ModelServiceConfig
from ..io import to_rgb_on_gray


def _center_crop_to_square(
    img: Image.Image,
    side_length: Optional[int] = None,
) -> Image.Image:
    img = to_rgb_on_gray(img)
    w, h = img.size
    max_side = min(w, h)
    if side_length is None:
        side = max_side
    else:
        if side_length <= 0:
            raise ValueError(f"side_length must be > 0, got {side_length}")
        if side_length > max_side:
            raise ValueError(
                f"side_length={side_length} exceeds image min side {max_side}"
            )
        side = side_length
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


class ImageSimilarityMetric(ABC):

    @abstractmethod
    def __call__(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute the similarity between two images."""
        pass


class VLMScorer:
    """Material scorer backed by Qwen3-VL-Reranker.

    The score is ``sigmoid(logit_yes - logit_no)`` read off the reranker's final
    hidden state, i.e. a continuous relevance value in [0, 1] where higher is a
    better match. Unlike ``ImageSimilarityMetric`` it takes a reference image plus
    a bbox and several renders of one candidate material at once::

        scorer = VLMScorer(adapter_path="/path/to/lora/checkpoint")
        score = scorer.score(
            ref_image="scene.png",
            bbox=[469, 59, 660, 766],
            candidate_images=["plane.png", "bright_ball.png", "dark_ball.png"],
        )
    """

    SYSTEM_PROMPT = (
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )
    # Injected by ms-swift's reranker template during training; absent from the raw JSONL.
    DEFAULT_INSTRUCT = (
        'Given a search query, retrieve relevant candidates that answer the query.'
    )
    # Mirrors the training data's user message, minus the leading <image> that the
    # ref image block supplies.
    DEFAULT_QUERY = (
        'Focus on the material inside <bbox> in the reference image. '
        'Retrieve candidate material renders that are visually similar '
        'to the marked region.'
    )

    # Image token budget, matching the Qwen3-VL-Reranker training config.
    _IMAGE_FACTOR = 32  # patch_size * merge_size
    _MIN_PIXELS = 4 * 32 * 32        # 4 visual tokens
    _MAX_PIXELS = 1280 * 32 * 32     # 1280 visual tokens

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-Reranker-8B",
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        merge_lora: bool = True,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left",
        )

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            if merge_lora:
                model = model.merge_and_unload()

        # 1-D scoring head: score = sigmoid((W_yes - W_no) · h_last).
        vocab = self.processor.tokenizer.get_vocab()
        yes_id, no_id = vocab["yes"], vocab["no"]
        self._score_weight = (
            model.lm_head.weight[yes_id] - model.lm_head.weight[no_id]
        ).detach().clone()

        self.backbone = model.model
        self.backbone.eval()
        self._score_weight = self._score_weight.to(self.backbone.device).to(dtype)
        self.device = self.backbone.device
        self.dtype = dtype

    @staticmethod
    def _format_bbox(bbox: List[int]) -> str:
        points = [f"({x},{y})" for x, y in zip(bbox[::2], bbox[1::2])]
        return f'<|box_start|>{",".join(points)}<|box_end|>'

    @staticmethod
    def _load_image(img: Union[str, Image.Image]) -> Image.Image:
        if isinstance(img, str):
            with Image.open(img) as pim:
                return to_rgb_on_gray(pim).copy()
        return to_rgb_on_gray(img)

    def _to_image_block(self, img: Union[str, Image.Image]) -> dict:
        return {
            "type": "image",
            "image": self._load_image(img),
            "min_pixels": self._MIN_PIXELS,
            "max_pixels": self._MAX_PIXELS,
        }

    def _build_messages(
        self,
        ref_image: Union[str, Image.Image],
        bbox: List[int],
        candidate_images: List[Union[str, Image.Image]],
        query_text: str,
        instruct: str,
    ) -> list:
        query_text = query_text.replace("<bbox>", self._format_bbox(bbox))

        user_content: list = [
            {"type": "text", "text": f"<Instruct>: {instruct}<Query>:"},
            self._to_image_block(ref_image),
            {"type": "text", "text": query_text + "\n<Document>:"},
        ]
        for img in candidate_images:
            user_content.append(self._to_image_block(img))

        return [
            {"role": "system",
             "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _collect_images(messages: list) -> List[Image.Image]:
        """Images in chat order; the processor pairs them positionally with the text."""
        images: List[Image.Image] = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "image" and "image" in block:
                    images.append(block["image"])
        return images

    @torch.no_grad()
    def score(
        self,
        ref_image: Union[str, Image.Image],
        bbox: List[int],
        candidate_images: List[Union[str, Image.Image]],
        query_text: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> float:
        """Score one candidate set against a reference image region.

        ``bbox`` is ``[x1, y1, x2, y2]`` in reference-image pixels. Returns a score
        in [0, 1], higher = better match.
        """
        if query_text is None:
            query_text = self.DEFAULT_QUERY
        if instruct is None:
            instruct = self.DEFAULT_INSTRUCT

        messages = self._build_messages(
            ref_image, bbox, candidate_images, query_text, instruct,
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images = self._collect_images(messages)

        inputs = self.processor(
            text=[text],
            images=images,
            min_pixels=self._MIN_PIXELS,
            max_pixels=self._MAX_PIXELS,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state[:, -1]        # (1, hidden)
        logit = (last_hidden * self._score_weight).sum(dim=-1)  # (1,)
        return torch.sigmoid(logit).float().item()

    def score_batch(
        self,
        ref_image: Union[str, Image.Image],
        bbox: List[int],
        candidates: List[List[Union[str, Image.Image]]],
        query_text: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> List[float]:
        """Score each candidate set in *candidates* independently, in order."""
        return [
            self.score(ref_image, bbox, cand, query_text, instruct)
            for cand in candidates
        ]


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# Process-level cache: the scorer puts an 8B Qwen3-VL-Reranker on the GPU, so
# concurrent pipelines must share one instance.
_SCORER_CACHE: Dict[tuple, VLMScorer] = {}
_SCORER_LOCK = threading.Lock()


def get_image_similarity_metric(
    config: ImageSimilarityMetricConfig,
    services: Optional[Dict[str, ModelServiceConfig]] = None,
) -> VLMScorer:
    dtype = _DTYPE_MAP.get(config.dtype)
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype '{config.dtype}'. "
            f"Choose from {list(_DTYPE_MAP)}."
        )
    key = (
        config.model_path,
        config.adapter_path,
        config.device,
        config.dtype,
        config.merge_lora,
    )
    with _SCORER_LOCK:
        scorer = _SCORER_CACHE.get(key)
        if scorer is None:
            scorer = VLMScorer(
                model_path=config.model_path,
                adapter_path=config.adapter_path,
                device=config.device,
                dtype=dtype,
                merge_lora=config.merge_lora,
            )
            _SCORER_CACHE[key] = scorer
    return scorer


def _selftest() -> None:
    """Smoke test: load VLMScorer and rank candidate dirs against one reference."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="VLMScorer self-test")
    parser.add_argument(
        "--model-path", default="Qwen/Qwen3-VL-Reranker-8B",
        help="Base model path or HF id.",
    )
    parser.add_argument(
        "--adapter-path", default=None,
        help="Optional LoRA adapter path.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ref-image", required=True,
        help="Path to reference scene image.",
    )
    parser.add_argument(
        "--bbox", required=True,
        help="Comma-separated x1,y1,x2,y2 marking the target region.",
    )
    parser.add_argument(
        "--candidate-dirs", nargs="+", required=True,
        help="One or more dirs, each containing plane.png / bright_ball.png / dark_ball.png.",
    )
    args = parser.parse_args()

    bbox = [int(x) for x in args.bbox.split(",")]
    assert len(bbox) == 4, f"bbox must be x1,y1,x2,y2, got {args.bbox}"

    import os

    print("=" * 70)
    print(f"Model:     {args.model_path}")
    if args.adapter_path:
        print(f"Adapter:   {args.adapter_path}")
    print(f"Ref image: {args.ref_image}")
    print(f"BBox:      {bbox}")
    print("-" * 70)

    t0 = time.time()
    scorer = VLMScorer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device,
    )
    print(f"Loaded model in {time.time() - t0:.1f}s")
    print("-" * 70)

    views = ("plane.png", "bright_ball.png", "dark_ball.png")
    results = []
    for cand_dir in args.candidate_dirs:
        cand_imgs = [os.path.join(cand_dir, v) for v in views]
        missing = [p for p in cand_imgs if not os.path.exists(p)]
        if missing:
            print(f"  SKIP {cand_dir}: missing {[os.path.basename(m) for m in missing]}")
            continue
        t0 = time.time()
        score = scorer.score(
            ref_image=args.ref_image,
            bbox=bbox,
            candidate_images=cand_imgs,
        )
        dt = time.time() - t0
        results.append((os.path.basename(cand_dir.rstrip("/")), score, dt))

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Rank  Score   Time    Candidate")
    print(f"  {'-' * 65}")
    for rank, (name, score, dt) in enumerate(results, 1):
        print(f"  #{rank:<3} {score:.4f}  {dt:5.2f}s  {name}")
    print("=" * 70)


if __name__ == "__main__":
    _selftest()
