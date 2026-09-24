"""Dataset batch runner.

Drives the same :class:`ShaderPipeline` as ``cli.py`` over a JSONL dataset index,
writing each item to ``<tag>/<unique_name>/`` so per-item ``usage.json`` files stay
separate. Consumes the four ``pipeline.dataset_*`` config keys.

Usage:
    python -m shader_agent.batch --config config/x.yaml --tag outputs/eval
    python -m shader_agent.batch --config config/x.yaml --tag outputs/eval \\
        --limit 4 --skip-existing
"""
from __future__ import annotations

import argparse
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from . import find_project_root
from .config import Config
from .dataset import DatasetItem, JsonlDataset
from .io import get_logger

# ``ShaderPipeline`` pulls in metrics -> torch, so it is imported inside :func:`main`
# instead of at module scope. That keeps the selection and kwargs-mapping logic here
# importable on a machine without a GPU stack.
if TYPE_CHECKING:
    from .pipeline import ShaderPipeline

logger = get_logger(__name__)


def _item_dir(tag: str, unique_name: str) -> Path:
    return Path(tag) / unique_name


def _has_best(tag: str, unique_name: str) -> bool:
    """Whether a previous run carried this item all the way through.

    Gates on ``usage.json``, which ``usage_session`` writes in its ``finally`` block,
    rather than on ``*_best.py``, which appears as soon as any trial produces a graph.
    A run killed mid-trial leaves the latter behind, and --skip-existing would then
    treat a half-finished sample as done.
    """
    d = _item_dir(tag, unique_name)
    return d.is_dir() and (d / "usage.json").is_file()


def resolve_dataset_path(cfg: Config) -> Path:
    raw = cfg.pipeline.dataset_indices_path
    if not raw:
        raise SystemExit(
            "pipeline.dataset_indices_path is not set — batch mode needs a JSONL index."
        )
    p = Path(raw)
    return p if p.is_absolute() else find_project_root() / p


def build_run_kwargs(
    item: DatasetItem,
    base_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    return {
        "text_description": item.textual_prompt,
        "image_path": str(item.image_path(base_dir)),
        "additional_image_paths": [str(p) for p in item.additional_image_paths(base_dir)],
        "run_name": item.unique_name,
        "tag": tag,
    }


def select_items(
    dataset: JsonlDataset, tag: str, skip_existing: bool,
) -> List[DatasetItem]:
    items = list(dataset)
    if skip_existing:
        kept, skipped = [], []
        for item in items:
            (skipped if _has_best(tag, item.unique_name) else kept).append(item)
        if skipped:
            logger.info(
                "Skipping %d item(s) with an existing usage.json: %s",
                len(skipped), ", ".join(i.unique_name for i in skipped),
            )
        items = kept

    # Clear the dirs we are about to (re)run so a half-finished earlier attempt cannot
    # leave stale artifacts mixed in with the new ones.
    wiped = []
    for item in items:
        d = _item_dir(tag, item.unique_name)
        if d.exists():
            shutil.rmtree(d)
            wiped.append(item.unique_name)
    if wiped:
        logger.info("Wiped %d stale item dir(s) under %s", len(wiped), tag)
    return items


def _run_one(pipeline: "ShaderPipeline", kwargs: Dict[str, Any], name: str) -> bool:
    try:
        result = pipeline.run(**kwargs)
    except Exception as exc:
        logger.exception("Pipeline crashed for %s: %s", name, exc)
        return False
    if result:
        logger.info("Success: %s -> %s", name, result)
        return True
    logger.error("Pipeline produced no result for %s.", name)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ShaderAgent over a dataset index.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True,
                    help="Output dir; each item lands in <tag>/<unique_name>/.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap to N items after offset (overrides config).")
    ap.add_argument("--offset", type=int, default=None,
                    help="Skip the first N items (overrides config).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip items that already have usage.json under <tag>.")
    args = ap.parse_args()

    from .pipeline import ShaderPipeline  # deferred: pulls in torch

    cfg = Config(args.config)
    dataset = JsonlDataset.from_jsonl(
        resolve_dataset_path(cfg),
        limit=args.limit if args.limit is not None else cfg.pipeline.dataset_limit,
        offset=args.offset if args.offset is not None else cfg.pipeline.dataset_offset,
    )
    items = select_items(dataset, args.tag, args.skip_existing)
    if not items:
        logger.info("Nothing to run.")
        return 0

    workers = max(1, int(cfg.pipeline.dataset_workers))
    logger.info(
        "Batch: %d item(s), %d worker(s), tuning=%s",
        len(items), workers,
        cfg.pipeline.param_tuning_enabled,
    )

    succeeded = 0
    if workers <= 1:
        pipeline = ShaderPipeline(cfg)
        for item in items:
            succeeded += _run_one(
                pipeline,
                build_run_kwargs(item, dataset.base_dir, args.tag),
                item.unique_name,
            )
    else:
        # One ShaderPipeline per worker: pipelines hold per-run agent state and are not
        # thread-safe. The 8B VLMScorer is memoized in metrics.py, so the extra
        # instances do not each load their own copy.
        if cfg.pipeline.param_tuning_enabled:
            logger.warning(
                "dataset_workers=%d with param tuning on: each tuning round spawns "
                "Blender daemons (capped by SHADER_PIPELINE_TUNING_MAX_CONCURRENT) "
                "— watch GPU memory.", workers,
            )
        pool: "queue.Queue[ShaderPipeline]" = queue.Queue()
        for _ in range(workers):
            pool.put(ShaderPipeline(cfg))

        def task(item: DatasetItem) -> bool:
            p = pool.get()
            try:
                return _run_one(
                    p,
                    build_run_kwargs(item, dataset.base_dir, args.tag),
                    item.unique_name,
                )
            finally:
                pool.put(p)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(task, it): it for it in items}
            for fut in as_completed(futures):
                succeeded += bool(fut.result())

    logger.info("Batch done: %d/%d succeeded.", succeeded, len(items))
    return 0 if succeeded == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
