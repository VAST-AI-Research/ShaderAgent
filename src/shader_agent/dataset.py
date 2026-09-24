from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class DatasetItem:
    img_name: str
    textual_prompt: str
    sub_dir: str
    unique_name: str
    additional_image_names: list[str] = field(default_factory=list)

    def image_path(self, base_dir: Path) -> Path:
        return base_dir / self.sub_dir / self.img_name

    def additional_image_paths(self, base_dir: Path) -> list[Path]:
        return [base_dir / self.sub_dir / name for name in self.additional_image_names]


class JsonlDataset:
    def __init__(
        self,
        items: Sequence[DatasetItem],
        base_dir: Path,
        limit: int | None = None,
        offset: int | None = None,
    ):
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        self._items = list(items)
        self._base_dir = base_dir

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self._items)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._items[idx]

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str | Path,
        base_dir: str | Path | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> "JsonlDataset":
        jsonl_path = Path(jsonl_path)
        base_dir = Path(base_dir) if base_dir is not None else jsonl_path.parent

        items: list[DatasetItem] = []
        with jsonl_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                img_name = data["img_name"]
                sub_dir = data["sub_dir"]
                textual_prompt = data.get("textual_prompt") or ""
                additional = data.get("additional_image_names") or []
                if isinstance(additional, str):
                    additional = [additional]
                unique_name = data["unique_name"]
                items.append(
                    DatasetItem(
                        img_name=img_name,
                        textual_prompt=textual_prompt,
                        sub_dir=sub_dir,
                        unique_name=unique_name,
                        additional_image_names=list(additional),
                    )
                )

        return cls(items=items, base_dir=base_dir, limit=limit, offset=offset)


if __name__ == '__main__':
    dataset = JsonlDataset.from_jsonl("data/index.jsonl")
    for item in dataset:
        print(f"Image: {item.image_path(dataset.base_dir)}")
        print(f"Prompt: {item.textual_prompt}")
        print(f"Additional Images: {item.additional_image_paths(dataset.base_dir)}")
        print("-----")
