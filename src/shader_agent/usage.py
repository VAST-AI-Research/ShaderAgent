"""Per-run LLM/VLM usage tracking.

:class:`UsageCollector` is bound through a :class:`contextvars.ContextVar` so
parallel batch workers each record into their own session.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .io import get_logger

logger = get_logger(__name__)


@dataclass
class _ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageCollector:
    """Thread-safe accumulator of per-model token usage for one pipeline run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: Dict[str, _ModelUsage] = {}
        self._notes: Dict[str, Any] = {}
        self._start = time.monotonic()
        self._elapsed: Optional[float] = None

    def record(self, label: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            usage = self._by_model.setdefault(label, _ModelUsage())
            usage.calls += 1
            usage.input_tokens += int(input_tokens or 0)
            usage.output_tokens += int(output_tokens or 0)

    def note(self, key: str, value: Any) -> None:
        """Record a run-level fact (e.g. outcome, retries) beside the token counts,
        so one run yields one record of both cost and result.
        """
        with self._lock:
            self._notes[key] = value

    def bump(self, key: str, amount: int = 1) -> None:
        """Increment a run-level counter (e.g. DSL validation retries)."""
        with self._lock:
            self._notes[key] = int(self._notes.get(key, 0)) + amount

    def stop(self) -> None:
        if self._elapsed is None:
            self._elapsed = time.monotonic() - self._start

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            per_model = {
                label: {
                    "calls": u.calls,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "total_tokens": u.total_tokens,
                }
                for label, u in self._by_model.items()
            }
        total_calls = sum(v["calls"] for v in per_model.values())
        total_in = sum(v["input_tokens"] for v in per_model.values())
        total_out = sum(v["output_tokens"] for v in per_model.values())
        elapsed = self._elapsed if self._elapsed is not None else time.monotonic() - self._start
        with self._lock:
            notes = dict(self._notes)
        return {
            "wall_clock_s": round(elapsed, 3),
            "total_calls": total_calls,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "by_model": per_model,
            **notes,
        }

    def dump(self, path: str) -> None:
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("Usage written to %s", path)
        except Exception as exc:  # bookkeeping must never break a run
            logger.warning("Failed to write usage to %s: %s", path, exc)


# A ContextVar, not a process-global: batch.py runs `dataset_workers` pipelines in a
# ThreadPoolExecutor, each in its own usage_session. With a single global slot the
# last starter won and the first finisher reset it to None, so the other workers
# recorded nowhere and wrote all-zero usage.json files.
_active: "ContextVar[Optional[UsageCollector]]" = ContextVar(
    "shader_agent_usage_collector", default=None,
)


def get_active() -> Optional[UsageCollector]:
    return _active.get()


@contextmanager
def usage_session(save_dir: str):
    """Activate a fresh collector for one run; dump ``usage.json`` on exit.

    Dumps on success *and* on exception, so failed runs still report what they spent.
    """
    collector = UsageCollector()
    token = _active.set(collector)
    try:
        yield collector
    finally:
        collector.stop()
        collector.dump(os.path.join(save_dir, "usage.json"))
        _active.reset(token)


class InstrumentedModel:
    """Transparent proxy over a callable model that tallies token usage.

    Every attribute access is delegated to the wrapped model; only the three call
    paths that yield a ``ChatMessage`` are intercepted.
    """

    def __init__(self, inner: Any, label: str) -> None:
        # Bypass __setattr__ so these land on the proxy, not the inner model.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_label", label)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def _record(self, result: Any) -> None:
        collector = get_active()
        if collector is None:
            return
        token_usage = getattr(result, "token_usage", None)
        if token_usage is not None:
            collector.record(
                self._label,
                getattr(token_usage, "input_tokens", 0) or 0,
                getattr(token_usage, "output_tokens", 0) or 0,
            )
        else:
            collector.record(self._label, 0, 0)  # still count the call

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._inner(*args, **kwargs)
        self._record(result)
        return result

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        result = self._inner.generate(*args, **kwargs)
        self._record(result)
        return result

    def generate_stream(self, *args: Any, **kwargs: Any):
        last = None
        for event in self._inner.generate_stream(*args, **kwargs):
            last = event
            yield event
        # Streamed usage, if any, rides on the final delta event.
        collector = get_active()
        if last is not None and collector is not None:
            token_usage = getattr(last, "token_usage", None)
            if token_usage is not None:
                collector.record(
                    self._label,
                    getattr(token_usage, "input_tokens", 0) or 0,
                    getattr(token_usage, "output_tokens", 0) or 0,
                )
