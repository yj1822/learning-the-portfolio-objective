from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from implementable_frontier.models.backtest import (
    BacktestResult,
    BacktestState,
)


class CheckpointStore:
    def __init__(self, root: str | Path, *, resume: bool = True) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.resume = bool(resume)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        self.progress_log = self.root / "progress.jsonl"

    def candidate_id(self, method: str, parameters: dict[str, object]) -> str:
        payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{method}-{digest}"

    def has(self, namespace: str, item_id: str, year: int) -> bool:
        exists = self.exists(namespace, item_id, year)
        if self.resume and exists:
            self.hits += 1
            return True
        self.misses += 1
        return False

    def exists(self, namespace: str, item_id: str, year: int) -> bool:
        return (self._directory(namespace, item_id, year) / "complete.json").exists()

    def load_returns(self, namespace: str, item_id: str, year: int) -> pd.DataFrame:
        directory = self._directory(namespace, item_id, year)
        if not (directory / "complete.json").exists():
            raise FileNotFoundError(
                f"Incomplete checkpoint: {namespace}/{item_id}/{int(year)}"
            )
        return pd.read_parquet(directory / "returns.parquet")

    def load_state(self, namespace: str, item_id: str, year: int) -> BacktestState:
        directory = self._directory(namespace, item_id, year)
        if not (directory / "complete.json").exists():
            raise FileNotFoundError(
                f"Incomplete checkpoint: {namespace}/{item_id}/{int(year)}"
            )
        return _load_state(directory)

    def save(
        self,
        namespace: str,
        item_id: str,
        year: int,
        result: BacktestResult,
        *,
        metadata: dict[str, object] | None = None,
        include_artifacts: bool = False,
    ) -> None:
        directory = self._directory(namespace, item_id, year)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(result.returns, directory / "returns.parquet")
        _save_state(result.final_state, directory)
        if include_artifacts:
            _atomic_parquet(result.predictions, directory / "predictions.parquet")
            _atomic_parquet(result.weights, directory / "weights.parquet")
        payload = {
            "namespace": namespace,
            "item_id": item_id,
            "year": int(year),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "warnings": result.warnings,
            "risk_models_used": sorted(result.risk_models_used),
            "adjustment_modes": sorted(result.adjustment_modes),
            "metadata": metadata or {},
        }
        _atomic_json(payload, directory / "complete.json")
        self.writes += 1
        self.log("complete", namespace=namespace, item_id=item_id, year=year)

    def load(
        self,
        namespace: str,
        item_id: str,
        year: int,
        *,
        include_artifacts: bool = False,
    ) -> BacktestResult:
        directory = self._directory(namespace, item_id, year)
        metadata = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        return BacktestResult(
            predictions=(
                pd.read_parquet(directory / "predictions.parquet")
                if include_artifacts and (directory / "predictions.parquet").exists()
                else pd.DataFrame()
            ),
            weights=(
                pd.read_parquet(directory / "weights.parquet")
                if include_artifacts and (directory / "weights.parquet").exists()
                else pd.DataFrame()
            ),
            returns=pd.read_parquet(directory / "returns.parquet"),
            warnings=[str(value) for value in metadata.get("warnings", [])],
            risk_models_used=set(metadata.get("risk_models_used", [])),
            adjustment_modes=set(metadata.get("adjustment_modes", [])),
            final_state=_load_state(directory),
        )

    def record_error(
        self,
        namespace: str,
        item_id: str,
        year: int,
        error: Exception,
    ) -> None:
        self.errors += 1
        directory = self._directory(namespace, item_id, year)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            {
                "namespace": namespace,
                "item_id": item_id,
                "year": int(year),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
            directory / "error.json",
        )
        self.log(
            "error",
            namespace=namespace,
            item_id=item_id,
            year=year,
            error=f"{type(error).__name__}: {error}",
        )

    def info(self) -> dict[str, float | int | bool]:
        requests = self.hits + self.misses
        return {
            "resume_enabled": self.resume,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "requests": requests,
            "hit_rate": self.hits / requests if requests else 0.0,
            "completed_checkpoints": len(list(self.root.rglob("complete.json"))),
        }

    def write_manifest(self, extra: dict[str, object] | None = None) -> Path:
        path = self.root / "checkpoint_manifest.json"
        _atomic_json({**self.info(), **(extra or {})}, path)
        return path

    def log(self, event: str, **values: object) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        with self.progress_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _directory(self, namespace: str, item_id: str, year: int) -> Path:
        return self.root / namespace / item_id / str(int(year))


def _save_state(state: BacktestState, directory: Path) -> None:
    indices = state.weights.index.union(state.forward_returns.index).union(
        state.lambda_adv.index
    )
    frame = pd.DataFrame(
        {
            "permno": indices.astype("int64"),
            "weight": state.weights.reindex(indices).to_numpy(float),
            "forward_return": state.forward_returns.reindex(indices).to_numpy(float),
            "lambda_adv": state.lambda_adv.reindex(indices).to_numpy(float),
        }
    )
    _atomic_parquet(frame, directory / "state.parquet")
    _atomic_json(
        {"market_growth": state.market_growth}, directory / "state_metadata.json"
    )


def _load_state(directory: Path) -> BacktestState:
    frame = pd.read_parquet(directory / "state.parquet")
    index = pd.Index(frame["permno"].astype(int), dtype="int64")
    metadata = json.loads(
        (directory / "state_metadata.json").read_text(encoding="utf-8")
    )
    return BacktestState(
        weights=pd.Series(frame["weight"].to_numpy(float), index=index).dropna(),
        forward_returns=pd.Series(
            frame["forward_return"].to_numpy(float), index=index
        ).dropna(),
        lambda_adv=pd.Series(frame["lambda_adv"].to_numpy(float), index=index).dropna(),
        market_growth=(
            float(metadata["market_growth"])
            if metadata.get("market_growth") is not None
            else None
        ),
    )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(values: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)
