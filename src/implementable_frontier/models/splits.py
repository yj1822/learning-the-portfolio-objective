from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExperimentSplit:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_year: int | None = None

    def masks(self, dates: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        values = pd.to_datetime(dates)
        train = values.between(self.train_start, self.train_end)
        validation = values.between(self.validation_start, self.validation_end)
        test = values.between(self.test_start, self.test_end)
        if (train & validation).any() or (train & test).any() or (validation & test).any():
            raise ValueError(f"Split {self.name} contains overlapping periods")
        return train, validation, test


def fast_split(config: dict[str, object]) -> ExperimentSplit:
    values = config["fast_split"]
    return ExperimentSplit(
        name="fast_split",
        train_start=pd.Timestamp(values["train_start"]),
        train_end=pd.Timestamp(values["train_end"]),
        validation_start=pd.Timestamp(values["validation_start"]),
        validation_end=pd.Timestamp(values["validation_end"]),
        test_start=pd.Timestamp(values["test_start"]),
        test_end=pd.Timestamp(values["test_end"]),
    )


def rolling_yearly_splits(config: dict[str, object]) -> list[ExperimentSplit]:
    values = config["rolling_yearly"]
    analysis_start = pd.Timestamp(values["analysis_start"])
    validation_years = int(values.get("validation_years", 3))
    output: list[ExperimentSplit] = []
    for year in range(int(values["first_test_year"]), int(values["last_test_year"]) + 1):
        validation_start_year = year - validation_years
        output.append(
            ExperimentSplit(
                name=f"rolling_{year}",
                train_start=analysis_start,
                train_end=pd.Timestamp(f"{validation_start_year - 1}-12-31"),
                validation_start=pd.Timestamp(f"{validation_start_year}-01-01"),
                validation_end=pd.Timestamp(f"{year - 1}-12-31"),
                test_start=pd.Timestamp(f"{year}-01-01"),
                test_end=pd.Timestamp(f"{year}-12-31"),
                test_year=year,
            )
        )
    return output


def experiment_splits(config: dict[str, object]) -> list[ExperimentSplit]:
    if config["protocol"] == "fast_split":
        return [fast_split(config)]
    return rolling_yearly_splits(config)
