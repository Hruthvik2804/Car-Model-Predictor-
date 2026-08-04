"""Lightweight checks that need no ML dependencies.

Validates the zero-shot catalogue and the class-name convention used by the
custom-training pipeline. Runs in CI with only the Python standard library.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "vehicle_catalog.json"
REQUIRED_KEYS = {"manufacturer", "model", "year_range"}


def _load_catalog() -> list[dict]:
    with CATALOG.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_catalog_is_non_empty_list() -> None:
    data = _load_catalog()
    assert isinstance(data, list) and data, "catalog must be a non-empty list"


def test_every_entry_has_required_keys() -> None:
    for i, entry in enumerate(_load_catalog()):
        assert isinstance(entry, dict), f"entry {i} is not an object"
        missing = REQUIRED_KEYS - entry.keys()
        assert not missing, f"entry {i} missing keys: {sorted(missing)}"
        for key in REQUIRED_KEYS:
            assert str(entry[key]).strip(), f"entry {i} has empty {key!r}"


def test_no_duplicate_vehicles() -> None:
    seen = set()
    for entry in _load_catalog():
        key = (entry["manufacturer"].lower(), entry["model"].lower())
        assert key not in seen, f"duplicate vehicle: {key}"
        seen.add(key)


def test_class_name_convention_round_trips() -> None:
    # Manufacturer__Model__YearRange must split cleanly into three parts.
    for entry in _load_catalog():
        year = entry["year_range"].replace("–", "-")  # normalise en-dash
        class_name = f"{entry['manufacturer']}__{entry['model']}__{year}"
        parts = class_name.split("__")
        assert len(parts) == 3, f"bad class name: {class_name}"
