#!/usr/bin/env python3
"""
Validate a generated submission.csv against the sample submission format.

This is a smoke test for structural compatibility, not exact content matching.
The abstractive text is model-generated, so exact row-by-row equality with the
sample submission is not expected.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


EXPECTED_COLUMNS = ["ID", "abstractive", "refs"]
REFS_PATTERN = re.compile(r"^P\d+(?:,\s*P\d+)*$")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_structure(df: pd.DataFrame) -> list[str]:
    errors = []

    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(
            f"column order mismatch: expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
        )

    if "ID" in df.columns:
        ids = df["ID"]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
            errors.append("ID column contains empty values")
        if ids.astype(str).duplicated().any():
            errors.append("ID column contains duplicates")

    if "abstractive" in df.columns:
        abstractive = df["abstractive"]
        if abstractive.isna().any() or abstractive.astype(str).str.strip().eq("").any():
            errors.append("abstractive column contains empty values")

    if "refs" in df.columns:
        refs = df["refs"].fillna("").astype(str).str.strip()
        invalid_refs = refs[
            (refs != "") & ~refs.str.match(REFS_PATTERN)
        ]
        if not invalid_refs.empty:
            errors.append(
                "refs column has invalid format in rows: "
                + ", ".join(invalid_refs.index.astype(str).tolist()[:10])
            )

    return errors


def compare_to_template(candidate: pd.DataFrame, template: pd.DataFrame) -> list[str]:
    notes = []

    if list(candidate.columns) != list(template.columns):
        notes.append("candidate columns differ from sample submission columns")

    if len(candidate) != len(template):
        notes.append(
            f"row count differs from sample submission: candidate={len(candidate)}, template={len(template)}"
        )
        return notes

    if "ID" in candidate.columns and "ID" in template.columns:
        if not candidate["ID"].astype(str).equals(template["ID"].astype(str)):
            mismatch = candidate["ID"].astype(str).ne(template["ID"].astype(str))
            first_bad = mismatch[mismatch].index.tolist()[:10]
            notes.append(
                "ID order differs from sample submission at rows: "
                + ", ".join(str(i) for i in first_bad)
            )

    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate submission.csv format")
    parser.add_argument(
        "--candidate",
        "-c",
        type=Path,
        default=config.OUTPUT_DIR / "submission.csv",
        help="Path to the generated submission CSV",
    )
    parser.add_argument(
        "--template",
        "-t",
        type=Path,
        default=Path("data/eval_sample/submission.csv"),
        help="Path to the sample submission CSV",
    )
    args = parser.parse_args()

    if not args.candidate.exists():
        print(f"FAIL: candidate file not found: {args.candidate}")
        return 1

    if not args.template.exists():
        print(f"FAIL: template file not found: {args.template}")
        return 1

    candidate = load_csv(args.candidate)
    template = load_csv(args.template)

    errors = validate_structure(candidate)
    notes = compare_to_template(candidate, template)

    if errors:
        print("FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("PASS: submission structure is valid")
    print(f"- candidate: {args.candidate}")
    print(f"- columns: {list(candidate.columns)}")
    print(f"- rows: {len(candidate)}")

    if notes:
        print("Template comparison notes:")
        for note in notes:
            print(f"- {note}")
    else:
        print("Template comparison: exact row layout matches the sample submission")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
