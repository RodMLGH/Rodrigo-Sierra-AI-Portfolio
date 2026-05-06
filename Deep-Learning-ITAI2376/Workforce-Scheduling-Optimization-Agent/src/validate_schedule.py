"""
validate_schedule.py

This script validates and summarizes the schedule produced by scheduler.py.

Input:
- outputs/schedule.csv
- outputs/schedule_summary.json
- data/employees.csv

Output:
- terminal validation summary
- outputs/schedule_validation_summary.json

Purpose:
This step checks whether the OR-Tools scheduling output is reasonable before
passing it to the LLM explanation layer.
"""


# Imports

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EMPLOYEE_FILE = DATA_DIR / "employees.csv"
SCHEDULE_FILE = OUTPUTS_DIR / "schedule.csv"
SUMMARY_FILE = OUTPUTS_DIR / "schedule_summary.json"
VALIDATION_OUTPUT_FILE = OUTPUTS_DIR / "schedule_validation_summary.json"

SHIFT_HOURS = 8


# Data loading functions

def load_schedule() -> pd.DataFrame:
    """Load generated schedule."""
    if not SCHEDULE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SCHEDULE_FILE}. "
            "Run python src/scheduler.py first."
        )

    return pd.read_csv(SCHEDULE_FILE)


def load_employees() -> pd.DataFrame:
    """Load employee dataset."""
    if not EMPLOYEE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {EMPLOYEE_FILE}. "
            "Run python src/generate_data.py first."
        )

    return pd.read_csv(EMPLOYEE_FILE)


def load_schedule_summary() -> dict:
    """Load schedule summary JSON file."""
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SUMMARY_FILE}. "
            "Run python src/scheduler.py first."
        )

    with open(SUMMARY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# Validation helpers functions

def summarize_by_day(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize assigned and unfilled positions by day."""
    summary = (
        schedule_df
        .groupby(["date", "day_of_week", "assignment_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "Assigned" not in summary.columns:
        summary["Assigned"] = 0

    if "Unfilled" not in summary.columns:
        summary["Unfilled"] = 0

    summary["total_positions"] = summary["Assigned"] + summary["Unfilled"]
    summary["coverage_rate"] = (
        summary["Assigned"] / summary["total_positions"]
    ).round(4)

    return summary


def summarize_by_shift(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize assigned and unfilled positions by shift."""
    summary = (
        schedule_df
        .groupby(["shift", "assignment_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "Assigned" not in summary.columns:
        summary["Assigned"] = 0

    if "Unfilled" not in summary.columns:
        summary["Unfilled"] = 0

    summary["total_positions"] = summary["Assigned"] + summary["Unfilled"]
    summary["coverage_rate"] = (
        summary["Assigned"] / summary["total_positions"]
    ).round(4)

    return summary


def summarize_by_skill(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize assigned and unfilled positions by required skill."""
    summary = (
        schedule_df
        .groupby(["skill_required", "assignment_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "Assigned" not in summary.columns:
        summary["Assigned"] = 0

    if "Unfilled" not in summary.columns:
        summary["Unfilled"] = 0

    summary["total_positions"] = summary["Assigned"] + summary["Unfilled"]
    summary["coverage_rate"] = (
        summary["Assigned"] / summary["total_positions"]
    ).round(4)

    return summary.sort_values("coverage_rate")


def summarize_employee_hours(schedule_df: pd.DataFrame, employees_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate scheduled hours per employee and compare against max weekly hours."""
    assigned_df = schedule_df[schedule_df["assignment_status"] == "Assigned"].copy()

    if assigned_df.empty:
        return pd.DataFrame()

    hours_summary = (
        assigned_df
        .groupby(["employee_id", "employee_name"])
        .agg(
            assigned_shifts=("shift", "count"),
            scheduled_hours=("shift_hours", "sum"),
            estimated_cost=("shift_cost", "sum")
        )
        .reset_index()
    )

    employee_limits = employees_df[[
        "employee_id",
        "max_hours_per_week",
        "primary_skill",
        "secondary_skill",
        "preferred_shift",
        "hourly_rate"
    ]].copy()

    hours_summary = hours_summary.merge(
        employee_limits,
        on="employee_id",
        how="left"
    )

    hours_summary["remaining_hours"] = (
        hours_summary["max_hours_per_week"] - hours_summary["scheduled_hours"]
    )

    hours_summary["exceeds_max_hours"] = (
        hours_summary["scheduled_hours"] > hours_summary["max_hours_per_week"]
    )

    return hours_summary.sort_values(
        ["scheduled_hours", "employee_name"],
        ascending=[False, True]
    )


def check_max_hour_violations(employee_hours_df: pd.DataFrame) -> int:
    """Count max-hour violations."""
    if employee_hours_df.empty:
        return 0

    return int(employee_hours_df["exceeds_max_hours"].sum())


def get_top_unfilled_areas(schedule_df: pd.DataFrame) -> dict:
    """Identify the biggest unfilled areas."""
    unfilled_df = schedule_df[schedule_df["assignment_status"] == "Unfilled"].copy()

    if unfilled_df.empty:
        return {
            "top_unfilled_skills": [],
            "top_unfilled_shifts": [],
            "top_unfilled_days": []
        }

    top_unfilled_skills = (
        unfilled_df["skill_required"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    top_unfilled_shifts = (
        unfilled_df["shift"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    top_unfilled_days = (
        unfilled_df["day_of_week"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    return {
        "top_unfilled_skills": top_unfilled_skills,
        "top_unfilled_shifts": top_unfilled_shifts,
        "top_unfilled_days": top_unfilled_days
    }


# Main validation function

def main() -> None:
    """Run schedule validation."""
    schedule_df = load_schedule()
    employees_df = load_employees()
    schedule_summary = load_schedule_summary()

    print("SCHEDULE VALIDATION")
    print("-" * 60)

    print("\nOriginal schedule summary:")
    for key, value in schedule_summary.items():
        print(f"- {key}: {value}")

    by_day = summarize_by_day(schedule_df)
    by_shift = summarize_by_shift(schedule_df)
    by_skill = summarize_by_skill(schedule_df)
    employee_hours = summarize_employee_hours(schedule_df, employees_df)
    max_hour_violations = check_max_hour_violations(employee_hours)
    unfilled_areas = get_top_unfilled_areas(schedule_df)

    print("\nCoverage by day:")
    print(by_day.to_string(index=False))

    print("\nCoverage by shift:")
    print(by_shift.to_string(index=False))

    print("\nCoverage by skill:")
    print(by_skill.to_string(index=False))

    print("\nEmployee hours summary:")
    print(employee_hours.to_string(index=False))

    print("\nValidation checks:")
    print(f"- Max-hour violations: {max_hour_violations}")

    if max_hour_violations == 0:
        print("- Max-hour rule: PASSED")
    else:
        print("- Max-hour rule: FAILED")

    print("\nTop unfilled areas:")
    print(f"- Skills: {unfilled_areas['top_unfilled_skills']}")
    print(f"- Shifts: {unfilled_areas['top_unfilled_shifts']}")
    print(f"- Days: {unfilled_areas['top_unfilled_days']}")

    validation_summary = {
        "schedule_summary": schedule_summary,
        "max_hour_violations": max_hour_violations,
        "top_unfilled_areas": unfilled_areas,
        "coverage_by_day": by_day.to_dict(orient="records"),
        "coverage_by_shift": by_shift.to_dict(orient="records"),
        "coverage_by_skill": by_skill.to_dict(orient="records"),
        "employee_hours": employee_hours.to_dict(orient="records"),
    }

    with open(VALIDATION_OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(validation_summary, file, indent=2)

    print(f"\nValidation summary saved to: {VALIDATION_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
