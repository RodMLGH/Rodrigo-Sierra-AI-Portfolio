"""
Validate the synthetic demand history dataset.

This script checks whether the generated demand data contains
useful patterns by shift, weekend, peak day, and required skill.
"""


# Imports

import pandas as pd


# Load data

df = pd.read_csv("data/demand_history.csv")


# Basic checks

print("DATASET BASIC CHECK")
print("-" * 50)
print("Rows:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Unique shifts:", df["shift"].unique())
print("Unique skills:", df["skill_required"].unique())
print()


# Demand by shift

print("DEMAND BY SHIFT")
print("-" * 50)
shift_summary = (
    df.groupby("shift")
    .agg(
        avg_required_staff=("required_staff", "mean"),
        avg_production_units=("production_units", "mean"),
        min_required_staff=("required_staff", "min"),
        max_required_staff=("required_staff", "max"),
        rows=("required_staff", "count")
    )
    .reset_index()
)
print(shift_summary)
print()


# Weekend pattern

print("WEEKEND PATTERN")
print("-" * 50)
weekend_summary = (
    df.groupby("is_weekend")
    .agg(
        avg_required_staff=("required_staff", "mean"),
        avg_production_units=("production_units", "mean"),
        rows=("required_staff", "count")
    )
    .reset_index()
)
print(weekend_summary)
print()


# Peak-day pattern

print("PEAK DAY PATTERN")
print("-" * 50)
peak_summary = (
    df.groupby("is_peak_day")
    .agg(
        avg_required_staff=("required_staff", "mean"),
        avg_production_units=("production_units", "mean"),
        rows=("required_staff", "count")
    )
    .reset_index()
)
print(peak_summary)
print()


# Demand by skill

print("DEMAND BY SKILL")
print("-" * 50)
skill_summary = (
    df.groupby("skill_required")
    .agg(
        avg_required_staff=("required_staff", "mean"),
        min_required_staff=("required_staff", "min"),
        max_required_staff=("required_staff", "max"),
        rows=("required_staff", "count")
    )
    .reset_index()
)
print(skill_summary)
