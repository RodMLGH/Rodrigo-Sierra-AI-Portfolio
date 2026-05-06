"""
generate_data.py

This script generates synthetic datasets for the Workforce Scheduling
Optimization Multi-Agent System.

It creates:
1. data/employees.csv
2. data/demand_history.csv

Project scenario:
A 24/7 manufacturing facility that schedules employees across
Morning, Afternoon, and Night shifts based on production demand,
skills, availability, and weekly hour limits.

Version 1 uses only a subset of fields for the working prototype,
but the datasets include additional future-ready fields for later
versions of the project.
"""

# Imports

from __future__ import annotations

import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Configuration

RANDOM_SEED = 42

NUM_EMPLOYEES = 20
NUM_WEEKS = 16

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EMPLOYEE_FILE = DATA_DIR / "employees.csv"
DEMAND_FILE = DATA_DIR / "demand_history.csv"

START_DATE = datetime(2026, 1, 5)  # Monday

SHIFTS = ["Morning", "Afternoon", "Night"]

SHIFT_HOURS = {
    "Morning": "06:00-14:00",
    "Afternoon": "14:00-22:00",
    "Night": "22:00-06:00",
}

SKILLS = [
    "Machine Operator",
    "Quality Inspector",
    "Maintenance Technician",
    "Packaging Operator",
]

PRODUCTION_LINES = ["Line A", "Line B", "Line C"]

PRODUCT_TYPES = [
    "Standard Parts",
    "Custom Components",
    "Packaged Goods",
]

EDUCATION_LEVELS = [
    "High School",
    "Associate Degree",
    "Technical Certificate",
    "Bachelor Degree",
]

EMPLOYMENT_TYPES = [
    "Full-Time",
    "Part-Time",
]

DEPARTMENTS = [
    "Production",
    "Quality Control",
    "Maintenance",
    "Packaging",
]

CERTIFICATION_LEVELS = [
    "Basic",
    "Intermediate",
    "Advanced",
]


# Helper functions

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible synthetic data."""
    random.seed(seed)
    np.random.seed(seed)


def create_project_folders() -> None:
    """Create required project folders if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def random_phone_number(index: int) -> str:
    """
    Generate a fake phone number.

    The 555 prefix is commonly used for fictional numbers in examples.
    """
    return f"555-01{index:02d}"


def random_address(index: int) -> str:
    """Generate a fictional manufacturing-area address."""
    street_numbers = [1001, 1205, 1330, 1488, 1602, 1750, 1881, 1945]
    street_names = [
        "Industrial Pkwy",
        "Manufacturing Dr",
        "Logistics Ave",
        "Production Blvd",
        "Factory Lane",
    ]
    return f"{random.choice(street_numbers) + index} {random.choice(street_names)}"


def choose_available_days(employment_type: str) -> str:
    """
    Generate available days as a comma-separated string.

    Full-time employees usually have 5-6 available days.
    Part-time employees usually have 3-4 available days.
    """
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    if employment_type == "Full-Time":
        num_days = random.choice([5, 5, 6])
    else:
        num_days = random.choice([3, 4])

    selected_days = sorted(
        random.sample(days, num_days),
        key=lambda day: days.index(day)
    )
    return ",".join(selected_days)


def choose_available_shifts(preferred_shift: str) -> str:
    """
    Generate available shifts as a comma-separated string.

    Most employees can work their preferred shift and possibly one more.
    """
    possible_shifts = SHIFTS.copy()

    if random.random() < 0.65:
        # Preferred shift plus one optional shift
        remaining = [shift for shift in possible_shifts if shift != preferred_shift]
        selected = [preferred_shift, random.choice(remaining)]
    else:
        # More flexible employee
        selected = random.sample(possible_shifts, random.choice([2, 3]))

    selected = sorted(selected, key=lambda shift: SHIFTS.index(shift))
    return ",".join(selected)


def skill_to_department(skill: str) -> str:
    """Map primary skill to department."""
    mapping = {
        "Machine Operator": "Production",
        "Quality Inspector": "Quality Control",
        "Maintenance Technician": "Maintenance",
        "Packaging Operator": "Packaging",
    }
    return mapping[skill]


def skill_base_hourly_rate(skill: str) -> float:
    """Return a base hourly rate by skill."""
    base_rates = {
        "Machine Operator": 24.00,
        "Quality Inspector": 26.00,
        "Maintenance Technician": 30.00,
        "Packaging Operator": 21.00,
    }
    return base_rates[skill]


def calculate_annual_salary(hourly_rate: float, employment_type: str) -> int:
    """
    Estimate annual salary from hourly rate.

    Full-time is estimated at 40 hours/week.
    Part-time is estimated at 24 hours/week.
    """
    weekly_hours = 40 if employment_type == "Full-Time" else 24
    return int(hourly_rate * weekly_hours * 52)


# Employee data generation

def generate_employees(num_employees: int = NUM_EMPLOYEES) -> pd.DataFrame:
    """
    Generate a synthetic employee dataset.

    Version 1 active fields:
    - employee_id
    - name
    - primary_skill
    - secondary_skill
    - available_days
    - available_shifts
    - max_hours_per_week
    - preferred_shift
    - hourly_rate
    - status

    Future-ready fields:
    - gender
    - phone_number
    - address
    - education_level
    - years_experience
    - annual_salary
    - employment_type
    - department
    - supervisor
    - certification_level
    - certification_expiration
    - hire_date
    - performance_rating
    """
    first_names = [
        "Ana", "James", "Sofia", "Daniel", "Maria",
        "Carlos", "Emily", "David", "Laura", "Michael",
        "Isabella", "Kevin", "Camila", "Robert", "Valentina",
        "Jose", "Natalie", "Andres", "Grace", "Samuel",
    ]

    last_names = [
        "Torres", "Miller", "Rivera", "Kim", "Garcia",
        "Johnson", "Brown", "Lopez", "Davis", "Martinez",
        "Wilson", "Anderson", "Thomas", "Moore", "Clark",
        "Harris", "Lewis", "Young", "Walker", "Scott",
    ]

    supervisors = [
        "Patricia Gomez",
        "William Carter",
        "Angela Reed",
        "Brian Thompson",
    ]

    employees = []

    # Force a balanced initial distribution of skills.
    primary_skill_pool = (SKILLS * ((num_employees // len(SKILLS)) + 1))[:num_employees]
    random.shuffle(primary_skill_pool)

    for i in range(1, num_employees + 1):
        employee_id = f"E{i:03d}"
        name = f"{first_names[i - 1]} {last_names[i - 1]}"

        gender = random.choice(["Female", "Male", "Non-specified"])

        primary_skill = primary_skill_pool[i - 1]
        secondary_skill_options = [skill for skill in SKILLS if skill != primary_skill]
        secondary_skill = random.choice(secondary_skill_options)

        employment_type = random.choice(["Full-Time", "Full-Time", "Full-Time", "Part-Time"])
        max_hours_per_week = 40 if employment_type == "Full-Time" else random.choice([24, 28, 32])

        preferred_shift = random.choice(SHIFTS)
        available_days = choose_available_days(employment_type)
        available_shifts = choose_available_shifts(preferred_shift)

        base_rate = skill_base_hourly_rate(primary_skill)
        years_experience = random.randint(1, 15)
        experience_bonus = min(years_experience * 0.35, 5.0)
        hourly_rate = round(base_rate + experience_bonus + random.uniform(-1.0, 1.5), 2)

        annual_salary = calculate_annual_salary(hourly_rate, employment_type)

        hire_year = random.randint(2015, 2025)
        hire_month = random.randint(1, 12)
        hire_day = random.randint(1, 28)
        hire_date = datetime(hire_year, hire_month, hire_day).strftime("%Y-%m-%d")

        cert_expiration = datetime(
            random.randint(2026, 2028),
            random.randint(1, 12),
            random.randint(1, 28)
        ).strftime("%Y-%m-%d")

        performance_rating = round(random.uniform(3.2, 5.0), 1)

        # Keep all active for Version 1 to avoid unnecessary scheduling issues.
        status = "Active"

        employees.append({
            "employee_id": employee_id,
            "name": name,
            "gender": gender,
            "phone_number": random_phone_number(i),
            "address": random_address(i),
            "education_level": random.choice(EDUCATION_LEVELS),
            "years_experience": years_experience,
            "employment_type": employment_type,
            "department": skill_to_department(primary_skill),
            "supervisor": random.choice(supervisors),
            "primary_skill": primary_skill,
            "secondary_skill": secondary_skill,
            "certification_level": random.choice(CERTIFICATION_LEVELS),
            "certification_expiration": cert_expiration,
            "hire_date": hire_date,
            "max_hours_per_week": max_hours_per_week,
            "available_days": available_days,
            "available_shifts": available_shifts,
            "preferred_shift": preferred_shift,
            "hourly_rate": hourly_rate,
            "annual_salary": annual_salary,
            "performance_rating": performance_rating,
            "status": status,
        })

    return pd.DataFrame(employees)


# Demand data generation

def calculate_required_staff(
    shift: str,
    skill_required: str,
    is_weekend: int,
    is_peak_day: int,
    production_units: int,
    machine_status: str,
    raw_material_availability: float,
    planned_downtime: int,
    supplier_delay: int,
) -> int:
    """
    Calculate required staff using synthetic business logic.

    This function creates realistic patterns for the LSTM to learn.
    """
    base_by_skill = {
        "Machine Operator": 3,
        "Quality Inspector": 1,
        "Maintenance Technician": 1,
        "Packaging Operator": 2,
    }

    shift_modifier = {
        "Morning": 1,
        "Afternoon": 1,
        "Night": 0,
    }

    staff = base_by_skill[skill_required] + shift_modifier[shift]

    # Higher weekend demand for manufacturing catch-up and shipping needs.
    if is_weekend:
        staff += 1 if skill_required in ["Machine Operator", "Packaging Operator"] else 0

    # Peak days require extra staffing.
    if is_peak_day:
        staff += 1

    # More units means more staffing.
    if production_units > 650:
        staff += 1

    # If a machine is down or downtime is planned, fewer operators may be required,
    # but maintenance may need more staff.
    if machine_status == "Down":
        if skill_required == "Machine Operator":
            staff = max(1, staff - 1)
        if skill_required == "Maintenance Technician":
            staff += 1

    if planned_downtime:
        if skill_required == "Maintenance Technician":
            staff += 1
        else:
            staff = max(1, staff - 1)

    # Material shortages or supplier delays can reduce production staffing needs.
    if raw_material_availability < 0.70 or supplier_delay:
        if skill_required in ["Machine Operator", "Packaging Operator"]:
            staff = max(1, staff - 1)

    return max(1, int(staff))


def generate_demand_history(num_weeks: int = NUM_WEEKS) -> pd.DataFrame:
    """
    Generate a synthetic manufacturing demand history dataset.

    Version 1 active fields:
    - date
    - day_of_week
    - shift
    - skill_required
    - required_staff
    - production_units
    - is_weekend
    - is_peak_day

    Future-ready fields:
    - week_number
    - production_line
    - product_type
    - machine_id
    - machine_status
    - order_backlog
    - priority_level
    - raw_material_availability
    - raw_material_cost_index
    - planned_downtime
    - absenteeism_rate
    - overtime_allowed
    - weather_disruption
    - supplier_delay
    """
    rows = []

    total_days = num_weeks * 7
    day_names_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_names_full = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for day_offset in range(total_days):
        current_date = START_DATE + timedelta(days=day_offset)
        day_index = current_date.weekday()
        day_of_week = day_names_full[day_index]
        day_short = day_names_short[day_index]
        week_number = (day_offset // 7) + 1

        is_weekend = 1 if day_short in ["Sat", "Sun"] else 0

        # Peak days occur more often near the end of the week and randomly.
        is_peak_day = 1 if (
            day_short in ["Thu", "Fri"] and random.random() < 0.45
        ) or random.random() < 0.10 else 0

        for shift in SHIFTS:
            for production_line in PRODUCTION_LINES:
                product_type = random.choice(PRODUCT_TYPES)
                machine_id = f"M-{production_line[-1]}{random.randint(1, 4)}"

                machine_status = random.choices(
                    ["Running", "Maintenance", "Down"],
                    weights=[0.82, 0.13, 0.05],
                    k=1
                )[0]

                raw_material_availability = round(random.uniform(0.60, 1.00), 2)
                raw_material_cost_index = round(random.uniform(0.85, 1.25), 2)
                order_backlog = random.randint(0, 80)

                priority_level = random.choices(
                    ["Low", "Medium", "High"],
                    weights=[0.25, 0.55, 0.20],
                    k=1
                )[0]

                planned_downtime = 1 if machine_status == "Maintenance" else 0
                absenteeism_rate = round(random.uniform(0.01, 0.12), 2)
                overtime_allowed = 1 if is_peak_day or priority_level == "High" else 0
                weather_disruption = 1 if random.random() < 0.04 else 0
                supplier_delay = 1 if raw_material_availability < 0.68 or random.random() < 0.05 else 0

                # Production units include patterns:
                # - Morning and Afternoon usually higher than Night.
                # - Weekends and peak days may increase demand.
                # - Machine issues and supply delays reduce output.
                base_units = {
                    "Morning": 520,
                    "Afternoon": 500,
                    "Night": 380,
                }[shift]

                if is_weekend:
                    base_units += 60

                if is_peak_day:
                    base_units += 120

                if priority_level == "High":
                    base_units += 80

                if machine_status == "Down":
                    base_units -= 150
                elif machine_status == "Maintenance":
                    base_units -= 80

                if supplier_delay:
                    base_units -= 70

                if weather_disruption:
                    base_units -= 40

                noise = random.randint(-45, 45)
                production_units = max(150, base_units + noise + order_backlog // 2)

                # Create one row per skill requirement.
                for skill_required in SKILLS:
                    required_staff = calculate_required_staff(
                        shift=shift,
                        skill_required=skill_required,
                        is_weekend=is_weekend,
                        is_peak_day=is_peak_day,
                        production_units=production_units,
                        machine_status=machine_status,
                        raw_material_availability=raw_material_availability,
                        planned_downtime=planned_downtime,
                        supplier_delay=supplier_delay,
                    )

                    rows.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "day_of_week": day_of_week,
                        "week_number": week_number,
                        "shift": shift,
                        "production_line": production_line,
                        "product_type": product_type,
                        "machine_id": machine_id,
                        "machine_status": machine_status,
                        "skill_required": skill_required,
                        "required_staff": required_staff,
                        "production_units": production_units,
                        "order_backlog": order_backlog,
                        "priority_level": priority_level,
                        "raw_material_availability": raw_material_availability,
                        "raw_material_cost_index": raw_material_cost_index,
                        "planned_downtime": planned_downtime,
                        "absenteeism_rate": absenteeism_rate,
                        "overtime_allowed": overtime_allowed,
                        "is_weekend": is_weekend,
                        "is_peak_day": is_peak_day,
                        "weather_disruption": weather_disruption,
                        "supplier_delay": supplier_delay,
                    })

    return pd.DataFrame(rows)


# Main function

def main() -> None:
    """Generate and save all synthetic datasets."""
    set_random_seed()
    create_project_folders()

    employees_df = generate_employees()
    demand_df = generate_demand_history()

    employees_df.to_csv(EMPLOYEE_FILE, index=False)
    demand_df.to_csv(DEMAND_FILE, index=False)

    print("Synthetic datasets generated successfully.")
    print(f"Employees file: {EMPLOYEE_FILE}")
    print(f"Demand history file: {DEMAND_FILE}")
    print()
    print("Dataset summary:")
    print(f"- Employees: {len(employees_df)} rows")
    print(f"- Demand history: {len(demand_df)} rows")
    print(f"- Date range: {demand_df['date'].min()} to {demand_df['date'].max()}")
    print(f"- Skills: {', '.join(SKILLS)}")
    print(f"- Shifts: {', '.join(SHIFTS)}")
    print()
    print("Employee sample:")
    print(employees_df.head(3).to_string(index=False))
    print()
    print("Demand sample:")
    print(demand_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
