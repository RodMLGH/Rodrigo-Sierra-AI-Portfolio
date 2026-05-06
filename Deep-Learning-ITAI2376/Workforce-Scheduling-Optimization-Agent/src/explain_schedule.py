"""
explain_schedule.py

Step 4: LLM Explanation Layer

This script reads the OR-Tools schedule output and validation summary,
then uses Gemini to generate a manager-friendly explanation.

Inputs:
- outputs/schedule.csv
- outputs/schedule_summary.json
- outputs/schedule_validation_summary.json

Outputs:
- outputs/schedule_explanation.txt
- outputs/schedule_explanation.json

Project role:
This script represents the explanation and reasoning layer of Agent 2
(Scheduling & Explanation Agent).
"""


# Imports

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SCHEDULE_CSV_FILE = OUTPUTS_DIR / "schedule.csv"
SCHEDULE_SUMMARY_FILE = OUTPUTS_DIR / "schedule_summary.json"
VALIDATION_SUMMARY_FILE = OUTPUTS_DIR / "schedule_validation_summary.json"

EXPLANATION_TXT_FILE = OUTPUTS_DIR / "schedule_explanation.txt"
EXPLANATION_JSON_FILE = OUTPUTS_DIR / "schedule_explanation.json"


# Data loading fucntion

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_schedule_sample(max_rows: int = 20) -> list[dict]:
    """
    Load a small sample of the schedule for the LLM.

    We do not send the entire CSV because a concise sample is enough
    for explanation and keeps the prompt smaller.
    """
    if not SCHEDULE_CSV_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {SCHEDULE_CSV_FILE}. "
            "Run python src/scheduler.py first."
        )

    schedule_df = pd.read_csv(SCHEDULE_CSV_FILE)

    # Keep a balanced sample: assigned + unfilled examples.
    assigned = schedule_df[schedule_df["assignment_status"] == "Assigned"].head(max_rows // 2)
    unfilled = schedule_df[schedule_df["assignment_status"] == "Unfilled"].head(max_rows // 2)

    sample_df = pd.concat([assigned, unfilled], ignore_index=True)

    return sample_df.to_dict(orient="records")


# Prompt building function

def build_explanation_prompt(
    schedule_summary: Dict[str, Any],
    validation_summary: Dict[str, Any],
    schedule_sample: list[dict],
) -> str:
    """
    Build a structured prompt for the LLM.

    The prompt asks Gemini to explain the schedule like an operations
    manager report, not like raw technical output.
    """

    prompt = f"""
You are an operations analyst explaining the output of an AI workforce scheduling system.

Project context:
- Industry: Manufacturing
- Scenario: 24/7 production facility
- Agent 1: Demand Forecasting Agent using an LSTM model
- Agent 2: Scheduling & Explanation Agent using OR-Tools and an LLM
- The schedule is generated using synthetic data
- OR-Tools creates the schedule
- The LLM explains the schedule in plain language

Your task:
Write a clear manager-friendly explanation of the schedule.

Use ONLY the information provided below.
Do not invent employee names, percentages, constraints, or causes not supported by the data.

Schedule summary:
{json.dumps(schedule_summary, indent=2)}

Validation summary:
{json.dumps(validation_summary, indent=2)}

Schedule sample:
{json.dumps(schedule_sample, indent=2)}

Please produce the response with the following sections:

1. Executive Summary
Briefly explain what the scheduling system produced and the overall coverage rate.

2. What Worked Well
Explain the strongest parts of the schedule, including any well-covered days, shifts, or skills.

3. Main Staffing Gaps
Explain where the unfilled positions occurred, using the validation data.

4. Constraint Impact
Explain how employee availability, skill matching, and max-hour limits affected the final schedule.

5. Manager Recommendations
Provide practical recommendations. Examples may include cross-training, increasing availability for difficult shifts, hiring for shortage skills, or adjusting demand expectations.

6. Limitations
Mention that this is a proof-of-concept using synthetic data and a 3-day scheduling window.

Style requirements:
- Use professional but simple English.
- Do not be overly technical.
- Use bullet points where helpful.
- Keep the explanation suitable for a class demo.
"""
    return prompt.strip()


# Gemini call function

def generate_with_gemini(prompt: str) -> str:
    """
    Generate explanation using Gemini.

    Includes retry logic for temporary service overload errors.
    """
    import time

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    if genai is None:
        raise ImportError(
            "google-genai is not installed. Run: python -m pip install google-genai"
        )

    client = genai.Client(api_key=api_key)

    max_attempts = 3
    wait_seconds = 10

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Gemini attempt {attempt} of {max_attempts}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text

        except Exception as error:
            last_error = error
            print(f"Attempt {attempt} failed: {error}")

            if attempt < max_attempts:
                print(f"Waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini failed after {max_attempts} attempts. Last error: {last_error}"
    )


def generate_fallback_explanation(
    schedule_summary: Dict[str, Any],
    validation_summary: Dict[str, Any],
) -> str:
    """
    Generate a simple local explanation if Gemini is not available.

    This is only for debugging. The final demo should use Gemini.
    """
    coverage_rate = schedule_summary.get("coverage_rate", 0)
    total_required = schedule_summary.get("total_required_positions", 0)
    total_assigned = schedule_summary.get("total_assigned_positions", 0)
    total_unfilled = schedule_summary.get("total_unfilled_positions", 0)
    labor_cost = schedule_summary.get("estimated_labor_cost", 0)

    unfilled = validation_summary.get("top_unfilled_areas", {})
    skills = unfilled.get("top_unfilled_skills", {})
    shifts = unfilled.get("top_unfilled_shifts", {})
    days = unfilled.get("top_unfilled_days", {})

    max_hour_violations = validation_summary.get("max_hour_violations", "unknown")

    return f"""
1. Executive Summary
The scheduling system generated a 3-day manufacturing workforce schedule.
It covered {total_assigned} of {total_required} required positions, for an overall coverage rate of {coverage_rate:.2%}.
There were {total_unfilled} unfilled positions. The estimated labor cost is ${labor_cost:,.2f}.

2. What Worked Well
The optimizer produced a valid schedule and respected max weekly hour limits.
Max-hour violations: {max_hour_violations}.

3. Main Staffing Gaps
The top unfilled skill areas were: {skills}.
The top unfilled shifts were: {shifts}.
The top unfilled days were: {days}.

4. Constraint Impact
The schedule was affected by employee availability, skill matching, and maximum weekly hour constraints.
Because the optimizer was not allowed to assign unavailable or unqualified employees, some positions remained unfilled.

5. Manager Recommendations
Managers could improve coverage by increasing employee flexibility, cross-training workers, hiring for shortage skills, or adjusting production demand during difficult shifts.

6. Limitations
This is a proof-of-concept using synthetic data and a 3-day scheduling window.
""".strip()


# Save outputs 

def save_explanation(explanation_text: str, used_gemini: bool) -> None:
    """Save explanation as TXT and JSON."""
    EXPLANATION_TXT_FILE.write_text(explanation_text, encoding="utf-8")

    explanation_payload = {
        "used_gemini": used_gemini,
        "explanation": explanation_text,
    }

    with open(EXPLANATION_JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(explanation_payload, file, indent=2)


# Main function

def main() -> None:
    """Run the explanation layer."""
    print("Loading schedule summary...")
    schedule_summary = load_json_file(SCHEDULE_SUMMARY_FILE)

    print("Loading validation summary...")
    validation_summary = load_json_file(VALIDATION_SUMMARY_FILE)

    print("Loading schedule sample...")
    schedule_sample = load_schedule_sample()

    print("Building explanation prompt...")
    prompt = build_explanation_prompt(
        schedule_summary=schedule_summary,
        validation_summary=validation_summary,
        schedule_sample=schedule_sample,
    )

    print("Generating explanation...")
    try:
        explanation = generate_with_gemini(prompt)
        used_gemini = True
        print("Gemini explanation generated successfully.")
    except Exception as error:
        used_gemini = False
        print("Gemini generation failed.")
        print(f"Reason: {error}")
        print("Using local fallback explanation for debugging.")
        explanation = generate_fallback_explanation(
            schedule_summary=schedule_summary,
            validation_summary=validation_summary,
        )

    save_explanation(explanation, used_gemini)

    print("\nExplanation saved successfully.")
    print(f"Text file: {EXPLANATION_TXT_FILE}")
    print(f"JSON file: {EXPLANATION_JSON_FILE}")

    print("\nExplanation preview:")
    print("-" * 60)
    print(explanation[:2000])


if __name__ == "__main__":
    main()
