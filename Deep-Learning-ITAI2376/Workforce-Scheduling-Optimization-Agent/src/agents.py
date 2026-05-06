"""
agents.py

Step 5: CrewAI Multi-Agent Integration

This script coordinates the working project components as a two-agent system:

Agent 1:
- Demand Forecasting Agent
- Runs the LSTM forecasting pipeline

Agent 2:
- Scheduling & Explanation Agent
- Runs OR-Tools scheduling
- Runs schedule validation
- Runs Gemini explanation layer

Inputs:
- data/employees.csv
- data/demand_history.csv

Outputs:
- outputs/forecast.json
- outputs/schedule.json
- outputs/schedule_summary.json
- outputs/schedule_validation_summary.json
- outputs/schedule_explanation.txt
- outputs/schedule_explanation.json

Run from project root:
python src/agents.py
"""


# Imports

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Type

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FORECAST_FILE = OUTPUTS_DIR / "forecast.json"
SCHEDULE_FILE = OUTPUTS_DIR / "schedule.json"
SCHEDULE_SUMMARY_FILE = OUTPUTS_DIR / "schedule_summary.json"
VALIDATION_SUMMARY_FILE = OUTPUTS_DIR / "schedule_validation_summary.json"
EXPLANATION_FILE = OUTPUTS_DIR / "schedule_explanation.txt"

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


# Helper function to run project scripts

def run_python_script(script_name: str) -> str:
    """
    Run an existing project script as a subprocess.

    This keeps the project modular:
    - forecast_model.py handles LSTM forecasting
    - scheduler.py handles OR-Tools scheduling
    - validate_schedule.py handles validation
    - explain_schedule.py handles Gemini explanation
    """
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_message = (
            f"Script {script_name} failed.\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
        raise RuntimeError(error_message)

    return result.stdout


# Tool input schema

class EmptyToolInput(BaseModel):
    """No input is required for this tool."""
    confirmation: str = Field(
        default="run",
        description="A simple confirmation string. Use 'run'."
    )


# CrewAI Tools

class DemandForecastingTool(BaseTool):
    name: str = "Demand Forecasting Tool"
    description: str = (
        "Runs the LSTM demand forecasting pipeline. "
        "It reads demand_history.csv and creates outputs/forecast.json."
    )
    args_schema: Type[BaseModel] = EmptyToolInput

    def _run(self, confirmation: str = "run") -> str:
        output = run_python_script("forecast_model.py")

        if not FORECAST_FILE.exists():
            raise FileNotFoundError("Forecast file was not created.")

        return (
            "Demand forecasting completed successfully.\n"
            f"Forecast file created: {FORECAST_FILE}\n\n"
            f"Script output:\n{output[-2000:]}"
        )


class SchedulingTool(BaseTool):
    name: str = "Scheduling Optimization Tool"
    description: str = (
        "Runs the OR-Tools scheduling engine. "
        "It reads employees.csv and forecast.json, then creates schedule outputs."
    )
    args_schema: Type[BaseModel] = EmptyToolInput

    def _run(self, confirmation: str = "run") -> str:
        output = run_python_script("scheduler.py")

        if not SCHEDULE_FILE.exists():
            raise FileNotFoundError("Schedule file was not created.")

        return (
            "Scheduling optimization completed successfully.\n"
            f"Schedule file created: {SCHEDULE_FILE}\n\n"
            f"Script output:\n{output[-2000:]}"
        )


class ScheduleValidationTool(BaseTool):
    name: str = "Schedule Validation Tool"
    description: str = (
        "Validates the generated schedule. "
        "It creates outputs/schedule_validation_summary.json."
    )
    args_schema: Type[BaseModel] = EmptyToolInput

    def _run(self, confirmation: str = "run") -> str:
        output = run_python_script("validate_schedule.py")

        if not VALIDATION_SUMMARY_FILE.exists():
            raise FileNotFoundError("Schedule validation summary was not created.")

        return (
            "Schedule validation completed successfully.\n"
            f"Validation file created: {VALIDATION_SUMMARY_FILE}\n\n"
            f"Script output:\n{output[-2000:]}"
        )


class ScheduleExplanationTool(BaseTool):
    name: str = "Schedule Explanation Tool"
    description: str = (
        "Runs the LLM explanation layer. "
        "It reads schedule and validation outputs and creates schedule_explanation.txt."
    )
    args_schema: Type[BaseModel] = EmptyToolInput

    def _run(self, confirmation: str = "run") -> str:
        output = run_python_script("explain_schedule.py")

        if not EXPLANATION_FILE.exists():
            raise FileNotFoundError("Schedule explanation file was not created.")

        return (
            "Schedule explanation completed successfully.\n"
            f"Explanation file created: {EXPLANATION_FILE}\n\n"
            f"Script output:\n{output[-2000:]}"
        )


# LLM setup

def build_llm() -> LLM:
    """
    Build the LLM used by CrewAI agents.

    CrewAI uses the LLM for agent reasoning and task coordination.
    The technical forecasting is still done by the LSTM, and the scheduling
    optimization is still done by OR-Tools.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    return LLM(
        model=f"gemini/{GEMINI_MODEL}",
        api_key=GEMINI_API_KEY,
        temperature=0.2,
    )


# Agent definitions

def create_agents(llm: LLM) -> tuple[Agent, Agent]:
    """Create the two CrewAI agents."""

    demand_forecasting_agent = Agent(
        role="Demand Forecasting Agent",
        goal=(
            "Generate a staffing demand forecast using the LSTM forecasting tool."
        ),
        backstory=(
            "You are an operations analytics specialist for a manufacturing plant. "
            "Your responsibility is to forecast staffing needs from historical "
            "production demand patterns."
        ),
        tools=[DemandForecastingTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    scheduling_explanation_agent = Agent(
        role="Scheduling and Explanation Agent",
        goal=(
            "Create a workforce schedule using OR-Tools, validate the schedule, "
            "and generate a manager-friendly explanation."
        ),
        backstory=(
            "You are a workforce scheduling specialist. You use optimization tools "
            "to assign qualified employees to shifts while respecting availability, "
            "skills, and maximum weekly hours. You also explain results clearly "
            "for operations managers."
        ),
        tools=[
            SchedulingTool(),
            ScheduleValidationTool(),
            ScheduleExplanationTool(),
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return demand_forecasting_agent, scheduling_explanation_agent


# Task definitions

def create_tasks(
    demand_forecasting_agent: Agent,
    scheduling_explanation_agent: Agent,
) -> tuple[Task, Task, Task, Task]:
    """Create the CrewAI tasks."""

    forecast_task = Task(
        description=(
            "Run the Demand Forecasting Tool to create the next-week staffing "
            "forecast. Confirm that outputs/forecast.json is created. "
            "Do not manually invent the forecast. Use the tool."
        ),
        expected_output=(
            "A confirmation that the LSTM forecasting pipeline ran successfully "
            "and created outputs/forecast.json."
        ),
        agent=demand_forecasting_agent,
    )

    scheduling_task = Task(
        description=(
            "Run only the Scheduling Optimization Tool. "
            "Use the forecast created by the previous task and the employee data "
            "to generate the workforce schedule. Confirm that outputs/schedule.csv, "
            "outputs/schedule.json, and outputs/schedule_summary.json are created."
        ),
        expected_output=(
            "A confirmation that the OR-Tools scheduling engine ran successfully "
            "and created the schedule output files."
        ),
        agent=scheduling_explanation_agent,
        context=[forecast_task],
    )

    validation_task = Task(
        description=(
            "Run only the Schedule Validation Tool. "
            "Validate the schedule created by the scheduling task. "
            "Confirm that outputs/schedule_validation_summary.json is created."
        ),
        expected_output=(
            "A confirmation that the schedule validation summary was generated successfully."
        ),
        agent=scheduling_explanation_agent,
        context=[scheduling_task],
    )

    explanation_task = Task(
        description=(
            "Run only the Schedule Explanation Tool. "
            "Use the schedule summary and validation summary to generate the "
            "manager-friendly explanation. Confirm that outputs/schedule_explanation.txt "
            "and outputs/schedule_explanation.json are created."
        ),
        expected_output=(
            "A final confirmation that the LLM explanation was generated successfully."
        ),
        agent=scheduling_explanation_agent,
        context=[validation_task],
    )

    return forecast_task, scheduling_task, validation_task, explanation_task


# Crew main function

def main() -> None:
    """Run the two-agent CrewAI workflow."""

    print("Starting CrewAI multi-agent workflow...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Gemini model for CrewAI: {GEMINI_MODEL}")

    llm = build_llm()

    demand_agent, scheduling_agent = create_agents(llm)
    forecast_task, scheduling_task, validation_task, explanation_task = create_tasks(
        demand_agent,
        scheduling_agent
    )

    crew = Crew(
        agents=[demand_agent, scheduling_agent],
        tasks=[
            forecast_task,
            scheduling_task,
            validation_task,
            explanation_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\nCrewAI workflow completed.")
    print("-" * 60)
    print(result)

    print("\nExpected output files:")
    expected_files = [
        FORECAST_FILE,
        OUTPUTS_DIR / "forecast.csv",
        OUTPUTS_DIR / "schedule.csv",
        SCHEDULE_FILE,
        SCHEDULE_SUMMARY_FILE,
        VALIDATION_SUMMARY_FILE,
        EXPLANATION_FILE,
        OUTPUTS_DIR / "schedule_explanation.json",
    ]

    for file_path in expected_files:
        status = "FOUND" if file_path.exists() else "MISSING"
        print(f"- {status}: {file_path}")


if __name__ == "__main__":
    main()
