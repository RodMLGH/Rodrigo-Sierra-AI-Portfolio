# ITAI-2376-workforce-scheduling-agent
This repository contains the code, synthetic data, and documentation for my ITAI 2376 Deep Learning final project.

# Workforce Scheduling Optimization — Multi-Agent System

## Project Overview

This project is an AI-based workforce scheduling optimization system for a synthetic manufacturing environment. It simulates a 24/7 production facility where employees must be assigned to shifts based on forecasted demand, skills, availability, maximum weekly hours, and operational constraints.

The system uses a two-agent architecture coordinated with CrewAI:

1. **Demand Forecasting Agent**: Predicts staffing demand using an LSTM model.
2. **Scheduling & Explanation Agent**: Creates a schedule using OR-Tools, validates the schedule, and explains the results using Gemini.

## What the Project Does

The system performs the following workflow:

1. Generates synthetic employee and demand datasets.
2. Trains an LSTM model to forecast staffing needs.
3. Uses OR-Tools to assign employees to shifts.
4. Validates schedule coverage and constraint compliance.
5. Generates a manager-friendly explanation using Gemini.
6. Coordinates the workflow through CrewAI.

## Tools and Models Used

- **CrewAI**: Multi-agent orchestration.
- **TensorFlow / Keras**: LSTM demand forecasting model.
- **Google OR-Tools**: Constraint-based scheduling optimization.
- **Google Gemini API**: Schedule explanation.
- **pandas / NumPy**: Data processing.
- **scikit-learn**: Scaling and evaluation.
- **python-dotenv**: API key management.

## Project Structure

```
workforce_scheduling_agent/
├── data/
│   ├── employees.csv
│   └── demand_history.csv
├── outputs/                              # Generated after running the workflow
│   ├── forecast.json
│   ├── forecast.csv
│   ├── schedule.json
│   ├── schedule.csv
│   ├── schedule_summary.json
│   ├── schedule_validation_summary.json
│   ├── schedule_explanation.txt
│   └── schedule_explanation.json
├── src/
│   ├── generate_data.py
│   ├── validate_data.py
│   ├── forecast_model.py
│   ├── scheduler.py
│   ├── validate_schedule.py
│   ├── explain_schedule.py
│   └── agents.py
├── workflow/
│   ├── workf_sched_optim_flow.png
├── .gitignore
├── requirements.txt
└── README.md
```

Create a local .env file in the project root. This file is not included in the repository.

## Installation

From the project root, install the required packages:
`python -m pip install -r requirements.txt`

If needed, install the packages manually:
`python -m pip install pandas numpy scikit-learn tensorflow ortools google-genai python-dotenv crewai`

This project was implemented and tested as a local Python application using a `requirements.txt` file for dependencies and a `.env` file for API credentials.

## Gemini API Setup

Create a `.env` file in the project root. This file is not included in the repository for security reasons.

GEMINI_API_KEY=your_api_key_here

GEMINI_MODEL=gemini-2.5-flash-lite

## How to Run the Project

After installing the required packages and adding a Gemini API key, the full workflow can be run from the project root using: 
`python src\agents.py`
This runs the full CrewAI workflow from Demand Forecasting Agent to Scheduling and Explanation Agent.

## Output Files

After running the workflow, the project generates the following output files:

- `outputs/forecast.json` and `outputs/forecast.csv` (staffing demand forecast)
- `outputs/schedule.json` and `outputs/schedule.csv` (optimized workforce schedule)
- `outputs/schedule_summary.json` (high-level schedule metrics)
- `outputs/schedule_validation_summary.json` (coverage and constraint validation)
- `outputs/schedule_explanation.txt` and `outputs/schedule_explanation.json` (manager-friendly explanation)

## Notes About Synthetic Data

All data is synthetic. No real employee records, phone numbers, addresses, salary data, or proprietary company information are used. Synthetic data allows the project to demonstrate the full workflow safely while avoiding privacy concerns.


## Current Limitations

This project is a proof of concept. The first version schedules only a 3-day window, uses synthetic data, and applies a simplified set of constraints. Some positions may remain unfilled if there are not enough available or qualified employees.


## Future Improvements

Future versions could expand the schedule to 7 days, activate more employee fields such as experience and certifications, include more advanced manufacturing variables, improve the LSTM model, add preference-based scheduling, and build a Streamlit interface.


## Main Demo Command

`python src\agents.py`
