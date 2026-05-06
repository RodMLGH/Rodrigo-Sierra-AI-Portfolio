# Rodrigo-Sierra-AI-Resources-Portfolio
This repository is for storing the documentation for my ITAI2277 AI Resources class. 

# AI-Powered Integrated Recruitment System

## Project Overview
This project is an AI-powered integrated recruitment system designed to reduce manual resume screening and improve consistency in candidate evaluation. The system uses n8n as the workflow orchestrator to connect Gmail, Google Drive, Airtable, Google Calendar, and OpenAI GPT models.

The system receives resumes by email, extracts candidate information, analyzes resumes using an LLM, matches candidates to job openings, sends questionnaires, evaluates responses, and shortlists candidates who meet the required score threshold.

## Problem Statement
Recruiters often receive many resumes for a single job opening. Manually reviewing each resume is time-consuming, inconsistent, and prone to human error. Resumes are also unstructured documents, which makes it difficult to compare candidates fairly.

This project addresses that problem by automating resume intake, candidate analysis, job matching, questionnaire evaluation, and shortlisting.

## Solution Summary
The solution is divided into two main workflows:

1. Resume Screening and Candidate Profiling
   - Triggered when a candidate sends a resume by email.
   - Extracts resume attachments.
   - Processes and validates resume text.
   - Uses an AI model to extract structured candidate information.
   - Stores candidate records and logs in Airtable.

2. Interview Evaluation and Final Decision
   - Triggered when a candidate submits questionnaire responses.
   - Retrieves the candidate record from Airtable.
   - Uses an AI model to evaluate the response.
   - Calculates the candidate score.
   - Updates the candidate status.
   - Sends notifications and supports interview scheduling.

## Technologies Used
- n8n: Workflow orchestration
- Gmail API: Email trigger and candidate communication
- Google Drive: Resume storage
- Airtable: Candidate database, logs, job data, and recruiter dashboard
- OpenAI GPT-4o / GPT-4o-mini: Resume analysis, questionnaire generation, and response evaluation
- Google Calendar: Interview scheduling

## AI Areas Integrated
- Natural Language Processing: Resume parsing, skill extraction, response evaluation
- Deep Learning: Use of transformer-based GPT models
- Machine Learning Concepts: Candidate scoring, ranking, threshold-based shortlisting
- Systems Integration: Connecting multiple cloud services through n8n

## Repository Contents
- `/screenshots`: System screenshots for documentation
- `/docs`: Presentation
- `/sample-data`: Sample resume and response files for testing

## How to Test the Project
1. Import the workflow JSON files into n8n.
2. Configure the required credentials:
   - Gmail
   - Google Drive
   - Airtable
   - OpenAI API
   - Google Calendar
3. Prepare the Airtable base with the required tables:
   - Analyzed CVs
   - AI_Logs
   - Job Openings
4. Send a test email with a sample resume attachment.
5. Confirm that the workflow extracts and analyzes the resume.
6. Check Airtable to confirm that a candidate record was created.
7. Send a sample questionnaire response.
8. Confirm that the second workflow evaluates the response and updates the candidate status.

## Important Notes
This project does not train a custom machine learning model. Instead, it integrates existing AI models into a functional recruitment workflow. The goal is to demonstrate practical AI deployment, automation, and system integration.

The AI supports candidate shortlisting, but the final hiring decision remains with human recruiters.

## Limitations
- Requires active API credentials and internet access.
- AI output quality depends on the model response.
- Current input method is email-based.
- Final hiring decisions require human review.

## Future Improvements
- Candidate web portal
- Recruiter analytics dashboard
- Multilingual resume support
- More advanced reporting
- Integration with HR management systems
