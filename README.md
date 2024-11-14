# AI Services Hub - README

## Overview
The **AI Services Hub** is a FastAPI-based web service designed for various AI-powered functionalities, including:
- Parsing resumes and extracting job-relevant information.
- Generating tailored summaries based on a given job description and resume.
- Matching mentor-mentee profiles based on factors like experience, semantic embedding similarity, and skill overlap.

This project leverages **Groq** for model interactions (specifically with LLaMA) and includes custom processing functions for embedding generation, skill extraction, and matching.

## Features
- **Resume Parsing and Summary Generation**: Parses resumes in PDF format and generates structured summaries and skill lists tailored to job descriptions.
- **Mentor-Mentee Matching**: Matches mentor-mentee profiles based on compatibility scores calculated from experience, skills, and semantic similarity.

## Dependencies
This project relies on several libraries, including:
- `FastAPI`, `Pydantic`, `Uvicorn`, `starlette.middleware.cors`, and `HTTPException` for API creation and middleware setup.
- `Pandas`, `NumPy`, and `Sklearn` for data manipulation and machine learning operations.
- `dotenv` for managing environment variables.
- `requests` for interacting with external APIs.
- `nest_asyncio` for enabling nested asynchronous event loops.
- `llama_parse` and `groq` for handling model-based NLP tasks.

## Environment Setup
1. **Python Version**: Ensure you have Python 3.8+ installed.
2. **Environment Variables**:
   - Create a `.env` file in the root directory.
   - Define the following environment variables:
     - `GROQ_API_KEY`: Your API key for the Groq service.
     - `LLAMA_CLOUD_API_KEY`: API key for accessing LLaMA through Groq.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
- `main.py`: Contains the FastAPI application, route definitions, and core processing classes.
- `llama_parse` and `groq`: Model interaction modules for text parsing and embedding generation.
- `Profile` and `ImprovedMentorMatchingSystem` classes: Define user profiles and mentor-mentee matching logic.

## Usage

### 1. Start the Server
To run the server on `http://localhost:8000`, execute:
```bash
uvicorn main:app --reload
```

### 2. Endpoints

#### Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Description**: Basic check for server status.
- **Response**: `{"message": "Welcome to the AI Services Hub"}`

#### `/generate_summary`
- **Method**: `POST`
- **Parameters**:
  - `user_id` (Form): The unique ID of the user.
  - `resume` (File): The PDF file containing the resume.
  - `job_description` (Form): A description of the job for tailored summary generation.
- **Description**: Accepts a resume PDF and job description, extracts content, and returns a JSON structured summary.
- **Response**: JSON object with `id`, `summary`, `experience`, and `skills` fields, based on the resume and job description.

#### `/data-to-df/`
- **Method**: `POST`
- **Parameters**: JSON payload with profile data in tabular format.
- **Description**: Converts JSON data to a DataFrame, processes profiles, and returns optimal mentor-mentee matches.
- **Response**: JSON object with `intro_text` and `pairs`, showing matched mentor-mentee pairs and other relevant information.

### 3. Classes and Functions

#### `Profile`
- **Fields**:
  - `id`: Unique identifier.
  - `experience_years`: Years of experience.
  - `summary`: Brief professional summary.
  - `job_experience`: Work experience description.
- **Methods**:
  - `__post_init__`: Initializes sets for skills, domains, and seniority keywords.

#### `ImprovedMentorMatchingSystem`
- **Purpose**: Computes scores to match profiles based on experience, semantic similarity, and skills.
- **Main Methods**:
  - `process_profile(profile)`: Processes a single profile, generating embeddings and extracting skills.
  - `calculate_match_score(mentor, mentee)`: Calculates the final match score between profiles based on weights.
  - `find_matches(profiles)`: Finds optimal matches for a list of profiles.

### 4. GroqProcessor
Responsible for generating semantic embeddings and extracting skills using the LLaMA model. Key functions:
- `generate_embedding(text)`: Generates an embedding for a text input.
- `extract_skills(text)`: Identifies skills from a predefined list based on keyword matching.

### 5. Error Handling and Logging
Errors are logged, and relevant HTTP exceptions are raised for issues such as:
- File reading and parsing errors during resume extraction.
- Failures in generating model responses.

## Example Requests
1. **Generating a Summary**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/generate_summary' \
     -F 'user_id=123' \
     -F 'resume=@/path/to/resume.pdf' \
     -F 'job_description="Data Scientist with Python and ML experience"'
   ```

2. **Matching Mentor-Mentee Profiles**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/data-to-df/' \
     -H 'Content-Type: application/json' \
     -d '{"profiles": [{"id": "1", "experience_years": 5, "summary": "...", "job_experience": "..."}, ...]}'
   ```


## Future Enhancements
Potential future improvements include:
- Enhanced skill extraction using NLP-based named entity recognition.
- Dynamic expansion of the skills keyword list based on job market trends.
- Integration with a front-end application for interactive user experiences.

## Troubleshooting
- **Environment Variables Not Loaded**: Ensure `.env` file is in the root directory and includes the required API keys.
- **File Parsing Errors**: Verify the PDF format and content readability.