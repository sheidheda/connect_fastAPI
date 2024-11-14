import nest_asyncio
import asyncio
import logging
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from llama_parse import LlamaParse
import os
from dotenv import load_dotenv
from groq import Groq
import json
import uvicorn
from starlette.middleware.cors import CORSMiddleware


from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Set, Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO
from fastapi import Request


# Apply nest_asyncio to allow nested event loops ~Sheidheda
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Services Hub")

app.add_middleware(
CORSMiddleware,
allow_origins=["*"], 
allow_credentials=True,
allow_methods=["*"], 
allow_headers=["*"],
)

# Set up Groq client and LLaMA model
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "llama-3.2-3b-preview"  # Specify the LLaMA model you are using

# Model input class for JSON response
class JobDescription(BaseModel):
    description: str

# =============================
# Triggerfish functions
# ==============================

@dataclass
class Profile:
    id: str
    experience_years: float
    summary: str
    job_experience: str
    extracted_skills: Set[str] = None
    industry_domains: Set[str] = None
    seniority_keywords: Set[str] = None
    semantic_embedding: np.ndarray = None

    def __post_init__(self):
        self.extracted_skills = set()
        self.industry_domains = set()
        self.seniority_keywords = set()

def create_profiles_from_dataframe(df: pd.DataFrame) -> List[Profile]:
    profiles = []
    for _, row in df.iterrows():
        profile = Profile(
            id=row['id'],
            experience_years=row['experience_years'],
            summary=row['summary'],
            job_experience=row['job_experience']
        )
        profiles.append(profile)
    return profiles


class GroqProcessor:
    def __init__(self, model_name="llama3-8b-8192", api_key=None):
        self.model = model_name
        self.api_base = "https://api.groq.com/openai/v1"
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text using Groq's Llama model."""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(
            f"{self.api_base}/embeddings",  # Adjust this endpoint as per Groq's API docs
            headers=headers,
            json={
                "model": self.model,
                "prompt": text
            }
        )
        if response.status_code == 200:
            return np.array(response.json()['embedding'])
        else:
            return np.zeros(8192)  # Fallback to zero vector with 8192 dimensions for Groq's model

    def extract_skills(self, text: str) -> Set[str]:
        """Use simple keyword matching for skills extraction."""
        # Common tech skills keywords - expand this list as needed
        common_skills = {
            'python', 'java', 'javascript', 'ml', 'ai', 'deep learning',
            'machine learning', 'data science', 'cloud', 'aws', 'azure',
            'devops', 'docker', 'kubernetes', 'sql', 'nosql', 'react',
            'angular', 'vue', 'node', 'express', 'django', 'flask'
        }
        
        text_lower = text.lower()
        return {skill for skill in common_skills if skill in text_lower}

class ImprovedMentorMatchingSystem:
    def __init__(self, 
                 experience_weight: float = 0.4,
                 semantic_weight: float = 0.3,
                 skill_weight: float = 0.3,
                 min_experience_gap: float = 1.0,
                 max_experience_gap: float = 10.0,
                 model_name: str = "llama3-8b-8192",
                 api_key: str = None):
        self.weights = {
            'experience': experience_weight,
            'semantic': semantic_weight,
            'skill': skill_weight
        }
        self.experience_gaps = {
            'min': min_experience_gap,
            'max': max_experience_gap
        }
        self.llm_processor = GroqProcessor(model_name, api_key)

    def process_profile(self, profile: Profile) -> Profile:
        """Process a single profile."""
        combined_text = f"{profile.summary} {profile.job_experience}"
        
        # Generate embedding using Groq's Llama model
        profile.semantic_embedding = self.llm_processor.generate_embedding(combined_text)
        
        # Extract skills using simple keyword matching
        profile.extracted_skills = self.llm_processor.extract_skills(combined_text)
        
        return profile

    def calculate_experience_score(self, mentor_exp: float, mentee_exp: float) -> float:
        """Calculate experience compatibility score."""
        gap = mentor_exp - mentee_exp
        
        if gap < self.experience_gaps['min'] or gap > self.experience_gaps['max']:
            return 0.0
        
        return 1.0 - (gap - self.experience_gaps['min']) / (self.experience_gaps['max'] - self.experience_gaps['min'])

    def calculate_semantic_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate semantic similarity between embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0])

    def calculate_match_score(self, mentor: Profile, mentee: Profile) -> Tuple[float, Dict]:
        """Calculate match score."""
        # Calculate experience score
        exp_score = self.calculate_experience_score(mentor.experience_years, mentee.experience_years)
        
        # Calculate semantic similarity
        sem_score = self.calculate_semantic_similarity(mentor.semantic_embedding, mentee.semantic_embedding)
        
        # Calculate skill overlap score
        skill_score = len(mentor.extracted_skills.intersection(mentee.extracted_skills)) / \
                     max(len(mentor.extracted_skills.union(mentee.extracted_skills)), 1)
        
        # Calculate weighted final score
        scores = {
            'experience': exp_score,
            'semantic': sem_score,
            'skill': skill_score
        }
        
        final_score = sum(score * self.weights[category] for category, score in scores.items())
        
        return final_score, scores

    def find_matches(self, profiles: List[Profile]) -> List[Tuple[str, str, float, Dict]]:
        """Find optimal mentor-mentee matches."""
        # Process all profiles
        processed_profiles = [self.process_profile(profile) for profile in profiles]
        
        # Calculate all possible matches
        matches = []
        used_profiles = set()
        
        # Generate all possible pairs and their scores
        pairs = []
        for i, mentor in enumerate(processed_profiles):
            for j, mentee in enumerate(processed_profiles):
                if i != j and mentor.experience_years > mentee.experience_years:
                    score, breakdown = self.calculate_match_score(mentor, mentee)
                    if score > 0:
                        pairs.append((mentor.id, mentee.id, score, breakdown))
        
        # Sort pairs by score and select best non-overlapping pairs
        pairs.sort(key=lambda x: x[2], reverse=True)
        for mentor_id, mentee_id, score, breakdown in pairs:
            if mentor_id not in used_profiles and mentee_id not in used_profiles:
                matches.append((mentor_id, mentee_id, score, breakdown))
                used_profiles.add(mentor_id)
                used_profiles.add(mentee_id)
        
        return matches

def get_matches_json(matches: List[Tuple[str, str, float, Dict]], profiles: List[Profile]) -> dict:
    """Convert matches to a JSON-compatible dictionary with the specified format."""
    intro_text = [
        f"Successfully loaded {len(profiles)} profiles from data_job_exp.csv",
        f"{len(matches)} pairs were matched"
    ]
    
    pairs = [{"mentor_id": mentor_id, "mentee_id": mentee_id} for mentor_id, mentee_id, _, _ in matches]
    
    output_data = {
        "intro_text": intro_text,
        "pairs": pairs
    }
    
    return output_data

# ==============================
# / Triggerfish Functions
# ==============================


# ==============================
# Root Endpoint
# ==============================

@app.get("/", name="Root Endpoint")
async def root():
    return {"message": "Welcome to the AI Services Hub"}


# ==============================
# Sheidheda Endpoint
# ==============================

# Endpoint to handle the resume PDF and job description
@app.post("/generate_summary")
async def generate_summary(user_id: str = Form(...), resume: UploadFile = File(...), job_description: str = Form(...)):
    
    # Step 1: Extract the resume file bytes
    try:
        resume_bytes = await resume.read()  # Get the bytes of the uploaded file
        logging.info("Resume file received, starting PDF parsing.")
        
        # Initialize the LlamaParse parser
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),  # Fetch API key from environment variables
            result_type="markdown",  # Use markdown format for the extracted content
            num_workers=4,  # Number of workers for API calls
            verbose=True,
            language="en",  # Language for the resume
        )
        
        # Use LlamaParse to process the uploaded PDF file
        extra_info = {"file_name": resume.filename}
        documents = parser.load_data(resume_bytes, extra_info=extra_info)

        # If no documents were extracted, raise an error
        if not documents:
            logging.error("Failed to extract text from the PDF.")
            raise HTTPException(status_code=400, detail="No text found in the PDF")
        
        logging.info(f"Text successfully extracted from the resume. Parsing {len(documents)} document(s).")

        # Step 2: Extract the markdown content from the documents
        resume_markdown = "\n".join([doc.text for doc in documents])

    except Exception as e:
        logging.error(f"Error during PDF extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")

    # Step 3: Prepare the prompt for the LLaMA model to generate a summary
    prompt = f"""
    Given the resume content in markdown format: 
    ```{resume_markdown}``` 
    and the job description: "{job_description}", 
    provide the following in a JSON format:
    
    {{
        "id": [{user_id}]
        "summary": ["generate 3 different detailed personless summaries tailored to fit the job description and the resume attached"],
        "experience": ["give relevant experiences based on the resume with detailed actionable kpis in that have good structure and sub-keys: Jobtitle, company, location, duration, key-responsibility in two sentences and 3 kpis as items only"],
        "skills": ["List the 10 most relevant skills for the job given the applicant's resume and the JD"]
    }}
    Make sure the summaries are concise and are at least 90% relevant to the resume and the job description.
    Make sure my output is complete
    """
    
    # Log the prompt being sent to the model
    logging.info(f"Sending prompt to Groq API: {prompt[:100]}...")  # Logging the first 100 characters of the prompt

    # Step 4: Generate response using the LLaMA model via Groq client ~Sheidheda
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.5,
            max_tokens=3024,
            stop=None
        )
        
        logging.info("Response received from Groq API.")


        # Extract the content from the response
        message_content = response.choices[0].message.content.strip()

        # Extract the JSON part wrapped inside the markdown code block
        json_str = message_content.split('```json')[1].split('```')[0].strip()

        # Convert the JSON string to a Python dictionary
        parsed_json = json.loads(json_str)
        
        return json.dumps(parsed_json, indent=4) # This will include the full response from the Groq API

    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")
    
    finally:
        # Schedule shutdown after processing the request
        asyncio.create_task(shutdown_server())




async def shutdown_server():
    """
    Function to gracefully shut down the FastAPI server after handling a request.
    """
    logging.info("Shutting down server after processing request.")
    await asyncio.sleep(1)  # Short delay before shutdown
    await app.shutdown()


# ==============================
# Triggerfish Endpoint
# ==============================

@app.post("/data-to-df/")
async def json_to_dataframe(request: Request):
    # Receive JSON blob from the request body
    json_data = await request.json()
    
    # Convert JSON data to a DataFrame
    df = pd.DataFrame(json_data)

    profiles = create_profiles_from_dataframe(df)

    matcher = ImprovedMentorMatchingSystem(
        experience_weight=0.4,
        semantic_weight=0.3,
        skill_weight=0.3,
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    matches = matcher.find_matches(profiles)

    json_data = get_matches_json(matches, profiles)
    return json_data





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
