from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from processmanager import ProcessManager
from dbmanager import DatabaseManager
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os

app = FastAPI(title="JD Resume Matching API", version="1.1")
db = DatabaseManager()
manager = ProcessManager(source="source", processed="processed")
model = SentenceTransformer("all-MiniLM-L6-v2")


class SkillInput(BaseModel):
    skills: str
    threshold: float = 0.5


@app.get("/")
def root():
    return {"message": "JD Resume Matching API is running!"}


@app.post("/process_files/")
async def process_files(files: List[UploadFile] = File(...)):
    os.makedirs("source", exist_ok=True)
    for file in files:
        path = os.path.join("source", file.filename)
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
    manager.extract_jds()
    return {"status": "success", "message": f"{len(files)} file(s) processed successfully"}


@app.get("/jobs/")
def get_all_jobs():
    df = db.fetch_jobs("Jobs")
    if df.empty:
        return {"message": "No job descriptions found"}
    return df.to_dict(orient="records")


@app.post("/match_jobs/")
def match_jobs(input: SkillInput):
    df = db.fetch_jobs("Jobs")
    if df.empty:
        return {"message": "No job descriptions found"}

    skills_text = input.skills.lower().strip()
    jd_skills = df["SKILLS"].fillna("").astype(str).tolist()

    resume_emb = model.encode([skills_text], convert_to_tensor=True)
    jd_embs = model.encode(jd_skills, convert_to_tensor=True)

    similarities = util.cos_sim(resume_emb, jd_embs)[0].cpu().numpy()
    df["SIMILARITY"] = similarities

    matched = df[df["SIMILARITY"] >= input.threshold].sort_values(by="SIMILARITY", ascending=False)
    return matched.to_dict(orient="records")

@app.get("/job_insights/locations")
def top_job_locations():
    df = db.fetch_jobs("Jobs")
    if df.empty or "JOB LOCATION" not in df.columns:
        return {"message": "No location data found"}

    top_locs = (
    df["JOB LOCATION"]
    .dropna()
    .value_counts()
    .head(5)
    .reset_index()
    )
    top_locs.columns = ["location", "count"]
    return top_locs.to_dict(orient="records")


@app.get("/job_insights/skills")
def top_skills():
    df = db.fetch_jobs("Jobs")
    if df.empty or "SKILLS" not in df.columns:
        return {"message": "No skill data found"}
    all_skills = (
        df["SKILLS"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.split(",")
    )

    skills_flat = [s.strip() for sublist in all_skills for s in sublist if s.strip()]
    skills_df = (
        pd.Series(skills_flat)
        .value_counts()
        .head(10)
        .reset_index()
        .rename(columns={"index": "skill", 0: "count"})
    )
    return skills_df.to_dict(orient="records")
