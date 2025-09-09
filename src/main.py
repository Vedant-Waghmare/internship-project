import os
import shutil
import pandas as pd
from filereader import FileReader
from jobparser import JobParser
from csvexporter import CSVExporter
from featureextractor import FeatureExtractor

SOURCE = "source"
OUTPUT = "destination/jobs.csv"
PROCESSED = "processed"

if not os.path.exists(PROCESSED):
    os.makedirs(PROCESSED, exist_ok=True)

already_processed = set()
if os.path.exists(OUTPUT):
    try:
        df_existing = pd.read_csv(OUTPUT)
        if "FILENAME" in df_existing.columns:
            already_processed = set(df_existing["FILENAME"].dropna().astype(str))
    except Exception as e:
        print(f"Error reading existing CSV: {e}")

all_data = []

for file in os.listdir(SOURCE):
    filepath = os.path.join(SOURCE, file)
    if file in already_processed:
        print(f"Skipping {file} (already processed)")
        continue

    if os.path.isfile(filepath) and file.lower().endswith(('.pdf', '.docx', '.csv', '.xlsx', '.txt', '.xls')):
        try:
            # Read
            reader = FileReader(filepath)
            raw_text = reader.read()

            job_texts = [raw_text]
            if file.lower().endswith(('.csv', '.xls', '.xlsx')):
                df = pd.read_csv(filepath) if file.lower().endswith('.csv') else pd.read_excel(filepath)
                job_texts = df.fillna("").astype(str).agg(" ".join, axis=1).tolist()

            for idx, text in enumerate(job_texts, start=1):
                #Cleaning
                parser = JobParser(text)
                clean_text = parser.clean_text()

                # Extract
                extractor = FeatureExtractor(text)
                info = extractor.extract()

                # Append
                filename_entry = f"{file}_row{idx}" if len(job_texts) > 1 else file
                all_data.append({
                    "FILENAME": filename_entry,
                    #"JOB DESCRIPTION": clean_text,
                    "COMPANY": info["company"],
                    "JOB ROLE": info["job_role"],
                    "EMPLOYMENT TYPE": info.get("employment_type", "NA"),
                    "JOB LOCATION": info["job_location"],
                    "EXPERIENCE": info["experience"],
                    "MIN EXPERIENCE": info["min_exp"],
                    "MAX EXPERIENCE": info["max_exp"],
                    "SKILLS": info["skills"],
                    "TECH SKILLS": info.get("tech_skills", "NA"),
                    "SOFT SKILLS": info.get("soft_skills", "NA"),
                    "QUALIFICATION": info["qualification"],
                    "WORK MODE": info["work_mode"],
                    "SALARY": info["salary"],
                    "JOB TYPE": info["job_type"],
                    "RESPONSIBILITIES": info["responsibilities"]
                })

            # Move processed file
            shutil.move(filepath, os.path.join(PROCESSED, file))
            print(f"Processed: {file} & moved to processed folder")

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Exporting to CSV
exporter = CSVExporter(OUTPUT)
exporter.save(all_data)
print(f"\nAll files processed. Output saved to {OUTPUT}")
