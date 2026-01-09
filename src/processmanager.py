import os
import shutil
from filereader import FileReader
from jobparser import JobParser
from featureextractor import FeatureExtractor
from dbmanager import DatabaseManager

ALLOWED_EXTS = ('.pdf', '.docx', '.csv', '.xlsx', '.txt', '.xls')

class ProcessManager:
    def __init__(self, source="source", processed="processed"):
        self.SOURCE = source
        self.PROCESSED = processed
        os.makedirs(self.SOURCE, exist_ok=True)
        os.makedirs(self.PROCESSED, exist_ok=True)
        self.db = DatabaseManager()

    def extract_jds(self):
        all_rows = []
        files = [f for f in os.listdir(self.SOURCE) if os.path.isfile(os.path.join(self.SOURCE, f))]
        if not files:
            print("No files in source folder.")
            return

        for file in files:
            filepath = os.path.join(self.SOURCE, file)
            if not file.lower().endswith(ALLOWED_EXTS):
                print(f"Skipping unsupported file: {file}")
                continue
            try:
                reader = FileReader(filepath)
                raw_text = reader.read()
                if not raw_text.strip():
                    print(f"Empty content for {file}, skipping.")
                    shutil.move(filepath, os.path.join(self.PROCESSED, file))
                    continue

                texts = [raw_text]
                if file.lower().endswith(('.csv', '.xls', '.xlsx')):
                    import pandas as pd
                    try:
                        df = pd.read_csv(filepath) if file.lower().endswith('.csv') else pd.read_excel(filepath)
                        df = df.fillna("").astype(str)
                        texts = df.agg(" ".join, axis=1).tolist()
                    except Exception:
                        texts = [raw_text]

                for idx, t in enumerate(texts, start=1):
                    parser = JobParser(t)
                    cleaned = parser.clean_text()

                    extractor = FeatureExtractor(t)
                    info = extractor.extract()

                    filename_entry = f"{file}_row{idx}" if len(texts) > 1 else file
                    row = {
                        "FILENAME": filename_entry,
                        "JOB DESCRIPTION": cleaned,
                        "COMPANY": info.get("company", "NA"),
                        "JOB ROLE": info.get("job_role", "NA"),
                        "EMPLOYMENT TYPE": info.get("employment_type", "NA"),
                        "JOB LOCATION": info.get("job_location", "NA"),
                        "EXPERIENCE": info.get("experience", "NA"),
                        "MIN EXPERIENCE": info.get("min_exp", None),
                        "MAX EXPERIENCE": info.get("max_exp", None),
                        "SKILLS": info.get("skills", "NA"),
                        "TECH SKILLS": info.get("tech_skills", "NA"),
                        "SOFT SKILLS": info.get("soft_skills", "NA"),
                        "QUALIFICATION": info.get("qualification", "NA"),
                        "WORK MODE": info.get("work_mode", "NA"),
                        "SALARY": info.get("salary", "NA"),
                        "JOB TYPE": info.get("job_type", "NA"),
                        "RESPONSIBILITIES": info.get("responsibilities", "NA")
                    }
                    all_rows.append(row)

                shutil.move(filepath, os.path.join(self.PROCESSED, file))
                print(f"Processed and moved to processed folder: {file}")

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        if all_rows:
            self.db.insert_jobs(all_rows, "Jobs")
        else:
            print("No JD rows to insert.")
