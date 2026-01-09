import os
import re
import math
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from filereader import FileReader
from featureextractor import FeatureExtractor
from sqlalchemy import text


class JDResumeEvaluator:
    def __init__(self, db, resume_folder="resumes", threshold=0.15):
        self.db = db
        self.resume_folder = resume_folder
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def _ensure_resumes_schema(self):
        sql = """
        IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_JDResumeComparison_Resumes')
            ALTER TABLE dbo.JDResumeComparison DROP CONSTRAINT FK_JDResumeComparison_Resumes;

        IF OBJECT_ID('dbo.Resumes','U') IS NOT NULL
            DROP TABLE dbo.Resumes;

        CREATE TABLE dbo.Resumes (
            ResumeID INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            FILENAME NVARCHAR(255) NOT NULL,
            CANDIDATE_NAME NVARCHAR(255) NULL,
            SKILLS NVARCHAR(MAX) NULL,
            EXPERIENCE NVARCHAR(255) NULL,
            EDUCATION NVARCHAR(255) NULL
        );
        """
        with self.db.engine.begin() as conn:
            conn.exec_driver_sql(sql)

    def _ensure_jdresumecomparison_schema(self):
        try:
            self.db._ensure_jobs_schema("Jobs")
        except Exception:
            pass

        sql = """
        IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_JDResumeComparison_Jobs')
            ALTER TABLE dbo.JDResumeComparison DROP CONSTRAINT FK_JDResumeComparison_Jobs;
        IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_JDResumeComparison_Resumes')
            ALTER TABLE dbo.JDResumeComparison DROP CONSTRAINT FK_JDResumeComparison_Resumes;
        IF OBJECT_ID('dbo.JDResumeComparison','U') IS NOT NULL
            DROP TABLE dbo.JDResumeComparison;

        CREATE TABLE dbo.JDResumeComparison (
            JDID INT NOT NULL,
            RESUMEID INT NOT NULL,
            COSINESIMILARITY FLOAT NULL
        );

        ALTER TABLE dbo.JDResumeComparison ADD CONSTRAINT FK_JDResumeComparison_Jobs
            FOREIGN KEY (JDID) REFERENCES dbo.Jobs(ID);
        ALTER TABLE dbo.JDResumeComparison ADD CONSTRAINT FK_JDResumeComparison_Resumes
            FOREIGN KEY (RESUMEID) REFERENCES dbo.Resumes(ResumeID);
        """
        with self.db.engine.begin() as conn:
            conn.exec_driver_sql(sql)

    def _ensure_jd_skill_weights_schema(self):
        try:
            self.db._ensure_jobs_schema("Jobs")
        except Exception:
            pass

        sql = """
        IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_JDSkillWeights_Jobs')
            ALTER TABLE dbo.JDSkillWeights DROP CONSTRAINT FK_JDSkillWeights_Jobs;
        IF OBJECT_ID('dbo.JDSkillWeights','U') IS NOT NULL
            DROP TABLE dbo.JDSkillWeights;

        CREATE TABLE dbo.JDSkillWeights (
            JOBID INT NOT NULL,
            [EXTRACTED SKILLS] NVARCHAR(MAX) NULL,
            [SKILL COUNT] NVARCHAR(MAX) NULL,
            [SKILL WEIGHT] NVARCHAR(MAX) NULL,
            [IDF] NVARCHAR(MAX) NULL,
            [TF-IDF] NVARCHAR(MAX) NULL
        );

        ALTER TABLE dbo.JDSkillWeights ADD CONSTRAINT FK_JDSkillWeights_Jobs
            FOREIGN KEY ([JOBID]) REFERENCES dbo.Jobs(ID);
        """
        with self.db.engine.begin() as conn:
            conn.exec_driver_sql(sql)

    def create_resumes_table(self):
        resumes = []
        files = sorted([
            f for f in os.listdir(self.resume_folder)
            if os.path.isfile(os.path.join(self.resume_folder, f))
        ])

        for file in files:
            filepath = os.path.join(self.resume_folder, file)
            reader = FileReader(filepath)
            raw_text = reader.read()

            extractor = FeatureExtractor(raw_text)
            info = extractor.extract()

            resumes.append({
                "FILENAME": file,
                "CANDIDATE_NAME": info.get("name", "NA"),
                "SKILLS": info.get("skills", "NA"),
                "EXPERIENCE": info.get("experience", "NA"),
                "EDUCATION": info.get("education", "NA")
            })

        df = pd.DataFrame(resumes)
        self._ensure_resumes_schema()
        with self.db.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM dbo.Resumes")
        df.to_sql("Resumes", self.db.engine, if_exists="append", index=False)
        print(f"Created/Updated 'Resumes' table with {len(df)} entries.")
        return df

    def create_skill_master_table(self, jobs_df, resumes_df):
        all_skills = set()

        for col in ["SKILLS", "TECH SKILLS", "SOFT SKILLS"]:
            if col in jobs_df.columns:
                all_skills.update(
                    [s.strip().lower() for s in ",".join(jobs_df[col].dropna()).split(",") if s.strip()]
                )

        if "SKILLS" in resumes_df.columns:
            all_skills.update(
                [s.strip().lower() for s in ",".join(resumes_df["SKILLS"].dropna()).split(",") if s.strip()]
            )

        skill_records = []
        for skill in sorted(all_skills):
            skill_type = "technical" if any(
                kw in skill for kw in ["python", "java", "sql", "cloud", "react", "ai", "data", "ml"]
            ) else "non-technical"
            skill_records.append({"SkillName": skill, "SkillType": skill_type})

        df = pd.DataFrame(skill_records).reset_index().rename(columns={"index": "SkillID"})
        df.to_sql("SkillMaster", self.db.engine, if_exists="replace", index=False)
        print(f"Created/Updated 'SkillMaster' with {len(df)} unique skills.")
        return df

    def create_comparison_table(self):
        jobs_all = self.db.fetch_jobs("Jobs")
        if not {"ID", "SKILLS"}.issubset(set(jobs_all.columns)):
            print("Missing required columns 'ID' or 'SKILLS' in Jobs table.")
            return pd.DataFrame()

        jobs_df = jobs_all[["ID", "SKILLS"]].fillna("")
        resumes_df = pd.read_sql("SELECT ResumeID, SKILLS FROM dbo.Resumes", self.db.engine)

        if jobs_df.empty or resumes_df.empty:
            print("No jobs or resumes to compare.")
            return pd.DataFrame()

        jd_embs = self.model.encode(jobs_df["SKILLS"].astype(str).tolist(), convert_to_tensor=True)
        res_embs = self.model.encode(resumes_df["SKILLS"].astype(str).tolist(), convert_to_tensor=True)
        sim_matrix = util.cos_sim(jd_embs, res_embs)

        results = []
        for j_idx, j_row in jobs_df.iterrows():
            for r_idx, r_row in resumes_df.iterrows():
                sim = float(sim_matrix[j_idx, r_idx])
                results.append({
                    "JDID": int(j_row["ID"]),
                    "RESUMEID": int(r_row["ResumeID"]),
                    "COSINESIMILARITY": round(sim, 3)
                })

        df = pd.DataFrame(results)
        self._ensure_jdresumecomparison_schema()
        with self.db.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM dbo.JDResumeComparison")
        df.to_sql("JDResumeComparison", self.db.engine, if_exists="append", index=False)
        print(f"Created 'JDResumeComparison' with {len(df)} entries.")
        return df

    def create_jd_skill_weights_table(self):
        jobs_df = self.db.fetch_jobs("Jobs").fillna("")
        if "ID" not in jobs_df.columns:
            raise KeyError("Jobs table must contain 'ID' column.")

        job_count = len(jobs_df)
        df_counter = {}
        job_stats = {}

        for _, row in jobs_df.iterrows():
            job_id = int(row["ID"])
            jd_text = str(row.get("JOB DESCRIPTION", "")).lower()
            skills_raw = str(row.get("SKILLS", "")).lower()

            skills_pretty = [s.strip() for s in skills_raw.split(",") if s.strip()]
            if not skills_pretty:
                continue

            seen = set()
            skills_norm = []
            for s in skills_pretty:
                if s not in seen:
                    seen.add(s)
                    skills_norm.append(s)

            embeddings = self.model.encode(skills_norm, convert_to_tensor=True)
            sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

            if len(skills_norm) > 1:
                weights = ((sim_matrix.sum(axis=1) - 1.0) / (len(skills_norm) - 1)).tolist()
            else:
                weights = [1.0]
            min_w, max_w = min(weights), max(weights)
            denom = (max_w - min_w) if max_w != min_w else 1.0
            normalized = [round((w - min_w) / denom, 4) for w in weights]
            weight_map, tf_map, norm_map = {}, {}, {}
            for pretty, norm, w in zip(skills_pretty, skills_norm, normalized):
                pattern = r"(?<!\w)" + re.escape(norm.lower()) + r"(?!\w)"
                tf_val = len(re.findall(pattern, jd_text.lower(), flags=re.IGNORECASE))
                if tf_val > 0:
                    df_counter.setdefault(norm, set()).add(job_id)
                weight_map[pretty] = w
                tf_map[pretty] = tf_val
                norm_map[pretty] = norm

            job_stats[job_id] = {
                "skills_pretty": skills_pretty,
                "weight": weight_map,
                "tf": tf_map,
                "norm_map": norm_map
            }
        idf_global = {}
        for norm_skill, jobs_with_skill in df_counter.items():
            df_n = len(jobs_with_skill)
            idf_global[norm_skill] = round(math.log((1 + job_count) / (1 + df_n)) + 1, 6)

        records = []
        for job_id, stats in job_stats.items():
            idf_map = {pretty: idf_global.get(stats["norm_map"][pretty], 0.0) for pretty in stats["skills_pretty"]}
            tfidf_map = {
                pretty: round(stats["tf"][pretty] * idf_map[pretty], 6)
                for pretty in stats["skills_pretty"]
            }
            records.append({
                "JOBID": job_id,
                "EXTRACTED SKILLS": ", ".join(stats["skills_pretty"]),
                "SKILL COUNT": json.dumps(stats["tf"], ensure_ascii=False),
                "SKILL WEIGHT": json.dumps(stats["weight"], ensure_ascii=False),
                "IDF": json.dumps(idf_map, ensure_ascii=False),
                "TF-IDF": json.dumps(tfidf_map, ensure_ascii=False)
            })

        df = pd.DataFrame(records)
        self._ensure_jd_skill_weights_schema()
        with self.db.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM dbo.JDSkillWeights")
        df.to_sql("JDSkillWeights", self.db.engine, if_exists="append", index=False)
        print(f"Created 'JDSkillWeights' with {len(df)} records.")
        return df

    def run_full_pipeline(self):
        print("\nStarting JD–Resume Evaluation Pipeline...\n")
        resumes_df = self.create_resumes_table()
        jobs_df = self.db.fetch_jobs("Jobs")
        self.create_skill_master_table(jobs_df, resumes_df)
        self.create_comparison_table()
        self.create_jd_skill_weights_table()
        print("\nAll tables created successfully.\n")
