import pandas as pd
from sentence_transformers import SentenceTransformer, util

class JobFilter:
    def __init__(self, db_manager, model_name="all-MiniLM-L6-v2", threshold=0.5):
        self.db = db_manager
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def _load_data(self):
        df = self.db.fetch_jobs("Jobs")
        if df is None:
            df = pd.DataFrame()
        return df

    def filter_by_experience(self, df, user_exp):
        if df.empty:
            return df
        for col in ["MIN EXPERIENCE", "MAX EXPERIENCE"]:
            if col not in df.columns:
                df[col] = None
        df["MIN EXPERIENCE"] = pd.to_numeric(df["MIN EXPERIENCE"], errors="coerce").fillna(0).astype(int)
        df["MAX EXPERIENCE"] = pd.to_numeric(df["MAX EXPERIENCE"], errors="coerce").fillna(100).astype(int)
        return df[(df["MIN EXPERIENCE"] <= user_exp) & (df["MAX EXPERIENCE"] >= user_exp)]

    def filter_by_skills(self, df, user_skills):
        if df.empty:
            return df
        user_sentence = " ".join(user_skills) if isinstance(user_skills, (list, tuple)) else str(user_skills)
        if not user_sentence.strip():
            return df

        user_emb = self.model.encode(user_sentence, convert_to_tensor=True)
        matched_rows = []
        for _, row in df.iterrows():
            jd_skills = str(row.get("TECH SKILLS", "") or row.get("SKILLS", "") or "")
            if jd_skills.strip().upper() in ("", "NA"):
                continue
            jd_sentence = "Job requires skills: " + jd_skills
            jd_emb = self.model.encode(jd_sentence, convert_to_tensor=True)

            sim = float(util.cos_sim(user_emb, jd_emb).item())
            if sim >= self.threshold:
                row["SIMILARITY"] = round(sim, 4)
                matched_rows.append(row)

        if not matched_rows:
            return pd.DataFrame()

        return pd.DataFrame(matched_rows)

    def filter_jobs(self, user_exp, user_skills):
        df = self._load_data()
        if df.empty:
            print("[JobFilter] No jobs in DB.")
            return pd.DataFrame()

        exp_filtered = self.filter_by_experience(df, user_exp)
        if exp_filtered.empty:
            print("[JobFilter] No jobs matched experience filters.")
            return pd.DataFrame()

        final = self.filter_by_skills(exp_filtered, user_skills)
        if final.empty:
            print("[JobFilter] No jobs matched skills filters.")
            return pd.DataFrame()

        return final.sort_values(by="SIMILARITY", ascending=False)[["FILENAME", "COMPANY", "JOB ROLE", "TECH SKILLS", "SIMILARITY"]]
