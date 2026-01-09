from sentence_transformers import SentenceTransformer, util

class SkillMatcher:
    def __init__(self, threshold=0.6):
        self.model = SentenceTransformer("TechWolf/JobBERT-v2")
        self.threshold = threshold

    def normalize(self, skill):
        return skill.lower().strip()

    def similarity_match(self, jd_skills, resume_skills):
        sim_scores = util.cos_sim(jd_skills, resume_skills)
        max_score = sim_scores.max().item()
        return max_score >= self.threshold, max_score

    def compare(self, resume_skills, jd_skills):
        resume_skills = [self.normalize(s) for s in resume_skills]
        jd_skills = [self.normalize(s) for s in jd_skills]

        if not jd_skills:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "accuracy": 0,
                "similarity_scores": []
            }

        resume_embs = self.model.encode(resume_skills, convert_to_tensor=True)
        jd_embs = self.model.encode(jd_skills, convert_to_tensor=True)

        matched, missing, similarity_scores = [], [], []

        for i, jd_skill in enumerate(jd_skills):
            decision, score = self.similarity_match(jd_embs[i], resume_embs)
            if decision:
                matched.append(jd_skill)
                similarity_scores.append(score)
            else:
                missing.append(jd_skill)

        total_predictions = len(matched) + len(missing)
        accuracy = (len(matched) / total_predictions) * 100 if total_predictions else 0

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "accuracy": round(accuracy, 2),
            "similarity_scores": [round(s, 4) for s in similarity_scores]
        }
