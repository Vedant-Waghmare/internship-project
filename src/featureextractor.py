import re
import os
import spacy
import pandas as pd
from transformers import pipeline

nlp = spacy.load("en_core_web_lg")

class FeatureExtractor:

    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    LEGAL_SUFFIXES = ["Ltd", "Limited", "Pvt", "LLC", "Inc", "Corporation", "Technologies", "Company", "Enterprises"]

    def __init__(self, text):
        self.text = text
        self.doc = nlp(text)
        
        self.dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "datasets",
            "skills_en.csv"
        )

        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            self.skill_keywords = df["skill"].dropna().astype(str).tolist()
            self.tech_keywords = df[df["type"].str.lower() == "tech"]["skill"].tolist()
            self.soft_keywords = df[df["type"].str.lower() == "soft"]["skill"].tolist()
        else:
            self.skill_keywords, self.tech_keywords, self.soft_keywords = [], [], []

    def extract(self):
        company = self.extract_company()
        role = self.extract_jobrole()
        employment_type = self.extract_employmenttype()
        location = self.extract_location()
        experience = self.extract_experience()
        min_exp, max_exp = self.extract_experience_range(experience)
        skills = self.extract_skills()
        qualification = self.extract_qualification()
        work_mode = self.extract_workmode()
        salary = self.extract_salary()
        job_type = self.extract_jobtype()
        responsibilities = self.extract_responsibilities()
        tech_skills = self.extract_techskills()
        soft_skills = self.extract_softskills()

        return {
            "company": company or "",
            "job_role": role or "",
            "employment_type": employment_type,
            "job_location": location if location else "",
            "experience": experience or "",
            "min_exp": min_exp if min_exp is not None else "NA",
            "max_exp": max_exp if max_exp is not None else "NA",
            "skills": ", ".join(skills) if isinstance(skills, list) else skills,
            "qualification": qualification or "",
            "work_mode": work_mode or "",
            "salary": salary or "",
            "job_type": job_type or "",
            "responsibilities": responsibilities or "",
            "tech_skills": tech_skills,
            "soft_skills": soft_skills
        }

    def extract_company(self):
        ner_results = FeatureExtractor.ner_model(self.text)
        org_candidates = [ent["word"].strip() for ent in ner_results if ent["entity_group"] == "ORG"]

        org_candidates = list(set(org_candidates))
        org_candidates = sorted(org_candidates, key=len, reverse=True)
        org_candidates = [c for c in org_candidates if len(c) > 2 and c.lower() not in ["hr", "recruitment", "team"]]

        for c in org_candidates:
            if any(suffix in c for suffix in FeatureExtractor.LEGAL_SUFFIXES):
                return c

        if org_candidates:
            return org_candidates[0]

        spacy_orgs = [ent.text for ent in self.doc.ents if ent.label_ == "ORG"]
        if spacy_orgs:
            return max(spacy_orgs, key=len).strip()

        match = re.search(r"(?:Company|Organization|Employer)[:\-]\s*([A-Za-z&.,\s]+)", self.text, flags=re.I)
        if match:
            return match.group(1).strip()

        match2 = re.search(r"\b([A-Z][A-Za-z& ]+(?:Ltd|Limited|Pvt|Corporation|Inc|Company))\b", self.text)
        if match2:
            return match2.group(1).strip()

        blacklist = ["Job Description", "Job Role", "Role", "Position", "Responsibilities"]
        match3 = re.findall(r"\b([A-Z][A-Za-z& ]{2,})\b", self.text)
        for candidate in match3:
            if candidate not in blacklist and len(candidate.split()) > 1:
                return candidate.strip()

        return "NA"

    def extract_jobrole(self):
        text = self.text
        if not text:
            return "NA"
        pattern = r"(?i)(?:Job Title|Role|Designation|Position|Title)[:\-]?\s*(.*?)(?:\n|$)"
        match = re.search(pattern, text)
        if match:
            role_text = match.group(1).strip()
            role_text = re.sub(r"^(we are (looking|hiring)|looking for|hiring for|openings for)\s+", "", role_text, flags=re.I)
            role_text = re.sub(r"^(Role|Designation|Position)\s*[:\-]?\s*", "", role_text, flags=re.I)
            role_text = re.split(r"[,\-;]", role_text)[0].strip()
            return role_text[:100]
        keywords = r"(Engineer|Analyst|Developer|Manager|Scientist|Architect|Intern|Trainee|Specialist)"
        match2 = re.search(rf"([A-Z][A-Za-z\s]+{keywords}[A-Za-z\s]*)", text)
        if match2:
            return match2.group(1).strip()
        return "NA"

    def extract_employmenttype(self):
        text = self.text.lower()

        if re.search(r"\bintern(ship)?\b", text):
            return "Internship"
        elif re.search(r"\bfull[- ]?time\b", text):
            return "Full-time"
        elif re.search(r"\bpart[- ]?time\b", text):
            return "Part-time"
        elif re.search(r"\bcontract\b", text):
            return "Contract"
        elif re.search(r"\bfresher\b", text):
            return "Fresher"
        return "NA"

    def extract_location(self):
        text = self.text
        text_lower = text.lower()

        location_patterns = [
            r"(?:\blocation|work\s+location|office\s+location|job\s+location|based\s+in|city|workplace|office)"
            r":?\s*([A-Za-z\s]+?)(?:\s|$|,|;|\n|\r)"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1).strip()
                if len(location) > 3:
                    return location.title()

        remote_patterns = [
            r'\b(remote|work from home|wfh|telecommute|anywhere in the world)\b'
        ]
        for pattern in remote_patterns:
            if re.search(pattern, text_lower):
                return "Remote"
        p_patterns = [
            r'\bpan\b'
        ]
        for pattern in p_patterns:
            if re.search(pattern, text_lower):
                return "Pan"
        for ent in self.doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                return ent.text
        return "NA"

    def extract_experience(self):
        text = self.text

        regex_patterns = [
            r'(\d+\s*(?:-|to)\s*\d+\s*(?:years?|yrs?))',
            r'(\d+\+?\s*(?:years?|yrs?))',
            r'minimum\s+of\s+(\d+\s*(?:years?|yrs?))',
            r'upto\s+(\d+\s*(?:years?|yrs?))'
        ]

        for pattern in regex_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    def extract_experience_range(self, experience):
        if not experience or experience == "NA":
            return "NA", "NA"
        if experience.lower() == "fresher":
            return 0, 0

        match = re.match(r'(\d+)\s*(?:-|to)\s*(\d+)', experience)
        if match:
            return int(match.group(1)), int(match.group(2))

        match = re.match(r'(\d+)', experience)
        if match:
            val = int(match.group(1))
            return val, val
        return "NA", "NA"

        if re.search(r"\bfresher(s)?\b", text, re.IGNORECASE):
            return "Fresher"
        return "NA"

    def extract_skills(self):
        if not self.skill_keywords:
            return []

        text_lower = self.text.lower()
        found_skills = [k for k in self.skill_keywords if k.lower() in text_lower]

        return list(set(found_skills)) if found_skills else []

    def extract_techskills(self):
        all_skills = self.extract_skills()
        found = [s for s in all_skills if s in self.tech_keywords]
        return ", ".join(found) if found else "NA"

    def extract_softskills(self):
        all_skills = self.extract_skills()
        found = [s for s in all_skills if s in self.soft_keywords]
        return ", ".join(found) if found else "NA"

    def extract_qualification(self):
        text = self.text

        degree_pattern = r"\b(?:B\.?\s?E\.?|B\.?\s?Tech|M\.?\s?Tech|B\.?\s?Sc|M\.?\s?Sc|MBA|PGDM|Ph\.?\s?D|Diploma|B\.?\s?Com|M\.?\s?Com|CA|Bachelor|Master)\b"
        degree_match = re.search(degree_pattern, text, re.I)
        degree = degree_match.group() if degree_match else None

        stream_pattern = r"\b(?:Computer|Information|Electronics|Mechanical|Civil|Data Science|Biotechnology|IT|CS|Engg)\b"
        stream_match = re.search(stream_pattern, text, re.I)
        stream = stream_match.group() if stream_match else None

        if degree and stream:
            return f"{degree} in {stream}"
        elif degree:
            return degree
        else:
            return "NA"

    def extract_workmode(self):
        text = self.text
        if re.search(r"remote|work from home", text, re.I):
            return "Remote"
        if re.search(r"hybrid", text, re.I):
            return "Hybrid"
        if re.search(r"onsite|on-site|office", text, re.I):
            return "Onsite"
        return "NA"

    def extract_salary(self):
        text = self.text
        regex_patterns = [
            r'\b(?:₹|INR)\s?\d{1,3}(?:,\d{3})*(?:\s*(?:-|\sto)\s*(?:₹|INR)?\d{1,3}(?:,\d{3})*)?\s*(?:lpa|lakhs?|per\s*annum|pa|per\s*month)?\b',
            r'\b\d+(?:\.\d+)?\s*(?:-|\sto)\s*\d+(?:\.\d+)?\s*(?:lpa|lakhs?|per\s*annum|pa|per\s*month)\b',
            r'\$\s?\d+(?:,\d+)*(?:\s*-\s*\$?\d+(?:,\d+)*)?\s*(?:usd|per\s*month|per\s*annum)\b',
            r'\b\d+(?:\.\d+)?\s*(?:lpa|lakhs?|per\s*annum|pa|per\s*month)\b'
        ]

        text_lower = text.lower()
        for pattern in regex_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return re.sub(r'[^0-9a-zA-Z\s\-\.,₹$]', '', match.group(0)).strip()

        for ent in self.doc.ents:
            if ent.label_ == "MONEY" and re.search(r'\d', ent.text):
                return re.sub(r'[^0-9a-zA-Z\s\-\.,₹$]', '', ent.text).strip()

        return "NA"

    def extract_jobtype(self):
        text = self.text
        if re.search(r"developer|engineer|scientist|architect|analyst|security|ai|ml|cloud|data", text, re.I):
            return "Tech"
        if re.search(r"hr|sales|marketing|finance|operations|account|trainee|manager", text, re.I):
            return "Non-Tech"
        return "NA"

    def extract_responsibilities(self, max_chars=300):
        text = self.text
        pattern = r"(?i)(Responsibilities|Key Responsibilities|Roles and Responsibilities|Duties|What You'll Do|Your Role|Tasks|Job Duties|Role and Responsibilities|Your Responsibilities)[:\-]?\s*(.*?)(?=\n\s*\n|Requirements|Qualifications|Skills|Experience|Eligibility|Benefits|How to Apply|About|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            res_text = " ".join([m[1].strip() for m in matches])
            res_text = re.sub(r'\n{2,}', '\n', res_text).strip()
            if len(res_text) > max_chars:
                res_text = res_text[:max_chars].rstrip() + "..."
            return res_text

        return "NA"