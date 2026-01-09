import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os

API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Job Recommender", layout="wide")

# Dark mode styling
dark_mode = st.toggle("Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #0e1117; color: #fafafa; }
        .stTextInput, .stTextArea, .stSelectbox, .stButton>button {
            background-color: #262730 !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

st.title("Job Recommendation System")
st.write("Enter your skills and explore insights from job descriptions in your database.")

# ==========================
# 📁 FILE UPLOAD SECTION
# ==========================
st.subheader("Upload and Process New Job Description Files")
uploaded_files = st.file_uploader(
    "Upload your Job Description files",
    type=["pdf", "docx", "csv", "xlsx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Process Uploaded Files"):
        with st.spinner("Processing job descriptions..."):
            files = [("files", (f.name, f, f.type)) for f in uploaded_files]
            res = requests.post(f"{API_URL}/process_files/", files=files)
        if res.status_code == 200:
            st.success(res.json().get("message", "Files processed successfully!"))
        else:
            st.error("Error processing files.")


# ==========================
# 🔍 SKILL MATCHING SECTION
# ==========================
st.subheader("Enter Your Skills")
skills_input = st.text_area("Enter your skills:")

threshold = st.slider("Minimum Similarity Threshold", 0.0, 1.0, 0.5, 0.05)

st.subheader("Filter Jobs")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    location_filter = st.text_input("Location Filter (e.g., Pune, Remote)").strip()
with col2:
    experience_filter = st.selectbox("Experience Level", ["All", "0-2", "3-5", "6-10", "10+"])
with col3:
    work_mode_filter = st.selectbox("Work Mode", ["All", "Remote", "Onsite", "Hybrid"])
with col4:
    job_type_filter = st.selectbox("Job Type", ["All", "Tech", "Non-Tech"])
with col5:
    employment_type_filter = st.selectbox(
        "Employment Type", ["All", "Internship", "Full-time", "Part-time", "Contract", "Fresher"]
    )

if st.button("Find Matching Jobs"):
    if not skills_input.strip():
        st.warning("Please enter some skills first.")
    else:
        with st.spinner("Fetching matching jobs..."):
            res = requests.post(
                f"{API_URL}/match_jobs/",
                json={"skills": skills_input, "threshold": threshold}
            )

        if res.status_code != 200:
            st.error("Error contacting API.")
        else:
            data = res.json()
            if not data or (isinstance(data, dict) and "message" in data):
                st.warning("No matching jobs found.")
            else:
                jobs_df = pd.DataFrame(data)

                if location_filter:
                    jobs_df = jobs_df[
                        jobs_df["JOB LOCATION"].str.contains(location_filter, case=False, na=False)
                    ]

                if experience_filter != "All":
                    def match_exp(min_exp, max_exp, filt):
                        if filt == "0-2":
                            return max_exp <= 2
                        if filt == "3-5":
                            return (min_exp >= 3 and max_exp <= 5)
                        if filt == "6-10":
                            return (min_exp >= 6 and max_exp <= 10)
                        if filt == "10+":
                            return min_exp >= 10
                        return True
                    jobs_df = jobs_df[
                        jobs_df.apply(lambda r: match_exp(r["MIN EXPERIENCE"], r["MAX EXPERIENCE"], experience_filter), axis=1)
                    ]

                if work_mode_filter != "All":
                    jobs_df = jobs_df[
                        jobs_df["WORK MODE"].str.contains(work_mode_filter, case=False, na=False)
                    ]

                if job_type_filter != "All":
                    jobs_df = jobs_df[
                        jobs_df["JOB TYPE"].str.contains(job_type_filter, case=False, na=False)
                    ]

                if employment_type_filter != "All":
                    jobs_df = jobs_df[
                        jobs_df["EMPLOYMENT TYPE"].str.contains(employment_type_filter, case=False, na=False)
                    ]

                jobs_df = jobs_df.sort_values(by="SIMILARITY", ascending=False)

                if jobs_df.empty:
                    st.warning("No suitable jobs found with the selected filters.")
                else:
                    st.success(f"Found {len(jobs_df)} matching jobs!")
                    display_columns = [
                        "ID", "FILENAME", "COMPANY", "JOB ROLE",
                        "JOB LOCATION", "EXPERIENCE", "WORK MODE",
                        "EMPLOYMENT TYPE", "JOB TYPE", "SIMILARITY", "SKILLS"
                    ]
                    display_columns = [c for c in display_columns if c in jobs_df.columns]
                    st.dataframe(jobs_df[display_columns], use_container_width=True)
                    st.download_button(
                        label="⬇ Download Results as CSV",
                        data=jobs_df[display_columns].to_csv(index=False).encode("utf-8"),
                        file_name="matched_jobs.csv",
                        mime="text/csv"
                    )


# ==========================
# 📊 VISUALIZATION SECTION
# ==========================
st.divider()
st.subheader("📈 Job Insights Dashboard")

if st.button("Show Job Insights"):
    col1, col2 = st.columns(2)

    with st.spinner("Loading analytics..."):
        # --- Top Locations ---
        res_loc = requests.get(f"{API_URL}/job_insights/locations")
        if res_loc.status_code == 200:
            loc_data = res_loc.json()
            if loc_data and isinstance(loc_data, list):
                loc_df = pd.DataFrame(loc_data)
                fig_loc = px.bar(
                    loc_df,
                    x="location",
                    y="count",
                    color="count",
                    text="count",
                    title="Top 5 Job Locations",
                )
                fig_loc.update_layout(xaxis_title="", yaxis_title="Number of Jobs")
                with col1:
                    st.plotly_chart(fig_loc, use_container_width=True)
        else:
            st.error("Could not load job location data.")

        # --- Top Skills ---
        res_skill = requests.get(f"{API_URL}/job_insights/skills")
        if res_skill.status_code == 200:
            skill_data = res_skill.json()
            if skill_data and isinstance(skill_data, list):
                skill_df = pd.DataFrame(skill_data)
                fig_sk = px.bar(
                    skill_df,
                    x="skill",
                    y="count",
                    color="count",
                    text="count",
                    title="Top Trending Skills",
                )
                fig_sk.update_layout(xaxis_title="", yaxis_title="Frequency")
                with col2:
                    st.plotly_chart(fig_sk, use_container_width=True)
        else:
            st.error("Could not load skills data.")
