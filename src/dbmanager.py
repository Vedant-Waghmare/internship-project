import pandas as pd
import sqlalchemy as sa

class DatabaseManager:
    def __init__(self, db_name="JobPortal", server="localhost\\SQLEXPRESS"):
        # Connection string
        connection_string = (
            f"mssql+pyodbc://@{server}/{db_name}"
            "?driver=ODBC+Driver+18+for+SQL+Server"
            "&trusted_connection=yes"
            "&TrustServerCertificate=yes"
        )
        self.engine = sa.create_engine(connection_string)
        try:
            df = pd.read_sql("SELECT 1 AS test", self.engine)
            print("Connection successful")
        except Exception as e:
            print("Connection failed:", e)

    def _ensure_jobs_schema(self, table_name="Jobs"):
        sql = f"""
IF OBJECT_ID('dbo.{table_name}','U') IS NULL
BEGIN
    CREATE TABLE dbo.{table_name} (
        ID INT IDENTITY(1,1) PRIMARY KEY,
        [FILENAME] NVARCHAR(255) NULL,
        [JOB DESCRIPTION] NVARCHAR(MAX) NULL,
        [COMPANY] NVARCHAR(255) NULL,
        [JOB ROLE] NVARCHAR(255) NULL,
        [EMPLOYMENT TYPE] NVARCHAR(255) NULL,
        [JOB LOCATION] NVARCHAR(255) NULL,
        [EXPERIENCE] NVARCHAR(255) NULL,
        [MIN EXPERIENCE] INT NULL,
        [MAX EXPERIENCE] INT NULL,
        [SKILLS] NVARCHAR(MAX) NULL,
        [TECH SKILLS] NVARCHAR(MAX) NULL,
        [SOFT SKILLS] NVARCHAR(MAX) NULL,
        [QUALIFICATION] NVARCHAR(255) NULL,
        [WORK MODE] NVARCHAR(255) NULL,
        [SALARY] NVARCHAR(255) NULL,
        [JOB TYPE] NVARCHAR(255) NULL,
        [RESPONSIBILITIES] NVARCHAR(MAX) NULL
    );
END
ELSE
BEGIN
    IF COL_LENGTH('dbo.{table_name}','ID') IS NULL
    BEGIN
        ALTER TABLE dbo.{table_name} ADD ID INT IDENTITY(1,1);
        IF OBJECT_ID('dbo.PK_{table_name}','PK') IS NULL
            ALTER TABLE dbo.{table_name} ADD CONSTRAINT PK_{table_name} PRIMARY KEY (ID);
    END
END
"""
        with self.engine.begin() as conn:
            conn.exec_driver_sql(sql)

    def insert_jobs(self, data, table_name="Jobs"):
        if not data:
            print("No data to insert")
            return

        for item in data:
            for k, v in item.items():
                if isinstance(v, list):
                    item[k] = ", ".join(v)
                elif v is None:
                    item[k] = "NA"
                elif isinstance(v, str):
                    item[k] = v.strip()

        df = pd.DataFrame(data)

        df["MIN EXPERIENCE"] = pd.to_numeric(df["MIN EXPERIENCE"], errors="coerce").fillna(0).astype(int)
        df["MAX EXPERIENCE"] = pd.to_numeric(df["MAX EXPERIENCE"], errors="coerce").fillna(50).astype(int)

        columns = ["FILENAME","JOB DESCRIPTION","COMPANY","JOB ROLE","EMPLOYMENT TYPE",
                   "JOB LOCATION","EXPERIENCE","MIN EXPERIENCE","MAX EXPERIENCE",
                   "SKILLS","TECH SKILLS","SOFT SKILLS","QUALIFICATION",
                   "WORK MODE","SALARY","JOB TYPE","RESPONSIBILITIES"]
        df = df[columns]

        try:
            self._ensure_jobs_schema(table_name)
            df.to_sql(table_name, self.engine, if_exists="append", index=False)
            print(f"{len(df)} records inserted into {table_name}")
        except Exception as e:
            print("Error inserting jobs:", e)

    def fetch_jobs(self, table_name="Jobs"):
        try:
            self._ensure_jobs_schema(table_name)
            query = f"SELECT * FROM dbo.{table_name}"
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print("Error fetching jobs:", e)
            return pd.DataFrame()
