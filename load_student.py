import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

df = pd.read_csv("students.csv")

with engine.connect() as conn:
    for _, row in df.iterrows():
        stmt = insert("students").values(
            usn=row["usn"],
            name=row["name"],
            semester=row["semester"],
            branch=row["branch"],
            phone=row["phone"],
            email=row["email"]
        )
        # IF usn already exists → skip inserting
        stmt = stmt.on_conflict_do_nothing(index_elements=["usn"])
        conn.execute(stmt)

print("Student upload completed — duplicates skipped automatically.")
