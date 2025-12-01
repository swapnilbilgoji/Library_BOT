import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

df = pd.read_csv("Library_data.csv")

with engine.connect() as conn:
    for _, row in df.iterrows():
        stmt = insert("books").values(
            accession_no=row["AccessionNo"],
            title=row["Title"],
            author=row["Author"],
            subject=row["Subject"],
            total_copies=1 if "total_copies" not in row else row["total_copies"],
            available_copies=1 if "available_copies" not in row else row["available_copies"],
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["accession_no"])
        conn.execute(stmt)

print("Book upload completed — duplicates skipped automatically.")
