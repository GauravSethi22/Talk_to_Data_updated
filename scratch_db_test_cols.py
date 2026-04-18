import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(r"C:\Users\Gaurav Sethi\Desktop\Talk_to_Data\.env")

conn = psycopg2.connect(
    dbname=os.environ.get("DB_NAME", "postgres"),
    user=os.environ.get("DB_USER", "postgres"),
    password=os.environ.get("DB_PASSWORD", "1234"),
    host=os.environ.get("DB_HOST", "localhost"),
    port=os.environ.get("DB_PORT", "5432")
)
cursor = conn.cursor()

cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'student_text_to_sql_dataset';")
cols = cursor.fetchall()
print("Columns in student_text_to_sql_dataset:", [c[0] for c in cols])

conn.close()
