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

try:
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = cursor.fetchall()
    print("Tables in DB:", [t[0] for t in tables])
except Exception as e:
    print("Error:", e)
    
try:
    query = "SELECT name, d.name AS department_name, attendance FROM student_text_to_sql_dataset_(2) s JOIN department d ON s.departmentid = d.departmentid WHERE marks > 85 AND attendance > 90"
    cursor.execute(query)
    print("Query success!")
except Exception as e:
    print("Query Error:", e)

conn.close()
