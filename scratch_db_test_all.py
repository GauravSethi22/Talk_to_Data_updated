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

cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
tables = [t[0] for t in cursor.fetchall()]

print("Tables in DB:")
for table in tables:
    cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}';")
    cols = cursor.fetchall()
    print(f"  - {table}: {[c[0] for c in cols]}")

conn.close()
