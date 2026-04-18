import sys
import os
sys.path.append(r"C:\Users\Gaurav Sethi\Desktop\Talk_to_Data")
from main_pipeline import AIQuerySystem

pipeline = AIQuerySystem()
query = "List the names of students, their department names, and attendance percentage for students who scored more than 85 marks and have attendance above 90"

print("\n--- ROUTING ---")
routing = pipeline.router.route(query)
print(routing)

print("\n--- SCHEMAS RETRIEVED ---")
schemas = pipeline.tag.retrieve_schemas(query, top_k=10)
for s in schemas:
    print(s.table_name)
    
print("\n--- SQL EXECUTION (IF ANY) ---")
schema_context = "\n\n".join([s.to_document() for s in schemas])
sql_result = pipeline.sql_engine.execute(query, schema_context)
print("Valid:", sql_result.success)
print("Query:", sql_result.query)
print("Errors:", sql_result.validation_errors)
