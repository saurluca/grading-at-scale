# %%
import json
import pandas as pd
import os

# Read the JSON files
with open("data/privacy/tasks.json", "r") as f:
    tasks = json.load(f)

with open("data/privacy/chunks.json", "r") as f:
    chunks = json.load(f)

# Create a dictionary to map chunk_id to chunk_text
chunk_dict = {chunk["id"]: chunk["chunk_text"] for chunk in chunks}

# Process tasks and extract the required data
data = []
for task in tasks:
    # Get the question
    question = task["question"]

    # Get the answer (assuming there's only one correct answer)
    answer = task["answer_options"][0]["answer"] if task["answer_options"] else ""

    # Get the chunk_text using chunk_id
    chunk_id = task["chunk_id"]
    chunk_text = chunk_dict.get(chunk_id, "")

    data.append({"question": question, "answer": answer, "chunk_text": chunk_text})

# Create DataFrame
df = pd.DataFrame(data)

# Ensure the output directory exists
os.makedirs("data/privacy", exist_ok=True)

# Save to CSV
output_path = "data/privacy/privacy_data.csv"
df.to_csv(output_path, index=False)

print(f"Created CSV with {len(df)} rows")
print(f"Saved to: {output_path}")
print("\nFirst few rows:")
print(df.head())

df
