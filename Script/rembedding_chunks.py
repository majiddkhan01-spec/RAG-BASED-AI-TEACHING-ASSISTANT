import requests
import json
import pandas as pd
import os
import joblib

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text_list
        }
    )

    response = r.json()

    if "embeddings" not in response:
        raise ValueError(f"Embedding API Error: {response}")

    return response["embeddings"]


dataset = os.listdir("jsons")

chunk_id = 0
all_chunks = []

for data in dataset:
    with open(f"jsons/{data}", "r", encoding="utf-8") as f:
        file_data = json.load(f)

    texts = [chunk["text"] for chunk in file_data[0]["chunks"]]

    embeddings = create_embedding(texts)

    for i, chunk in enumerate(file_data[0]["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        all_chunks.append(chunk)

df = pd.DataFrame.from_records(all_chunks)

joblib.dump(df, "embedding.joblib")

print("✅ Embeddings regenerated successfully!")
print("Total chunks:", len(df))
print("Embedding dimension:", len(df["embedding"].iloc[0]))