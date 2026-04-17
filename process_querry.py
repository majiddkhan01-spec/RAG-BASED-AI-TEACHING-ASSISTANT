import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text   # ✅ CORRECT
        }
    )

    response = r.json()

    if "embedding" not in response:
        raise ValueError(f"Embedding Error: {response}")

    return response["embedding"]


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }
    )

    response = r.json()

    if "response" not in response:
        raise ValueError(f"LLM Error: {response}")

    return response["response"]


df = joblib.load("embedding.joblib")

query = input("Ask query: ")

query_embedding = create_embedding(query)

similarities = cosine_similarity(
    np.vstack(df["embedding"].values),
    [query_embedding]
).flatten()

top_k = 5
top_indices = np.argsort(similarities)[::-1][:top_k]
new_df = df.loc[top_indices]

prompt = f"""
You are an AI teaching assistant for the Sigma Web Development course.

Use ONLY the provided subtitle chunks to answer.

Subtitle chunks:
{retrieved_chunks[['title','number','start','end','text']].to_json(orient="records")}

User Question: {user_query}

Instructions:
- Do NOT say you don't have access.
- Answer strictly from the given chunks.
- Mention video number, title, and time range.
- Keep answer concise.
"""

response = inference(prompt)

print(response)

with open("response.txt", "w") as f:
    f.write(response)