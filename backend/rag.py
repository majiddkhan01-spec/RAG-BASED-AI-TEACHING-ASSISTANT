import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = joblib.load("embedding.joblib")

def create_embedding(text):
    r = requests.post(
        "http://ollama:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )
    return r.json()["embeddings"][0]

def inference(prompt):
    r = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]

def answer_query(query):

    query_embedding = create_embedding(query)

    similarities = cosine_similarity(
        np.vstack(df["embedding"].values),
        [query_embedding]
    ).flatten()

    top_k = 5
    top_indices = np.argsort(similarities)[::-1][:top_k]
    chunks = df.loc[top_indices]

    prompt = f"""
Use the following video subtitles to answer.

{chunks[['title','number','start','end','text']].to_json(orient="records")}

Question: {query}
"""

    return inference(prompt)