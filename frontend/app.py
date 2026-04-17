import streamlit as st
import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

st.title("🎓 AI Teaching Assistant (RAG Based)")
st.write("Ask questions related to Sigma Web Development Course")

@st.cache_resource
def load_data():
    return joblib.load("C:\\Users\\Majid\\Downloads\\AI_assistent\\backend\\embedding.joblib")

df = load_data() 

def create_embedding(text):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )

    response = r.json()

    if "embeddings" not in response or len(response["embeddings"]) == 0:
        st.error(f"Embedding API Error: {response}")
        return None

    return response["embeddings"][0]  


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
        st.error(f"LLM API Error: {response}")
        return None

    return response["response"]


user_query = st.text_input("Ask your question:")

if st.button("Get Answer"):

    if user_query.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching relevant video chunks..."):

        query_embedding = create_embedding(user_query)

        if query_embedding is None:
            st.stop()

        similarities = cosine_similarity(
            np.vstack(df["embedding"].values),
            [query_embedding]
        ).flatten()

        top_k = 5
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_chunks = df.loc[top_indices]

        prompt = f"""
I am teaching web development using Sigma Web Development course.

Here are the relevant subtitle chunks:
{retrieved_chunks[['title','number','start','end','text']].to_json(orient="records")}

--------------------------------------
User Question: {user_query}

You must:
- Tell which video
- From what start time to end time
- Briefly explain what is taught
- If unrelated question → say you only answer course-related questions.
- Do NOT show reasoning.
"""

        response = inference(prompt)

        if response is None:
            st.stop()

    st.success("Answer Generated ✅")
    st.write(response)

    with st.expander("🔎 Retrieved Video Chunks"):
        st.dataframe(
        retrieved_chunks[['title','number','start','end','text']]
        )