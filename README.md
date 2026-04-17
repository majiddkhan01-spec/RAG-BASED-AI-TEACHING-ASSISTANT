🎓 RAG-Based AI Teaching Assistant

An AI-powered teaching assistant that answers questions based on the Sigma Web Development Course using a Retrieval-Augmented Generation (RAG) pipeline.

🚀 Features
🔍 Semantic search using embeddings
🤖 LLM-powered answer generation (via Ollama)
📚 Context-aware responses from video subtitles
🎥 Returns:
Video title
Timestamp (start → end)
Explanation of the concept
⚡ FastAPI backend + Streamlit frontend
🧠 Tech Stack
Backend: FastAPI
Frontend: Streamlit
Embeddings: nomic-embed-text (Ollama)
LLM: tinyllama (Ollama)
Vector Similarity: Cosine Similarity (sklearn)
Storage: Precomputed embeddings (joblib)
📁 Project Structure
AI_assistent/
│
├── backend/
│   ├── rag.py              # Core RAG pipeline
│   ├── main.py             # FastAPI app
│   └── embedding.joblib    # Stored embeddings
│
├── frontend/
│   └── app.py              # Streamlit UI
│
├── README.md
└── .gitignore
⚙️ How It Works
User enters a query
Query is converted into an embedding
Cosine similarity is computed against stored embeddings
Top relevant chunks are retrieved
LLM generates a response using retrieved context
🛠️ Setup Instructions
🔹 1. Clone the repo
git clone https://github.com/majiddkhan01-spec/RAG-BASED-AI-TEACHING-ASSISTANT.git
cd RAG-BASED-AI-TEACHING-ASSISTANT
🔹 2. Install dependencies
pip install -r requirements.txt

(Create one if not present — I can help if needed)

🔹 3. Install & Run Ollama

Download from: https://ollama.com/download

Start server:

ollama serve
🔹 4. Pull required models
ollama pull nomic-embed-text
ollama pull tinyllama
🔹 5. Run Backend (FastAPI)
cd backend
uvicorn main:app --reload

API runs at:

http://localhost:8000
🔹 6. Run Frontend (Streamlit)
cd frontend
streamlit run app.py
📌 API Endpoints
GET /
{
  "message": "AI Teaching Assistant API"
}
POST /ask
{
  "question": "What is SEO?"
}
💡 Example Query

"Where is SEO mentioned?"

Output:
Video name
Timestamp
Explanation of SEO concept
⚠️ Notes

Make sure Ollama is running at:

http://localhost:11434

Do NOT use:

http://ollama:11434

unless using Docker

🔮 Future Improvements
Add vector database (FAISS / Pinecone)
Improve prompt engineering
Add chat history (memory)
Deploy on cloud (AWS / GCP)
Add UI enhancements
🤝 Contribution

Feel free to fork and improve this project.

📄 License

This project is for educational purposes.

👨‍💻 Author

Majid Khan

⭐ If you like this project

Give it a ⭐ on GitHub!
