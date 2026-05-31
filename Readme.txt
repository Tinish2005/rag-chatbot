# PDF Chatbot using RAG

A conversational AI app that lets you chat with any PDF document using RAG (Retrieval Augmented Generation).

## Tech Stack
- **Streamlit** — UI
- **LangChain** — RAG pipeline
- **ChromaDB** — Vector store
- **Google Gemini** — Embeddings
- **Mistral AI** — LLM

## Setup

1. Clone the repo
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. Install dependencies
   pip install -r requirements.txt

3. Add your API keys
   - Rename `.env.example` to `.env`
   - Add your keys inside

4. Run the app
   streamlit run app.py