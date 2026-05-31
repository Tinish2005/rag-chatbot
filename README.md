# 📄 PDF Chatbot using RAG

A conversational AI application that allows users to chat with PDF documents using Retrieval-Augmented Generation (RAG). The chatbot retrieves relevant information from uploaded PDFs and generates accurate, context-aware responses.

## 🚀 Features

* Chat with PDF documents
* Retrieval-Augmented Generation (RAG)
* Vector search using ChromaDB
* Context-aware responses
* Streamlit-based user interface

## 🛠️ Tech Stack

* **Streamlit** — User Interface
* **LangChain** — RAG Pipeline
* **ChromaDB** — Vector Database
* **Google Gemini** — Embedding Model
* **Mistral AI** — Large Language Model

## 📂 Project Structure

```text
app.py
main.py
create_database.py
document_loaders/
retrievers/
vector_store/
requirements.txt
```

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Tinish2005/rag-chatbot.git
cd rag-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Rename `.env.example` to `.env` and add your API keys:

```env
MISTRAL_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### 4. Create the Vector Database

```bash
python create_database.py
```

### 5. Run the Application

```bash
streamlit run app.py
```

## 👨‍💻 Author

Tinish K
