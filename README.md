# 🤖 Conversational RAG Chatbot

A conversational AI chatbot built with LangChain that answers questions from your own documents using Retrieval Augmented Generation (RAG). It remembers chat history and gives context-aware answers.

---

## 🧠 How It Works

```
User Question
     │
     ▼
LLM rephrases question using chat history
     │
     ▼
Retriever searches ChromaDB for relevant chunks
     │
     ▼
LLM answers using retrieved context + history
     │
     ▼
Answer ✅
```

---

## 🏗️ Project Structure

```
conversational-rag-langchain/
│
├── data/                   
│   ├── apple.txt
│   ├── google.txt
|   ├── microsoft.txt
│   └── sample.txt
│
├── src/
│   ├── main.py             
│   ├── loader.py           
│   ├── retriever.py        
│   └── chain.py            
│
├── .env                    
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | RAG chain framework |
| ChromaDB | Local vector store |
| HuggingFace Embeddings | Free embeddings (all-MiniLM-L6-v2) |
| OpenAI GPT-3.5 | LLM for answering |
| Python Dotenv | Managing API keys |

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/conversational-rag-langchain.git
cd conversational-rag-langchain
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key in `.env`**
```
OPENAI_API_KEY=your-key-here
```

**5. Run**
```bash
python src/main.py
```

---

## 💬 Example

```
You: Who founded Google?
Answer: Google was founded by Larry Page and Sergey Brin in 1998.

You: Who is the CEO?
Answer: The CEO of Google is Sundar Pichai.

You: What was their revenue in 2023?
Answer: Google's revenue in 2023 was 307 billion dollars.
```

---

## ✅ Key Features

- Conversational memory — remembers previous questions
- History aware retrieval — rephrases questions using chat history
- Local vector store — no cloud needed
- Free embeddings — uses HuggingFace, no API cost
- Multi document support — load multiple files at once

---

## 🛠️ Future Improvements

- [ ] Add UI with Streamlit
- [ ] Support PDF, CSV, DOCX files
- [ ] Add LangSmith tracing
- [ ] Deploy to cloud