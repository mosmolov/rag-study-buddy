# PDF-RAG: A Retrieval-Augmented Generation System for PDFs

PDF-RAG is a Streamlit application that allows users to upload PDF documents, process them for semantic search, and ask questions about their content. The system uses a Retrieval-Augmented Generation (RAG) approach to provide accurate, contextual answers based on the document content.

## 🌟 Features

- **PDF Upload**: Upload one or multiple PDF documents
- **Document Processing**: Extract, chunk, and embed document content
- **Semantic Search**: Query your documents using natural language
- **AI-Powered Responses**: Get contextual answers generated by an LLM (Language Learning Model)
- **Response Reasoning**: View the AI's reasoning process for transparency
- **Database Management**: Clear the document database when needed

## 🛠️ Architecture

The system uses a Retrieval-Augmented Generation (RAG) architecture:

1. **Ingestion Pipeline**: PDFs are processed, text is extracted and chunked semantically
2. **Vector Database**: Chunks are embedded and stored in a Qdrant vector database
3. **Retrieval System**: User queries retrieve the most relevant document chunks
4. **Generation**: An LLM generates answers based on the retrieved context

## 📋 Requirements

- Python 3.8+
- Qdrant account (for vector database)
- Ollama (for local LLM access)

## 📦 Dependencies

```
streamlit
qdrant-client
ollama
python-dotenv
pypdf2
scikit-learn
numpy
nltk
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-rag.git
   cd pdf-rag
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Qdrant credentials:
   ```
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

5. Make sure Ollama is installed and running with the required models:
   ```bash
   # Install models
   ollama pull nomic-embed-text
   ollama pull deepseek-r1:1.5b
   ```

## 🖥️ Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Upload PDF files using the file uploader

4. Click "Process PDFs" to extract, chunk, and embed the document content

5. Enter your questions in the query field and click "Search"

6. View the AI's response, reasoning, and supporting context

## ⚙️ Configuration

You can adjust system parameters in `config.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 1024)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `COLLECTION_NAME`: Name of the Qdrant collection (default: "documents")
- `RETRIEVAL_LIMIT`: Number of chunks to retrieve per query (default: 7)
- `EMBEDDING_MODEL`: Model for generating embeddings (default: "nomic-embed-text")
- `LLM_MODEL`: Model for answering queries (default: "deepseek-r1:1.5b")
- `STREAM_RESPONSE`: Enable streaming responses (default: False)

## 🔍 How It Works

### Ingestion Process

1. PDFs are uploaded and text is extracted
2. Text is split into semantically meaningful chunks using NLTK
3. Embeddings are generated for each chunk using the specified embedding model
4. Chunks and their embeddings are stored in Qdrant

### Query Process

1. The user enters a natural language question
2. The query is embedded and used to search for relevant chunks in Qdrant
3. Retrieved chunks are combined to create context for the LLM
4. The LLM generates a response based on the retrieved context
5. The system displays the answer, reasoning, and supporting context

## 🧩 Project Structure

- `app.py`: Main Streamlit application
- `config.py`: Configuration parameters
- `ingestion.py`: PDF processing and storage
- `query.py`: Document retrieval and LLM query functions
- `utils.py`: Utility functions for chunking and embeddings
- `requirements.txt`: Project dependencies