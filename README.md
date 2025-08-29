# RAG Chatbot with Google Gemini and Pinecone

This project is a chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on the content of a provided PDF document. The core AI components are powered by Google's Gemini Pro and Google's embedding models.

## Features

-   **PDF Ingestion**: Processes and indexes a PDF document.
-   **Vector Storage**: Uses **Pinecone** as the vector database.
-   **AI Models**: Uses **Gemini flash lite** for generation and Google's `embedding-001` for embeddings.
-   **Source Citations**: Displays the page number for each source document chunk used to generate the answer.
-   **Guardrails**: The prompt instructs the bot to answer *only* from the provided document context.
-   **Web UI**: A simple and clean user interface built with **Streamlit**.
-   **Containerized**: The application is fully containerized with **Docker** for easy setup and deployment.

---

## Setup & How to Run

### **Prerequisites**
-   Docker must be installed and running.
-   You have an API key from **Google AI Studio** and API keys for **Pinecone**.

### **Step 1: Configuration**

1.  Clone this repository.
2.  Place your PDF document inside the `source_document/` folder.
3.  Create a `.env` file in the root of the project and add your API keys:
    ```
    # Get from Google AI Studio
    GOOGLE_API_KEY="AIza..."

    # Get from Pinecone
    PINECONE_API_KEY="..."
    PINECONE_ENVIRONMENT="your-pinecone-env"
    ```

### **Step 2: Build the Docker Image**

Open a terminal in the project root and run:
```bash
docker build -t rag-chatbot-gemini .
```

### **Step 3: Run the Ingestion Script**

This command processes your PDF and populates your Pinecone index. This only needs to be done once per document.

```bash
docker run --rm --env-file .env rag-chatbot-gemini python ingest.py
```

### **Step 4: Run the Chatbot Application**

Start the Streamlit web application.

```bash
docker run -d -p 8501:8501 --env-file .env --name rag-gemini-app rag-chatbot-gemini
```
### To stop the app
```bash
docker stop rag-gemini-app
docker rm rag-gemini-app
```

### **Step 5: Access the Chatbot**

Open your web browser and navigate to:
[http://localhost:8501](http://localhost:8501)

---

## Design Notes

-   **Chunking Strategy**: I used `RecursiveCharacterTextSplitter` with a `chunk_size` of 1000 and `chunk_overlap` of 100. This is a balanced approach that maintains semantic context within chunks.
-   **Embedding Model**: `models/embedding-001` from Google was chosen. It is a strong, general-purpose text embedding model.
-   **LLM**: `gemini-2.5-flash-lite` is used for its strong reasoning capabilities and ability to follow the strict instructions in the prompt.
-   **Prompt Engineering**: The prompt explicitly forbids the LLM from using outside knowledge and forces it to rely only on the retrieved context. This is the primary guardrail against hallucinations.
-   **Limitations**: The quality of answers is dependent on the retrieval step. If relevant context is not retrieved, the model cannot provide a correct answer. The citations refer to the retrieved text chunk's page, which is generally accurate but not guaranteed to be the only page with relevant info.