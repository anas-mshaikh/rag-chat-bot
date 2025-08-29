import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pdf_path = "source_document/attention_is_all_you_need.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} pages from the PDF.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = text_splitter.split_documents(documents)
print(f"Split the document into {len(chunked_docs)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("Initialized Google Embeddings model.")

index_name = "chatbot"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print("Index created successfully.")
else:
    print(f"Index '{index_name}' already exists.")


print(f"Storing document chunks in Pinecone index: '{index_name}'...")
PineconeVectorStore.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    index_name=index_name
)

print(f"Successfully stored {len(chunked_docs)} chunks.")