import streamlit as st
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()
st.set_page_config(page_title="Gemini RAG Bot", page_icon="🤖")
st.title("🤖 RAG Chatbot with Gemini & Pinecone")

@st.cache_resource
def get_vectorstore():
    """Initializes and returns the Pinecone vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_name = "chatbot"

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    return vectorstore

prompt_template = """
You are an assistant for question-answering tasks.
Use only the following pieces of retrieved context to answer the question.
If you don't know the answer from the context, just say that you don't know.
Do not make up an answer. Keep the answer concise.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, convert_system_message_to_human=True)

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_question = st.text_input("Ask a question about your document:")

if user_question:
    with st.spinner("Thinking..."):
        retrieved_docs = retriever.invoke(user_question)
        response = rag_chain.invoke(user_question)
        
        st.subheader("Answer:")
        st.write(response)

        st.subheader("Sources:")
        for i, doc in enumerate(retrieved_docs):
            page_number = doc.metadata.get('page', 'N/A')
            citation = f"Source {i+1} (Page: {page_number})"
            with st.expander(citation):
                st.write(doc.page_content)