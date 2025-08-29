# app.py
import os
import streamlit as st
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

asyncio.set_event_loop(asyncio.new_event_loop())

# --- Setup ---
load_dotenv()
st.set_page_config(page_title="Gemini RAG Bot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot with History & Sources")

# --- RAG Logic ---

@st.cache_resource
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_name = "chatbot"
    return PineconeVectorStore.from_existing_index(index_name, embeddings)

# --- The RAG Chain ---
prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer from the context, just say that you don't know.
Do not make up an answer. Keep the answer concise.
Use the chat history to understand the context of the question.

Chat History:
{chat_history}

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

def format_chat_history(messages):
    return "\n".join(f"{msg.type}: {msg.content}" for msg in messages)

rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: format_chat_history(x["chat_history"])
    }
    | RunnablePassthrough.assign(
        answer=(
            prompt
            | llm
            | StrOutputParser()
        )
    )
)

# --- Streamlit UI and Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I can answer questions about your document.")]

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)

if user_question := st.chat_input("Ask a question..."):
    st.session_state.messages.append(HumanMessage(content=user_question))
    with st.chat_message("human"):
        st.write(user_question)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            result = rag_chain.invoke({
                "question": user_question,
                "chat_history": st.session_state.messages
            })
            
            response = result["answer"]
            source_documents = result["context"]
            
            st.write(response)

            with st.expander("Show Sources"):
                for i, doc in enumerate(source_documents):
                    page_number = doc.metadata.get('page', 'N/A')
                    citation = f"Source {i+1} (Page: {page_number})"
                    st.write(citation)
                    st.write(doc.page_content)
    
    st.session_state.messages.append(AIMessage(content=response))