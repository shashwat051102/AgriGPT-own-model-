import os
import streamlit as st
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def get_astra_vectorstore():
    Astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    Astra_DB_ID = os.getenv("ASTRA_DB_ID")
    if not Astra_token or not Astra_DB_ID:
        st.warning("Astra DB credentials are missing. Similarity search will not work.")
        return None
    cassio.init(database_id=Astra_DB_ID, token=Astra_token)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Cassandra(
        embedding=embedding,
        table_name="plant_responses",
        session=None,
        keyspace=None
    )

def store_response(vectorstore, query, response):
    if not vectorstore:
        return
    content = response.raw if hasattr(response, "raw") else str(response)
    document = Document(page_content=content, metadata={"query": query})
    vectorstore.add_documents([document])

def similarity_search(vectorstore, query, k=3):
    if not vectorstore:
        return "Astra DB is not configured."
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant treatments found in Astra DB."
    return "\n\n".join([doc.page_content for doc in docs])