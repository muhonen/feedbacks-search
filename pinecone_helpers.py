from pinecone import Pinecone
import streamlit as st

def init_pinecone():
    """Initializes the Pinecone client and connects to the index."""
    # Initialize Pinecone client with the correct region
    pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"], environment="us-west1-gcp")
    
    # Connect to the Pinecone index
    index_name = "feedbacks-index"
        
    # Connect to the index
    index = pinecone.Index(index_name)
    return index

def query_pinecone(index, embedding, top_k=5):
    """Queries Pinecone with the given embedding and returns the top-k results."""
    query_result = index.query(
    vector=embedding,
    top_k=top_k,
    include_metadata=True
)   
    return query_result['matches']