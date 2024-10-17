from pinecone import Pinecone, ServerlessSpec
import os
import streamlit as st

# Get Pinecone API key
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
  name="feedbacks-index",
  dimension=1536,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)