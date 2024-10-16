from pinecone import Pinecone
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai_helpers import initialize_client
import pandas as pd
from tqdm.auto import tqdm
from openai_helpers import get_embedding

# Load environment variables from .env file
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize pinecone and OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
client = initialize_client(OPENAI_API_KEY)

# Connect to pinecone index index
index = pc.Index("feedbacks-index")

# Load your data
df = pd.read_csv('feedbacks-search\data\if.fi_reviews.csv', sep=';', encoding='utf-16', index_col=0)

# Prepare data for upsert
batch_size = 100  # Adjust based on your needs
for i in tqdm(range(0, len(df), batch_size)):
    # Get a batch of data
    i_end = min(i+batch_size, len(df))
    batch = df.iloc[i:i_end]
    
    # Prepare the batch for upsert
    ids = batch.index.astype(str).tolist()
    texts = batch['review_text'].tolist()
    
    # Get embeddings for the batch
    embeddings = [get_embedding(text) for text in texts]
    
    # Prepare metadata
    metadata = batch.to_dict(orient='records')
    
    # Create upsert data
    upsert_data = list(zip(ids, embeddings, metadata))
    
    # Upsert to Pinecone
    index.upsert(vectors=upsert_data)