import streamlit as st
from openai_helpers import initialize_client, get_embedding, summarize_feedbacks, expand_query
from pinecone_helpers import init_pinecone, query_pinecone
import pandas as pd

# Function to display results in a table format
def display_feedbacks(results):
    # Extract the review text and ratings from the result metadata
    feedback_data = [
        {
            "Vastaavuus (0-1)": result['score'],
            "Palaute": result['metadata']['review_text'],
            "Arvosana (1-5)": result['metadata']['rating']
        }
        for result in results
    ]
    
    # Create a DataFrame for a clean display
    feedback_df = pd.DataFrame(feedback_data)
    st.table(feedback_df)


# Show title and description
st.title("🔍 Löydä sinua kiinnostavat asiakaspalautteet nopeasti!💬")
st.write(
    "Tervetuloa asiakaspalautteiden hakuun! Tämä työkalu auttaa sinua löytämään juuri ne palautteet, joista olet kiinnostunut. "
    "💡 Käytä hakukenttää kirjoittaaksesi kysymyksesi ja etsi asiakaspalautteita helposti. "
    "🔑 Aloita syöttämällä OpenAI API-avain.\n\n"
    "Data sisältää IF vakuutusyhtiön asiakaspalautteita trustpilot.com:sta."
)

# Input for OpenAI API key
openai_api_key = st.text_input("Syötä OpenAI API-avain:", type="password")

# Check if the user has entered a valid API key
if openai_api_key:
    # Initialize Pinecone and OpenAI
    index = init_pinecone()
    client = initialize_client(openai_api_key)

    # Streamlit app layout
    st.subheader("Asiakaspalautteiden haku")

    # Input box for user query
    user_query = st.text_input("Kirjoita kysymyksesi asiakaspalautteista:")
    
    # Select mode (Feedback Only or Summarize)
    mode = st.selectbox("Toiminto:", ("Etsi palautteita", "Hae tiivistelmä tietyistä palautteista"))

    # Button to trigger search
    if st.button("Hae palautteita"):
        status_placeholder = st.empty()

        # Check if the query is not empty and has a minimum character length (e.g., 10 characters)
        if len(user_query.strip()) < 10 and len(user_query.strip().split()) < 3:
            status_placeholder.write("Kysymyksen on oltava vähintään 10 merkkiä pitkä ja sisältää vähintään kolme sanaa. Tarkista kysymyksesi.")
        else:
            # Clear any previous messages
            status_placeholder.empty()

            # Show loading spinner
            with st.spinner("Tutkitaan palautteita..."):
                
                # Expand user query for hopefully better results
                user_query = expand_query(user_query, client)

                # Generate embedding using OpenAI
                embedding = get_embedding(user_query, client)

                # Display results or summarize feedbacks
                if mode == "Etsi palautteita":
                    status_placeholder.write("Haetaan halutut asiakaspalautteet...")
                    results = query_pinecone(index, embedding)
                    
                    # Clear the message before displaying feedback
                    status_placeholder.empty()

                    # Display feedbacks in a table format
                    display_feedbacks(results)
                else:
                    status_placeholder.write("Luodaan tiivistelmää...")
                    results = query_pinecone(index, embedding, top_k=20)
                    summarized_feedback = summarize_feedbacks(user_query, [review['metadata']['review_text'] for review in results], client)
                    
                    # Clear the message before displaying summary
                    status_placeholder.empty()
                    st.text_area("Tiivistelmä: ", summarized_feedback.content, height=200)
else:
    st.write("Anna OpenAI API-avain aloittaaksesi.")