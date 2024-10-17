import streamlit as st
from openai_helpers import initialize_client, get_embedding, summarize_feedbacks
from pinecone_helpers import init_pinecone, query_pinecone

# Show title and description
st.title("🔍 Löydä sinua kiinnostavat asiakaspalautteet nopeasti!💬")
st.write(
    "Tervetuloa asiakaspalautteiden hakuun! Tämä työkalu auttaa sinua löytämään juuri ne palautteet, joista olet kiinnostunut. "
    "💡 Käytä hakukenttää kirjoittaaksesi kysymyksesi ja etsi asiakaspalautteita helposti. "
    "🔑 Aloita syöttämällä OpenAI API-avain."
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
    user_query = st.text_input("Kirjoita kysymyksesi asiakaspalautteesta:")

    # Select mode (Feedback Only or Summarize)
    mode = st.selectbox("Toiminto:", ("Etsi palautteita", "Hae tiivistelmä tietyistä palautteista"))

    # Button to trigger search
    if st.button("Hae palautteita"):
        # Check if the query is not empty and has a minimum character length (e.g., 10 characters)
        if len(user_query.strip()) < 10:
            st.write("Kysymyksen on oltava vähintään 10 merkkiä pitkä. Tarkista kysymyksesi.")
        else:
            # Generate embedding using OpenAI
            st.write("Tutkitaan palautteita...")
            embedding = get_embedding(user_query, client)

            # Display results or summarize feedbacks
            if mode == "Etsi palautteita":
                st.write("Haetaan halutut asiakaspalautteet...")
                results = query_pinecone(index, embedding)
                for result in results:
                    st.write(f"{result['metadata']['review_text']}")
            else:
                st.write("Luodaan tiivistelmää...")
                results = query_pinecone(index, embedding, top_k=20)
                summarized_feedback = summarize_feedbacks(user_query, [review['metadata']['review_text'] for review in results], client)
                st.write(summarized_feedback.content)
else:
    st.write("Anna OpenAI API-avain aloittaaksesi.")