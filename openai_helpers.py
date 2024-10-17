from openai import OpenAI

# Initialize OpenaAI client
def initialize_client(api_key):
    client = OpenAI()
    client.api_key = api_key
    return client

# Function to get OpenAI embeddings
def get_embedding(text, client, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def expand_query(user_query, client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": """
             Olet avulias assistentti, joka laajentaa käyttäjän kyselyä siihen liittyvillä sanoilla ja termeillä, 
             jotta palautteiden haku toimii paremmin. Palautteet ovat vakuutusyhtiön palautteita. 
             Keskity laajentamaan hakua siltä osin mitä käyttäjä spesifisesti hakee, 
             älä lisää yleisiä termejä."""},
            {"role": "user", "content": f"""
             Laajenna tätä kyselyä: 
             #### '{user_query}' #### 
             """
             }
        ]
    )
    expanded_query = response.choices[0].message.content
    return expanded_query


def summarize_feedbacks(user_query, feedbacks, client):
    """Summarizes a list of feedbacks using GPT-4o-mini."""
    feedback_text = "\n".join(feedbacks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
                Olet avulias assistantti, joka vastaa asiakkaan kysymykseen tekemällä tiivistelmän palautteista. 
                Otat tiivistelmään mukaan vain ne palautteet, jotka vastaavat asiakkaan kysymykseen.
                Palautat vastauksen selkeässä tekstimuodossa.
                """
            },
            {
                "role": "user",
                "content": f"""
                Tee tiivistelmä palautteista ja vastaa asiakkaan kysymykseen käyttäen relevantteja palautteita.
                
                #### 
                Asiakkaan kysymys: 
                {user_query}
                ####

                ####
                Palautteet: 
                {feedback_text}
                ####
                """
            }
        ],
    )
    return response.choices[0].message
