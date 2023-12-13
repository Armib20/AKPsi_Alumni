import streamlit as st
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
    
client = QdrantClient(url="https://62b16dce-f46e-4244-9271-2e203f098417.us-east4-0.gcp.cloud.qdrant.io:6333",
                      api_key="WxYn_uwVASfEYywVW0mpbPfTWHkaUEQ0jOyUYUuZ0jgsytez6NAU0g")

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def query_database(question):
  list = []
  list.append(question)
  query = model.encode(list)[0]
  search_result = client.search(
    collection_name="AKPsi_Alumni",
    query_vector=query,
    limit=3,
    with_payload=True
  )
  for a in search_result:
    payload = a.payload
    score = a.score
    st.write(payload['name'])
    st.write(f"Email: {payload['email']}")
    st.write(f"Company: {payload['company']}")
    st.write(f"Similarity Score: {100 * round(score, 4)}%")
    st.divider()

def main():
  st.set_page_config(page_title="AKPsi AlumniGPT")
  st.header("AKPsi AlumniGPT")
  user_question = st.text_input("Enter a question about alumni that you would like answered")
  if st.button("Query"):
    query_database(user_question)
  
if __name__ == '__main__':
  main()
