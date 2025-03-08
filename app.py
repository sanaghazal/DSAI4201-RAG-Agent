import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss
import re

# Set up your Mistral API key
api_key = "aW27WXAwHm1vjDpMMaMWDl8RUdYwyJuw"
os.environ["MISTRAL_API_KEY"] = api_key

# Dictionary of policies with name and URL
policies = {
    "Academic Annual Leave": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
    "Academic Appraisal": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
    "Intellectual Property": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
    "Credit Hour": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "Program Accreditation": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy",
    "Student Conduct": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "Graduate Final Grade": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "Examination Rules": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
    "International Student": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "Student Attendance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Graduate Academic Standing": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
    "Student Engagement": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "Graduate Admissions": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "Student Appeals": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "Scholarship and Financial Assistance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Right to Refuse Service": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/right-refuse-service-procedure",
    "Library Study Room Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/library-study-room-booking-procedure",
    "Digital Media Centre Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/digital-media-centre-booking",
    "Use of Library Space": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy"
}

# Fetching and parsing policy data
def fetch_policy_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        
        # Look for text content in div tags (adjust selector based on website structure)
        content_div = soup.find("div", class_="main-content") or soup.find("div", {"id": "content"}) or soup.find("div")
        
        if content_div:
            text = content_div.get_text(separator=' ', strip=True)
            # Clean up text - remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return "Could not extract content from the policy page."
    except Exception as e:
        return f"Error fetching policy data: {str(e)}"

# Chunking function to break text into smaller parts with overlap
def chunk_text(text, chunk_size=512, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 200:  # Only keep chunks with substantial content
            chunks.append(chunk)
    return chunks

# Get embeddings for text chunks
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

# Classify the intent of a question to determine relevant policy
def classify_intent(query):
    client = Mistral(api_key=api_key)
    
    # Create a prompt for policy classification
    policy_names = "\n".join([f"- {name}" for name in policies.keys()])
    prompt = f"""
    Given the following question: "{query}"
    
    Which of the following policies is most relevant to answer this question?
    
    {policy_names}
    
    Return only the exact name of the most relevant policy from the list above without any explanation or additional text.
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(model="mistral-large-latest", messages=messages)
    
    classified_policy = response.choices[0].message.content.strip()
    
    # Try to match the response to one of our policy names
    for policy_name in policies.keys():
        if policy_name.lower() in classified_policy.lower() or classified_policy.lower() in policy_name.lower():
            return policy_name
    
    # If no direct match is found, return the first policy as a fallback
    # (in a production system, you'd want better error handling)
    return list(policies.keys())[0]

# Initialize FAISS index
def create_faiss_index(embeddings):
    # Convert the embeddings to a 2D NumPy array
    embedding_vectors = np.array([embedding.embedding for embedding in embeddings])
    
    # Ensure the shape is (num_embeddings, embedding_size)
    d = embedding_vectors.shape[1]  # embedding size (dimensionality)
    
    # Create the FAISS index and add the embeddings
    index = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIDMap(index)
    
    # Add the embeddings with an id for each (FAISS requires ids)
    faiss_index.add_with_ids(embedding_vectors, np.array(range(len(embedding_vectors))))
    
    return faiss_index

# Search for the most relevant chunks based on query embedding
def search_relevant_chunks(faiss_index, query_embedding, k=3):
    D, I = faiss_index.search(query_embedding, k)
    return I

# Mistral model to generate answers based on context
def mistral_answer(query, context, policy_name):
    prompt = f"""
    Context information from the {policy_name} policy is below.
    ---------------------
    {context}
    ---------------------
    Based ONLY on the context information provided and not prior knowledge, please answer the following query about the {policy_name} policy.
    If the answer cannot be found in the context, please state that the information is not available in the policy document.
    
    Query: {query}
    
    Answer:
    """
    
    client = Mistral(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

# Caching policy data to avoid repeated fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_policy_data(url):
    return fetch_policy_data(url)

# Caching embeddings to avoid recomputation
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_policy_embeddings(policy_text):
    chunks = chunk_text(policy_text)
    embeddings = get_text_embedding(chunks)
    return chunks, embeddings

# Streamlit Interface
def main():
    st.title('UDST Policy Chatbot')
    st.markdown('Ask me any question regarding the UDST policies!')
    
    # Input box for query
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner('Processing your question...'):
            # Step 1: Classify the intent to determine which policy is relevant
            policy_name = classify_intent(query)
            
            # Get the URL for the identified policy
            policy_url = policies[policy_name]
            
            # Display which policy was identified
            st.info(f"Your question relates to the *{policy_name}* policy.")
            
            # Step 2: Fetch and process the relevant policy
            policy_text = get_policy_data(policy_url)
            chunks, embeddings = get_policy_embeddings(policy_text)
            
            # Create FAISS index for the policy chunks
            faiss_index = create_faiss_index(embeddings)
            
            # Step 3: Embed the user query and search for relevant chunks
            query_embedding = np.array([get_text_embedding([query])[0].embedding])
            I = search_relevant_chunks(faiss_index, query_embedding, k=3)
            retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
            context = " ".join(retrieved_chunks)
            
            # Step 4: Generate answer from the model
            answer = mistral_answer(query, context, policy_name)
            
            # Display the answer
            st.markdown("### Answer:")
            st.markdown(answer)
            
            # Optionally show the source (for debugging)
            with st.expander("Show source policy text"):
                st.markdown(f"*Policy:* {policy_name}")
                st.markdown(f"*URL:* {policy_url}")
                st.text_area("Relevant sections:", context, height=200)

if __name__ == "__main__":
    main()
