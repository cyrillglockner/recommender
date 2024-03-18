import streamlit as st
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Streamlit app title
st.title('User Recommendation System')

# API key and model initialization
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def read_users():
    with open('user_data.csv', 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def generate_embeddings(input_data):
    embeddings = model.encode(input_data)
    return embeddings

def find_nearest_neighbors(query, vectors, k=1):
    similarities = cosine_similarity([query], vectors)[0]
    indices = np.argsort(-similarities)[:k]
    return indices

def find_user_info_by_id(user_id, user_data):
    return next((line for line in user_data if line.startswith(user_id + ",")), None)

def craft_prompt(user_name, relevant_users):
    question = f"Find reasons for connecting user {user_name} with the supplied relevant members of the climate community, and describe these."
    context = "\n\n".join(relevant_users)
    instruction = "Include user ID and names in the output but exclude the user itself. Always include all relevant users in your output."
    return f"Question: {question}\n\nContext:\n\n{context}\n\nInstruction:\n{instruction}"

# User ID input
user_id_input = st.text_input('Enter User ID:', '')

if st.button('Find Recommendations'):
    if user_id_input:
        user_data = read_users()
        embeddings = generate_embeddings(user_data)
        user_info = find_user_info_by_id(user_id_input, user_data)
        
        if user_info:
            query_text = user_info
            query_embedding = generate_embeddings(query_text)
            relevant_user_indices = find_nearest_neighbors(query_embedding, embeddings, k=6)
            relevant_users = [user_data[i] for i in relevant_user_indices if not user_data[i].startswith(user_id_input)]
            
            user_name = user_info.split(',')[1] if len(user_info.split(',')) > 1 else "Unknown"
            prompt = craft_prompt(user_name, relevant_users)
            
            # Make API request to OpenAI
            headers = {
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': 'gpt-3.5-turbo',
                'max_tokens': 2000,
                'messages': [
                    {"role": "system", "content": "This is a user recommendation services for the selected user and the user database in the csv file."},
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            
            if response.status_code == 200:
                generated_text = response.json()['choices'][0]['message']['content']
                st.text_area("Recommendations:", generated_text.strip(), height=300)
            else:
                st.error(f"Error: {response.status_code}\n{response.text}")
        else:
            st.error('User ID not found. Please try again.')
    else:
        st.error('Please enter a valid User ID.')
