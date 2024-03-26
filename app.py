import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

# Set global variables
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
API_KEY = os.getenv('OPENAI_API_KEY')


# Function Definitions (adapted for Streamlit)

@st.cache_resource  # 👈 Add the caching decorator
def load_model(model_name):
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(model_name)

# Initialize model
model = load_model(MODEL_NAME)

@st.cache_data  # 👈 Add the caching decorator
def preprocess_data(df: pd.DataFrame) -> list:
    """Process the DataFrame to include labels into each datafield."""
    vectorized_data = []
    for _, row in df.iterrows():
        row_data = " ".join([f"{label}:{value}" for label, value in row.items()])
        vectorized_data.append(row_data)
    return vectorized_data

@st.cache_resource  # 👈 Add the caching decorator
def generate_embeddings(_model, input_data: list[str]) -> np.ndarray:
    """Generate embeddings for the input data."""
    return _model.encode(input_data)

def find_user_info_by_id(user_id: str, df: pd.DataFrame) -> str:
    """Find and return user information by user ID."""
    user_info = df[df['ID'].astype(str) == str(user_id)]
    if user_info.empty:
        return ""
    else:
        return " ".join([f"{label}:{value}" for label, value in user_info.iloc[0].items()])

def find_nearest_neighbors(query: np.ndarray, vectors: np.ndarray, k: int = 1) -> list[int]:
    """Find the k-nearest neighbors for the query."""
    similarities = cosine_similarity(query, vectors)[0]
    indices = np.argsort(-similarities)[:k]
    return indices.tolist()

def craft_prompt_and_get_response(user_name: str, relevant_users: list):
    """Craft a prompt and make an API request for recommendations."""
    question = f"Identify reasons for connecting user {user_name} with the supplied relevant members of the community, include areas of collaboration, and describe these."
    context = "\n\n".join(relevant_users)
    instruction = "Create a response for each of the relevant users but not for the user {user_name} itself. Format the output as a table with a comprehensive and readable description."
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nInstruction:\n{instruction}"

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 2000,
        'temperature': 0.7,
        'messages': [
            {"role": "system", "content": "This is a user recommendation service for the selected user and the user database in the CSV file."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    if response.status_code == 200:
        generated_text = response.json()['choices'][0]['message']['content']
        return generated_text.strip()
    else:
        return "Error with GPT response: " + response.text

# Streamlit App UI

st.title("User Recommender System")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
user_id_input = st.text_input("Enter User ID", "")

if uploaded_file and user_id_input:
    df = pd.read_csv(uploaded_file)
    vectorized_data = preprocess_data(df)
    embeddings = generate_embeddings(model, vectorized_data)

    user_info = find_user_info_by_id(user_id_input, df)
    if user_info:
        user_name = user_info.split(' First Name:')[1].split(' ')[0]
        query_embedding = generate_embeddings(model, [user_info])
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        relevant_user_indices = find_nearest_neighbors(query_embedding, embeddings, k=5)
        relevant_users = [vectorized_data[i] for i in relevant_user_indices]

        generated_response = craft_prompt_and_get_response(user_name, relevant_users)
        st.write(generated_response)
    else:
        st.write("User ID not found.")
