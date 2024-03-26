#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import os
import sys
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union

# Global variables
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'

# Function Definitions

def read_data(filename: str) -> pd.DataFrame:
    """Load user data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(filename)

def preprocess_data(df: pd.DataFrame) -> list:
    """Process the DataFrame to include labels into each data field."""
    vectorized_data = []
    for _, row in df.iterrows():
        row_data = " ".join([f"{label}:{value}" for label, value in row.items()])
        vectorized_data.append(row_data)
    return vectorized_data

def generate_embeddings(input_data: Union[str, list[str]]) -> np.ndarray:
    """Generate embeddings for the input data using a SentenceTransformer model."""
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(input_data)

def find_user_info_by_id(user_id: str, user_data: pd.DataFrame) -> str:
    """Find and return user information by user ID."""
    # Ensure user_id is treated as a string
    user_id_str = str(user_id)
    user_info = user_data[user_data['ID'].astype(str) == user_id_str]
    if user_info.empty:
        return ""
    else:
        return " ".join([f"{label}:{value}" for label, value in user_info.iloc[0].items()])

def find_nearest_neighbors(query: np.ndarray, vectors: np.ndarray, k: int = 1) -> np.ndarray:
    # Compute cosine similarities between the query and all embeddings
    similarities = cosine_similarity(query, vectors)[0]
    # Get indices of k highest values
    indices = np.argsort(-similarities)[:k]
    return indices

def craft_prompt_and_get_response(user_name: str, relevant_users: list) -> None:
    """Craft a prompt with relevant user data and make an API request for recommendations."""
    question = f"Identify reasons for connecting user {user_name} with the supplied relevant members of the community, include areas of collaboration, and describe these."
    context = "\n\n".join(relevant_users)  # Ensuring separation between users for clarity
    instruction = "Create a response for each of the relevant users but not for the user {user_name} itself. Format the output as a table with a comprehensive and readable description."
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nInstruction:\n{instruction}"

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    url = 'https://api.openai.com/v1/chat/completions'

    system_message = "This is a user recommendation service for the selected user and the user database in the CSV file."
    data = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 2000,
        'temperature': 0.7,
        'messages': [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        generated_text = response.json()['choices'][0]['message']['content']
        print(generated_text.strip())
    else:
        print("Error:", response.status_code, response.text)

# Main Execution Block

def main():
        # Check if enough arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python recommender.py <filename> <user_id>")
        sys.exit(1)

    # Parse command-line arguments
    filename = sys.argv[1]
    user_id = sys.argv[2]


    # Check if the filename exists
    if not os.path.exists(filename):
        print(f"Error: The file {filename} does not exist.")
        sys.exit(1)

    df = read_data(filename)

    vectorized_data = preprocess_data(df)
    embeddings = generate_embeddings(vectorized_data)

    user_info = find_user_info_by_id(user_id, df)
    if not user_info:
        print(f"User ID {user_id} not found.")
        sys.exit(2)

    user_name = user_info.split(' First Name:')[1].split(' ')[0]
    query_embedding = generate_embeddings([user_info])
    if query_embedding.ndim == 1:
     query_embedding = query_embedding.reshape(1, -1)
    
    relevant_user_indices = find_nearest_neighbors(query_embedding, embeddings, k=6)
    relevant_users = [vectorized_data[i] for i in relevant_user_indices]

    craft_prompt_and_get_response(user_name, relevant_users)

if __name__ == "__main__":
    main()
