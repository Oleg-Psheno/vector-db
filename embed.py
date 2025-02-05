import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Load environment variables
load_dotenv()

# Initialize the client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Create embeddings using new syntax
response = client.embeddings.create(
    input=["Кот сидит на окне", "Собака лежит на диване", "The cat is sitting on the window"],
    model="text-embedding-ada-002"
)

# Access embeddings in the new format
embedding1 = response.data[0].embedding
embedding2 = response.data[1].embedding
embedding3 = response.data[2].embedding

# Print the raw embeddings for inspection
print("Embedding 1:", embedding1[:5], "...")  # Show first 5 dimensions
print("Embedding 2:", embedding2[:5], "...")
print("Embedding 3:", embedding3[:5], "...")
# Calculate cosine similarity between the two embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(embedding1, embedding2)
print(f"Similarity between the sentences: {similarity}")

similarity2 = cosine_similarity(embedding1, embedding3)
print(f"Similarity between the sentences 1 and 3: {similarity2}")
