import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Load the free semantic search model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert project descriptions into embeddings
import numpy as np

# Load previously saved project records and embeddings
projects = pd.read_csv("all_unique_projects.csv")
embeddings = np.load("all_project_embeddings.npy")

print("Loaded", len(projects), "saved projects")

# Ask a question
question = input("What projects are you interested in? ")
question_embedding = model.encode(question)

# Find the five most similar projects
scores = util.cos_sim(question_embedding, embeddings)[0]
top_results = scores.topk(5)

for score, index in zip(top_results.values, top_results.indices):
    project = projects.iloc[index.item()]
    print("\nProject:", project["Project Title"])
    print("Similarity:", round(score.item(), 3))
    print("Description:", str(project["Public Description"])[:400])

from openai import OpenAI

client = OpenAI()

# Collect the five retrieved projects as evidence
context = ""

for score, index in zip(top_results.values, top_results.indices):
    project = projects.iloc[index.item()]

    context += (
        f"\nProject number: {project['Project Number']}"
        f"\nTitle: {project['Project Title']}"
        f"\nDescription: {str(project['Public Description'])[:800]}\n"
    )

# Ask OpenAI to answer using only the retrieved evidence
response = client.responses.create(
    model="gpt-4o-mini",
    instructions=(
        "You are an assistant helping users explore Innovate UK funded projects. "
        "Answer using only the supplied project descriptions. "
        "Mention relevant project titles and project numbers. "
        "If the evidence is insufficient, say so. "
        "Do not invent project details."
    ),
    input=f"Question: {question}\n\nRetrieved projects:\n{context}",
    max_output_tokens=350
)

print("\n--- AI-GENERATED ANSWER ---")
print(response.output_text) 