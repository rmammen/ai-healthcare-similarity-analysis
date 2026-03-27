import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Create feature dataset
data = {
    "Tool": [
        "ChatGPT",
        "Ada Health",
        "Buoy Health",
        "WebMD",
        "Mayo Clinic",
        "MedlinePlus",
        "NHS 111"
    ],
    "AI": [1, 1, 1, 0, 0, 0, 0],
    "Symptom_Checker": [0, 1, 1, 1, 1, 0, 1],
    "Education": [1, 0, 0, 1, 1, 1, 0],
    "Navigation": [1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
df.set_index("Tool", inplace=True)

print("Feature dataset:")
print(df)

# Compute cosine similarity
similarity_matrix = cosine_similarity(df)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df.index,
    columns=df.index
)

print("\nCosine similarity matrix:")
print(similarity_df.round(3))

# Save similarity matrix
similarity_df.to_csv("similarity_matrix.csv")

# Function to get ranked similar tools
def get_top_similar(tool_name, similarity_df):
    scores = similarity_df[tool_name].sort_values(ascending=False)
    scores = scores.drop(tool_name)  # remove self-match
    return scores

queries = ["Ada Health", "ChatGPT", "WebMD"]

for query in queries:
    print(f"\nMost similar tools to {query}:")
    print(get_top_similar(query, similarity_df).round(3))