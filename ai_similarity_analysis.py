# Build the dataset
import pandas as pd

data = {
    "Tool": [
        "ChatGPT", "Ada Health", "Buoy Health", "WebMD", "Mayo Clinic",
        "MedlinePlus", "NHS 111", "Cleveland Clinic", "Healthline",
        "K Health", "Symptoma", "Teladoc", "Zocdoc"
    ],
    "AI_Based": [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    "Symptom_Checker": [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    "Patient_Education": [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    "Care_Navigation": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    "Telehealth": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    "Appointment_Booking": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)
df.to_csv("health_tool_features.csv", index=False)

print(df)

# Compute Cosine Similarity 
from sklearn.metrics.pairwise import cosine_similarity

features_df = df.set_index("Tool")

similarity_matrix = cosine_similarity(features_df)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=features_df.index,
    columns=features_df.index
)

print(similarity_df.round(3))
similarity_df.to_csv("similarity_matrix.csv")

# Get top 10 similar tools for each query 
def get_top_similar(tool_name, similarity_df, top_n=10):
    scores = similarity_df[tool_name].sort_values(ascending=False)
    scores = scores.drop(tool_name)
    return scores.head(top_n)

queries = ["Ada Health", "ChatGPT", "WebMD"]

for query in queries:
    print(f"\nTop similar tools to {query}:")
    print(get_top_similar(query, similarity_df).round(3))

# Save ranked result to CSV 
all_rankings = []

for query in queries:
    top_results = get_top_similar(query, similarity_df, top_n=10)
    for rank, (tool, score) in enumerate(top_results.items(), start=1):
        all_rankings.append({
            "Query_Tool": query,
            "Rank": rank,
            "Similar_Tool": tool,
            "Cosine_Similarity": round(score, 3)
        })

rankings_df = pd.DataFrame(all_rankings)
rankings_df.to_csv("top_similarity_rankings.csv", index=False)

print(rankings_df)

# Create a heatmap
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(similarity_df, interpolation="nearest", aspect="auto")
plt.colorbar(label="Cosine Similarity")
plt.xticks(range(len(similarity_df.columns)), similarity_df.columns, rotation=90)
plt.yticks(range(len(similarity_df.index)), similarity_df.index)
plt.title("Cosine Similarity Between Digital Healthcare Tools")
plt.tight_layout()
plt.show()
