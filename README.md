# AI Healthcare Similarity Analysis

This project analyzes the similarity between digital healthcare tools based on their features.

## Objective
The goal of this analysis is to identify which healthcare platforms are most similar to each other using feature-based similarity.

## Dataset
The dataset includes the following platforms:
- ChatGPT  
- Ada Health  
- Buoy Health  
- WebMD  
- Mayo Clinic  
- MedlinePlus  
- NHS 111  

Each platform is represented using binary features:
- AI-based interaction  
- Symptom checking  
- Patient education  
- Care navigation  

## Method
Similarity between tools was measured using **cosine similarity**. Each platform was represented as a feature vector, and similarity scores were calculated based on how closely those vectors aligned.

## Results
The analysis shows:
- Ada Health and Buoy Health are the most similar tools  
- WebMD is most similar to Mayo Clinic and MedlinePlus  
- ChatGPT shows moderate similarity across multiple tools  

## Visualization

![Image 3-26-26 at 11 31 PM](https://github.com/user-attachments/assets/918f59f4-b8e2-46ba-a6cd-953d422a7733)


## Files Included

- `health_tool_features.csv` – dataset  
- `similarity_matrix.csv` – similarity scores  
- `top_similarity_rankings.csv` – ranked results  
- `ai_similarity_analysis.py` – Python code  
- `similarity_heatmap.png` – visualization  

## Tools Used
- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
