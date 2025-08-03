import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def average_pairwise_cosine_similarity(embeddings):
    if len(embeddings) < 2:
        return None  # Not enough data
    # Stack into matrix of shape [n_runs, hidden_dim]
    embedding_matrix = np.vstack(embeddings)
    # Compute cosine similarity matrix: shape [n_runs, n_runs]
    similarity_matrix = cosine_similarity(embedding_matrix)
    # Extract upper triangle (excluding diagonal) for unique pairs
    n = len(embeddings)
    upper_triangle_indices = np.triu_indices(n, k=1)
    upper_triangle_values = similarity_matrix[upper_triangle_indices]
    # Return average similarity
    return np.mean(upper_triangle_values)

def process_file(fname):
    match = re.match(r"prompt(\d+)_q(\d+)\.pkl", fname)
    if not match:
        return None
    prompt_id = int(match.group(1))
    question_id = int(match.group(2))
    file_path = os.path.join("Github", fname)

    try:
        with open(file_path, "rb") as f:
            result = pickle.load(f)

        # === Internal Repeatability ===
        # Retrieve H_bar (average entropy over runs), negate to match LaTeX definition
        H_bar_content = result["internal_repeatability"]["H_bar_per_run"]
        internal_repeatability_score = -H_bar_content if H_bar_content is not None else None

        # === Semantic Repeatability ===
        # Cosine similarity across all embeddings (stopword-filtered outputs)
        embeddings = result["semantic_repeatability"]["embeddings_per_run"]
        semantic_repeatability_score = average_pairwise_cosine_similarity(embeddings)

        return {
            "prompt_id": prompt_id,
            "question_id": question_id,
            "semantic_repeatability": semantic_repeatability_score,
            "internal_repeatability": internal_repeatability_score
        }

    except Exception as e:
        return {"error": f"Error loading {file_path}: {e}"}

# Change to your directory
result_dir = "..."
files = [f for f in os.listdir(result_dir) if f.endswith(".pkl") and re.match(r"prompt(\d+)_q(\d+)\.pkl", f)]

records = []
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your system
    futures = {executor.submit(process_file, f): f for f in files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        result = future.result()
        if result is None:
            continue
        if "error" in result:
            print(result["error"])
        else:
            records.append(result)

# Save the results
df = pd.DataFrame.from_records(records)
df.to_csv("./Semantic_and_Internal_Repeatability_Scores.csv", index=False)
