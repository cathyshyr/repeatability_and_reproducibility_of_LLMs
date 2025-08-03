from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm

def load_embeddings_data(result_dir):
    files = [f for f in os.listdir(result_dir) if f.endswith(".pkl") and re.match(r"prompt(\d+)_q(\d+)\.pkl", f)]
    records = []

    for f in tqdm(files):
        match = re.match(r"prompt(\d+)_q(\d+)\.pkl", f)
        if not match:
            continue
        prompt_id = int(match.group(1))
        question_id = int(match.group(2))
        file_path = os.path.join(result_dir, f)
        try:
            with open(file_path, "rb") as pf:
                result = pickle.load(pf)
                emb_list = result.get("semantic_repeatability", {}).get("embeddings_per_run", None)
                if emb_list is not None:
                    records.append({
                        "prompt_id": prompt_id,
                        "question_id": question_id,
                        "embeddings_per_run": emb_list
                    })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return pd.DataFrame(records)


def reproducibility_with_averaged_embeddings(df):
    records = []
    for question_id, group in df.groupby("question_id"):
        prompt_avg_embeddings = []

        for _, row in group.iterrows():
            embs = row["embeddings_per_run"]
            if embs and len(embs) > 0:
                avg_emb = np.mean(np.vstack(embs), axis=0)
                prompt_avg_embeddings.append((row["prompt_id"], avg_emb))

        for (p1, e1), (p2, e2) in combinations(prompt_avg_embeddings, 2):
            sim = cosine_similarity([e1], [e2])[0, 0]
            records.append({
                "question_id": question_id,
                "prompt1": p1,
                "prompt2": p2,
                "cosine_similarity": sim
            })

    return pd.DataFrame(records)


def load_entropy_data(result_dir):
    files = [f for f in os.listdir(result_dir) if f.endswith(".pkl") and re.match(r"prompt(\d+)_q(\d+)\.pkl", f)]
    records = []

    for f in tqdm(files):
        match = re.match(r"prompt(\d+)_q(\d+)\.pkl", f)
        if not match:
            continue
        prompt_id = int(match.group(1))
        question_id = int(match.group(2))
        file_path = os.path.join(result_dir, f)
        try:
            with open(file_path, "rb") as pf:
                result = pickle.load(pf)
                H_bar = result.get("internal_repeatability", {}).get("H_bar_per_run", None)
                if H_bar is not None:
                    records.append({
                        "prompt_id": prompt_id,
                        "question_id": question_id,
                        "H_bar": H_bar
                    })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return pd.DataFrame(records)


# === Compute internal reproducibility score ===
def compute_internal_reproducibility(entropies):
    if len(entropies) < 2:
        return None
    diffs = [abs(e1 - e2) for e1, e2 in combinations(entropies, 2)]
    return -np.mean(diffs)  # negate for "larger = more reproducible"


def reproducibility_score_per_question(entropy_df):
    records = []
    for question_id, group in entropy_df.groupby("question_id"):
        entropy_vals = group["H_bar"].dropna().values
        score = compute_internal_reproducibility(entropy_vals)
        if score is not None:
            records.append({
                "question_id": question_id,
                "internal_reproducibility": score
            })
    return pd.DataFrame(records)

# Change to your directory
result_dir = "..."

# Semantic reproducibility
df_embs = load_embeddings_data(result_dir)
df_sem_rpd = reproducibility_with_averaged_embeddings(df_embs)
df_sem_rpd.to_csv("Semantic_Reproducibility_Scores.csv", index=False)

# Internal reproducibility
df_entropy = load_entropy_data(result_dir)
df_int_rpd = reproducibility_score_per_question(df_entropy)
df_int_rpd.to_csv("Internal_Reproducibility_Scores.csv", index=False)
