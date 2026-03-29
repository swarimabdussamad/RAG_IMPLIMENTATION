import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

print("Loading HS data...")

load_dotenv()

moci = pd.read_excel("data/moci_hs.xlsx", header=1, dtype=str)

moci.columns = [
    "hs_code",
    "hs_ar",
    "hs_en"
]

moci["hs_code"] = moci["hs_code"].str.zfill(12)

print("Setting up OpenAI client...")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"

print("Creating text for embedding...")

moci["text"] = moci["hs_en"]

# Remove rows with missing descriptions
print(f"Total rows before cleaning: {len(moci)}")
moci = moci.dropna(subset=["text"])
moci = moci[moci["text"].str.strip() != ""]
moci = moci.reset_index(drop=True)
print(f"Total rows after cleaning: {len(moci)}")

print(f"Generating vectors using {EMBEDDING_MODEL}...")

# Batch embed all descriptions (OpenAI allows up to 2048 inputs per request)
all_vectors = []
batch_size = 2000

for i in range(0, len(moci), batch_size):
    batch = moci["text"].iloc[i:i+batch_size].tolist()
    print(f"Processing batch {i//batch_size + 1}/{(len(moci)-1)//batch_size + 1}...")
    
    response = client.embeddings.create(
        input=batch,
        model=EMBEDDING_MODEL
    )
    
    batch_vectors = [item.embedding for item in response.data]
    all_vectors.extend(batch_vectors)

vectors = np.array(all_vectors)

print("Vector shape:", vectors.shape)

np.save("data/hs_vectors.npy", vectors)

moci.to_csv("data/moci_clean.csv", index=False)

print("DONE")