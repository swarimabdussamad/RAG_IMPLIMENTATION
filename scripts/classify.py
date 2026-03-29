import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5.4"
CONFIDENCE_THRESHOLD = 0.52
HIGH_CONFIDENCE_THRESHOLD = 0.65
BATCH_SIZE = 100
MAX_WORKERS = 20  # Increased for faster parallel processing

# Structured output schema
class Classification(BaseModel):
    hs_code: str
    reason: str
    confidence: str  # HIGH, MEDIUM, or LOW

print("Loading data...")
products = pd.read_excel("data/products.xlsx", header=1, dtype=str)
products.columns = ["id", "part_number", "product_name", "company_category", "hs_code"]

moci = pd.read_csv("data/moci_clean.csv", dtype=str)
moci["hs_code"] = moci["hs_code"].str.zfill(12)
hs_vectors = np.load("data/hs_vectors.npy")

# Resume logic
resume_from = 0
output_file = "data/products_classified_optimized.xlsx"
if os.path.exists(output_file):
    print(f"Found existing {output_file} - resuming...")
    existing = pd.read_excel(output_file, dtype=str)
    
    if "hs_code" in existing.columns and len(existing) == len(products):
        classified_count = (existing["hs_code"].notna() & (existing["hs_code"] != "")).sum()
        resume_from = classified_count
        print(f"Resuming from product {resume_from + 1}")
        
        products["new_hs_code"] = existing["hs_code"].fillna("")
        products["classification_reason"] = existing["classification_reason"].fillna("") if "classification_reason" in existing.columns else ""
        products["confidence_status"] = existing["confidence_status"].fillna("") if "confidence_status" in existing.columns else ""
        products["similarity_score"] = existing["similarity_score"].fillna("") if "similarity_score" in existing.columns else ""
    else:
        products["new_hs_code"] = ""
        products["classification_reason"] = ""
        products["confidence_status"] = ""
        products["similarity_score"] = ""
else:
    print(f"Starting fresh - saving to {output_file}...")
    products["new_hs_code"] = ""
    products["classification_reason"] = ""
    products["confidence_status"] = ""
    products["similarity_score"] = ""

print(f"Total: {len(products)} | Classified: {resume_from} | Remaining: {len(products) - resume_from}\n")

def classify_single_product(item):
    """Classify a single product using LLM with structured output"""
    idx = item['idx']
    row = item['row']
    candidates_text = "\n".join(item['candidates'][:10])  # Top 10 only
    
    prompt = f"""You are an HS code classification expert. Classify this product into the best HS code from the candidates provided.

Product: {row['product_name']}
Category: {row['company_category']}

Top Candidates:
{candidates_text}

CRITICAL CLASSIFICATION RULES:
1. Identify the ACTUAL PRODUCT being sold, not its packaging or container
   - "Rice in woven bags" → classify as RICE, not bags/packaging
   - "Oil in plastic bottles" → classify as OIL, not plastic containers

2. Determine the PROCESSING STATE accurately:
   - Paddy rice = rice still in husk (inedible, from field)
   - Husked/Brown rice = husk removed, bran layer intact (brownish)
   - Semi-milled/Milled/White rice = fully processed, white, ready to cook
   - DEFAULT: Unless product explicitly says "brown rice" or "paddy", assume it's MILLED WHITE RICE

3. Match by SPECIFIC function and material:
   - Use the most specific HS code that matches the product's primary purpose
   - Avoid generic household categories if specific industrial/food categories exist

4. Group SIMILAR items into common parent categories:
   - Multiple fruit types → Fresh Fruits category
   - Various fasteners → Fasteners category
   - Don't micro-classify unless product name is very specific

5. CONFIDENCE levels:
   - HIGH: Clear match with specific product type
   - MEDIUM: Reasonable match but some ambiguity
   - LOW: No good match or significant uncertainty

Select the best HS code from the candidates above."""
    
    try:
        schema = Classification.model_json_schema()
        schema["additionalProperties"] = False
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "classification",
                "strict": True,
                "schema": schema
            }},
            temperature=0.3
        )
        
        result = Classification.model_validate_json(response.choices[0].message.content)
        
        return {
            'idx': idx,
            'hs_code': result.hs_code,
            'reason': result.reason,
            'confidence': result.confidence,
            'similarity': item['similarity']
        }
    except Exception as e:
        print(f"  ⚠ Error classifying product {idx}: {e}")
        return {
            'idx': idx,
            'hs_code': '',
            'reason': f'Error: {str(e)}',
            'confidence': 'LOW',
            'similarity': item['similarity']
        }

# Process in batches
total_batches = (len(products) + BATCH_SIZE - 1) // BATCH_SIZE
start_batch = resume_from // BATCH_SIZE

for batch_num in range(start_batch, total_batches):
    batch_start_time = time.time()
    
    start_idx = batch_num * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(products))
    batch = products.iloc[start_idx:end_idx]
    
    print(f"\n{'='*80}")
    print(f"BATCH {batch_num + 1}/{total_batches} - Products {start_idx+1} to {end_idx}")
    print(f"{'='*80}")
    
    # Find products needing classification
    needs_classification = []
    for i, (idx, row) in enumerate(batch.iterrows()):
        if not products.at[idx, "new_hs_code"] or products.at[idx, "new_hs_code"] == "":
            needs_classification.append((i, idx, row))
    
    if not needs_classification:
        print("Already classified, skipping...")
        continue
    
    # Get embeddings
    embed_start = time.time()
    print(f"Embeddings for {len(needs_classification)} products...")
    product_texts = [f"{row['product_name']} {row['company_category']}" for _, _, row in needs_classification]
    
    embedding_response = client.embeddings.create(
        input=product_texts,
        model=EMBEDDING_MODEL
    )
    print(f"  ✓ Done in {time.time() - embed_start:.1f}s")
    
    # Calculate similarities and prepare for LLM
    sim_start = time.time()
    print("Calculating similarities...")
    llm_tasks = []
    
    for i, (orig_i, idx, row) in enumerate(needs_classification):
        product_vector = np.array([embedding_response.data[i].embedding])
        scores = cosine_similarity(product_vector, hs_vectors)[0]
        top_indices = np.argsort(scores)[-10:][::-1]  # Top 10 for faster LLM
        best_score = scores[top_indices[0]]
        
        products.at[idx, "similarity_score"] = str(round(best_score, 4))
        
        candidates = []
        for j in top_indices:
            hs_row = moci.iloc[j]
            candidates.append(f"{hs_row['hs_code']}: {hs_row['hs_en']}")
        
        llm_tasks.append({
            'idx': idx,
            'row': row,
            'candidates': candidates,
            'similarity': best_score
        })
    
    print(f"  ✓ Done in {time.time() - sim_start:.1f}s")
    
    # Parallel LLM classification
    llm_start = time.time()
    print(f"LLM classification ({len(llm_tasks)} products, {MAX_WORKERS} workers)...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(classify_single_product, task) for task in llm_tasks]
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            idx = result['idx']
            
            # Validate and ensure HS code is exactly 12 digits
            hs_code = result['hs_code'].strip()
            if hs_code:
                # Remove any non-digit characters
                hs_code = ''.join(filter(str.isdigit, hs_code))
                # Pad to 12 digits
                hs_code = hs_code.zfill(12)
                # Validate length
                if len(hs_code) != 12:
                    print(f"  ⚠ Invalid HS code length for product {idx}: {hs_code} (expected 12 digits)")
                    hs_code = hs_code[:12].zfill(12)  # Truncate or pad to 12
            
            products.at[idx, "new_hs_code"] = hs_code
            products.at[idx, "classification_reason"] = result['reason']
            
            # Update confidence status
            if result['confidence'] == 'LOW' or result['similarity'] < CONFIDENCE_THRESHOLD:
                products.at[idx, "confidence_status"] = "LOW CONFIDENCE - NEEDS REVIEW"
            else:
                products.at[idx, "confidence_status"] = "OK"
            
            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{len(llm_tasks)}")
    
    print(f"  ✓ Done in {time.time() - llm_start:.1f}s")
    
    # Save progress with validation
    products["hs_code"] = products["new_hs_code"]
    
    # Validate only classified HS codes (non-empty)
    classified_mask = (products["hs_code"].notna()) & (products["hs_code"] != "")
    invalid_codes = products[classified_mask & (products["hs_code"].str.len() != 12)]
    if len(invalid_codes) > 0:
        print(f"  ⚠ Warning: {len(invalid_codes)} classified products have invalid HS code length")
        print(f"    Examples: {invalid_codes['hs_code'].head().tolist()}")
    
    output_df = products[["id", "part_number", "product_name", "company_category", "hs_code", "classification_reason", "confidence_status", "similarity_score"]]
    output_df.to_excel(output_file, index=False)
    
    batch_time = time.time() - batch_start_time
    print(f"✓ Batch complete in {batch_time:.1f}s ({end_idx}/{len(products)} total)")

print("\n" + "="*80)
print("CLASSIFICATION COMPLETE")
print("="*80)

# Summary
total = len(products)
low_conf = len(products[products["confidence_status"].str.contains("LOW CONFIDENCE", na=False)])
high_conf = total - low_conf

print(f"\nTotal: {total}")
print(f"High confidence: {high_conf} ({round(high_conf/total*100, 1)}%)")
print(f"Low confidence: {low_conf} ({round(low_conf/total*100, 1)}%)")
