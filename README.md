hscode

# HS Code Classification using Embeddings + LLM (RAG Approach)
===

# 

# \## Problem

# 

# Due to government compliance requirements, we needed to classify \~22,000 ERP products into official HS codes provided by MOCI.  

# Many products did not have HS codes because this was not previously mandatory in our ERP system.

# 

# Manual classification would be:

# \- Time consuming

# \- Error prone

# \- Operationally expensive

# 

# So an AI-assisted classification pipeline was developed.

# 

# \---

# 

# \## Approach

# 

# The solution follows a simple Retrieval Augmented Generation (RAG) pattern:

# 

# 1\. Clean HS master data

# 2\. Generate embeddings for HS descriptions

# 3\. Generate embeddings for product descriptions

# 4\. Use cosine similarity to find top HS candidates

# 5\. Use LLM to select the best HS code from candidates

# 6\. Add confidence flags for review cases

# 

# Pipeline:

# 

# Product → Embedding → Similarity Search → Top Candidates → LLM Decision → HS Code

# 

# \---

# 

# \## Important Design Decision

# 

# In practice, \*\*LLM is not always required\*\*.

# 

# Embedding similarity alone can already give very strong matches for many products.

# 

# Example optimization strategy:

# 

# \- If similarity > 0.75 → Accept directly (skip LLM)

# \- If similarity between 0.50–0.75 → Use LLM verification

# \- If similarity < 0.50 → Manual review

# 

# This can significantly improve speed and reduce cost.

# 

# Current implementation still uses LLM for better quality and validation, but similarity-only classification can be used when performance is critical.

# 

# \---

# 

# \## Why combine Embedding + LLM?

# 

# Embedding similarity gives:

# \- Speed

# \- Scalability

# \- Low cost

# 

# LLM gives:

# \- Reasoning

# \- Better disambiguation

# \- Quality control

# \- Explanation capability

# 

# Tradeoff:

# 

# Embedding only → Fast but less intelligent  

# Embedding + LLM → Slower but more accurate  

# 

# Best practice is usually a hybrid approach.

# 

# \---

# 

# \## Performance Optimizations Implemented

# 

# \- Batch embedding generation

# \- Parallel LLM processing

# \- Resume capability

# \- Confidence scoring

# \- Candidate filtering (Top N retrieval)

# \- Data cleaning before vectorization

# 

# \---

# 

# \## Key Learning

# 

# Real RAG systems are mostly about:

# \- Retrieval quality

# \- Data cleaning

# \- Threshold tuning

# \- Confidence handling

# \- Business rules

# 

# The model is only one small part of the system.

# 

# \---

# 

# \## Future Improvements

# 

# Possible improvements:

# 

# \- Similarity threshold auto tuning

# \- Hybrid keyword + vector search

# \- Fine tuned classification model

# \- Feedback learning from corrections

# \- Vector database integration

# \- Active learning loop

# 

# \---

# 

# \## Tech Stack

# 

# Python  

# Pandas  

# NumPy  

# OpenAI Embeddings  

# Cosine Similarity  

# ThreadPool parallel processing  

# 

# \---

# 

# \## Key Takeaway

# 

# For classification problems:

# 

# \*\*Good retrieval often matters more than complex generation.\*\*

# 

# In many real cases:

# > Better retrieval = Less LLM dependency

# 

# This project demonstrates how RAG can solve real ERP data standardization problems.

