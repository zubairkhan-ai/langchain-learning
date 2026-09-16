from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------
# 1. Sample text
# --------------------------------

sample = """
Farmers were working hard in the fields, preparing the soil and planting
seeds for the next season.

The sun was bright, and the air smelled of earth and fresh grass.

The Indian Premier League (IPL) is the biggest cricket league in the world.
People all over the world watch the exciting matches.

Cricket is played between two teams of eleven players.
The batting team tries to score runs while the bowling team tries to take wickets.

Artificial intelligence is changing the way people work and learn.
Modern AI systems can analyze large amounts of information and help automate
many different tasks.
"""


# --------------------------------
# 2. Create local embeddings
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 3. Initialize SemanticChunker
# --------------------------------

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)


# --------------------------------
# 4. Split the text
# --------------------------------

chunks = splitter.split_text(sample)


# --------------------------------
# 5. Print results
# --------------------------------

print("Number of chunks:", len(chunks))


for i, chunk in enumerate(chunks):

    print(f"\n========== CHUNK {i + 1} ==========")

    print(chunk)