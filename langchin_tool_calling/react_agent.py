# ============================================================
# LOCAL REACT-STYLE AGENT
# TinyLlama + Hugging Face + DuckDuckGo
# No OpenAI API key required
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from transformers import pipeline

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace
)

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import HumanMessage


# ============================================================
# 2. LOAD TINYLLAMA
# ============================================================

print("\n============================================================")
print("LOADING TINYLLAMA")
print("============================================================")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

text_pipeline = pipeline(
    task="text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=250,
    do_sample=False,
    return_full_text=False,
    clean_up_tokenization_spaces=False
)

print("TinyLlama loaded successfully.")


# ============================================================
# 3. CREATE LANGCHAIN LLM
# ============================================================

llm_pipeline = HuggingFacePipeline(
    pipeline=text_pipeline
)

llm = ChatHuggingFace(
    llm=llm_pipeline
)

print("LangChain model created successfully.")


# ============================================================
# 4. CREATE DUCKDUCKGO SEARCH TOOL
# ============================================================

search_tool = DuckDuckGoSearchRun()

print("DuckDuckGo search tool created successfully.")


# ============================================================
# 5. DEFINE REACT-STYLE AGENT FUNCTION
# ============================================================

def run_react_agent(question: str) -> str:
    """
    A simple local ReAct-style agent.

    The agent performs:
    1. Thought
    2. Action: DuckDuckGo search
    3. Observation: Search results
    4. Final answer
    """

    print("\n============================================================")
    print("AGENT STARTED")
    print("============================================================")

    print("\nUser question:")
    print(question)

    # --------------------------------------------------------
    # STEP 1: THOUGHT
    # --------------------------------------------------------

    print("\nThought:")
    print("I need current information, so I will use web search.")

    # --------------------------------------------------------
    # STEP 2: ACTION
    # --------------------------------------------------------

    print("\nAction:")
    print("DuckDuckGoSearchRun")

    search_query = question

    search_results = search_tool.invoke(
        search_query
    )

    # --------------------------------------------------------
    # STEP 3: OBSERVATION
    # --------------------------------------------------------

    print("\nObservation:")
    print(search_results)

    # --------------------------------------------------------
    # STEP 4: ASK TINYLLAMA TO CREATE FINAL ANSWER
    # --------------------------------------------------------

    final_prompt = f"""
You are a helpful assistant.

Answer the user's question using the search results below.

User question:
{question}

Search results:
{search_results}

Instructions:
- Give a clear and concise answer.
- Do not invent facts.
- If the search results are incomplete, say so.
- Do not show your internal reasoning.
"""

    print("\n============================================================")
    print("GENERATING FINAL ANSWER")
    print("============================================================")

    response = llm.invoke(
        [
            HumanMessage(
                content=final_prompt
            )
        ]
    )

    return response.content


# ============================================================
# 6. RUN THE AGENT
# ============================================================

question = (
    "What are three ways to travel from Goa to Delhi?"
)

answer = run_react_agent(
    question
)


# ============================================================
# 7. PRINT FINAL ANSWER
# ============================================================

print("\n============================================================")
print("FINAL ANSWER")
print("============================================================")

print(answer)