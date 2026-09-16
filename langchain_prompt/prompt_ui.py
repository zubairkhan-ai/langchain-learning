import streamlit as st
import json

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate


# =====================================================
# LOCAL TINYLLAMA
# =====================================================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=300,
    return_full_text=False
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# =====================================================
# LOAD template.json
# =====================================================

with open("template.json", "r", encoding="utf-8") as f:
    data = json.load(f)

template_text = data["template"]


# =====================================================
# PROTECT CURLY BRACES
# =====================================================

template_text = template_text.replace("{", "{{")
template_text = template_text.replace("}", "}}")

template_text = template_text.replace(
    "{{paper_input}}",
    "{paper_input}"
)

template_text = template_text.replace(
    "{{style_input}}",
    "{style_input}"
)

template_text = template_text.replace(
    "{{length_input}}",
    "{length_input}"
)


# =====================================================
# CREATE PROMPT TEMPLATE
# =====================================================

template = PromptTemplate(
    template=template_text,
    input_variables=[
        "paper_input",
        "style_input",
        "length_input"
    ]
)


# =====================================================
# STREAMLIT UI
# =====================================================

st.header("Research Tool")


paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "ResNet: Deep Residual Learning for Image Recognition"
    ]
)


style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented"
    ]
)


length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed Explanation)"
    ]
)


# =====================================================
# LCEL CHAIN
# =====================================================

chain = template | model


# =====================================================
# BUTTON
# =====================================================

if st.button("Summarize"):

    if paper_input == "Select...":

        st.warning("Please select a research paper.")

    else:

        with st.spinner("TinyLlama is thinking..."):

            result = chain.invoke({
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input
            })

        answer = result.content

        # Remove special tokens
        answer = answer.replace("</s>", "")
        answer = answer.replace("<|assistant|>", "")
        answer = answer.replace("<|user|>", "")
        answer = answer.replace("<|system|>", "")

        st.subheader("Summary")

        st.write(answer.strip())