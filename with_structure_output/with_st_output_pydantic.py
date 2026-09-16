from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.output_parsers import JsonOutputParser

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ==========================================
# TINYLLAMA
# ==========================================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device=-1,
    max_new_tokens=300,
    do_sample=False,
    return_full_text=False,
    clean_up_tokenization_spaces=False
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

model = ChatHuggingFace(
    llm=llm
)


# ==========================================
# PYDANTIC STRUCTURE
# ==========================================

class Review(BaseModel):

    key_themes: list[str] = Field(
        description="Write all important themes discussed in the review"
    )

    summary: str = Field(
        description="Write a short summary of the review"
    )

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        default="neutral",
        description="Return the sentiment of the review"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Write all advantages mentioned in the review"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write all disadvantages mentioned in the review"
    )


# ==========================================
# OUTPUT PARSER
# ==========================================

parser = JsonOutputParser()


# ==========================================
# REVIEW
# ==========================================

review = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say,
it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes
everything lightning fast—whether I’m gaming, multitasking, or editing
photos. The 5000mAh battery easily lasts a full day even with heavy use,
and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches,
though I don't use it often. What really blew me away is the 200MP camera—
the night mode is stunning, capturing crisp, vibrant images even in low
light. Zooming up to 100x actually works well for distant objects, but
anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed
use. Also, Samsung’s One UI still comes with bloatware—why do I need five
different Samsung apps for things Google already provides? The $1,300
price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
"""


# ==========================================
# PROMPT
# ==========================================

prompt = f"""
You are a review analyzer.

Analyze the following review:

{review}

Return ONLY valid JSON.

The JSON must contain exactly these five fields:

"key_themes"
"summary"
"sentiment"
"pros"
"cons"

Rules:

1. key_themes must be a list of important themes from the review.
2. summary must be a short summary of the review.
3. sentiment must be exactly "positive", "negative", or "neutral".
4. pros must be a list of advantages mentioned in the review.
5. cons must be a list of disadvantages mentioned in the review.
6. Do not invent information.
7. Do not explain anything.
8. Do not write anything before or after the JSON.

Return JSON only.

Example structure:

{{
    "key_themes": [
        "performance",
        "camera",
        "battery"
    ],
    "summary": "The Samsung Galaxy S24 Ultra offers excellent performance, camera quality, battery life, and S-Pen support, but it is heavy, expensive, and contains bloatware.",
    "sentiment": "positive",
    "pros": [
        "powerful processor",
        "excellent camera",
        "long battery life",
        "S-Pen support"
    ],
    "cons": [
        "heavy and bulky",
        "bloatware",
        "expensive"
    ]
}}
"""


# ==========================================
# MODEL INVOCATION
# ==========================================

result = model.invoke(prompt)


print("\n==============================")
print("RAW RESPONSE")
print("==============================")

print(result.content)


# ==========================================
# PARSE JSON
# ==========================================

try:

    parsed_result = parser.parse(result.content)

    # TinyLlama sometimes returns a list
    if isinstance(parsed_result, list):
        parsed_result = parsed_result[0]


    # ==========================================
    # PYDANTIC VALIDATION
    # ==========================================

    review_result = Review(**parsed_result)


    # ==========================================
    # DISPLAY OUTPUT
    # ==========================================

    print("\n==============================")
    print("PYDANTIC OUTPUT")
    print("==============================")


    print("\nKey Themes:")

    for theme in review_result.key_themes:
        print("-", theme)


    print("\nSummary:")

    print(review_result.summary)


    print("\nSentiment:")

    print(review_result.sentiment)


    print("\nPros:")

    if review_result.pros:

        for pro in review_result.pros:
            print("-", pro)

    else:

        print("None")


    print("\nCons:")

    if review_result.cons:

        for con in review_result.cons:
            print("-", con)

    else:

        print("None")


except Exception as e:

    print("\n==============================")
    print("ERROR")
    print("==============================")

    print(e)