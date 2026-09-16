# ============================================================
# CURRENCY CONVERSION TOOL
# LangChain @tool + ExchangeRate-API
# No OpenAI API key required
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import requests

from dotenv import load_dotenv
from langchain_core.tools import tool


# ============================================================
# 2. LOAD .ENV FILE
# ============================================================

load_dotenv()


# ============================================================
# 3. READ API KEY FROM .ENV
# ============================================================

EXCHANGE_RATE_API_KEY = os.getenv(
    "EXCHANGE_RATE_API_KEY"
)


# ============================================================
# 4. CREATE CURRENCY TOOL
# ============================================================

@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> float:
    """
    Fetch the currency conversion factor between
    a base currency and a target currency.
    """

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not EXCHANGE_RATE_API_KEY:
        raise ValueError(
            "EXCHANGE_RATE_API_KEY is missing. "
            "Add it to the .env file."
        )

    # --------------------------------------------------------
    # Clean currency codes
    # --------------------------------------------------------

    base_currency = base_currency.upper().strip()
    target_currency = target_currency.upper().strip()

    # --------------------------------------------------------
    # Create dynamic API URL
    # --------------------------------------------------------

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{EXCHANGE_RATE_API_KEY}/pair/"
        f"{base_currency}/"
        f"{target_currency}"
    )

    # --------------------------------------------------------
    # Send API request
    # --------------------------------------------------------

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # Convert response to JSON
    # --------------------------------------------------------

    data = response.json()

    # --------------------------------------------------------
    # Check API result
    # --------------------------------------------------------

    if data.get("result") != "success":
        raise ValueError(
            f"Currency API error: "
            f"{data.get('error-type', 'Unknown error')}"
        )

    # --------------------------------------------------------
    # Return conversion factor
    # --------------------------------------------------------

    return float(data["conversion_rate"])


# ============================================================
# 5. DISPLAY TOOL INFORMATION
# ============================================================

print("\n============================================================")
print("CURRENCY TOOL INFORMATION")
print("============================================================")

print("Tool name:", get_conversion_factor.name)

print("\nTool description:")
print(get_conversion_factor.description)

print("\nTool arguments:")
print(get_conversion_factor.args)


# ============================================================
# 6. INVOKE TOOL
# ============================================================

print("\n============================================================")
print("CURRENCY CONVERSION")
print("============================================================")

result = get_conversion_factor.invoke({
    "base_currency": "USD",
    "target_currency": "INR"
})


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\n1 USD =", result, "INR")