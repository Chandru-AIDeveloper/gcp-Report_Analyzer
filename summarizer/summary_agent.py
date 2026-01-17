import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def generate_summary(data):
    """
    Generate a summary + suggestions + chart data using OpenAI.
    Uses the stuff approach for speed (responses under 5s).
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )

    # --- Input Handling ---
    if "raw_text" in data:
        report_input = data["raw_text"]
    elif "json_data" in data:
        report_input = json.dumps(data["json_data"], indent=2)
    else:
        raise ValueError("Input must include either 'raw_text' or 'json_data'.")

    # --- Text Splitting & Truncation ---
    # We truncate to ensure we stay within reasonable token limits for speed
    report_input = report_input[:15000]
    docs = [Document(page_content=report_input)]

    final_prompt_template = ChatPromptTemplate.from_template(
    """
    You are an **AI Insights Analyst**.

    Your responsibility is to analyze the provided input data and produce
    **high-quality business insights** in clear, human-readable language.

    The input may include:
    - Raw or nested JSON
    - Tables or key-value data
    - Transactional or analytical records
    - Mixed structured and unstructured text

    ---------------------------------------------------------
    STRICT RULES (DO NOT VIOLATE)
    ---------------------------------------------------------
    1. DO NOT repeat or rewrite the raw input data
    2. DO NOT list all fields, keys, or records
    3. DO NOT assume or invent missing data
    4. DO NOT output JSON, markdown tables, or code
    5. Interpret only what is present in the input
    6. Focus on **meaning, trends, patterns, and implications**
    7. Keep the output professional, concise, and insightful

    ---------------------------------------------------------
    INPUT DATA:
    {text}
    ---------------------------------------------------------

    Your output MUST contain ONLY the sections below,
    written in **plain readable text**.

    =========================================================
    📄 SUMMARY – KEY INSIGHTS
    =========================================================
    Write **5–8 meaningful insights** that explain what the data actually shows.

    Each insight should:
    - Describe patterns, trends, or distributions
    - Highlight significant numerical behavior (if present)
    - Point out inconsistencies, anomalies, or gaps
    - Translate raw data into **real-world understanding**
    - Explain *what this data implies*, not what it contains

    Avoid generic statements.
    Each insight should add **new analytical value**.

    Example tone:
    • A clear concentration of transactions indicates…
    • Repeated values across records suggest…
    • Missing or inconsistent fields may impact…
    • The distribution shows a strong imbalance toward…

    =========================================================
    🔍 SUGGESTIONS – ACTIONABLE RECOMMENDATIONS
    =========================================================
    Provide **4–6 practical recommendations** derived directly from the insights.

    Rules:
    - Start each suggestion with a **strong action verb**
    - Address improvements, risks, efficiency, or opportunities
    - Tie each suggestion back to an insight
    - Avoid vague advice (e.g., “improve quality”)

    Example starters:
    • Improve …
    • Standardize …
    • Reduce …
    • Strengthen …
    • Optimize …
    • Monitor …

    =========================================================
    📊 CHART-FRIENDLY OBSERVATIONS
    =========================================================
    Only include this section if the data supports aggregation.

    Present **clear, readable values** suitable for charts
    (Pie / Bar / Line), but NOT in JSON format.

    Rules:
    - Use only inferred or counted values from the input
    - Do not guess or fabricate numbers
    - Keep labels simple and meaningful

    Example:
    Category Distribution:
    - Approved → 18
    - Pending → 7
    - Rejected → 3

    OR

    Monthly Trend:
    - January → High activity
    - February → Moderate decline
    - March → Sharp increase

    =========================================================
    FINAL OUTPUT FORMAT (TEXT ONLY)
    =========================================================

    #### Summary:
    • Insight 1  
    • Insight 2  
    • Insight 3  

    #### Suggestions:
    • Recommendation 1  
    • Recommendation 2  
    • Recommendation 3  

    #### Chart-Friendly Observations:
    • Category A → Value  
    • Category B → Value
    """
    )

    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=final_prompt_template,
        verbose=False
    )

    response = chain.invoke({"input_documents": docs})
    return response["output_text"]
