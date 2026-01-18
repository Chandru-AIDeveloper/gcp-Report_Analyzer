import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def generate_summary(data):
    """
    Generate a summary + suggestions + chart data using OpenAI.
    Handles small text via Stuff and large text via Map-Reduce.
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

    # --- Case 1: Small text (Stuff) ---
    if len(report_input) < 10000:
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
    4. DO NOT output JSON, markdown tables, or code (EXCEPT for Chart Data)
    5. Interpret only what is present in the input
    6. Focus on **meaning, trends, patterns, and implications**
    7. Keep the output professional, concise, and insightful

    ---------------------------------------------------------
    INPUT DATA:
    {report}
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
    📊 CHART DATA
    =========================================================
    Only include **real, derivable categories** from the data.

    Rules:
    - NO placeholder values
    - NO assumptions
    - Use ONLY inferred counts or categories mentioned in summaries
    - Output ONLY in the format below (no extra text)

    chart_type: pie
    labels: ["Label A", "Label B", "Label C"]
    values: [10, 6, 3]

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

    #### Chart Data:
    chart_type: pie
    labels: ["Category A", "Category B"]
    values: [10, 20]
    """
        )

        chain = final_prompt_template | llm
        response = chain.invoke({"report": report_input})
        return response.content if hasattr(response, "content") else str(response)

    # --- Case 2: Large text (Map-Reduce) ---
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
        docs = [Document(page_content=x) for x in text_splitter.split_text(report_input)]

        map_prompt_template = """
        Summarize the following text or JSON section.
        - Do NOT repeat raw JSON keys/values.
        - Extract only meaningful insights.
        - Identify patterns, frequent values, and structure.

        "{text}"

        SHORT SUMMARY:
        """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        reduce_template_str = """
You are an **AI Insight Synthesizer**.

Your task is to merge multiple partial summaries into **one final, high-quality insight report**.
The partial summaries are already derived from the same source data.

---------------------------------------------------------
STRICT RULES (DO NOT VIOLATE)
---------------------------------------------------------
1. DO NOT repeat or restate the original input or partial summaries
2. DO NOT copy sentences verbatim from the summaries
3. DO NOT hallucinate or invent values, categories, or trends
4. DO NOT introduce new fields or assumptions
5. DO NOT output raw JSON from the input
6. Keep the output concise, analytical, and meaningful
7. Only infer what is clearly supported by the summaries

---------------------------------------------------------
PARTIAL SUMMARIES:
{text}
---------------------------------------------------------

Your output MUST strictly follow the format below.

=========================================================
#### Summary:
=========================================================
Provide **4–6 combined insights** that:
- Merge overlapping ideas into stronger conclusions
- Highlight **important keywords** in bold
- Focus on patterns, trends, strengths, weaknesses, or gaps
- Explain *what the combined data implies*, not what it contains
- Avoid repeating similar points

Each point must add **new analytical value**.

=========================================================
#### Suggestions:
=========================================================
Provide **3–5 actionable recommendations**:
- Start each with a **strong action verb**
- Bold **key terms**
- Clearly relate to strengths, weaknesses, or improvement areas
- Keep recommendations practical and specific

Example starters:
- **Improve** …
- **Reduce** …
- **Standardize** …
- **Strengthen** …
- **Optimize** …

=========================================================
#### Chart Data:
=========================================================
Only include **real, derivable categories** from the summaries.

Rules:
- NO placeholder values
- NO assumptions
- Use ONLY inferred counts or categories mentioned in summaries
- Output ONLY in the format below (no extra text)

chart_type: pie
labels: ["Label A", "Label B", "Label C"]
values: [10, 6, 3]

=========================================================
FINAL OUTPUT MUST CONTAIN ONLY:
- Summary
- Suggestions
- Chart Data (if supported)
=========================================================
"""
        reduce_prompt = PromptTemplate(template=reduce_template_str, input_variables=["text"])

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=reduce_prompt,
            verbose=False
        )
        
        response = chain.invoke({"input_documents": docs})
        return response["output_text"]
