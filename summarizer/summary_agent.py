import os
import json
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def generate_summary(data):
    """
    Generate a summary + suggestions + chart data using Ollama.
    Handles large reports via the map-reduce approach.
    """

    llm = ChatOllama(
        model="gemma:2b",
        temperature=0.3
    )

    # --- Input Handling ---
    if "raw_text" in data:
        report_input = data["raw_text"]
    elif "json_data" in data:
        report_input = json.dumps(data["json_data"], indent=2)
    else:
        raise ValueError("Input must include either 'raw_text' or 'json_data'.")

    # --- Text Splitting ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=400
    )
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(report_input)]

    # --- Case 1: Small text ---
    if len(docs) == 1:

        final_prompt_template = ChatPromptTemplate.from_template(
            """
    You are an AI insights analyst. Your task is to read the input and produce
    a structured, readable analysis. The input may contain raw JSON, text, mixed
    data, nested arrays, or unstructured content.

    ---------------------------------------------------------
    RULES (Follow strictly)
    ---------------------------------------------------------
    - Do NOT copy or restate the entire JSON / exact raw text
    - Avoid listing every field or key
    - No hallucination or adding fields not present
    - Focus only on meaning, patterns, and useful findings
    - The response must be easy to read like a human-written summary
    - Final response must NOT be in JSON or code format — only text

    ---------------------------------------------------------
    Input Provided:
    {report}
    ---------------------------------------------------------

    Your Output Must Be:

    -------------------------
    📄 Summary (Insights)
    -------------------------
    Write 5–8 meaningful insights explaining what the data indicates.
    Focus on:
    - patterns or trends
    - common values or categories
    - numerical significance
    - anomalies or missing areas
    - real-world interpretation of what the data implies

    Write naturally, like a readable report paragraph or bullet points.

    -------------------------
    🔍 Suggestions (Improvements / Actions)
    -------------------------
    Provide 4–6 recommendations derived from the insights:
    - Start each suggestion with a strong action verb
    - Mention strengths, weaknesses, improvements, opportunities
    - Make it helpful and practical — avoid generic advice

    Example tone:
    ✓ Improve reporting accuracy by...
    ✓ Reduce inconsistencies by...
    ✓ Enhance performance through...

    -------------------------
    📊 Chart-Friendly Section (Readable – NOT JSON)
    -------------------------
    Identify categories/counts from data and present chart content clearly.
    No placeholders. Only actual inferred values.

    Example output:
    Chart (Pie) Based on Category Distribution:
    - Category A → 10
    - Category B → 6
    - Category C → 3

    ---------------------------------------------------------
    Final Expected Output Format (Plain Text Only):
    ---------------------------------------------------------

    #### Summary:
    • Insight 1
    • Insight 2
    • Insight 3
    ...

    #### Suggestions:
    • Recommendation 1
    • Recommendation 2
    • Recommendation 3
    ...
    """
        )

        chain = final_prompt_template | llm
        response = chain.invoke({"report": report_input})
        return response.content if hasattr(response, "content") else str(response)

    # --- Case 2: Large text (Map-Reduce) ---

    map_prompt_template = """
    Summarize the following text or JSON section.
    - Do NOT repeat raw JSON keys/values.
    - Extract only meaningful insights.
    - Identify patterns, frequent values, and structure.

    "{text}"

    SHORT SUMMARY:
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    reduce_template = """
    Merge the partial summaries into one final summary.
    DO NOT repeat input content.
    DO NOT hallucinate non-existing values.

    ### Partial Summaries:
    {text}

    ### Final Output Instructions

    #### Summary:
    - Provide 4–6 combined insights.
    - Be concise.
    - Highlight **important keywords**.
    - Avoid copying JSON.
    - The given insights should be meaningful.

    #### Suggestions:
    - Provide 3–5 actionable recommendations.
    - Start each with a verb.
    - Bold key terms.
    - Give a Suggestions like improvements strengths and Weaknesses.

    #### Chart Data:
    - Must derive real categories from the JSON/summary.
    - DO NOT use placeholder values.
    - Format only:

      chart_type: pie
      labels: ["label1", "label2", "label3"]
      values: [value1, value2, value3]

    ### FINAL OUTPUT FORMAT:
    #### Summary:
    - ...

    #### Suggestions:
    - ...
    """
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"])

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=True
    )

    response = chain.invoke({"input_documents": docs})
    return response["output_text"]
