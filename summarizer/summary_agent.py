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
            You are an AI assistant that analyzes structured or unstructured data.
            Input may contain:
            - JSON objects
            - Arrays of JSON
            - Nested JSON
            - Plain text
            - Mixed data

            ### VERY IMPORTANT RULES:
            - **Do NOT repeat the entire JSON content in the output**
            - **Do NOT list every field**
            - **Do NOT hallucinate or invent fields**
            - Focus only on patterns, insights, and meaningful information
            - Extract categories, counts, or key attributes WITHOUT copying raw JSON

            ### Input Data:
            {report}

            ### Your Tasks:

            1. **Summary Section**
               - Provide 4–6 insights.
               - Extract meaning from the JSON (patterns, categories, key metrics).
               - Be concise and avoid repeating raw JSON structures.

            2. **Suggestions Section**
               - Provide 3–5 actionable recommendations.
               - Start each with a verb (e.g., Improve, Reduce, Optimize).

            3. **Chart Data Section**
               - Derive labels and values ONLY from the input.
               - Identify categories or counts from JSON keys, unique values, or patterns.
               - DO NOT use placeholders.
               - Format MUST be:

                 chart_type: pie
                 labels: ["label1", "label2", "label3"]
                 values: [value1, value2, value3]

               - If numeric values do not exist, generate counts based on frequency.

            ### FINAL OUTPUT FORMAT:
            #### Summary:
            - ...

            #### Suggestions:
            - ...

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

    #### Suggestions:
    - Provide 3–5 actionable recommendations.
    - Start each with a verb.
    - Bold key terms.

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

    #### Chart Data:
    chart_type: pie
    labels: [...]
    values: [...]
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
