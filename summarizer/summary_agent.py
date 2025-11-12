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

    # ✅ Initialize Ollama LLM (Replace model name with your available local model)
    llm = ChatOllama(
        model="gemma",  # You can change this to 'llama3', 'llama3.1', etc.
        temperature=0.3
    )

    # --- Input Handling ---
    if "raw_text" in data:
        report_input = data["raw_text"]
    elif "json_data" in data:
        report_input = json.dumps(data["json_data"], indent=2)
    else:
        raise ValueError("Input must include either 'raw_text' or 'json_data'.")

    # --- Text Splitting (to avoid large context issues) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=400
    )
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(report_input)]

    # --- Case 1: Small text (single chunk) ---
    if len(docs) == 1:
        print("--- Small input detected, running single prompt ---")
        final_prompt_template = ChatPromptTemplate.from_template(
            """
            You are an AI assistant that analyzes reports and provides concise summaries,
            actionable suggestions, and structured chart data.

            ### Input Report:
            {report}

            ### Output Instructions:
            1. **Summary Section**:
               - Write 4–6 bullet points.
               - Keep each point short, simple, and informative.
               - **Bold important keywords**.

            2. **Suggestions Section**:
               - Provide 3–5 practical and actionable suggestions.
               - Each suggestion should start with a verb (e.g., "Improve", "Enhance").
               - **Bold key terms**.

            3. **Chart Data Section**:
               - Output exactly in this format:
                 chart_type: pie
                 labels: ["Category1", "Category2", "Category3"]
                 values: [10, 20, 30]
               - Choose meaningful categories and values from the report.

            ### Final Output Format:
            #### Summary:
            - ...

            #### Suggestions:
            - ...

            #### Chart Data:
            chart_type: pie
            labels: [...]
            values: [...]
            """
        )

        chain = final_prompt_template | llm
        response = chain.invoke({"report": report_input})
        return response.content if hasattr(response, "content") else str(response)

    # --- Case 2: Large text (multiple chunks → map-reduce) ---
    print(f"--- Large input detected, running map-reduce with {len(docs)} chunks ---")

    map_prompt_template = """
    Summarize the following section from a large report,
    focusing on key points, figures, and major findings.

    "{text}"

    CONCISE SUMMARY:
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    reduce_template = """
    You are an AI assistant that combines partial summaries into one comprehensive output.

    ### Partial Summaries:
    {text}

    ### Output Instructions:
    Based *only* on the summaries above, generate:

    1. **Summary Section**:
       - Write 4–6 bullet points.
       - Highlight **important terms**.

    2. **Suggestions Section**:
       - 3–5 actionable improvements.
       - Start each with a verb (e.g., "Optimize", "Increase", "Monitor").
       - Bold key concepts.

    3. **Chart Data Section**:
       - Must strictly follow:
         chart_type: pie
         labels: ["Category1", "Category2", "Category3"]
         values: [10, 20, 30]

    ### Final Output Format:
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