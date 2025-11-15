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
        print("--- Small input detected, running single prompt ---")

        final_prompt_template = ChatPromptTemplate.from_template(
            """
            You are an AI assistant that analyzes **any type of input data**, including:
            - Plain text
            - JSON objects
            - Arrays of JSON
            - Deep or nested JSON structures
            - Logs or mixed structured data

            Your task is to interpret the input correctly and extract meaningful insights.

            ### Input Data:
            {report}

            ### What you MUST generate:
            1. **Summary Section**:
               - Provide 4–6 key insights.
               - Extract meaning even from raw JSON.
               - Keep points clear and concise.
               - **Bold important keywords**.

            2. **Suggestions Section**:
               - Provide 3–5 actionable, practical improvements.
               - Start each suggestion with a verb.
               - **Bold key concepts**.

            3. **Chart Data Section**:
               - Must follow EXACT format:
                 chart_type: pie
                 labels: ["Category1", "Category2", "Category3"]
                 values: [10, 20, 30]
               - If numeric values are missing, make reasonable assumptions.

            ### Final Output Format (must follow EXACTLY):
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

    # --- Case 2: Large text (Map-Reduce) ---
    print(f"--- Large input detected, running map-reduce with {len(docs)} chunks ---")

    map_prompt_template = """
    Summarize the following section. 
    The input may be JSON or plain text. Extract only meaningful points.

    "{text}"

    SHORT SUMMARY:
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    reduce_template = """
    You are an AI assistant that merges partial summaries into one final structured output.
    The original input may contain JSON or plain text.

    ### Partial Summaries:
    {text}

    ### Final Output Instructions:

    1. **Summary Section**:
       - Provide 4–6 combined insights.
       - Highlight **important keywords**.

    2. **Suggestions Section**:
       - Provide 3–5 improvements.
       - Start each with an action verb.
       - Bold key concepts.

    3. **Chart Data Section**:
       Output EXACTLY in this format:
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
