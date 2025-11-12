import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def generate_summary(data):
    """
    Generate a summary + suggestions + chart data.
    Handles large reports by using the map-reduce method.
    """

    llm = ChatGroq(
        api_key=os.getenv("Groq_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    # --- Input Handling (from your original code) ---
    if "raw_text" in data:
        report_input = data["raw_text"]
    elif "json_data" in data:
        report_input = json.dumps(data["json_data"], indent=2)
    else:
        raise ValueError("Input must include either 'raw_text' or 'json_data'.")

    # --- Text Splitting (to prevent 413 error) ---
    # We split the large report_input into smaller documents
    # The Groq 8b model limit is high, but the TPM limit you hit was 6k.
    # Let's use a conservative chunk size (e.g., 8000 chars) to be safe.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, 
        chunk_overlap=400
    )
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(report_input)]

    # If the text is small, we can skip map-reduce.
    # We'll check if we only have 1 doc (chunk).
    if len(docs) == 1:
        print("--- Input is small, running simple chain ---")
        # Use your original template directly
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

            3. **Chart Data Section** (very important):
               - Output in the following exact format:
                 chart_type: pie
                 labels: ["Category1", "Category2", "Category3"]
                 values: [10, 20, 30]
               - Pick meaningful categories and values from the report.
               - Ensure labels and values count match.
            
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

    # --- Map-Reduce for Large Text (if len(docs) > 1) ---
    print(f"--- Input is large, running map-reduce with {len(docs)} chunks ---")

    # 1. "Map" Prompt: Summarizes each chunk individually.
    map_prompt_template = """
    You are a helpful assistant. Summarize the following text snippet from a larger report,
    focusing on key facts, figures, and important findings.

    "{text}"

    CONCISE SUMMARY:
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # 2. "Reduce" Prompt: This is your original, detailed prompt.
    # It will run *once* on the *combined summaries* from the map step.
    reduce_template = """
    You are an AI assistant that analyzes reports and provides concise summaries,
    actionable suggestions, and structured chart data.

    ### Input Report:
    You will be given a set of summaries from a larger report. Your job is to
    synthesize these summaries into a final output.
    
    {text}

    ### Output Instructions:
    Based *only* on the text provided above, generate the following:

    1. **Summary Section**:
       - Write 4–6 bullet points.
       - Keep each point short, simple, and informative.
       - **Bold important keywords**.

    2. **Suggestions Section**:
       - Provide 3–5 practical and actionable suggestions.
       - Each suggestion should start with a verb (e.g., "Improve", "Enhance").
       - **Bold key terms**.

    3. **Chart Data Section** (very important):
       - Output in the following exact format:
         chart_type: pie
         labels: ["Category1", "Category2", "Category3"]
         values: [10, 20, 30]
       - Pick meaningful categories and values from the report.
       - Ensure labels and values count match.
    
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

    # --- Load and Run the Map-Reduce Chain ---
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=True  # Set to False in production
    )

    # The chain returns a dictionary, e.g., {"output_text": "..."}
    response = chain.invoke({"input_documents": docs})
    
    return response["output_text"]