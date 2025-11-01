import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json

load_dotenv()

def generate_summary(data):
    """
    Generate a summary + suggestions + chart data.
    """

    llm = ChatGroq(
        api_key=os.getenv("Groq_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    template = ChatPromptTemplate.from_template(
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

    # Handle input type
    if "raw_text" in data:
        report_input = data["raw_text"]
    elif "json_data" in data:
        report_input = json.dumps(data["json_data"], indent=2)
    else:
        raise ValueError("Input must include either 'raw_text' or 'json_data'.")

    chain = template | llm
    response = chain.invoke({"report": report_input})

    return response.content if hasattr(response, "content") else str(response)




