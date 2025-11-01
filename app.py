import streamlit as st
import json
from parsers.pdf_parser import parse_pdf
from utils.file_utils import read_csv, read_excel
from summarizer.summary_agent import generate_summary
from models.predictor import predict_kpis
from visualizer.chart_generator import generate_charts
from recommender.suggestor import get_recommendations
from qa_agent.qa_bot import get_answer

st.set_page_config(page_title="AI Report Insight Agent", layout="wide")
st.title("AI Report Insight Agent")

# Enhanced session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ""
if "summary_response" not in st.session_state:
    st.session_state.summary_response = ""
if "chart_data" not in st.session_state:
    st.session_state.chart_data = None

data = None

# ------------------ File Uploader (PDF / CSV / Excel) ------------------ #
report_file = st.file_uploader("Upload a PDF / CSV / Excel report", type=["pdf", "csv", "xlsx", "xls"])

if report_file:
    # Reset state on new file upload
    st.session_state.messages = []
    st.session_state.context = ""
    st.session_state.summary_response = ""
    st.session_state.chart_data = None  # Reset chart data too

    ext = report_file.name.split(".")[-1].lower()

    try:
        if ext == "pdf":
            file_bytes = report_file.read()
            data = parse_pdf(file_bytes)
        elif ext == "csv":
            df = read_csv(report_file)
            data = {"df": df, "raw_text": df.to_string()}
        elif ext in ["xlsx", "xls"]:
            df = read_excel(report_file)
            data = {"df": df, "raw_text": df.to_string()}
        else:
            st.error("Unsupported file format.")

        if data:
            st.success("File parsed successfully!")
            
            # Initialize context immediately after successful parsing
            if "raw_text" in data:
                st.session_state.context = data["raw_text"]
            elif "df" in data:
                st.session_state.context = data["df"].to_string()
            else:
                st.session_state.context = str(data)
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# ------------------ JSON Uploader ------------------ #
json_file = st.file_uploader("Upload a JSON file", type=["json"])
json_input = st.text_area("Or paste your JSON here")

if json_file is not None:
    try:
        payload = json.load(json_file)
        data = {"json_data": payload if isinstance(payload, list) else [payload]}
        st.success("JSON file parsed successfully!")
        
        # Set context for JSON data
        st.session_state.context = json.dumps(data["json_data"], indent=2)
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {e.msg} (line {e.lineno}, column {e.colno})")
    except Exception as e:
        st.error(f"Error processing JSON file: {str(e)}")

elif json_input.strip():
    try:
        payload = json.loads(json_input)
        data = {"json_data": payload if isinstance(payload, list) else [payload]}
        st.success("JSON text parsed successfully!")
        
        # Set context for JSON data
        st.session_state.context = json.dumps(data["json_data"], indent=2)
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e.msg} (line {e.lineno}, column {e.colno})")
    except Exception as e:
        st.error(f"Error processing JSON text: {str(e)}")

# ------------------ Processing Section ------------------ #
if data:
    # Summary Section
    with st.expander("Summary", expanded=True):
        if not st.session_state.summary_response:  # Generate only once
            with st.spinner("Generating summary..."):
                try:
                    summary_text = generate_summary(data)
                    st.session_state.summary_response = summary_text
                    
                    # Enhanced context with both raw data and summary
                    base_context = data.get("raw_text", "") or json.dumps(data.get("json_data", {}), indent=2) if "json_data" in data else str(data)
                    st.session_state.context = base_context + "\n\nSUMMARY:\n" + summary_text
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")
                    st.session_state.summary_response = "Summary generation failed"

        # Display only summary + suggestions (hide chart data here)
        if st.session_state.summary_response:
            if "#### Chart Data:" in st.session_state.summary_response:
                summary_part = st.session_state.summary_response.split("#### Chart Data:")[0]
            else:
                summary_part = st.session_state.summary_response
            st.markdown(summary_part)

    # KPI Predictions Section
    with st.expander("KPI Predictions"):
        try:
            prediction = predict_kpis(data)
            st.json(prediction)
        except Exception as e:
            st.warning(f"KPI prediction failed: {e}")

    # Recommendations Section
    with st.expander("Suggestions"):
        try:
            suggestions = get_recommendations(data)
            st.json(suggestions)
        except Exception as e:
            st.warning(f"Suggestions generation failed: {e}")

    # Enhanced Charts Section
    with st.expander("Visual Insights"):
        # Generate and store chart data only once
        if not st.session_state.chart_data and st.session_state.summary_response:
            try:
                with st.spinner("Generating charts from summary..."):
                    chart_data = generate_charts(st.session_state.summary_response)
                    st.session_state.chart_data = chart_data
            except Exception as e:
                st.warning(f"Chart generation failed: {e}")
                st.session_state.chart_data = None
        
        # Display chart if available
        if st.session_state.chart_data and st.session_state.chart_data.get("chart"):
            st.image(st.session_state.chart_data["chart"].getvalue(), caption="Generated Chart")
            
            # Show additional chart information if available
            if hasattr(st.session_state.chart_data, 'get') and st.session_state.chart_data.get("chart_info"):
                chart_info = st.session_state.chart_data["chart_info"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Chart Type", chart_info.get("chart_type", "N/A").title())
                with col2:
                    st.metric("Data Points", len(chart_info.get("categories", [])))
        else:
            st.warning("No chart could be generated from the summary.")

else:
    st.info("Please upload a PDF, CSV, Excel, or JSON file to begin.")


# Enhanced get_answer wrapper function for backwards compatibility
def enhanced_get_answer(question, context, summary_data=None, chart_data=None):
    """Enhanced wrapper that handles both summary and chart queries"""
    try:
        if not context or not question:
            return "I need both a question and context to provide an answer."
        
        question_lower = question.lower()
        
        # Chart-related queries
        if any(word in question_lower for word in ['chart', 'graph', 'plot', 'visualiz', 'visual']):
            if chart_data and hasattr(chart_data, 'get') and chart_data.get("chart_info"):
                chart_info = chart_data["chart_info"]
                if "show" in question_lower or "display" in question_lower:
                    return {
                        "type": "chart_display",
                        "message": "Here's the chart generated from the summary data:",
                        "chart_data": chart_data
                    }
                elif "data" in question_lower or "values" in question_lower:
                    categories = chart_info.get("categories", [])
                    values = chart_info.get("values", [])
                    response = "**Chart Data:**\n\n"
                    for cat, val in zip(categories, values):
                        response += f"- **{cat}**: {val}\n"
                    return response
                elif "type" in question_lower:
                    return f"The chart type is: **{chart_info.get('chart_type', 'unknown')}**"
                else:
                    return "I can show you the chart, provide chart data values, or explain the chart type. What would you like to know?"
            else:
                # For chart queries when no chart data available, still try the original function
                return get_answer(question, context)
        
        # Summary-related queries  
        elif any(word in question_lower for word in ['summary', 'summarize', 'overview', 'key points']):
            if summary_data:
                # Extract just the summary part (before chart data)
                summary_text = summary_data.split("#### Chart Data:")[0].strip()
                if "brief" in question_lower or "short" in question_lower:
                    # Return first few lines for brief summary
                    lines = summary_text.split('\n')
                    brief = '\n'.join(lines[:3])
                    return f"**Brief Summary:**\n\n{brief}"
                else:
                    return f"**Complete Summary:**\n\n{summary_text}"
            else:
                # If no summary data, let the original function handle it
                return get_answer(question, context)
        
        # Data analysis queries
        elif any(word in question_lower for word in ['analyze', 'analysis', 'insights', 'findings']):
            response = "**Analysis Results:**\n\n"
            if summary_data:
                response += f"**Summary Available**: {len(summary_data)} characters of summary data\n"
            if chart_data and hasattr(chart_data, 'get') and chart_data.get("chart_info"):
                response += f"**Chart Available**: {chart_data['chart_info']['chart_type']} chart with {len(chart_data['chart_info']['categories'])} data points\n"
            response += "\nYou can ask me about:\n- Summary details\n- Chart visualization\n- Specific data points\n- Key insights"
            return response
        
        # Count/numerical queries
        elif any(word in question_lower for word in ['how many', 'count', 'number']):
            if chart_data and hasattr(chart_data, 'get') and chart_data.get("chart_info"):
                values = chart_data["chart_info"].get("values", [])
                categories = chart_data["chart_info"].get("categories", [])
                if values:
                    return f"**Data Counts:**\n- **Categories**: {len(categories)}\n- **Total Value**: {sum(values)}\n- **Average Value**: {sum(values)/len(values):.1f}"
            # Fallback to original function for counting questions
            return get_answer(question, context)
        
        # For all other queries, use the original get_answer function
        else:
            return get_answer(question, context)
            
    except Exception as e:
        return f"Error processing your question: {str(e)}"


# ------------------ Enhanced Chat Interface ------------------ #
st.header("Chat with the AI Agent")

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the report..."):
    if not data:
        st.warning("Please upload a document or JSON first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Enhanced Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Ensure we have context before calling get_answer
                    if not st.session_state.context:
                        # Fallback: create context from current data
                        if "raw_text" in data:
                            context_to_use = data["raw_text"]
                        elif "json_data" in data:
                            context_to_use = json.dumps(data["json_data"], indent=2)
                        elif "df" in data:
                            context_to_use = data["df"].to_string()
                        else:
                            context_to_use = str(data)
                    else:
                        context_to_use = st.session_state.context
                    
                    # Get summary and chart data from session state
                    summary_data = st.session_state.summary_response
                    chart_data = st.session_state.chart_data
                    
                    # Call enhanced get_answer function
                    response = enhanced_get_answer(prompt, context_to_use, summary_data, chart_data)
                    
                    # Handle special response types (like chart display requests)
                    if isinstance(response, dict) and response.get("type") == "chart_display":
                        st.markdown(response["message"])
                        if response.get("chart_data") and response["chart_data"].get("chart"):
                            st.image(response["chart_data"]["chart"].getvalue(), caption="Requested Chart")
                        response = "Chart displayed above!"
                    
                except Exception as e:
                    response = f"Error generating answer: {str(e)}"
                    st.error(f"Debug info: {str(e)}")
                    st.error(f"Context length: {len(st.session_state.context) if st.session_state.context else 0}")
                    st.error(f"Data available: {bool(data)}")
                    
                st.markdown(response)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
