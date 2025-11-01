import matplotlib
matplotlib.use('Agg')  # ← CRITICAL: Use non-interactive backend
import matplotlib.pyplot as plt
import re
from io import BytesIO
import ast

def generate_charts(summary_text, save_path: str = None):
    """
    Generate a chart (pie, bar, or line) based on structured data inside summary_text.
    
    Expected format in summary_text:
        chart_type: pie
        labels: ["A", "B", "C"]
        values: [10, 20, 30]
    
    Returns:
        dict: {"chart": BytesIO, "error": None} or {"chart": None, "error": str}
    """
    try:
        # Extract chart_type, labels, values
        chart_type_match = re.search(r"chart_type:\s*['\"]?(\w+)['\"]?", summary_text)
        labels_match = re.search(r"labels:\s*(\[.*?\])", summary_text, re.DOTALL)
        values_match = re.search(r"values:\s*(\[.*?\])", summary_text, re.DOTALL)

        if not all([chart_type_match, labels_match, values_match]):
            print("Missing chart_type, labels, or values in summary_text.")
            return {"chart": None, "error": "Missing chart data in summary"}

        chart_type = chart_type_match.group(1).lower()
        labels = ast.literal_eval(labels_match.group(1))
        values = ast.literal_eval(values_match.group(1))

        if not labels or not values or len(labels) != len(values):
            print("Labels and values are missing or mismatched.")
            return {"chart": None, "error": "Labels/values mismatch"}

        # Create chart
        fig, ax = plt.subplots(figsize=(8, 6))

        if chart_type == 'pie':
            wedges, texts, autotexts = ax.pie(
                values,
                autopct='%1.1f%%',
                startangle=140,
                textprops=dict(color="black")
            )
            ax.axis('equal')
            ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.setp(autotexts, size=8, weight="bold")
            ax.set_title("Summary Insights Distribution", pad=20)

        elif chart_type == 'bar':
            bars = ax.bar(labels, values, color="skyblue")
            ax.set_ylabel('Values')
            ax.set_title('Summary Data')
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval}', ha='center', va='bottom')

        elif chart_type == 'line':
            ax.plot(labels, values, marker="o", linestyle="-", color="green")
            ax.set_ylabel('Values')
            ax.set_title('Summary Data (Line Chart)')
            for i, v in enumerate(values):
                ax.text(i, v + 0.5, str(v), ha='center')

        else:
            print(f"Unsupported chart type: {chart_type}")
            plt.close(fig)
            return {"chart": None, "error": f"Unsupported chart type: {chart_type}"}

        # Save to file if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)

        # Save to BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        # CRITICAL: Close the figure to free memory
        plt.close(fig)

        return {"chart": buf, "error": None}

    except (SyntaxError, ValueError) as e:
        print(f"Parsing error: {e}")
        return {"chart": None, "error": f"Parsing error: {str(e)}"}
    except Exception as e:
        print(f"Error generating chart: {e}")
        return {"chart": None, "error": f"Chart generation error: {str(e)}"}

