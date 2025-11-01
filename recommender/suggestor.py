import numpy as np
from scipy import stats

def get_recommendations(data):

    
    # Initialize recommendations list
    recommendations = []
    
    # Check for keywords in raw text
    if "raw_text" in data:
        text = data["raw_text"].lower()
        if any(keyword in text for keyword in ["drop", "decline", "decrease", "down", "fall"]):
            recommendations.append("Investigate recent performance drop mentioned in report.")
        if any(keyword in text for keyword in ["increase", "growth", "rise", "up", "improve"]):
            recommendations.append("Positive trends mentioned in report - analyze success factors.")
        if any(keyword in text for keyword in ["risk", "issue", "problem", "concern"]):
            recommendations.append("Address potential risks highlighted in the report.")
    
    # Analyze data if available
    if "df" in data:
        df = data["df"]
        if df.shape[1] >= 2:
            try:
                # Get numeric values from second column
                values = df.iloc[:, 1].dropna().astype(float)
                
                if len(values) >= 3:  # Need at least 3 points for meaningful analysis
                    # Calculate trend using linear regression
                    x = np.arange(len(values))
                    slope, _, _, p_value, _ = stats.linregress(x, values)
                    
                    # Calculate percentage change
                    pct_change = (values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100
                    
                    # Calculate volatility (standard deviation)
                    volatility = values.std()
                    
                    # Analyze trend
                    if p_value < 0.05:  # Statistically significant trend
                        if slope > 0:
                            if pct_change > 20:
                                recommendations.append("Strong upward trend detected. Consider scaling successful initiatives.")
                            else:
                                recommendations.append("Moderate growth observed. Identify key drivers for further improvement.")
                        else:
                            if pct_change < -20:
                                recommendations.append("Significant decline detected. Immediate intervention required.")
                            else:
                                recommendations.append("Gradual decline observed. Investigate root causes.")
                    else:  # No statistically significant trend
                        if volatility > values.mean() * 0.3:  # High volatility
                            recommendations.append("High volatility in metrics. Stabilize processes for consistent performance.")
                        else:
                            recommendations.append("Metrics are stable. Focus on optimization and incremental improvements.")
                    
                    # Check for recent anomaly (last point significantly different from trend)
                    if len(values) >= 4:
                        # Predict last value based on previous points
                        prev_x = np.arange(len(values)-1)
                        prev_y = values.iloc[:-1]
                        prev_slope, _, _, _, _ = stats.linregress(prev_x, prev_y)
                        predicted_last = prev_y.iloc[-1] + prev_slope
                        
                        # If actual last value is more than 20% different from predicted
                        if abs(values.iloc[-1] - predicted_last) / abs(predicted_last) > 0.2:
                            if values.iloc[-1] > predicted_last:
                                recommendations.append("Recent unexpected improvement. Identify and replicate success factors.")
                            else:
                                recommendations.append("Recent unexpected decline. Investigate recent changes or events.")
                
                elif len(values) == 2:  # Only two points available
                    if values.iloc[-1] < values.iloc[0]:
                        recommendations.append("Recent drop detected in metrics. Investigate last phase performance.")
                    else:
                        recommendations.append("Recent improvement observed. Continue current strategies.")
            except Exception as e:
                recommendations.append(f"Data analysis error: {str(e)}")
    
    # If no specific recommendations were generated
    if not recommendations:
        return {"action": "No critical issues detected. Continue monitoring metrics."}
    
    # Combine all recommendations
    return {"action": " | ".join(recommendations)}


