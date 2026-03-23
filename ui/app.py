"""
Streamlit frontend for sentiment analysis.
"""
import streamlit as st
import requests
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    page_icon="📊",
    layout="wide",
)

# Styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        color: #1f77b4;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API endpoint
API_URL = "http://127.0.0.1:8000"


def fetch_available_models():
    """Fetch available models from API."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot connect to API server. Please make sure the API is running on http://127.0.0.1:8000"
        )
        return None
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return None


def predict_sentiment(text: str, model_name: str):
    """Call API to predict sentiment."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "model_name": model_name},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API server. Please make sure the API is running.")
        return None
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


def plot_probabilities(probabilities: Dict[str, float], sentiment: str):
    """Plot sentiment probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(probabilities.keys())
    values = list(probabilities.values())

    # Define colors for each sentiment
    colors = {"Negative": "#ef553b", "Neutral": "#636efa", "Positive": "#00cc96"}
    bar_colors = [colors.get(label, "#636efa") for label in labels]

    # Highlight the predicted sentiment
    bar_colors = [
        colors.get(label, "#636efa") if label != sentiment else "gold" for label in labels
    ]

    bars = ax.bar(labels, values, color=bar_colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Probability", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=12, fontweight="bold")
    ax.set_title("Sentiment Probability Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig


def main():
    """Main Streamlit app."""
    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 20px;'>
            <h1>📊 Financial News Sentiment Analysis</h1>
            <p style='font-size: 18px; color: #666;'>Analyze the sentiment of financial news using AI models</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Fetch available models
        models = fetch_available_models()

        if models:
            model_options = {model["name"]: model["key"] for model in models}
            selected_model_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                help="Choose which model to use for sentiment analysis",
            )
            selected_model = model_options[selected_model_name]
        else:
            st.warning("Models not available. Check if API is running.")
            return

        st.divider()
        st.markdown("### About")
        st.markdown(
            """
            This application uses trained models to classify financial news as:
            - **Negative** 📉
            - **Neutral** ➡️
            - **Positive** 📈
            """
        )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter News Text")
        text_input = st.text_area(
            "Enter financial news text for sentiment analysis:",
            height=150,
            placeholder="Example: Apple Inc. announced record quarterly earnings...",
        )

    with col2:
        st.subheader("🎯 Model Info")
        st.info(f"**Selected Model:** {selected_model_name}")

    st.divider()

    # Predict button
    if st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary"):
        if not text_input or not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔄 Analyzing sentiment..."):
                result = predict_sentiment(text_input.strip(), selected_model)

            if result:
                st.success("✅ Analysis completed!")

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Predicted Sentiment",
                        value=result["sentiment"],
                        delta=None,
                    )

                with col2:
                    sentiment_confidence = max(result["probabilities"].values())
                    st.metric(
                        label="Confidence",
                        value=f"{sentiment_confidence:.2%}",
                        delta=None,
                    )

                with col3:
                    st.metric(
                        label="Model Used",
                        value=result["model_used"],
                        delta=None,
                    )

                st.divider()

                # Display probability distribution
                st.subheader("📊 Probability Distribution")

                # Chart
                fig = plot_probabilities(result["probabilities"], result["sentiment"])
                st.pyplot(fig)

                # Detailed probabilities table
                st.subheader("📋 Detailed Results")

                prob_data = []
                for sentiment, prob in result["probabilities"].items():
                    is_predicted = "✓" if sentiment == result["sentiment"] else ""
                    prob_data.append(
                        {
                            "Sentiment": f"{sentiment} {is_predicted}",
                            "Probability": f"{prob:.4f}",
                            "Percentage": f"{prob:.2%}",
                        }
                    )

                df = pd.DataFrame(prob_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #999; font-size: 12px; padding: 20px;'>
            <p>💡 Tip: Try different models to compare their predictions</p>
            <p>🔗 API: http://127.0.0.1:8000</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
