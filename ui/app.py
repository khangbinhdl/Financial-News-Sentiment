"""
Streamlit frontend for sentiment analysis.
"""
import streamlit as st
import requests
import pandas as pd
from typing import Dict, List

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


def fetch_available_devices() -> List[str]:
    """Fetch available devices from API health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            payload = response.json()
            return payload.get("available_devices", ["cpu"])
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot connect to API server. Please make sure the API is running on http://127.0.0.1:8000"
        )
        return ["cpu"]
    except Exception as e:
        st.error(f"Error fetching devices: {str(e)}")
        return ["cpu"]
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return None


def predict_sentiment(text: str, model_name: str, device: str):
    """Call API to predict sentiment."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "model_name": model_name, "device": device},
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


def format_label(label: str) -> str:
    return label.replace("_", " ").title()




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

    models = fetch_available_models()
    if models:
        model_options = {model["name"]: model["key"] for model in models}
    else:
        st.warning("Models not available. Check if API is running.")
        return

    devices = fetch_available_devices()
    if not devices:
        devices = ["cpu"]
    available_devices = set(devices)
    device_catalog = [("CPU", "cpu"), ("CUDA", "cuda"), ("MPS", "mps")]
    device_options = []
    device_value_map = {}
    for label, value in device_catalog:
        if value in available_devices:
            option_label = label
        else:
            option_label = f"{label} (unavailable)"
        device_options.append(option_label)
        device_value_map[option_label] = value

    settings_col, info_col, status_col = st.columns([2, 1.5, 1])

    with settings_col:
        st.subheader("⚙️ Model & Device")
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            help="Choose which model to use for sentiment analysis",
        )
        selected_model = model_options[selected_model_name]
        selected_device_label = st.selectbox(
            "Select Device",
            options=device_options,
            help="Choose device for inference",
        )
        selected_device = device_value_map[selected_device_label]
        device_available = selected_device in available_devices
        if not device_available:
            st.warning("Selected device is not available on this machine.")

    with info_col:
        st.subheader("ℹ️ About")
        st.markdown(
            """
            This application uses trained models to classify financial news as:
            - **Negative** 📉
            - **Neutral** ➡️
            - **Positive** 📈
            """
        )

    with status_col:
        st.subheader("✅ Ready")
        st.metric(label="Model", value=selected_model_name)
        st.metric(label="Device", value=selected_device.upper())

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
        st.info(
            f"**Selected Model:** {selected_model_name}\n\n**Device:** {selected_device.upper()}"
        )

    st.divider()

    # Predict button
    if st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True,
        type="primary",
        disabled=not device_available,
    ):
        if not text_input or not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔄 Analyzing sentiment..."):
                result = predict_sentiment(
                    text_input.strip(), selected_model, selected_device
                )

            if result:
                st.success("✅ Analysis completed!")

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Predicted Sentiment",
                        value=format_label(result["sentiment"]),
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

                # Detailed probabilities table
                st.subheader("📋 Class Probabilities")

                prob_data = []
                for sentiment, prob in result["probabilities"].items():
                    is_predicted = "✓" if sentiment == result["sentiment"] else ""
                    prob_data.append(
                        {
                            "Sentiment": f"{format_label(sentiment)} {is_predicted}",
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
            <p>💡 Tip: Compare custom RNNs vs MiniLM on the same headline</p>
            <p>🔗 API: http://127.0.0.1:8000</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
