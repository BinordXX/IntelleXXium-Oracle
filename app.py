# app.py

import streamlit as st
import pandas as pd
from forecasts.future_predictor import predict_from_df


st.set_page_config(page_title="IntelleXXium Oracle", layout="centered")

# 🧙‍♂️ App Title
st.title("🔮 IntelleXXium Oracle: Business Forecasting AI")

# 📝 Description
st.markdown("""
Welcome to the **Oracle**. Upload your cleaned business data (CSV format), and let our trained neural prophet forecast your sales.
""")

# 📤 Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    try:
        # Read file
        input_df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(input_df.head())

        # Predict Button
        if st.button("Predict Future Sales"):
            with st.spinner("allow model a munuite to make predictions..."):
                result_df = predict_from_df(input_df)

            st.success("Prediction complete!:")
            st.dataframe(result_df[['predicted_sales']].head())

            # 📥 Allow download of full predictions
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv,
                file_name="predicted_sales.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("Awaiting your CSV file to begin the foresight...")
