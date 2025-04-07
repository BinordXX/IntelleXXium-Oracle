import streamlit as st
import pandas as pd
from forecasts.future_predictor import predict_from_df
import plotly.express as px

st.set_page_config(page_title="IntelleXXium Oracle", layout="centered")

# App title and description
st.title("🔮 IntelleXXium Oracle: Business Forecasting AI")
st.markdown("""
Upload your cleaned business data (CSV format) with only the features, and let our trained neural prophet forecast your sales.
""")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded CSV
        input_df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(input_df.head())

        # Predict Button
        if st.button("Predict Future Sales"):
            with st.spinner("The model is making predictions... Please wait a moment!"):
                result_df = predict_from_df(input_df)

            st.success("Prediction complete!")
            st.subheader("Predicted Sales:")
            st.dataframe(result_df[['predicted_sales']].head())

            # 📈 Trend Visualizations
            st.subheader("📊 Trend Visualizations")

            if 'order_date' in result_df.columns:
                result_df['order_date'] = pd.to_datetime(result_df['order_date'])
                result_df['month'] = result_df['order_date'].dt.month
                result_df['weekday'] = result_df['order_date'].dt.weekday
                result_df['year'] = result_df['order_date'].dt.year

                st.plotly_chart(px.line(result_df, x='order_date', y='predicted_sales', title='Predicted Sales Over Time'))

                st.plotly_chart(px.box(result_df, x='month', y='predicted_sales', title='Monthly Sales Distribution'))

                st.plotly_chart(px.box(result_df, x='weekday', y='predicted_sales', title='Sales by Day of the Week'))

            # Download button for predictions
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
