import streamlit as st
import pandas as pd
import os
import subprocess

st.set_page_config(page_title="Clinical Trial Dropout Predictor", layout="wide")
st.title("🧬 Clinical Trial Dropout Prediction App")

st.markdown("Upload a patient dataset to predict who is likely to drop out of a clinical trial.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Step 1: Raw Uploaded Data")
    st.dataframe(df.head())

    # Basic cleaning - fill missing with 'Unknown'
    df.fillna("Unknown", inplace=True)

    st.subheader("Step 2: Cleaned Data")
    st.dataframe(df.head())

    # Limit the number of rows to improve speed
    max_rows = st.slider("Number of patients to predict (lower is faster)", 1, min(50, len(df)), 10)
    df = df.head(max_rows)

    # Define prompt creation
    def create_prompt(row):
        return (
            f"Patient ID: {row['Patient_ID']}. "
            f"{row['Age']} years old {row['Gender']} of {row['Ethnicity']} ethnicity. "
            f"Education: {row['Education']}. Employment: {row['Employment']}. "
            f"Medical history includes {row['Medical_History']}. Allergies: {row['Allergies']}. "
            f"Comorbidities: {row['Comorbidities']}. Medication history: {row['Medication_History']}. "
            f"Alcohol use: {row['Alcohol']}, Sleep: {row['Sleep']}, Exercise: {row['Exercise']}. "
            f"Stress level is {row['Stress_Level']} and motivation is {row['Motivation']}. "
            f"Trial understanding: {row['Trial_Understanding']}. Personal goals: {row['Personal_Goals']}. "
            f"Family support: {row['Family_Support']}, Transportation: {row['Transportation_Access']}. "
            f"Based on this profile, will the patient drop out of the clinical trial? Reply only 'stay' or 'drop'."
        )

    df['Prompt'] = df.apply(create_prompt, axis=1)

    st.subheader("Step 3: Sample LLaMA Prompts")
    st.dataframe(df[['Patient_ID', 'Prompt']].head())

    st.subheader("Step 4: Get Predictions from LLaMA 3")

    if st.button("Run Predictions with LLaMA 3"):
        results = []
        for prompt in df['Prompt']:
            try:
                result = subprocess.run(
                    f'echo "{prompt}" | ollama run llama3',
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60
                )
                output = result.stdout.strip().lower()
                if "drop" in output:
                    results.append("Drop")
                elif "stay" in output:
                    results.append("Stay")
                else:
                    results.append("Uncertain")
            except Exception as e:
                results.append(f"Error: {e}")

        df['Prediction'] = results
        st.subheader("Step 5: Prediction Results")
        st.dataframe(df[['Patient_ID', 'Prediction']])

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
