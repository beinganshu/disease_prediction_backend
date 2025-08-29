from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import numpy as np
import re

# Load all assets
model = joblib.load("assets/RandomForest_model.pkl")
symptom_cols = joblib.load("assets/symptom_cols.pkl")
label_encoder = joblib.load("assets/label_encoder.pkl")
disease_symptom_map = joblib.load("assets/disease_symptom_map.pkl")

# Load disease details and precautions from Excel
disease_info_df = pd.read_excel("assets/disease_details_with_precautions.xlsx")
disease_info_map = {
    str(row["Disease"]).strip().lower(): f"**Description:** {row['Details']}\n\n**Precautions:**\n{row['Precautions']}"
    for _, row in disease_info_df.iterrows()
}

# Initialize FastAPI app
app = FastAPI(title="Disease Prediction Chatbot")

# Pydantic schema
class SymptomInput(BaseModel):
    message: str

# Helper function to extract known symptoms from user message
def extract_symptoms_from_text(text, known_symptoms):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9_ ]", "", text)
    found = []
    for s in known_symptoms:
        if s.replace("_", " ") in text:
            found.append(s)
    return found

# Convert symptoms to model input format
def symptoms_to_input_vector(symptoms, all_symptoms):
    input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for s in symptoms:
        if s in input_vector.columns:
            input_vector.at[0, s] = 1
    return input_vector

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>✅ Symptom Prediction API</h2>
    <p>This backend is running! Use the <code>/predict</code> endpoint to POST symptom input.</p>
    """

@app.post("/predict")
async def predict_symptoms(symptoms: SymptomInput):
    extracted = extract_symptoms_from_text(symptoms.message, symptom_cols)

    if not extracted:
        raise HTTPException(status_code=400, detail="No recognizable symptoms found.")

    input_vector = symptoms_to_input_vector(extracted, symptom_cols)
    probs = model.predict_proba(input_vector)[0]

    top_n = 7
    top_indices = np.argsort(probs)[::-1][:top_n]

    predictions = []
    for i in top_indices:
        disease = label_encoder.classes_[i]
        confidence = round(float(probs[i]), 4)

        # Normalize key for lookup
        norm_disease = disease.strip().lower()
        disease_info = disease_info_map.get(norm_disease)
        print(disease_info)
        known_symptoms = disease_symptom_map.get(disease, [])

        predictions.append({
            "disease": disease,
            "confidence": confidence,
            "precautions": disease_info,
            "known_symptoms": known_symptoms
        })

    return {
        "extracted_symptoms": extracted,
        "predictions": predictions
    }
