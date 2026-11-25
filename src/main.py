import json
import joblib
import numpy as np
from typing import Dict
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.database import SessionLocal, EvaluacionDemandaRuta, init_db

init_db()

app = FastAPI(
    title="API Movilidad RUTA Puebla - Predicción de Saturación",
    description="Sistema End-to-End con Datos Abiertos Puebla, ML Híbrido y PostgreSQL",
    version="1.0.0"
)

scaler = joblib.load("models/scaler.pkl")
iso_forest = joblib.load("models/iso_forest.pkl")
rf_model = joblib.load("models/hybrid_rf_model.pkl")
schema_metadata = joblib.load("models/schema_metadata.pkl")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class RutaPredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Diccionario con los valores numéricos de las características del dataset RUTA"
    )

@app.get("/", tags=["Health Check"])
def root():
    return {
        "status": "online",
        "dataset": "RUTA Puebla - 2do Trimestre 2024",
        "target_metric": schema_metadata["target_metric_col"]
    }

@app.get("/schema", tags=["Metadatos"])
def get_dataset_schema():
    return schema_metadata

@app.post("/predict", tags=["Inferencia Híbrida"])
def predict_ruta_demand(req: RutaPredictionRequest, db: Session = Depends(get_db)):
    try:
        feature_cols = schema_metadata["feature_columns"]
        ordered_values = list((float(req.features.get(col, 0.0)) for col in feature_cols))
        input_array = np.array(list((ordered_values,)))

        data_scaled = scaler.transform(input_array)
        anomaly_score = iso_forest.decision_function(data_scaled).reshape(-1, 1)
        hybrid_input = np.hstack((data_scaled, anomaly_score))

        prediction = int(rf_model.predict(hybrid_input).item())
        probability = float(rf_model.predict_proba(hybrid_input).flatten().tolist().pop(1))
        anom_val = float(anomaly_score.item())

        risk_level = "ALTO" if probability > 0.65 else ("MEDIO" if probability > 0.35 else "BAJO")

        registro = EvaluacionDemandaRuta(
            parametros_entrada=json.dumps(req.features, ensure_ascii=False),
            probabilidad_saturacion=round(probability, 4),
            indice_anomalia=round(anom_val, 4),
            nivel_alerta=risk_level,
            es_saturacion_critica=bool(prediction)
        )
        db.add(registro)
        db.commit()
        db.refresh(registro)

        return {
            "id_registro": registro.id,
            "es_saturacion_critica": bool(prediction),
            "probabilidad_saturacion": round(probability, 4),
            "indice_anomalia": round(anom_val, 4),
            "nivel_alerta": risk_level
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error en inferencia: {str(e)}")

@app.get("/historial", tags=["Analítica"])
def get_history(db: Session = Depends(get_db)):
    return db.query(EvaluacionDemandaRuta).order_by(EvaluacionDemandaRuta.id.desc()).limit(100).all()
