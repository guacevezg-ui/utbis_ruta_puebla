# Importa 'joblib' para guardar modelos serializados y 'numpy' para manipulación matricial
import joblib
import numpy as np
# Importa IsolationForest (No Supervisado) y RandomForestClassifier (Supervisado)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
# Importa StratifiedKFold (validación cruzada estratificada) y GridSearchCV (optimización)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# Importa métricas de evaluación de clasificación y área bajo la curva ROC
from sklearn.metrics import classification_report, roc_auc_score
# Importa las funciones de ETL y carga a S3 desde el módulo data_pipeline
from data_pipeline import run_etl_pipeline, upload_to_s3

def train_hybrid_ruta_model():
    # 1. Ejecuta el pipeline ETL y recibe los arreglos escalados de entrenamiento y prueba
    X_train, X_test, y_train, y_test = run_etl_pipeline()

    # 2. CAPA NO SUPERVISADA: DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST)
    print("[ML] Entrenando capa no supervisada (IsolationForest)...")
    # Configura 100 árboles de aislamiento esperando un 10% de eventos atípicos
    iso_forest = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    # Ajusta el modelo no supervisado exclusivamente sobre las características X_train
    iso_forest.fit(X_train)

    # decision_function() retorna un vector 1D (N,) con el puntaje de normalidad.
    # .reshape(-1, 1) redimensiona ese vector horizontal a una matriz columna 2D de forma (N, 1)
    train_anom = iso_forest.decision_function(X_train).reshape(-1, 1)
    test_anom = iso_forest.decision_function(X_test).reshape(-1, 1)

    # np.hstack() concatena horizontalmente la columna de anomalía a la derecha de X_train y X_test
    X_train_hybrid = np.hstack((X_train, train_anom))
    X_test_hybrid = np.hstack((X_test, test_anom))

    # 3. CAPA SUPERVISADA CON OPTIMIZACIÓN DE HIPERPARÁMETROS (GRIDSEARCHCV)
    print("[ML] Optimizando RandomForest con Validación Cruzada Estratificada (5-Folds)...")
    # Define la rejilla de hiperparámetros a evaluar durante la búsqueda exhaustiva

    param_grid = {
        "n_estimators": list((100, 200, 300)),
        "max_depth": list((6, 10, None)),
        "min_samples_split": list((2, 5))
    }

    # Configura 5 particiones estratificadas mezclando aleatoriamente los registros
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Instancia el clasificador base ponderando inversamente las frecuencias de clase
    base_rf = RandomForestClassifier(class_weight="balanced", random_state=42)

    # Configura GridSearchCV evaluando cada combinación mediante la métrica ROC-AUC en paralelo
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    # Entrena todas las combinaciones sobre la matriz enriquecida X_train_hybrid
    grid_search.fit(X_train_hybrid, y_train)
    # Extrae el estimador ganador con el mejor desempeño promedio en validación cruzada
    best_rf = grid_search.best_estimator_

    # Imprime los hiperparámetros óptimos encontrados y su puntaje ROC-AUC
    print(f"[OPTIMIZACIÓN] Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"[OPTIMIZACIÓN] Mejor ROC-AUC (CV): {grid_search.best_score_:.4f}")

    # 4. EVALUACIÓN FINAL SOBRE EL CONJUNTO DE PRUEBA (TEST SET)
    # Genera el arreglo 1D de predicciones binarias (0 o 1)
    y_pred = best_rf.predict(X_test_hybrid)
    # predict_proba() retorna matriz (N, 2); el corte [:, 1] extrae la probabilidad de clase 1
    y_proba = best_rf.predict_proba(X_test_hybrid)[:, 1]

    # Despliega el reporte estadístico completo en consola
    print("\n" + "="*55)
    print("   REPORTE DE EVALUACIÓN - MOVILIDAD RUTA PUEBLA   ")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=["Demanda Normal", "Saturación Crítica"]))
    print(f"ROC-AUC Score en Test: {roc_auc_score(y_test, y_proba):.4f}")
    print("="*55 + "\n")

    # 5. SERIALIZACIÓN DE ARTEFACTOS Y CARGA A AWS S3
    # Guarda ambos modelos entrenados en el directorio local 'models/'
    joblib.dump(iso_forest, "models/iso_forest.pkl")
    joblib.dump(best_rf, "models/hybrid_rf_model.pkl")

    # Sincroniza los modelos serializados hacia la carpeta 'artifacts/' en AWS S3
    upload_to_s3("models/iso_forest.pkl", "artifacts/iso_forest.pkl")
    upload_to_s3("models/hybrid_rf_model.pkl", "artifacts/hybrid_rf_model.pkl")

# Punto de entrada principal del script de entrenamiento
if __name__ == "__main__":
    train_hybrid_ruta_model()
