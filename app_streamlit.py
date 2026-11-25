# Importa Streamlit para interfaz web, Requests para peticiones HTTP, Pandas y Plotly para visualización
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Configura el título de la pestaña del navegador, ícono y disposición de pantalla completa
st.set_page_config(page_title="Movilidad RUTA Puebla - UTBIS", page_icon="🚌", layout="wide")

# Renderiza el encabezado principal y la narrativa alineada con los criterios SEAES
st.title("🚌 Analítica Predictiva de Movilidad Urbana — RUTA Puebla")
st.markdown("""
**Criterios Transversales SEAES: Responsabilidad Social e Inclusión.**  
Sistema basado en datos oficiales de **Carreteras de Cuota Puebla (2do Trimestre 2024)** para anticipar 
escenarios de saturación crítica en la Red Urbana de Transporte Articulado (RUTA) y optimizar la asignación de unidades.
""")

# Crea un campo en la barra lateral para configurar la IP pública de la instancia AWS EC2
st.sidebar.header("⚙️ Configuración Cloud AWS")
ec2_ip = st.sidebar.text_input("IP Pública EC2", value="localhost")
# Construye la URL base del servidor backend FastAPI
base_url = f"http://{ec2_ip}:8000"

# Organiza la aplicación en tres pestañas funcionales
tab1, tab2, tab3 = st.tabs([
    "🔮 Simulador de Demanda (API + PostgreSQL)",
    "📈 Exploración Datos Abiertos Puebla",
    "🗄️ Historial de Predicciones (BD)"
])

# ==============================================================================
# PESTAÑA 1: SIMULADOR INTERACTIVO DE INFERENCIA
# ==============================================================================
with tab1:
    st.subheader("Evaluación de Escenario Operativo en RUTA")
    try:
        # Consulta el esquema real del dataset al endpoint /schema de FastAPI
        schema_res = requests.get(f"{base_url}/schema", timeout=5)
        if schema_res.status_code == 200:
            schema = schema_res.json()
            feature_cols = schema["feature_columns"]
            cat_mappings = schema["category_mappings"]
            stats = schema["feature_stats"]

            # Muestra un banner informativo con la métrica analizada y el umbral crítico calculado
            st.info(f"Variable objetivo analizada: **{schema['target_metric_col']}** (Umbral de saturación crítica: ≥ {schema['umbral_critico']:,.0f} viajes)")

            # Diccionario que recopilará los valores ingresados por el usuario
            user_inputs = {}
            # Divide el formulario en dos columnas visuales
            cols = st.columns(2)

            # Genera dinámicamente un control de entrada por cada variable predictora
            for idx, col_name in enumerate(feature_cols):
                with cols[idx % 2]:
                    # Si la columna es categórica, despliega un menú con las etiquetas de texto originales
                    if col_name in cat_mappings:
                        mapping = cat_mappings[col_name]
                        # Invierte el diccionario {código: texto} a {texto: código entero}
                        label_to_code = {v: int(k) for k, v in mapping.items()}
                        selected_label = st.selectbox(f"{col_name}", list(label_to_code.keys()))
                        user_inputs[col_name] = float(label_to_code[selected_label])
                    else:
                        # Si es numérica, crea un campo numérico acotado por las estadísticas reales del dataset
                        min_v = float(stats[col_name]["min"])
                        max_v = float(stats[col_name]["max"])
                        mean_v = float(stats[col_name]["mean"])
                        user_inputs[col_name] = st.number_input(
                            f"{col_name}", min_value=min_v, max_value=max_v * 1.5, value=mean_v
                        )

            # Botón que dispara la petición POST hacia FastAPI (/predict)
            if st.button("🚀 Predecir Demanda y Guardar en PostgreSQL"):
                pred_res = requests.post(f"{base_url}/predict", json={"features": user_inputs}, timeout=5)
                if pred_res.status_code == 200:
                    res = pred_res.json()
                    st.success(f"✅ Evaluación #{res['id_registro']} almacenada correctamente en PostgreSQL.")

                    # Despliega tres tarjetas KPI con los resultados del modelo híbrido
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Probabilidad de Saturación", f"{res['probabilidad_saturacion']*100:.2f}%")
                    c2.metric("Índice Anomalía (IsolationForest)", f"{res['indice_anomalia']}")
                    c3.metric("Nivel de Alerta Operativa", res["nivel_alerta"])

                    # Muestra recomendación operativa accionable según el diagnóstico
                    if res["es_saturacion_critica"]:
                        st.error("🚨 **ALERTA DE SATURACIÓN CRÍTICA:** Se recomienda despachar unidades articuladas de refuerzo y habilitar carriles de desahogo en estaciones troncales.")
                    else:
                        st.info("🟢 **OPERACIÓN FLUÍDA:** La demanda proyectada se mantiene dentro de la capacidad instalada de la flota.")
        else:
            st.warning("No se pudo obtener el esquema desde FastAPI. Verifica que el backend esté activo.")
    except Exception as e:
        st.error(f"Error conectando con FastAPI en {base_url}: {e}")

# ==============================================================================
# PESTAÑA 2: EXPLORACIÓN DESCRIPTIVA DEL DATASET OFICIAL DE PUEBLA
# ==============================================================================
with tab2:
    st.subheader("Análisis Descriptivo — Dataset Oficial Carreteras de Cuota Puebla")
    try:
        # Carga desde disco local el respaldo crudo descargado en la capa bronze
        df_bronze = pd.read_csv("data/ruta_puebla_bronze.csv")
        st.write(f"Total de registros oficiales cargados: **{len(df_bronze)}**")
        # Renderiza una vista previa interactiva de los primeros 20 registros
        st.dataframe(df_bronze.head(20), use_container_width=True)

        # Detecta automáticamente columnas categóricas y numéricas para graficar
        cat_cols = df_bronze.select_dtypes(include=["object"]).columns.tolist()
        num_cols = df_bronze.select_dtypes(include=["number"]).columns.tolist()

        if cat_cols and num_cols:
            col_x = st.selectbox("Agrupar por categoría:", cat_cols, index=0)
            col_y = st.selectbox("Métrica de viajes/usuarios:", num_cols, index=len(num_cols)-1)
            # Agrupa los datos y construye un gráfico de barras interactivo con Plotly
            fig = px.bar(
                df_bronze.groupby(col_x, as_index=False)[col_y].sum(),
                x=col_x, y=col_y, color=col_x,
                title=f"Distribución de {col_y} por {col_x} (RUTA Puebla)"
            )
            st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Ejecuta primero `python src/train.py` para descargar el dataset oficial de Puebla.")

# ==============================================================================
# PESTAÑA 3: CONSULTA AL HISTORIAL EN POSTGRESQL
# ==============================================================================
with tab3:
    st.subheader("Bitácora de Inferencias en PostgreSQL")
    # Botón para consultar el historial actualizado desde la base de datos relacional
    if st.button("🔄 Refrescar Tabla PostgreSQL"):
        try:
            hist_res = requests.get(f"{base_url}/historial", timeout=5)
            if hist_res.status_code == 200 and len(hist_res.json()) > 0:
                # Convierte la respuesta JSON en un DataFrame y la muestra en tabla
                df_hist = pd.DataFrame(hist_res.json())
                st.dataframe(df_hist, use_container_width=True)
            else:
                st.info("No hay registros guardados todavía.")
        except Exception as e:
            st.error(f"Error consultando historial: {e}")
