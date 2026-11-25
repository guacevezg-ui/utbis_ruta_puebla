# Importa el módulo estándar 'os' para leer las variables de entorno del sistema
import os
# Importa 'json' para serializar diccionarios de Python a cadenas de texto JSON
import json
# Importa 'datetime' para generar marcas de tiempo (timestamps) en cada registro
from datetime import datetime
# Importa 'load_dotenv' para cargar las credenciales almacenadas en el archivo oculto .env
from dotenv import load_dotenv
# Importa 'create_engine' para abrir la conexión con PostgreSQL y los tipos de datos de columnas SQL
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
# Importa 'declarative_base' (clase madre de las tablas ORM) y 'sessionmaker' (gestor de sesiones)
from sqlalchemy.orm import declarative_base, sessionmaker

# Carga en memoria las variables definidas dentro del archivo .env
load_dotenv()

# Obtiene la cadena de conexión a PostgreSQL (usuario, contraseña, host y base de datos)
DATABASE_URL = os.getenv("DATABASE_URL")
# Crea el motor principal (Engine) que administra el pool de conexiones hacia PostgreSQL
engine = create_engine(DATABASE_URL)
# Crea la fábrica de sesiones; autocommit=False exige confirmar transacciones explícitamente con db.commit()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Instancia la clase base de la cual heredarán los modelos de tablas relacionales
Base = declarative_base()

# Define el modelo ORM 'EvaluacionDemandaRuta', que mapea a una tabla física en PostgreSQL
class EvaluacionDemandaRuta(Base):
    # Asigna el nombre exacto de la tabla dentro de la base de datos 'ruta_puebla_db'
    __tablename__ = "evaluaciones_demanda_ruta"

    # Columna 'id': entero autoincremental que actúa como llave primaria (Primary Key) e índice
    id = Column(Integer, primary_key=True, index=True)
    # Columna 'fecha_registro': almacena automáticamente la fecha y hora UTC al insertar la fila
    fecha_registro = Column(DateTime, default=datetime.utcnow)
    # Columna 'parametros_entrada': texto largo que guarda el JSON con las variables evaluadas
    parametros_entrada = Column(Text, nullable=False)
    # Columna 'probabilidad_saturacion': número decimal con la probabilidad devuelta por RandomForest
    probabilidad_saturacion = Column(Float, nullable=False)
    # Columna 'indice_anomalia': número decimal con el puntaje de normalidad/anomalía de IsolationForest
    indice_anomalia = Column(Float, nullable=False)
    # Columna 'nivel_alerta': cadena de texto de hasta 20 caracteres ('BAJO', 'MEDIO' o 'ALTO')
    nivel_alerta = Column(String(20), nullable=False)
    # Columna 'es_saturacion_critica': valor booleano (True/False) según el umbral operativo
    es_saturacion_critica = Column(Boolean, nullable=False)

# Función que verifica si las tablas declaradas existen en PostgreSQL y las crea si no existen
def init_db():
    Base.metadata.create_all(bind=engine)
