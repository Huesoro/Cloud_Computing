from fastapi import FastAPI
from pydantic import BaseModel
import xgboost
import pandas as pd
import pickle  
MODEL_DIR = "xgboost_model.pkl"
# Crear api
app = FastAPI(
    title="Predecir fecha modificada",
    description="API para predecir cuando un registro se modificara",
    version="1.0.0"
)

print("Cargando modelo...")
model = pickle.load(open(MODEL_DIR, "rb"))

# Entrada

class InputData(BaseModel):
    CustomerID: int
    Title: str
    Suffix: str
    SalesPerson: str


# Base
@app.get("/")
def home():

    return {
        "message": "El endpoint esta funcionando.  Para predecir usa POST /predecir. Mas informacion en /docs"
    }
# Prediccion

@app.post("/predecir")
def predict(data: InputData):

    # Comprobar categorias 
    if data.CustomerID <= 0:
        return {"error": "CustomerID debe ser un entero positivo"}
    elif data.Title not in ["Ms.", "Sr.", "Sra."]:
        return {"error": "Titulo no valido. Debe ser 'Ms.', 'Sr.' o 'Sra.'"}
    elif data.Suffix not in ["IV", "Jr.", "PhD", "Sr."]:
        return {"error": "Sufijo no valido. Debe ser 'IV', 'Jr.', 'PhD' o 'Sr.'"}
    elif data.SalesPerson not in ["garrett1", "jae0", "jillian0", "jose1", "linda3", "michael9", "pamela0", "shu0"]:
         return {"error": "Persona de ventas no valida. Debe ser uno de los siguientes: 'garrett1', 'jae0', 'jillian0', 'jose1', 'linda3', 'michael9', 'pamela0', 'shu0'"}


    # Diccionario para convertir cat to num
    title_map = {"Ms.": 0, "Sr.": 1, "Sra.": 2}
    suffix_map = {"IV": 0, "Jr.": 1, "PhD": 2, "Sr.": 3}
    salesperson_map = {"garrett1": 0, "jae0": 1, "jillian0": 2, "jose1": 3, "linda3": 4, "michael9": 5, "pamela0": 6, "shu0": 7}

    title_list = [0,0,0]
    suffix_list = [0,0,0,0]
    salesperson_list = [0,0,0,0,0,0,0,0]

    title_list[title_map[data.Title]] = 1
    suffix_list[suffix_map[data.Suffix]] = 1
    salesperson_list[salesperson_map[data.SalesPerson]] = 1

    model_input = [data.CustomerID] + title_list + suffix_list + salesperson_list

    prediction = model.predict([model_input])[0]

    # Convertir de Unix timestamp a fecha legible
    predicted_date = pd.to_datetime(prediction, unit='s')

    return {
        "Fecha_predicha": predicted_date.strftime('%Y-%m-%d %H:%M:%S')
    }