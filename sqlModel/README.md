# Predecir fecha modificada

API que predice **cuándo se modificará un registro de cliente** (`ModifiedDate`) de la tabla `SalesLT.Customer`.

## Archivos
`crear_modelo.ipynb`:  Extrae los datos de Azure SQL, entrena el modelo y lo guarda en `xgboost_model.pkl` 
`main.py`: La API que carga el modelo y expone el endpoint 
`usar_api.ipynb`: Ejemplo de cómo llamar al endpoint desde Python 
`xgboost_model.pkl`: Modelo ya entrenado

## El modelo

- **Datos:** `CustomerID`, `Title`, `Suffix`, `SalesPerson` y `ModifiedDate` de `SalesLT.Customer`.
- **Objetivo:** `ModifiedDate`, convertida a segundos Unix (desde 1970-01-01) para poder hacer regresión.
- **Preprocesamiento:** `Title`, `Suffix` y `SalesPerson` se convierten a variables one-hot.
- **Algoritmo:** `XGBRegressor` con 20 árboles, split 80/20 train/test.
- **Resultado:** R² ≈ -0.03. Las variables no tienen relación real con la fecha de modificación. No hay ninguna variable que tenga relacion aparente en la bd.

Documentación interactiva en `revisar entrega de canvas`.

## Endpoints

### `GET /`
Solo confirma que la API está funcionando.

### `POST /predecir`
Espera un JSON con 4 campos:

| Campo | Tipo | Valores aceptados |
|---|---|---|
| `CustomerID` | int | Entero positivo |
| `Title` | str | `"Ms."`, `"Sr."`, `"Sra."` |
| `Suffix` | str | `"IV"`, `"Jr."`, `"PhD"`, `"Sr."` |
| `SalesPerson` | str | `"garrett1"`, `"jae0"`, `"jillian0"`, `"jose1"`, `"linda3"`, `"michael9"`, `"pamela0"`, `"shu0"` |

**Ejemplo de entrada:**
```json
{
  "CustomerID": 30,
  "Title": "Ms.",
  "Suffix": "IV",
  "SalesPerson": "jillian0"
}
```

**Respuesta**
```json
{ "Fecha_predicha": "2006-08-01 00:00:00" }
```

Si algún valor no es válido, regresa un mensaje como:
```json
{ "error": "Titulo no valido. Debe ser 'Ms.', 'Sr.' o 'Sra.'" }
```

## Usarlo desde la libreta

Abre `usar_api.ipynb`, pon la URL de la API **terminando en `/predecir`** y corre las celdas:

```python
import requests

url = "revisar entrega de canvas"  

data = {"CustomerID": 30, "Title": "Ms.", "Suffix": "IV", "SalesPerson": "jillian0"}
response = requests.post(url, json=data)
print(response.json())
```

## Requerimientos
Conexion a internet y un ambiente de python

