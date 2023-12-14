from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Charger le modèle pré-entrainé
model = joblib.load("iris.pkl")

# Définition de la structure des données en entrée
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Route pour prédire la classe de fleur
@app.post("/predict/")
def predict_species(iris: IrisInput):
    data = [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
    data_np = np.array(data).reshape(1, -1)
    prediction = model.predict(data_np)
    predicted_class = prediction[0]
    if predicted_class == 0:
        species = 'Setosa'
    elif predicted_class == 1:
        species = 'Versicolor'
    else:
        species = 'Virginica'
    return {"predicted_species": species}
