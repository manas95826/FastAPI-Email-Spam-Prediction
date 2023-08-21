from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create a FastAPI app instance.
app = FastAPI()

# Load the pipeline model
pipeline_filename = 'email_spam_pipeline.pkl'
with open(pipeline_filename, 'rb') as pipeline_file:
    clf = pickle.load(pipeline_file)

# Define a Pydantic BaseModel class to represent the request input data.
class Item(BaseModel):
    text: str

# Create an API endpoint that listens to POST requests at "/predict/".
@app.post("/predict/")
async def predict(item: Item):
    # Extract the input text from the request.
    text = item.text
    prediction = clf.predict([text])[0]  #If no [0], output= [1]/[0], get rid of array with this
    probability = clf.predict_proba([text])[0]  #Similarly with this, if not [0], output = [[0.2,0.8]]
    return {"prediction": prediction, "probability": probability[prediction]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
