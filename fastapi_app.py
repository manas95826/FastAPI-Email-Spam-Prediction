from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = FastAPI()

# Load the pipeline model
pipeline_filename = 'email_spam_pipeline.pkl'
with open(pipeline_filename, 'rb') as pipeline_file:
    clf = pickle.load(pipeline_file)

class Item(BaseModel):
    text: str

@app.post("/predict/")
async def predict(item: Item):
    text = item.text
    prediction = clf.predict([text])[0]
    probability = clf.predict_proba([text])[0]
    return {"prediction": prediction, "probability": probability[prediction]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
