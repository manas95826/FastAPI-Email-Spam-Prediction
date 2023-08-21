# FastAPI Email Spam Prediction API

This is a simple FastAPI web API that predicts whether an input text is likely to be spam or not using a pre-trained machine learning model. The API takes an input text and provides a prediction along with the prediction probability.

## Getting Started

### Prerequisites
Before you begin, ensure you have the following:

- Python (>=3.6) installed on your machine.

### Installation
- Clone the repository:
```bash
git clone https://github.com/manas95826/FastAPI-Email-Spam-Prediction.git
```
- Navigate to the project directory:
```bash
cd FastAPI-Email-Spam-Prediction
```
- Install the required libraries:
```bash
pip install fastapi uvicorn pydantic pandas scikit-learn
```
- Run the FastAPI app:
```bash
uvicorn fastapi_app:app --reload
```
The API will be available at http://localhost:8000.

## Usage
### Making Predictions

1. Send a POST request to the /predict/ endpoint with a JSON payload containing the text parameter. For example:

```bash
{
    "text": "Congratulations, you've won a prize!"
}
```
2. The API will respond with a JSON containing the prediction result and the prediction probability.

## API Endpoints
- POST /predict/: Makes a spam prediction based on the input text and returns the prediction and probability.

## Built With
- FastAPI - The web framework used for building the API.
- Scikit-learn - The machine learning library used for predictions.
- Python - The programming language used for the API logic.

