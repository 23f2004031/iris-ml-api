# Iris ML Production API

This API serves a machine learning model trained on the Iris flower dataset to predict flower species (`setosa`, `versicolor`, or `virginica`) based on four numerical features: sepal length, sepal width, petal length, and petal width.

## Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train/export the model (if `model.pkl` is not present):
   ```bash
   python train.py
   ```

3. Run the API locally:
   ```bash
   uvicorn main:app --reload
   ```
   Open `http://localhost:8000/docs` in your browser.

## Endpoints

- `GET /health` - Returns the health status and whether `model.pkl` loaded properly.
- `POST /predict` - Accepts feature measurements and returns predictions.

### Example Request Body for `/predict`

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

### Example Response Body

```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```