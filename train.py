import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset and train
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

# Export model
joblib.dump(model, "model.pkl")
print("Saved model to model.pkl successfully.")