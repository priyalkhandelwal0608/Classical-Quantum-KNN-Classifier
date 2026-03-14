from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from quantum_knn import quantum_knn, classical_knn
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_dataset(df):
    # Assume last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classical accuracy
    classical_acc = classical_knn(X_train, y_train, X_test, y_test)

    # Quantum accuracy (subset for speed)
    predictions = [quantum_knn(x, X_train[:20], y_train[:20], k=3) for x in X_test[:10]]
    quantum_acc = np.mean(np.array(predictions) == y_test[:10])

    return classical_acc, quantum_acc

@app.route("/", methods=["GET", "POST"])
def index():
    classical_acc = None
    quantum_acc = None
    error_message = None

    if request.method == "POST":
        try:
            if "demo" in request.form:
                # Iris demo
                iris = load_iris()
                X = iris.data[:, :2]
                y = iris.target
                df = pd.DataFrame(np.column_stack([X, y]))
                classical_acc, quantum_acc = process_dataset(df)

            elif "file" in request.files:
                file = request.files["file"]
                if file.filename.endswith(".csv"):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filepath)
                    df = pd.read_csv(filepath)
                    classical_acc, quantum_acc = process_dataset(df)
                else:
                    error_message = "❌ Only CSV files are supported."
        except Exception:
            error_message = "⚠️ The dataset format is not correct. Please upload a clean CSV with numeric features and the target column as the last column."

    return render_template("index.html",
                           classical_acc=classical_acc,
                           quantum_acc=quantum_acc,
                           error_message=error_message)

@app.route("/download-sample")
def download_sample():
    # Create a richer sample dataset with 3 balanced classes
    data = {
        "feature1": [5.1, 4.9, 6.7, 5.6, 7.2, 6.4, 5.9, 6.3, 7.1],
        "feature2": [3.5, 3.0, 3.1, 2.9, 3.6, 3.2, 3.0, 2.8, 3.1],
        "feature3": [1.4, 1.4, 4.4, 3.6, 5.8, 4.5, 1.5, 4.7, 5.9],
        "feature4": [0.2, 0.2, 1.4, 1.3, 2.3, 1.5, 0.3, 1.6, 2.1],
        "target":   [0, 0, 1, 1, 2, 2, 0, 1, 2]
    }
    df = pd.DataFrame(data)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], "sample_dataset.csv")
    df.to_csv(filepath, index=False)
    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)