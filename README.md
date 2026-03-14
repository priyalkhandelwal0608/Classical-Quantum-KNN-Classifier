#  Classical vs Quantum KNN Classifier

A comparative demo of **Classical K‑Nearest Neighbors (KNN)** and a **Quantum KNN implementation** using Qiskit.  
This project showcases how quantum algorithms can be applied to machine learning, and compares their performance on datasets like the Iris dataset or user‑uploaded CSV files.

---

##  Features
-  **Upload your own dataset** (CSV format, numeric features + target column last).
-  **Iris demo mode** for quick testing.
-  **Classical KNN** using scikit‑learn.
-  **Quantum KNN** using Qiskit’s SWAP test for distance measurement.
-  **Modern UI** with attractive results cards and error handling.
-  **Downloadable sample dataset** to test the app instantly.

---

##  Project Structure
Classical-Quantum-KNN-Classifier/
│
├── app.py                  # Flask entry point: routes, dataset upload, demo mode
├── quantum_knn.py          # Classical & Quantum KNN implementations (Qiskit + scikit-learn)
│
├── templates/              # HTML templates for Flask
│   └── index.html          # Main frontend page (upload form, results cards)
│
├── static/                 # Static assets (CSS, JS, images)
│   └── style.css           # Styling for the web interface
│
├── uploads/                # Folder to store user-uploaded CSV datasets
│
├── requirements.txt        # Python dependencies (Flask, NumPy, Pandas, scikit-learn, Qiskit, etc.)
├── README.md               # Project documentation (overview, usage, dataset format)
├── Procfile                # For deployment (Heroku/Render/Railway)
