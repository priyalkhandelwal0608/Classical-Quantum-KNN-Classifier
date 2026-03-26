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
## Tech Stack

- **Quantum Computing:** Qiskit, Qiskit Aer Simulator  
- **Backend & Data Processing:** Python 3.x, NumPy  
- **Machine Learning:** Scikit-Learn (KNeighborsClassifier)  
- **Visualization (optional):** Matplotlib for plotting distances or results  

---

## Installation & Dependencies

To set up the project, run the following commands:


pip install -r requirements.txt
python app.py



---
## **Project Structure**

```text
.
├── **static/**                # CSS, JavaScript, and UI assets
├── **templates/**             # HTML files for the web interface (Flask)
├── **uploads/**               # Directory for user-provided data files
├── **app.py**                 # Main entry point for the Flask web application
├── **quantum_knn.py**         # Core logic for the Quantum-inspired KNN model
├── **requirements.txt**       # Python dependencies (Pandas, Scikit-Learn, FAISS, etc.)
└── **__pycache__/**           # Compiled Python files (ignored by Git)
