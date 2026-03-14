import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Normalize vector
def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

# Pad vector to nearest power of 2
def pad_to_power_of_two(v):
    length = len(v)
    next_pow2 = 2**int(np.ceil(np.log2(length)))
    return np.pad(v, (0, next_pow2 - length))

# Encode vector into quantum circuit
def encode_vector_to_circuit(v):
    v = normalize_vector(v)
    n_qubits = int(np.log2(len(v)))
    qc = QuantumCircuit(n_qubits)
    qc.initialize(v, range(n_qubits))
    return qc

# SWAP test
def swap_test(vec1, vec2):
    n = int(np.log2(len(vec1)))
    qc = QuantumCircuit(2*n + 1, 1)
    qc.initialize(normalize_vector(vec1), range(1, n+1))
    qc.initialize(normalize_vector(vec2), range(n+1, 2*n+1))
    qc.h(0)
    for i in range(n):
        qc.cswap(0, i+1, n+i+1)
    qc.h(0)
    qc.measure(0, 0)
    return qc

# Compute quantum distance
backend = AerSimulator()
def compute_distance(vec1, vec2, shots=256):
    vec1, vec2 = pad_to_power_of_two(vec1), pad_to_power_of_two(vec2)
    qc = swap_test(vec1, vec2)
    job = backend.run(qc, shots=shots)
    result = job.result().get_counts()
    prob_zero = result.get('0', 0) / shots
    return np.sqrt(1 - prob_zero)

# Quantum KNN
def quantum_knn(test_vec, X_train, y_train, k=3):
    test_vec = pad_to_power_of_two(test_vec)
    distances = [compute_distance(test_vec, pad_to_power_of_two(x)) for x in X_train]
    neighbors = np.argsort(distances)[:k]
    votes = [y_train[i] for i in neighbors]
    return max(set(votes), key=votes.count)

# Classical KNN
def classical_knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)