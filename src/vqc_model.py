"""
Variational Quantum Classifier (VQC) module for Q-Rescue.
Implements a 4-qubit parameterized quantum circuit using PennyLane
for multi-class flood-risk prediction (Low, Medium, High, Critical).
"""

import os
import joblib
import numpy as np
import pennylane as qml
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.preprocessing import QUANTUM_FEATURES, LABEL_MAPPING, REVERSE_LABEL_MAPPING

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

NUM_QUBITS = 4
NUM_LAYERS = 2

# Define simulation device
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="autograd", diff_method="best")
def quantum_circuit(features, weights):
    """
    4-Qubit Variational Quantum Circuit:
    1. Feature encoding via AngleEmbedding (RY rotations of scaled features in [0, pi]).
    2. Strongly entangling variational layers with parameterized RY, RZ rotations and circular CNOT entanglement.
    3. Multi-qubit Pauli-Z expectation value measurements for 4-class output.
    """
    # 1. State Preparation / Feature Encoding
    qml.AngleEmbedding(features=features, wires=range(NUM_QUBITS), rotation="Y")
    
    # 2. Variational parameterized layers
    for layer in range(weights.shape[0]):
        for q in range(NUM_QUBITS):
            qml.RY(weights[layer, q, 0], wires=q)
            qml.RZ(weights[layer, q, 1], wires=q)
        # Circular CNOT entanglement
        for q in range(NUM_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % NUM_QUBITS])
            
    # 3. Measurement: Pauli-Z expectation on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

def softmax(x: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Softmax with inverse temperature scaling beta."""
    e_x = np.exp(beta * (x - np.max(x, axis=-1, keepdims=True)))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class VariationalQuantumClassifier:
    """VQC multi-class model predicting Low, Medium, High, Critical risk."""

    def __init__(self, num_qubits: int = NUM_QUBITS, num_layers: int = NUM_LAYERS, 
                 lr: float = 0.05, epochs: int = 25, batch_size: int = 16, random_state: int = 42):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        np.random.seed(random_state)
        # Shape: (num_layers, num_qubits, 2)
        self.weights = 0.1 * np.random.randn(num_layers, num_qubits, 2)
        self.bias = np.zeros(num_qubits)
        self.is_fitted = False
        self.training_history = []

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Run batch of inputs through quantum circuit and return raw expectations."""
        raw_outputs = []
        for x in X:
            res = quantum_circuit(x, self.weights)
            # res is list of 4 expectations
            expvals = np.array([float(r) for r in res]) + self.bias
            raw_outputs.append(expvals)
        return np.array(raw_outputs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities using softmax over quantum expectation values."""
        raw_expvals = self._forward(X)
        return softmax(raw_expvals, beta=2.5)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class index with maximum probability."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_single_with_quantum_state(self, x: np.ndarray) -> Dict[str, Any]:
        """Detailed prediction for a single instance including raw quantum expectations."""
        res = quantum_circuit(x, self.weights)
        raw_expvals = np.array([float(r) for r in res]) + self.bias
        probs = softmax(raw_expvals, beta=2.5)
        pred_class = int(np.argmax(probs))
        
        return {
            "predicted_class_idx": pred_class,
            "predicted_risk_label": REVERSE_LABEL_MAPPING[pred_class],
            "confidence": float(probs[pred_class]),
            "class_probabilities": {
                REVERSE_LABEL_MAPPING[i]: round(float(probs[i]), 4) for i in range(4)
            },
            "raw_pauli_z_expectations": {
                f"Qubit_{i} ({REVERSE_LABEL_MAPPING[i]})": round(float(raw_expvals[i]), 4) for i in range(4)
            }
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              callback: Optional[callable] = None) -> "VariationalQuantumClassifier":
        """
        Train parameterized circuit weights using Nesterov Momentum / Gradient Descent.
        Loss function: Categorical Cross-Entropy with softmax.
        """
        opt = qml.NesterovMomentumOptimizer(stepsize=self.lr)
        n_samples = len(X_train)
        
        # One-hot encode targets
        y_onehot = np.zeros((n_samples, self.num_qubits))
        for i, val in enumerate(y_train):
            y_onehot[i, val] = 1.0

        params = self.weights
        self.training_history = []

        def cost_fn(w, x_batch, y_batch):
            loss = 0.0
            for x, y_true in zip(x_batch, y_batch):
                expvals = np.array(quantum_circuit(x, w)) + self.bias
                probs = np.exp(2.5 * expvals) / np.sum(np.exp(2.5 * expvals))
                # Cross-entropy loss with epsilon
                loss -= np.sum(y_true * np.log(np.clip(probs, 1e-7, 1.0)))
            return loss / len(x_batch)

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                x_batch = X_train[batch_idx]
                y_batch = y_onehot[batch_idx]
                
                params, batch_loss = opt.step_and_cost(lambda w: cost_fn(w, x_batch, y_batch), params)
                epoch_loss += float(batch_loss)
                n_batches += 1
                
            avg_loss = epoch_loss / max(1, n_batches)
            
            # Epoch accuracy
            self.weights = params
            preds = self.predict(X_train)
            acc = float(accuracy_score(y_train, preds))
            
            record = {"epoch": epoch + 1, "loss": round(avg_loss, 4), "train_accuracy": round(acc, 4)}
            self.training_history.append(record)
            
            if callback:
                callback(epoch + 1, self.epochs, avg_loss, acc)

        self.weights = params
        self.is_fitted = True
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate VQC on held-out test split."""
        y_pred = self.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]).tolist()
        
        report_dict = classification_report(
            y_test, y_pred,
            labels=[0, 1, 2, 3],
            target_names=["Low", "Medium", "High", "Critical"],
            output_dict=True,
            zero_division=0
        )
        
        return {
            "model_name": "Variational Quantum Classifier (VQC)",
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "training_history": self.training_history
        }

    def save(self, filepath: Optional[str] = None) -> str:
        """Save VQC model parameters to disk."""
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "vqc_model.joblib")
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: Optional[str] = None) -> "VariationalQuantumClassifier":
        """Load trained VQC model."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "vqc_model.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"VQC model file not found at {filepath}")
        return joblib.load(filepath)
