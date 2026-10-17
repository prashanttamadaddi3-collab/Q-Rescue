# Algorithmic & Mathematical Formulations — Q-Rescue

**Hybrid Quantum–Classical Framework for Flood Risk Prediction and Emergency Resource Allocation**

---

## 1. Variational Quantum Classifier (VQC)

### 1.1 State Preparation & Angle Embedding
Let $\vec{x} = [x_1, x_2, x_3, x_4]^T$ denote the four key continuous environmental features:
1. $x_1$: Cumulative 24h Rainfall ($\text{mm}$)
2. $x_2$: River Water Gauge Level ($\text{m}$)
3. $x_3$: Terrain Elevation ($\text{m AMSL}$)
4. $x_4$: Historical Inundation Indicator ($\{0, 1\}$)

Features are normalized to the angular interval $[0, \pi]$:
$$\tilde{x}_i = \pi \cdot \frac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}$$

The initial ground state $|0\rangle^{\otimes 4}$ is transformed into an encoded quantum state $|\psi(\vec{x})\rangle$ via single-qubit Pauli-Y rotation gates:
$$|\psi(\vec{x})\rangle = \bigotimes_{i=1}^4 R_Y(\tilde{x}_i) |0\rangle = \bigotimes_{i=1}^4 \left( \cos\frac{\tilde{x}_i}{2} |0\rangle + \sin\frac{\tilde{x}_i}{2} |1\rangle \right)$$

### 1.2 Parameterized Variational Ansatz
The ansatz consists of $L$ layers of parameterized single-qubit rotations followed by circular CNOT entangling gates:
$$U(\vec{\theta}) = \prod_{l=1}^L \left( U_{\text{ent}} \prod_{j=0}^3 R_Z(\theta_{l, j, 1}) R_Y(\theta_{l, j, 0}) \right)$$
where the circular entanglement operator is:
$$U_{\text{ent}} = \text{CNOT}_{3,0} \cdot \text{CNOT}_{2,3} \cdot \text{CNOT}_{1,2} \cdot \text{CNOT}_{0,1}$$

The output quantum state before measurement is:
$$|\Phi(\vec{x}, \vec{\theta})\rangle = U(\vec{\theta}) |\psi(\vec{x})\rangle$$

### 1.3 Pauli-Z Expectation Readout & Softmax
For multi-class classification into $C = 4$ risk classes ($\text{Low}=0, \text{Medium}=1, \text{High}=2, \text{Critical}=3$), we measure the expectation value of the Pauli-Z operator on each qubit wire $i \in \{0, 1, 2, 3\}$:
$$\langle Z_i \rangle = \langle \Phi(\vec{x}, \vec{\theta}) | Z_i | \Phi(\vec{x}, \vec{\theta}) \rangle \in [-1, +1]$$

A temperature-scaled softmax layer converts the expectation vector $\vec{z} = [\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_3 \rangle]^T$ into normalized class probabilities:
$$P(y = c \mid \vec{x}) = \frac{\exp(\beta \cdot \langle Z_c \rangle)}{\sum_{j=0}^3 \exp(\beta \cdot \langle Z_j \rangle)}, \quad \beta = 2.5$$

### 1.4 Loss Function & Optimization
Supervised training optimizes $\vec{\theta}$ by minimizing the Categorical Cross-Entropy loss over batch size $B$:
$$\mathcal{L}(\vec{\theta}) = -\frac{1}{B} \sum_{k=1}^B \sum_{c=0}^3 y_{k, c} \ln\left( P(y_k = c \mid \vec{x}_k) \right)$$
Gradients $\nabla_{\vec{\theta}} \mathcal{L}$ are computed analytically via the parameter-shift rule and updated using Nesterov Momentum / Adam optimization.

---

## 2. Quantum Approximate Optimization Algorithm (QAOA)

### 2.1 Problem Definition & Objective
Emergency logistics allocates $K$ types of scarce resources (Rescue teams, boats, ambulances, food, medical kits) across $N$ flood-affected candidate locations.

Let $x_i \in \{0, 1\}$ be the binary decision variable indicating whether location $i$ is actively prioritized for full emergency dispatch.

Each location $i$ has:
- Risk Weight $w_i \in \{0.5, 1.5, 3.0, 4.5\}$ for Low, Medium, High, Critical
- Population Exposure $\hat{P}_i = \frac{P_i}{\max P} \in [0, 1]$
- Travel Distance Penalty $\hat{D}_i = \frac{D_i}{\max D} \in [0, 1]$ from Belagavi DEOC
- Resource demand $d_{i, k}$ for resource type $k$
- Available central depot budget $B_k$ for resource type $k$

The constrained objective maximizes total rescue utility while respecting depot budgets:
$$\max_{\vec{x}} \sum_{i=1}^N \left( \alpha_1 w_i + \alpha_2 \hat{P}_i - \alpha_3 \hat{D}_i \right) x_i \quad \text{subject to} \quad \sum_{i=1}^N d_{i, k} x_i \le B_k \quad \forall k \in \{1, \dots, K\}$$

### 2.2 QUBO Formulation
We convert this into an unconstrained quadratic minimization problem by incorporating quadratic penalty terms $\lambda_k > 0$ for resource over-allocation:
$$\min_{\vec{x}} E(\vec{x}) = - \sum_{i=1}^N \text{Utility}_i x_i + \sum_{k=1}^K \lambda_k \left( \sum_{i=1}^N d_{i, k} x_i - B_k \right)^2$$

Expanding the quadratic penalty term:
$$\left( \sum_{i=1}^N d_{i, k} x_i - B_k \right)^2 = \sum_{i=1}^N (d_{i, k}^2 - 2 B_k d_{i, k}) x_i + 2 \sum_{i < j} d_{i, k} d_{j, k} x_i x_j + B_k^2$$

Thus, the QUBO is expressed in matrix-vector form:
$$E(\vec{x}) = \vec{x}^T Q \vec{x} + \vec{L}^T \vec{x} + \text{const}$$

### 2.3 Mapping to Ising Spin Hamiltonian
Using the canonical mapping from binary variables $x_i \in \{0, 1\}$ to quantum Pauli-Z spin operators $Z_i \in \{+1, -1\}$:
$$x_i = \frac{I - Z_i}{2}$$

Substituting into the QUBO yields the Problem/Cost Hamiltonian $H_C$:
$$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{i j} Z_i Z_j$$
where:
$$h_i = -\frac{L_i}{2} - \sum_{j \ne i} \frac{Q_{i j}}{4}, \qquad J_{i j} = \frac{Q_{i j}}{4}$$

The Transverse-Field Mixer Hamiltonian is:
$$H_M = \sum_{i=1}^N X_i$$

### 2.4 QAOA Circuit Evolution
1. **Initial State:** Equal superposition of all $2^N$ possible allocation configurations:
   $$|\psi_0\rangle = |+\rangle^{\otimes N} = \frac{1}{\sqrt{2^N}} \sum_{\vec{x} \in \{0, 1\}^N} |\vec{x}\rangle$$

2. **Alternating Unitaries:** For $p$ QAOA steps with variational parameters $\vec{\gamma} = [\gamma_1, \dots, \gamma_p]$ and $\vec{\beta} = [\beta_1, \dots, \beta_p]$:
   $$|\psi(\vec{\gamma}, \vec{\beta})\rangle = \prod_{l=1}^p \left( e^{-i \beta_l H_M} e^{-i \gamma_l H_C} \right) |\psi_0\rangle$$

3. **Optimization:** Classical gradient-free or gradient-based optimizer (COBYLA / Adam) updates $(\vec{\gamma}, \vec{\beta})$ to minimize the energy expectation:
   $$\min_{\vec{\gamma}, \vec{\beta}} \langle \psi(\vec{\gamma}, \vec{\beta}) | H_C | \psi(\vec{\gamma}, \vec{\beta}) \rangle$$

4. **Bitstring Sampling:** Upon convergence, the circuit is measured in the computational $Z$-basis to extract the most probable bitstring $\vec{x}^* = \operatorname{argmax}_{\vec{x}} |\langle \vec{x} \mid \psi(\vec{\gamma}^*, \vec{\beta}^*) \rangle|^2$.

---

## 3. Classical Baselines

- **Random Forest Baseline:** An ensemble of 100 decision trees utilizing Gini impurity splits over all 8 classical features. Serves as a reference for predictive accuracy.
- **Priority Greedy Knapsack Baseline:** A deterministic classical benchmark that sorts locations strictly by $\text{Utility}_i$ and greedily allocates available depot stock until inventory is depleted.
