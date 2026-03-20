"""
QOKR - Quantum Orbital Kernel Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
LAMBDA_REG = 0.01
NUM_ORBITALS = 4


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def orbital_circuit(x, orbital_idx):
    qc = QuantumCircuit(NUM_QUBITS)

    for i in range(NUM_QUBITS):
        qc.ry(x[i] * (orbital_idx + 1), i)

    if orbital_idx % 4 == 0:
        for i in range(NUM_QUBITS - 1):
            qc.cx(i, i + 1)
    elif orbital_idx % 4 == 1:
        for i in range(NUM_QUBITS - 1):
            qc.cz(i, i + 1)
    elif orbital_idx % 4 == 2:
        for i in range(0, NUM_QUBITS - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, NUM_QUBITS - 1, 2):
            qc.cx(i, i + 1)
    else:
        for i in range(NUM_QUBITS - 1):
            qc.cx(i, i + 1)
        qc.cx(NUM_QUBITS - 1, 0)

    phase = (orbital_idx + 1) * np.pi / NUM_ORBITALS
    for i in range(NUM_QUBITS):
        qc.rz(phase * x[i], i)

    return qc


def compute_orbital_kernels(X_feats):
    n = len(X_feats)
    kernels = []

    for orb in range(NUM_ORBITALS):
        svs = []
        for feat in X_feats:
            circ = orbital_circuit(feat, orb)
            sv = Statevector.from_instruction(circ)
            svs.append(sv)

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                fid = abs(svs[i].inner(svs[j])) ** 2
                K[i, j] = fid
                K[j, i] = fid
        kernels.append(K)

    return kernels


def optimize_kernel_weights(kernels, y, lam=LAMBDA_REG):
    n_k = len(kernels)
    n = kernels[0].shape[0]

    weights = np.ones(n_k) / n_k
    best_err = np.inf
    best_w = weights.copy()

    for trial in range(50):
        K = sum(w * K_i for w, K_i in zip(weights, kernels))
        alpha = np.linalg.solve(K + lam * np.eye(n), y)
        pred = K @ alpha
        err = np.mean((pred - y) ** 2)

        if err < best_err:
            best_err = err
            best_w = weights.copy()

        grad = np.zeros(n_k)
        for k in range(n_k):
            pred_k = kernels[k] @ alpha
            grad[k] = -2.0 * np.mean((y - pred) * pred_k)

        weights = weights - 0.1 * grad
        weights = np.maximum(weights, 0.01)
        weights /= weights.sum()

    return best_w


def ridge_predict(K, y, lam=LAMBDA_REG):
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    return K @ alpha


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_feats = np.array([value_to_features(v) for v in range(n_states)])

    print(f"\n--- Quantum Orbital Kernels ({NUM_QUBITS}q, "
          f"{NUM_ORBITALS} orbitale) ---")
    kernels = compute_orbital_kernels(X_feats)
    for i, K in enumerate(kernels):
        print(f"  Orbital {i+1}: rang={np.linalg.matrix_rank(K)}")

    print(f"\n--- QOKR po pozicijama (adaptivne tezine) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        w = optimize_kernel_weights(kernels, y)
        K_opt = sum(wi * Ki for wi, Ki in zip(w, kernels))
        pred = ridge_predict(K_opt, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        w_str = ", ".join(f"{wi:.2f}" for wi in w)
        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} w=[{w_str}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QOKR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Quantum Orbital Kernels (5q, 4 orbitale) ---
  Orbital 1: rang=32
  Orbital 2: rang=31
  Orbital 3: rang=32
  Orbital 4: rang=31

--- QOKR po pozicijama (adaptivne tezine) ---
  Poz 1 w=[0.25, 0.25, 0.25, 0.25]: 1:0.167 | 2:0.147 | 3:0.130
  Poz 2 w=[0.25, 0.25, 0.25, 0.25]: 8:0.086 | 5:0.077 | 9:0.076
  Poz 3 w=[0.25, 0.25, 0.25, 0.25]: 13:0.064 | 12:0.063 | 14:0.062
  Poz 4 w=[0.25, 0.25, 0.25, 0.25]: 23:0.064 | 21:0.063 | 18:0.063
  Poz 5 w=[0.25, 0.25, 0.25, 0.25]: 29:0.065 | 26:0.064 | 27:0.063
  Poz 6 w=[0.25, 0.25, 0.25, 0.25]: 33:0.084 | 32:0.081 | 35:0.080
  Poz 7 w=[0.25, 0.25, 0.25, 0.25]: 7:0.182 | 38:0.153 | 37:0.133

==================================================
Predikcija (QOKR, deterministicki, seed=39):
[1, 8, x, y, z, 33, 38]
==================================================
"""


"""
QOKR - Quantum Orbital Kernel Regression

4 razlicite "orbitale" - kvantna kola sa razlicitim enkodiranjem i entanglement topologijama:
Orbital 1: linearni CX lanac
Orbital 2: CZ entanglement
Orbital 3: paran/neparan brick-layer CX
Orbital 4: ciklicni CX (ring)
Svaka orbitala ima razlicito Ry skaliranje i Rz fazne rotacije
Adaptivne tezine: gradient descent optimizuje koliko svaka orbitala doprinosi za svaku poziciju
Razlicite pozicije mogu imati razlicite optimalne kombinacije orbitala
Deterministicki, brz
"""

