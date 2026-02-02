# Implicit Euler for a Coupled Drying Model
**Newton–Gauss–Seidel vs Fixed-Point Iteration**

This project solves a coupled two-state drying model for:
- Moisture content **M(t)**
- Temperature **T(t)**

Time integration is done with **Implicit Euler**. The nonlinear implicit step at each time level is solved using:
1) **Fixed-point iteration**
2) **Newton iteration**, where the linear Newton system is solved using **Gauss–Seidel**

The repository includes:
- A Python implementation that runs all experiments and produces the required figures/tables
- The final report: **CP2_final.pdf**

---

## Model
We solve the ODE system on **t ∈ [0, 50]** with initial condition:
- **M(0) = 0.6**
- **T(0) = 20**

Parameters:
- k1 = 0.08, k2 = 0.15, k3 = 0.6
- alpha = 0.05, Ta = 60, Meq = 0.12

Dynamics:
- dM/dt = −k1 (M − Meq) exp(alpha (T − Ta))
- dT/dt = k2 (Ta − T) − k3 dM/dt

---

## Numerical Method
### Implicit Euler
For step size **h** and tn+1 = tn + h:
- y_{n+1} = y_n + h f(y_{n+1}, t_{n+1})
- Residual: F(y_{n+1}) = y_{n+1} − y_n − h f(y_{n+1}, t_{n+1})

Each time step solves:
- F(y_{n+1}) = 0

### Nonlinear Solvers
**Fixed-point iteration**
- y^{(k+1)} = y_n + h f(y^{(k)}, t_{n+1})

**Newton–Gauss–Seidel**
- J(y^{(k)}) Δ^{(k)} = −F(y^{(k)})
- y^{(k+1)} = y^{(k)} + Δ^{(k)}
- The 2×2 linear system is solved using Gauss–Seidel iterations.

The Jacobian uses the required closed-form formulas with:
- E = exp(alpha (T − Ta))

---

## Experiments Included
Running the script executes:
1) **Solver comparison at h = 0.1**
   - trajectories M(t), T(t)
   - phase portrait (T vs M)
   - iteration counts per step
   - residual convergence plots (selected steps)
   - performance table

2) **Accuracy / convergence study**
   - reference solution computed with very small step: **h_ref = 0.001**
   - tested step sizes: **h ∈ {0.20, 0.10, 0.05, 0.01}**
   - L2 and max errors for M and T
   - observed order (expected ~ 1 for implicit Euler)

3) **Cost–accuracy trade-off**
   - CPU time vs error scatter plot
   - efficiency table

---

## Requirements
- Python 3.9+ recommended
- Required:
  - `numpy`
  - `matplotlib`
- Optional (improves reference interpolation to cubic):
  - `scipy`
---

## How to Run

1. **(Optional) Create and activate a virtual environment**:
  
   python -m venv .venv
   source .venv/Scripts/activate  # on Git Bash / Windows
   2. **Install the dependencies** (if not already installed):
  
   pip install numpy matplotlib
   # optional:
   pip install scipy
   3. **Run all experiments and generate figures/tables**:
  
   python ode.py
   The script will open the required figures in separate windows and print performance/accuracy tables in the terminal.


