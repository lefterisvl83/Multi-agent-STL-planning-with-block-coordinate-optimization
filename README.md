# Efficient Multi-Agent Temporal Logic Planning
<p align="center">
  <img src="assets/Github_gif_intro.gif" width="800" title="STL Planning Animation">
  <br>
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg">
  <img src="https://img.shields.io/badge/JAX-Accelerated-orange.svg">
  <img src="https://img.shields.io/badge/License-MIT-green.svg">
</p>

This repository implements the optimization framework described in:  
**"Efficient Multi-Agent Temporal Logic Planning via Block-Coordinate Optimization and Smooth Penalty Functions"**

## 📄 Overview

Multi-agent planning under Signal Temporal Logic (STL) tasks is challenging due to collaboration-induced couplings among agents. This framework provides a scalable solution by:
1. **Smooth STL Semantics**: Under-approximating non-smooth STL robustness functions to allow gradient-based optimization.
2. **Block-Coordinate Optimization**: Directional updates and computations are performed at the individual agent level.
3. **Penalty Method**: Relaxation into unconstrained optimization using quadratic robustness penalty functions $R(**u**)$.

---

## 🤖 Multi-Agent System (MAS)

We consider a MAS with $M$ agents, where each agent $i \in \mathcal{V} = \{1, \dots, M\}$ follows the dynamics:

$$x_i(t+1) = f_i(x_i(t), u_i(t))$$

where $x_i(t) \in \mathbb{R}^{n_i}$ and $u_i(t) \in \mathbb{R}^{m_i}$ are the state and input vectors. Agents are grouped into **cliques** $\nu \in \mathcal{K}_\phi$ based on their collaborative tasks.

---

## 📝 Problem Formulation

The multi-agent STL planning synthesis is formulated as an optimization over the multi-agent control sequence $\mathbf{u} = (u(0), \dots, u(N-1))$, where $u(t) = (u_1(t), \dots, u_M(t))$ and $N = H^\phi$. 

Given the initial condition $x(0) = x_0$ and the dynamics $x(t+1) = f(x(t), u(t))$, the multi-agent trajectory $\mathbf{x} = (x(0), \dots, x(N))$ is explicitly determined by $\mathbf{u}$. We denote the total cost as $\mathcal{L}(\mathbf{u}) = \sum_{i \in \mathcal{V}} \mathcal{L}_i(\mathbf{u}_i)$, where:

$$\mathcal{L}_i(\mathbf{u}_i) = \sum_{t=0}^{N-1} \ell_i(x_i(t), u_i(t), t) + V_{f,i}(x_i(N))$$

In this formulation:
* $\mathbf{u}_i = (u_i(0), \dots, u_i(N-1))$ denotes the $i^{th}$ agent's inputs.
* $\ell_i$ is the **running cost**, penalizing state energy, control effort, or deviations from reference points.
* $V_{f,i}$ is the **terminal cost**, penalizing deviations from a desired terminal condition.
* $\rho^\phi(\mathbf{u})$ represents the **STL robustness**.

The multi-agent STL planning problem is defined as:

$$
\begin{aligned}
    &\text{Minimize}_{\mathbf{u}} && \mathcal{L}(\mathbf{u}) = \sum_{i \in \mathcal{V}} \mathcal{L}_i(\mathbf{u}_i) \\
    &\text{subject to} && \rho^\phi(\mathbf{u}) = \min_{\nu \in \mathcal{K}_\phi} \rho^{\phi_\nu}(\mathbf{u}_\nu) > 0
\end{aligned}
$$

The goal is to minimize a separable cost function $\mathcal{L}(**u**})$ while satisfying a multi-agent STL specification $\phi$.

### Cost Function
The total cost is the sum of per-agent objectives (e.g., control effort, state energy, deviation from desired trajectory, etc.):
$$\mathcal{L}(\mathbf{**u**}) = \sum_{i \in \mathcal{V}} \mathcal{L}_i(\mathbf{**u**}_i)$$

### STL Specifications
The global task is a conjunction of individual and collaborative formulas:
$$\phi = \bigwedge_{\nu \in \mathcal{K}_\phi} \phi_\nu$$
The satisfaction of $\phi$ is determined by the robustness function $\rho^\phi(\mathbf{u}) > 0$.

---

## 💡 Solution Approach

We relax the constrained problem into an unconstrained penalty-based program:

$$\min_{\mathbf{u}} \mathcal{F}(\mathbf{u}, \mu) = \mathcal{L}(\mathbf{u}) + \frac{\mu}{2} \sum_{\nu \in \mathcal{K}_\phi} \max(0, -\tilde{\rho}^{\phi_\nu}(\mathbf{u}))^2$$

where $\tilde{\rho}$ represents the **smooth STL semantics** and $\mu$ is the penalty parameter.



### Algorithm 1: Inner Loop (BCGD)
For a fixed penalty $\mu$, we use **Block-Coordinate Gradient Descent (BCGD)**:
* **Step:** Update agent $i$ while keeping all other agents $\mathbf{u}_{-i}$ fixed.
* **Update Rule:** $\mathbf{u}_i^{k+1} = \text{arg min}_{\mathbf{u}_i} \mathcal{F}(\mathbf{u}_i, \mathbf{u}_{-i}^k, \mu)$
* **Advantage:** Computation scales linearly with the number of agents.

### Algorithm 2: Outer Loop (Penalty Method)
1. **Initialize** penalty parameter $\mu_0$ and control sequence $\mathbf{u}_0$.
2. **Solve** the inner loop (Algorithm 1) to find a stationary point.
3. **Increase** $\mu$ (e.g., $\mu \leftarrow \beta \mu$) and repeat until $\rho^\phi(\mathbf{u}) \geq 0$.

---

## 🚀 Features
* **Scalability**: Handles large robot fleets by decomposing the optimization per agent.
* **Expressivity**: Supports all STL operators including *Always* ($\square$), *Eventually* ($\lozenge$), and *Until* ($\mathcal{U}$).
* **Convergence**: Provable convergence to stationary points of the penalized problem.

## 📜 Citation
If you use this code in your research, please cite:
```bibtex
@inproceedings{VlahakisCDC24,
  title={Efficient Multi-Agent Temporal Logic Planning via Block-Coordinate Optimization and Smooth Penalty Functions},
  author={Vlahakis, Eleftherios E. and Kordabad, Arash Bahari and Lindemann, Lars and Sopasakis, Pantelis and Soudjani, Sadegh and Dimarogonas, Dimos V.},
  booktitle={2024 IEEE 63rd Conference on Decision and Control (CDC)},
  year={2024}
}

