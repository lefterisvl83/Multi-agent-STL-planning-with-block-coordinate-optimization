# Multi-Agent Signal Temporal Logic (STL) Planning
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
3. **Penalty Method**: Relaxation into unconstrained optimization using quadratic robustness penalty functions $R(\mathbf{u})$ with penalty parameter $\lambda>0$.

---

## 🤖 Multi-Agent System (MAS)

### Dynamics
We consider a MAS with $M$ agents, where each agent $i \in \mathcal{V} = \{1, \dots, M\}$ follows the dynamics:

$$x_i(t+1) = f_i(x_i(t), u_i(t))$$

where $x_i(t) \in \mathbb{R}^{n_i}$ and $u_i(t) \in \mathbb{R}^{m_i}$ are the state and input vectors. 

> 🐍 Python Implementation: Unicycle Dynamics:
> We model each agent $i$ as a unicycle with states $x_i = [p_x, p_y, \theta]^\top$ and inputs $u_i = [v, \omega]^\top$ (linear velocity and angular velocity).
>
```python
import jax
import jax.numpy as jnp

@jax.jit
def unicycle_step(state, control, dt=0.1):
    """
    Non-linear unicycle motion model.
    state: [x, y, theta]
    control: [v, omega]
    """
    x, y, theta = state
    v, omega = control
    
    # Standard unicycle kinematics
    new_x = x + v * jnp.cos(theta) * dt
    new_y = y + v * jnp.sin(theta) * dt
    new_theta = theta + omega * dt
    
    return jnp.array([new_x, new_y, new_theta])

@jax.jit
def compute_unicycle_trajectory(u_i, x0_i, dt=0.1):
    """
    Predicts the horizon trajectory for a single unicycle agent.
    u_i: (N, 2) sequence of [v, omega]
    x0_i: (3,) initial state [x, y, theta]
    """
    def scan_op(state, ut):
        next_state = unicycle_step(state, ut, dt)
        return next_state, state # (carry, output)

    # jax.lax.scan is XLA-optimized for sequential state updates
    last_state, trajectory = jax.lax.scan(scan_op, x0_i, u_i)
    
    # Append the final state to complete the (N+1, 3) trajectory
    return jnp.vstack([trajectory, last_state])

# Vectorize for the entire MAS (M agents)
# in_axes=(0, 0) means we map over the first dimension of both u and x0
compute_MAS_trajectories = jax.vmap(compute_unicycle_trajectory, in_axes=(0, 0, None))
X_all = compute_MAS_trajectories(U, X0, dt)
```

### Multi-Agent STL specification
Agents are grouped into **cliques** $\nu \in \mathcal{K}_\phi$ based on their collaborative tasks. The MAS is subject to the specification

$$\phi = \bigwedge_{\nu \in \mathcal{K}_\phi} \phi_\nu$$

which is a conjunctive STL formula, where each conjunct $\phi_\nu$ is defined over the aggregate trajectory $\mathbf{x}_\nu(t)$ collecting the individual trajectories of the agents in the clique $\nu$. 

The set $\mathcal{K}_\phi$ collects all these cliques induced by $\phi$, and may include individual agents $(|\nu|=1)$ or group of agents $(1<|\nu|\leq |\mathcal{V}|)$. Note that different cliques may overlap in their agent sets, indicating that some agents participate in multiple collaborative tasks. 

> 🐍 Python Implementation: Clique collection example
>
```python
import jax.numpy as jnp

def get_clique_trajectories(X_all, K_phi):
    """
    Extracts aggregate trajectories x_nu for each clique nu in K_phi.
    X_all: (M, N+1, 3) array of all agent trajectories.
    K_phi: List of agent indices for each clique.
    """
    return [X_all[jnp.array(nu)] for nu in K_phi]

# --- Example: Specification with overlapping cliques ---
# phi = phi_0 ^ phi_1 ^ phi_{0,1}
K_phi = [
    [0],    # Clique nu_0: Agent 0 (e.g., Reach Goal A)
    [1],    # Clique nu_1: Agent 1 (e.g., Reach Goal B)
    [0, 1]  # Clique nu_2: Agents 0 & 1 (e.g., Collision Avoidance)
]

# 1. Compute global trajectories for the whole MAS
X_all = compute_MAS_trajectories(U, X0, dt)

# 2. Collect into aggregate trajectories x_nu
x_cliques = get_clique_trajectories(X_all, K_phi)

# 3. Access a single aggregate trajectory x_nu
# For the collaborative task (nu_2), x_collaboration contains paths for agents 0 and 1
x_collaboration = x_cliques[2]
 
# Output: (2, N+1, 3) -> (|nu|, steps, state_dim)
```

---

## 📝 Problem Formulation

The multi-agent STL planning synthesis is formulated as an optimization over the multi-agent control sequence $\mathbf{u} = (u(0), \dots, u(N-1))$, where $u(t) = (u_1(t), \dots, u_M(t))$, and $N = H^\phi$ is the horizon of the multi-agent formula $\phi$. 

Given the initial condition $x(0) = x_0$ and the dynamics $x(t+1) = f(x(t), u(t))$, the multi-agent trajectory $\mathbf{x} = (x(0), \dots, x(N))$ is explicitly determined by $\mathbf{u}$. We denote the total cost as $\mathcal{L}(\mathbf{u}) = \sum_{i \in \mathcal{V}} \mathcal{L}_i(\mathbf{u}_i)$, where:

$$\mathcal{L}_i(\mathbf{u}_i) = \sum_{t=0}^{N-1} \ell_i(x_i(t), u_i(t), t) + V_{f,i}(x_i(N))$$

In this formulation:
* $\mathbf{u}_i = (u_i(0), \dots, u_i(N-1))$ denotes the $i^{th}$ agent's inputs.
* $\ell_i$ is the **running cost**, penalizing state energy, control effort, or deviations from reference points.
* $V_{f,i}$ is the **terminal cost**, penalizing deviations from a desired terminal condition.
* $\rho^\phi(\mathbf{u})$ represents the **STL robustness function**.

The multi-agent STL planning problem is defined as:

$$
\begin{aligned}
    &\text{Minimize}_{\mathbf{u}} && \mathcal{L}(\mathbf{u}) = \sum_{i \in \mathcal{V}} \mathcal{L}_i(\mathbf{u}_i) \\
    &\text{subject to} && \rho^\phi(\mathbf{u}) = \min_{\nu \in \mathcal{K}_\phi} \rho^{\phi_\nu}(\mathbf{u}_\nu) > 0
\end{aligned}
$$

The goal is to minimize a separable cost function $\mathcal{L}(u)$ while satisfying a multi-agent STL specification $\phi$.

> 🐍 Python implementation: Running control cost $\mathcal{L}(u_i)$ for unicycle dynamics
>
```python
import jax
import jax.numpy as jnp

def unicycle_running_cost(state, control):
    """Penalizes control effort and deviation from a goal."""
    # control: [v, omega]
    control_penalty = 0.1 * control[0]**2 + 0.5 * control[1]**2
    return control_penalty

@jax.jit
def agent_total_cost(u_i, x0_i):
    """Computes L_i(u_i) for a single unicycle agent."""
    traj = compute_unicycle_trajectory(u_i, x0_i)
    
    # Sum running costs across the horizon
    total_run = jnp.sum(jax.vmap(unicycle_running_cost)(traj[:-1], u_i))
    return total_run
```

We use **softmin/softmax** to underapproximate min/max operators in $\min\rho^{\phi_\nu}(\mathbf{u}_\nu)$: 

$$\min \left(\mu_1,\ldots,\mu_q\right) 
\overset{^{\geq}}{\approx} 
-\frac{1}{\Gamma} \log\left(\sum_{j=1}^q \exp(-\Gamma \mu_j)\right)$$

$$\max \left(\mu_1,\ldots,\mu_q\right) 
\overset{^{\geq}}{\approx} 
\frac{\sum_{j=1}^q \mu_j \exp(\Gamma \mu_j)}{\sum_{j=1}^q \exp(\Gamma \mu_j)}$$

where $\Gamma>0$ is the smoothing parameter.

> 🐍 Python implementation of softmin/softmax:
> 
```python
import jax
import jax.numpy as jnp

def smooth_min(vec, G):
    m = jnp.min(vec)
    return -(1/G) * (jnp.log(jnp.sum(jnp.exp(-G * (vec - m)))) - G * m)

def smooth_max(vec, G):
    """
    Boltzmann operator (Softmax weighted average).
    Approximates max(vec) as a weighted average.
    """
    m = jnp.max(vec)
    # Log-Sum-Exp trick for the denominator/weights
    exp_weights = jnp.exp(G * (vec - m)) 
    sum_weights = jnp.sum(exp_weights)
    
    # The sum(mu * exp(G*mu)) / sum(exp(G*mu))
    return jnp.sum(vec * exp_weights) / (sum_weights)
```
---

## 💡 Solution Approach

We relax the constrained problem into an unconstrained penalty-based program (UPP):

$$\min_{\mathbf{u}} \mathcal{F}_\lambda(\mathbf{u}) = \mathcal{L}(\mathbf{u}) + \lambda R(**u**) \qquad \text{(UPP)}$$

with a penalty parameter $\lambda>0$ and a quadratic penalty function

$$ R(**u**) = \max(0,-\varrho^\phi_\Gamma(**u**))^2$$

This is a smooth penalty function with gradient

$$\nabla R(**u**) = -2\max(0,-\varrho^\phi_\Gamma(**u**))\nabla\varrho_\Gamma^\phi(**u**)$$

where $\varrho^\phi_\Gamma(u)$ represents the **smooth STL semantics** underapproximating $\min\rho^{\phi_\nu}(\mathbf{u}_\nu)$. 

> Python Implementation: Penalty Function & Smooth Robustness
>
```python
@jax.jit
def compute_penalty_R(u, X0, K_phi, G):
    """
    Computes the quadratic penalty R(u) = max(0, -rho_smooth)^2.
    """
    # 1. Integrate all agent trajectories
    X_all = compute_MAS_trajectories(u, X0)
    
    # 2. Extract aggregate trajectories for each clique
    x_cliques = get_clique_trajectories(X_all, K_phi)
    
    # 3. Evaluate smooth robustness for each clique phi_nu
    # Note: evaluate_phi_nu is a placeholder for your specific STL logic
    rho_cliques = jnp.array([evaluate_phi_nu(x_nu) for x_nu in x_cliques])
    
    # 4. Global smooth robustness (under-approximation of the min)
    rho_smooth = smooth_min(rho_cliques, G)
    
    # 5. Quadratic penalty on violation
    return jnp.maximum(0, -rho_smooth)**2
```


### Inner Loop (BCGD)
For a fixed penalty $\lambda$, we use **Block-Coordinate Gradient Descent (BCGD)**:
* **Step:** Update agent $i$ while keeping all other agents $\mathbf{u}_{-i}$ fixed.
* **Quadratic Approximation:** The quadratic model of $R$ evaluated at $\boldsymbol{u}^k$
is $$Q^{H}(\boldsymbol{u}^k, \boldsymbol{d}) \coloneqq \nabla R(\boldsymbol{u}^k)^\intercal \boldsymbol{d} + \frac{1}{2} \boldsymbol{d}^\intercal H^k \boldsymbol{d} \approx R(\boldsymbol{u}^k+\boldsymbol{d}) -R(\boldsymbol{u}^k)$$
where $H^k {\succ} 0$ is a Hessian approximation.
  
* **Block Update Direction:**

$$
\boldsymbol{d}_i^k=\arg\min_{\boldsymbol{d}} \lambda Q^{H}(\boldsymbol{u}^k, \boldsymbol{d})+\mathcal{L}(\boldsymbol{u}^k + \boldsymbol{d})|\boldsymbol{d}_j = 0,\; j \notin J^k \qquad \text{(BUD)}
$$

* **Update Rule:** $\mathbf{u}_i^{k+1} = \mathbf{u}_i^k + a^k d_i^k$
* **Advantage:** Computation scales linearly with the number of agents.

**Algorithm 1: Block Coordinate Gradient Descent (BCGD)**

| Step | Description |
|------|------------|
| **Input:** | $\boldsymbol{u}^0$ (initial iterate), $\lambda$ (penalty), $K$ (max iterations), $\epsilon_0 > 0$ |
| 1 | **for** $k = 0,1,\dots,K-1$ **do** |
| 2 | $\quad$ **if** $\lVert \nabla F_\lambda(\boldsymbol{u}^k) \rVert \le \epsilon_k$ **break** |
| 3 | $\quad$ Select agent blocks $J^k \subseteq \mathcal{V}$ (Gauss–Seidel or Gauss–Southwell) |
| 4 | $\quad$ Compute $\nabla R(\boldsymbol{u}^k)$ and select $H^k$ (e.g., $H^k=I$) |
| 5 | $\quad$ Compute block update direction $\boldsymbol{d}^k$ via (BUD) |
| 6 | $\quad$ Choose step size $\alpha^k$ using the Armijo rule |
| 7 | $\quad$ Update $\boldsymbol{u}^{k+1} = \boldsymbol{u}^k + \alpha^k \boldsymbol{d}^k$ |
| 8 | **end for** |
| **Output:** | $\boldsymbol{u}^{\star,k}$ |


> 🐍 **Python Implementation: JAX-Compatible BCGD**
>
> This function performs one full "epoch" of updates. By using `jax.lax.fori_loop`, we maintain the sequential dependency required by the Gauss-Seidel method while staying inside the XLA-compiled JIT boundary.

```python
import jax
import jax.numpy as jnp

@jax.jit
def bcgd_epoch(u_k, lam, gamma_smooth, key):
    num_agents = u_k.shape[0]
    
    # 1. Randomly shuffle agent indices for this epoch
    shuffled_indices = jax.random.permutation(key, jnp.arange(num_agents))
    
    def agent_update_body(i, current_u):
        idx = shuffled_indices[i]
        
        # 2. Re-evaluate gradient for the updated global trajectory
        grad_R = jax.grad(compute_penalty_R)(current_u, gamma_smooth)
        
        # 3. Compute Block Update Direction (BUD) for specific agent
        d_i = solve_local_subproblem(current_u[idx], grad_R[idx], lam)
        
        # 4. Armijo Line Search for this specific block update
        alpha_i = find_block_step_size(current_u, d_i, idx, lam)
        
        # 5. Apply update to the agent block using JAX's functional syntax
        return current_u.at[idx].set(current_u[idx] + alpha_i * d_i)

    # Execute sequential updates across all shuffled indices
    return jax.lax.fori_loop(0, num_agents, agent_update_body, u_k)
    
    return u_next
```


### Outer Loop (Penalty Method)
1. **Initialize** penalty parameter $\lambda_0$ and control sequence $\mathbf{u}_0$.
2. **Solve** the inner loop (Algorithm 1) to find a stationary point.
3. **Increase** $\lambda$ (e.g., $\lambda \leftarrow \beta \mu$) and repeat until $\rho^\phi(\mathbf{u}) > 0$.


**Algorithm 2: Penalty Method (PM)**

| Step | Description |
|------|------------|
| **Input:** | $\boldsymbol{u}^0$ (initial iterate), $\lambda^0$ (initial penalty), $K_{\mathrm{PM}}$ (max. iterations), $\eta_\lambda > 1$ (penalty update factor), $\epsilon_{\mathrm{infeas}} > 0$ (max. infeasibility), $\epsilon^0$ (initial tolerance), $\eta_\epsilon \in (0,1]$ (tolerance update parameter) |
| 1 | Set $\bar{\boldsymbol{u}}^0 = \boldsymbol{u}^0$ |
| 2 | **for** $k = 0,1,\dots,K_{\mathrm{PM}}-1$ **do** |
| 3 | $\quad$ **if** $R(\bar{\boldsymbol{u}}^{k}) < \epsilon_{\mathrm{infeas}}$ **break** |
| 4 | $\quad$ Use Algorithm&nbsp;1 (BCGD) to solve (UPP) with $\lambda = \lambda^k$, initial guess $\bar{\boldsymbol{u}}^k$, and tolerance $\epsilon^k$; obtain an $\epsilon^k$-approximate solution $\boldsymbol{u}^{\star,k}$ |
| 5 | $\quad$ Update $\lambda^{k+1} = \eta_\lambda \lambda^k$, $\bar{\boldsymbol{u}}^{k+1} = \boldsymbol{u}^{\star,k}$, $\epsilon^{k+1} = \eta_\epsilon \epsilon^k$ |
| 6 | **end for** |
| **Output:** | $\bar{\boldsymbol{u}}^{\star,k}$ |

>🐍 **Python Implementation: Penalty Method (Outer Loop)**
>
> The outer loop gradually increases the penalty parameter $\lambda$ to drive the swarm toward a feasible STL solution.

```python
def penalty_method_outer_loop(u_init, config):
    u_curr = u_init
    lam = config['lambda_0']
    key = jax.random.PRNGKey(0)
    
    for k in range(config['K_PM']):
        # Check STL robustness satisfaction (Early exit)
        rho_val = compute_real_stl_robustness(u_curr)
        if rho_val > 0:
            break
            
        # Inner loop: Run BCGD epochs until convergence for current lambda
        for inner_k in range(config['max_inner_iter']):
            key, subkey = jax.random.split(key)
            u_prev = u_curr
            u_curr = bcgd_epoch(u_curr, lam, config['gamma'], subkey)
            
            # Check for inner-loop stationarity
            if jnp.linalg.norm(u_curr - u_prev) < config['eps_inner']:
                break
        
        # Update Penalty Schedule
        lam *= config['eta_lam']  # Increase penalty weight
        
    return u_curr
```

---

## 🚀 Features
* **Simplicity**: Eliminates the need for Hessian $H^k$ computation. The quadratic approximation of the penalty function $R(u)$ can be simplified by setting $H_i^k = I$.  
* **Scalability**: Handles large robot fleets by directional updates at the agent-block level.
* **Convergence**: Provable convergence to stationary points of the penalized problem (UPP).
---
## 📚 References
Details of the smooth under approximations of the STL robustness function can be found in:
* **Y. Gilpin, V. Kurtz, and H. Lin**, “A smooth robustness measure of signal temporal logic for symbolic control,” IEEE Control Systems Letters, vol. 5, no. 1, pp. 241–246, 2020

The Block-Coordinate Gradient Descent (BCGD) implementation and its convergence properties are based on the framework established in: 
* **P. Tseng and S. Yun**, "A coordinate gradient descent method for nonsmooth separable minimization," Mathematical Programming, vol. 117, no. 1, pp. 387–423, 2009.

The Penalty Method (Outer Loop) is based on the framework in:
* **W. Sun and Y.-X. Yuan**, Optimization Theory and Methods: Nonlinear Programming, Springer, 2006. (See Section 10.2: The Simple Penalty Function Method).
---

## 📜 Citation
If you use this method/code in your research, please cite:
```bibtex
@inproceedings{VlahakisLCSS26,
  title={Efficient Multi-Agent Temporal Logic Planning via Block-Coordinate Optimization and Smooth Penalty Functions},
  author={Vlahakis, Eleftherios E. and Kordabad, Arash Bahari and Lindemann, Lars and Sopasakis, Pantelis and Soudjani, Sadegh and Dimarogonas, Dimos V.},
  booktitle={arxiv},
  year={2026}
}

