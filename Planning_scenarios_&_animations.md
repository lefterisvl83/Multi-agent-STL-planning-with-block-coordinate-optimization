# Numerical Validation: Ten-Robot Motion Planning

This section evaluates the **BCGD-PM** (Block Coordinate Gradient Descent - Penalty Method) framework using a complex multi-agent motion planning scenario. We compare its performance against traditional Mixed-Integer Programming (MIP) approaches across various dynamical models and task specifications.

---

## 1. Problem Setup
The evaluation uses a ten-robot workspace containing three obstacles, ten collection regions ($C_i$), and ten delivery regions ($D_i$). 

### Robot Dynamics
We consider discrete-time **unicycle dynamics** for each robot $i$:

* **State:** $(z_i(t), y_i(t)) \in \mathbb{R}^2$ (Cartesian position) and $\theta_i(t) \in \mathbb{R}$ (heading).
* **Controls:** $v_i(t)$ (linear velocity) and $\omega_i(t)$ (angular velocity).
* **Equations:**
  $$z_i(t+1)=z_i(t)+v_i(t)\cos\theta_i(t)$$
  $$y_i(t+1)=y_i(t)+v_i(t)\sin\theta_i(t)$$
  $$\theta_i(t+1)=\theta_i(t)+\omega_i(t)$$

### Collaborative Task Topology & Cliques $\nu \in \mathcal{K}_\phi$
The figure below illustrates the collaborative formulas $\phi_\nu$ defined for cliques of agents. Each node represents a robot, and colored edges represent specific joint STL tasks.

![Collaborative Graph](assets/clique_set_github.PNG)

### Task Specification (R2AM)
The baseline **R2AM** (Reach-twice-Avoid-Meet) task requires each robot to:
1.  **Avoid Obstacles:** $\square_{\mathcal{I}}\neg \mathcal{O}_l$ for all time.
2.  **Collect:** Visit region $C_i$ within $t \in [10, 50]$.
3.  **Deliver:** Reach region $D_i$ within $t \in [70, 100]$.
4.  **Collaborative Meeting:** Agents in specific cliques must approach each other ($\|x_i - x_j\| \leq 0.25$) within $t \in [0, 70]$.

---

## 2. Experimental Scenarios
We extend the baseline R2AM task into more restrictive scenarios to test the limits of the framework:

1.  **R2AMCA:** Adds global **Inter-agent Collision Avoidance** ($\|x_i(t) - x_j(t)\| \geq 0.01$).
2.  **RURAMCA (Reach-Until-Reach):** Incorporates the **Until** operator ($\mathcal{U}$), requiring robots to visit collection regions $C_i$ *until* they transition to delivery goals $D_i$, while maintaining all other constraints.

---

## 3. Performance Comparison
The following table compares the computational runtime (in seconds) between our proposed **BCGD** method and the **MIP** baseline.

| Scenario | Dynamics | BCGD-PM (Ours) | MIP [1, 2] |
| :--- | :--- | :--- | :--- |
| **R2AM** | Linear | **345s** | 425s |
| **R2AMCA** | Linear | **1291s** | > 2000s |
| **RURAMCA** | Linear | **1740s** | > 10000s |
| **R2AM** | Unicycle | **820s** | N/A |
| **R2AMCA** | Unicycle | **1852s** | N/A |
| **RURAMCA** | Unicycle | **2326s** | N/A |

---

## 4. Implementation Details
The algorithm is implemented in **Python** leveraging **JAX** for high-performance hardware acceleration and automatic differentiation.

* **Outer Loop:** Penalty Method (terminates when infeasibility $R(\mathbf{u}) \leq 5.0 \times 10^{-4}$).
* **Inner Loop:** BCGD with randomized block updates.
* **Update Rule:** Using a simple Hessian approximation $H^k = I$, we solve for the update direction $\mathbf{d}_i^k$ via the closed-form:
    $$\mathbf{d}_i^k = - (\lambda^k + 2)^{-1} \left( 2 \mathbf{u}_i^k + \lambda^k \nabla R(\mathbf{u}^k)_i \right)$$
* **Efficiency:** BCGD typically terminates in fewer than 2750 updates per outer iteration. The randomization of block updates (shuffling agent order) significantly helps in avoiding local minima compared to centralized penalty methods.

---

## 5. Visual Results
The results for the **RURAMCA** scenario under unicycle dynamics are illustrated below. The robots successfully fulfill temporal promises (visiting collection regions until transitioning to delivery) while ensuring collision-free motion and meeting collaborative tasks.

![Ten-Robot Trajectories](figures/RURAMCA_unic_fig_final.png)
*Figure 1: RURAMCA motion planning for ten robots with discrete-time unicycle dynamics.*
