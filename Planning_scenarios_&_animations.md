# Ten-Robot SLT Motion Planning (Numerical Validation)

We evaluate the **BCGD-PM** (Block Coordinate Gradient Descent - Penalty Method) framework using a complex multi-agent motion planning scenario. We compare its performance against traditional Mixed-Integer Programming (MIP) approaches across various task specifications as well as linear and unicycle dynamical models.

---

## Problem Description
The evaluation considers a ten-robot workspace containing three obstacles ($\mathcal{O}_1, \mathcal{O}_2, \mathcal{O}_3$), ten collection regions ($C_i$), and ten delivery regions ($D_i$). 

<p align="center">
  <img src="assets/workspace_layout.png" width="850">
  <br>
  <em>Figure: Workspace</em>
</p>

In the most challenging case, robots $(Ri)$ must satisfy a complex Signal Temporal Logic (STL) task:
* **Reachability:** Visit collection region $C_i$ and delivery region $D_i$ within specific time intervals.
* **Safety:** Avoid all obstacles $\mathcal{O}_l$ and maintain inter-agent collision avoidance.
* **Collaboration:** Meet peers belonging to the same cliques within the 100-step time horizon. 

### Robot Dynamics
We test performance under discrete-time **linear dynamics** as well as **unicycle dynamics** for each robot $i$ with discretization interval $\Delta t=1$ time unit:

**Linear Case (Single Integrator):**
In this simplified setup, the control inputs directly dictate the change in position.
* **State:** $x_i(t) = (z_i(t), y_i(t)) \in \mathbb{R}^2$ (Cartesian position).
* **Controls:** $u_i(t) = (u_{i,1}(t), u_{i,2}(t)) \in \mathbb{R}^2$ (velocity inputs).
* **Equations:** $x_i(t+1) = x_i(t) + \Delta t u_i(t)$

**Unicycle case:**
In this model, the robot's movement is restricted by its heading.
* **State:** $x_i(t)=(z_i(t), y_i(t)) \in \mathbb{R}^2$ (Cartesian position) and $\theta_i(t) \in \mathbb{R}$ (heading).
* **Controls:** $v_i(t)$ (linear velocity) and $\omega_i(t)$ (angular velocity).
* **Equations:**
$z_i(t+1)=z_i(t)+\Delta t v_i(t)\cos\theta_i(t)$, $y_i(t+1)=y_i(t)+\Delta t v_i(t)\sin\theta_i(t)$, and $\theta_i(t+1)=\theta_i(t)+\Delta t\omega_i(t)$

### Cost Function: $\mathcal{L}(u)=\sum_i \mathcal{L}_i(u_i)$:
To retain convexity of $\mathcal{L}_i$ we penalize only control effort setting 

$$\mathcal{L}_i(u_i) = \sum_{t=0}^{N-1} \ell_i(u_i(t),t)$$

* **Linear case:** $\ell_i(u_i(t),t)=u_{i,1}(t)^2+u_{i,2}(t)^2$

* **Unicycle case:** $\ell_i(u_i(t),t)=(w_1 v_i(t))^2+(w_2 \omega_i(t))^2$, where $w_1$ and $w_2$ are weighting factors balancing the emphasis between linear and angular velocity control effort.

### Collaborative Task Topology & Cliques $\nu \in \mathcal{K}_\phi$
The figure below illustrates the collaborative formulas $\phi_\nu$ defined for cliques of agents. Each node represents a robot, and colored edges represent specific joint (meeting) STL tasks.

<p align="center">
  <img src="assets/clique_set_github.PNG" width="550">
  <br>
  <em>Figure: Collaborative-task graph for the ten-robot system.</em>
</p>

---

## Multi-Agent STL scenarios
### Baseline specification **R2AM** (Reach-twice-Avoid-Meet):
1.  **Avoid Obstacles:** $\square_{\mathcal{I}}\neg \mathcal{O}_l$ for all time.
2.  **Collect:** Visit region $C_i$ within $t \in [10, 50]$.
3.  **Deliver:** Reach region $D_i$ within $t \in [70, 100]$.
4.  **Collaborative Meeting:** Robots belonging to the same clique $\nu$ are required to approach each other within a distance of $0.25$ units during the interval $t \in [0, 70]$. The collaborative predicate for a clique is defined as:
$$M_\nu \coloneqq \left( \min_{\kappa \in \nu} \mu_\kappa(x_\kappa) \geq 0 \right)$$

where:
* $x_\kappa = (z_q, y_q)_{q \in \kappa}$ represents the concatenated positions of the agents in team $\kappa$.
* $\kappa = (i, j) \subseteq \nu$ denotes any pair of robots within the clique.
* The predicate function is:  $\mu_\kappa(x_\kappa) = 0.25 - \left\| [I \quad -I] x_\kappa \right\|_2$

### **R2AMCA** (R2AM + Collision Avoidance)
We extend the baseline **R2AM** specification by incorporating inter-agent safety constraints:

1. **Inter-agent Collision Avoidance:** For every pair of agents $(i, j)$, a minimum safety distance must be maintained globally:
   $$\|x_i(t) - x_j(t)\|_2 \geq 0.01, \quad \forall t \in [0, 100]$$
2. **Standard Requirements:** Retains all Obstacle Avoidance, Collection, Delivery, and Collaborative Meeting constraints from the R2AM baseline.


### **RURAMCA** (Reach-Until-Reach + Meet + Collision Avoidance)
This scenario introduces more complex temporal dependencies by replacing simple reachability with the **Until** ($\mathcal{U}$) operator.

1. **Reach-Until-Reach Task:** Each robot $i$ must visit the collection region $C_i$ *until* it moves toward the delivery goal $D_i$, specified as
 
 $$(\lozenge_{[10,50]} C_i) U_{[0,50]} (\lozenge_{[10,50]} D_i)$$

3. **Collaboration and Safety:** All collaborative meeting requirements as well as global inter-agent collision avoidance and obstacle avoidance constraints are retained from **R2AMCA**.

---

## 3. Performance Comparison
The following table compares the computational runtime (in seconds) between our proposed **BCGD-PM** method and the **MIP** baseline. All benchmarks were performed on a KTH-13361 workstation equipped with a 13th Gen Intel(R) Core(TM) i7-1365U (1.80 GHz) processor and 16.0 GB RAM, running a 64-bit operating system.

---

| Scenario | Dynamics | BCGD-PM (Ours) | MIP [1, 2] |
| :--- | :--- | :--- | :--- |
| **R2AM** | Linear | **345s** | 425s |
| **R2AMCA** | Linear | **1291s** | > 2000s |
| **RURAMCA** | Linear | **1740s** | > 10000s |
| **R2AM** | Unicycle | **820s** | N/A |
| **R2AMCA** | Unicycle | **1852s** | N/A |
| **RURAMCA** | Unicycle | **2326s** | N/A |

---

## 4. Implementation Details & Parameter Selection
The algorithm is implemented in **Python** leveraging **JAX** for high-performance hardware acceleration and automatic differentiation.

* **Outer Loop:** Penalty Method (terminates when infeasibility $R(\mathbf{u}) \leq 5.0 \times 10^{-4}$).
* **Inner Loop:** BCGD with randomized block updates.
* **Update Rule:** Using a simple Hessian approximation $H^k = I$, the update direction $\mathbf{d}_i^k$ at each BCGD iteration attains a closed-form solution:

  $$\mathbf{d}_i^k = - (\lambda^k + 2)^{-1} \left( 2 \mathbf{u}_i^k + \lambda^k \nabla R(\mathbf{u}^k)_i \right)$$
  
* **Efficiency:** BCGD typically terminates in fewer than 2750 updates per outer iteration. The randomization of block updates (shuffling agent order) significantly helps in avoiding local minima compared to centralized penalty methods.

---

## 5. Visual Results
The results for the **RURAMCA** scenario under unicycle dynamics are illustrated below. The robots successfully fulfill temporal promises (visiting collection regions until transitioning to delivery) while ensuring collision-free motion and meeting collaborative tasks.

![Ten-Robot Trajectories](figures/RURAMCA_unic_fig_final.png)
*Figure 1: RURAMCA motion planning for ten robots with discrete-time unicycle dynamics.*
