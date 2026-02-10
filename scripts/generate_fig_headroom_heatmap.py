# sweep_stability_margin_vs_N.py
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Updated with accurate probing load per resource (fix for shared bottleneck)
def modified_pan_primal_equations(t, state, network, params):
    """
    An adapted version of the primal algorithm for the PAN simulation.
    This version knows about the scheduler and probing traffic.
    Modified to use network['probe_rate'] instead of scheduler.probe_rate.
    FIXED: Accurate per-resource probing load instead of uniform approximation.
    """
    # --- Unpack parameters and network components ---
    a_bar, b_r, beta_j = params['a_bar'], params['b_r'], params['beta_j']
    
    # IMPORTANT: We now only consider the paths the scheduler has made active
    active_paths = network['active_paths']
    resources = network['resources']
    path_list = network['path_list']
    
    # The state vector corresponds ONLY to the flow rates of the active paths
    flow_rates = {p.id: state[i] for i, p in enumerate(active_paths)}

    # --- Calculate intermediate values ---

    # Calculate z_j: total load on each resource
    z = {res_id: 0 for res_id in resources}
    # 1. Add data traffic from ACTIVE paths
    for p in active_paths:
        if flow_rates[p.id] > 0:
            for res in p.resources:
                z[res.id] += flow_rates[p.id]
    
    # 2. Add the "PROBE STORM" traffic from the scheduler ACCURATELY per resource
    inactive_paths = [p for p in path_list if p not in active_paths]
    probe_z = {res_id: 0.0 for res_id in resources}
    for p in inactive_paths:
        for res in p.resources:
            probe_z[res.id] += network['probe_rate']
    for res_id in z:
        z[res_id] += probe_z[res_id]

    # Calculate penalty_j for each resource (same as before)
    penalty_j = {
        res_id: (z[res_id] / res.capacity) ** beta_j
        for res_id, res in resources.items()
    }

    # Calculate lambda_r for each ACTIVE path
    lambda_r = {}
    for p in active_paths:
        prod_term = np.prod([(1 - penalty_j[res.id]) for res in p.resources])
        lambda_r[p.id] = 1 - prod_term
    
    # Calculate y_s: total flow from the source (from active paths only)
    y_s = sum(flow_rates.values())

    # --- Calculate the derivatives dx_r/dt for ACTIVE paths ---
    derivatives = []
    for i, p in enumerate(active_paths):
        x_r = state[i]
        
        inner_term = a_bar * (1 - lambda_r[p.id]) - b_r * y_s * lambda_r[p.id]
        dx_dt = (x_r / p.rtt) * inner_term
        
        if x_r <= 0 and dx_dt < 0:
            derivatives.append(0)
        else:
            derivatives.append(dx_dt)

    return derivatives

class Resource:
    """Represents a resource (e.g., a router or link) in the network."""
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity  # This is the C_j from the paper
        
        # This is for the dual algorithm, the resource price mu_j
        self.mu = 0.1 # Start with a small non-zero price

class Path:
    """Represents a route or path from a source to a destination."""
    def __init__(self, id, source_id, resources, rtt):
        self.id = id
        self.source_id = source_id # Which source this path belongs to
        # A list of Resource objects that this path uses
        self.resources = resources 
        self.rtt = rtt # This is the T_r from the paper

        # This is the variable for the flow rate x_r(t)
        self.x = 1.0 # Start with a flow of 1.0

class Source:
    """Represents a source of traffic."""
    def __init__(self, id):
        self.id = id
        
        # This is for the dual algorithm, the total source rate y_s(t)
        self.y = 1.0 # Start with a rate of 1.0

class FairnessEstimator:
    """
    Estimates Axiom 4 (Fairness) using the Markov Process 
    described in Scherrer et al. (Section 5.2).
    Simulates a population of agents to track variance of cwnd.
    """
    def __init__(self, num_agents=1000, alpha=1, beta=0.7):
        self.N = num_agents
        self.alpha = alpha # Additive Increase
        self.beta = beta   # Multiplicative Decrease
        # Initialize all agents with a small starting window
        self.cwnds = np.ones(num_agents) 

    def step(self, m, r, loss_probability=0.0):
        """
        Updates the distribution of windows for one time step.
        m: migration probability (responsiveness)
        r: reset softness (0=hard reset, 1=no reset)
        loss_probability: probability of loss event in this step
        """
        # Generate random values for all agents at once for performance
        rand_m = np.random.random(self.N)
        rand_loss = np.random.random(self.N)
        
        # Determine states for each agent
        migrating = rand_m < m
        # Loss only affects those NOT migrating (as per paper logic, migration takes precedence or happens concurrently)
        experiencing_loss = (rand_loss < loss_probability) & (~migrating)
        increasing = (~migrating) & (~experiencing_loss)

        # Apply updates
        # 1. Migration: Apply reset softness 'r'
        self.cwnds[migrating] = self.cwnds[migrating] * r
        
        # 2. Loss: Apply multiplicative decrease 'beta'
        self.cwnds[experiencing_loss] = self.cwnds[experiencing_loss] * self.beta
        
        # 3. Increase: Apply additive increase 'alpha'
        self.cwnds[increasing] += self.alpha

    def get_variance(self):
        """Returns the current fairness metric eta (variance)."""
        return np.var(self.cwnds)

class DynamicNaiveScheduler:
    """
    Periodically and abruptly changes active path set.
    High Responsiveness (m), Hard Resets (r).
    Decisions only at discrete epochs starting from decision_start_time.
    """
    def __init__(self, source_id, all_paths, probe_rate=0.01, active_set_size=5,
                 decision_interval=10.0, decision_start_time=200.0):
        self.source_id = source_id
        self.all_paths = [p for p in all_paths if p.source_id == source_id]
        self.probe_rate = probe_rate
        self.active_set_size = active_set_size
        
        # Deterministic sorting: RTT primary, path ID secondary for reproducibility
        self.all_paths.sort(key=lambda p: (p.rtt, p.id))
        self.active_paths = self.all_paths[:self.active_set_size]
        
        # Decision timing control
        self.decision_interval = decision_interval
        self.decision_start_time = decision_start_time
        self.last_decision_epoch = -1
        self.decision_times = []           # ← used in plotting for vertical lines
        
        # Fairness estimator (Axiom 4)
        self.fairness_estimator = FairnessEstimator()

    def get_active_paths(self):
        return self.active_paths

    def get_probing_load(self):
        num_inactive_paths = len(self.all_paths) - len(self.active_paths)
        return num_inactive_paths * self.probe_rate

    def update(self, t, dt=0.1, current_loss=0.0):
        # Always step the fairness estimator
        self.fairness_estimator.step(m=0.5, r=0.5, loss_probability=current_loss)

        # No decisions before decision_start_time
        if t < self.decision_start_time:
            return

        # Integer epoch number
        epoch = int(t // self.decision_interval)
        if epoch <= self.last_decision_epoch:
            return

        # New decision epoch
        self.last_decision_epoch = epoch
        self.decision_times.append(t)   # Record for plotting vertical lines
        # print(f"Decision epoch at t={t:.1f}")

        # Perceived RTTs with reproducible noise
        perceived = [
            (p, p.rtt + np.random.uniform(-0.5, 0.5))
            for p in self.all_paths
        ]
        perceived.sort(key=lambda x: x[1])
        new_best_paths = [item[0] for item in perceived[:self.active_set_size]]

        old_set = {p.id for p in self.active_paths}
        new_set = {p.id for p in new_best_paths}

        if old_set != new_set:
            # print("  -> Switching active paths!")
            self.active_paths = new_best_paths

    def get_fairness_metric(self):
        return self.fairness_estimator.get_variance()

def run_one_trial(N, DeltaT, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Running trial: N={N}, DeltaT={DeltaT}, Seed={seed}")

    # --- Shared-bottleneck topology (high-diversity setup) ---
    bottleneck = Resource(id='bottleneck', capacity=25.0)
    resources = {'bottleneck': bottleneck}
    for i in range(9):
        resources[f'j{i}'] = Resource(id=f'j{i}', capacity=30.0)
    resource_list = list(resources.values())

    s1 = Source(id='s1')
    paths = {}
    for i in range(N):
        path_id = f'r{i}'  # internal ID (r0, r1, ..., r{N-1})
        extra = random.sample([r for r in resource_list if r.id != 'bottleneck'], k=random.randint(0, 2))
        path_resources = [bottleneck] + extra
        path_rtt = random.uniform(1.0, 3.0)
        paths[path_id] = Path(id=path_id, source_id='s1', resources=path_resources, rtt=path_rtt)
    path_list = list(paths.values())

    # --- Naive greedy scheduler (outer loop) ---
    scheduler = DynamicNaiveScheduler(
        source_id='s1',
        all_paths=path_list,
        active_set_size=5,
        probe_rate=0.01,
        decision_interval=DeltaT,
        decision_start_time=200.0
    )

    network = {
        'sources': {'s1': s1},
        'resources': resources,
        'paths': paths,
        'path_list': path_list,
        'scheduler': scheduler,
        'active_paths': None,
        'probe_rate': 0.0  # Will be set in loop
    }
    params = {'a_bar': 0.02, 'b_r': 0.875, 'beta_j': 10}

    # --- Simulation parameters ---
    dt = 0.1
    t_max = 400.0
    time_points = np.arange(0, t_max + dt/2, dt)

    # Initial active set + small initial rates
    network['active_paths'] = scheduler.get_active_paths()
    current_state = np.array([0.1] * len(network['active_paths']))

    history = {
        'time': [],
        'total_throughput': [],
        'active_path_ids': [],
        'decision_epochs': []
    }

    for t in time_points:
        # Probing only active after scheduler starts
        probing_active = (t >= scheduler.decision_start_time)
        probe_rate_temp = scheduler.probe_rate if probing_active else 0.0
        network['probe_rate'] = probe_rate_temp

        # Approximate loss proxy for scheduler (based on penalty)
        current_loss_prob = 0.0
        if len(network['active_paths']) > 0:
            z_temp = {r.id: 0.0 for r in resource_list}
            for i, p in enumerate(network['active_paths']):
                if i < len(current_state):
                    for res in p.resources:
                        z_temp[res.id] += current_state[i]
            inactive_paths = [p for p in path_list if p not in network['active_paths']]
            probe_per_res = {r.id: 0.0 for r in resource_list}
            for p in inactive_paths:
                for res in p.resources:
                    probe_per_res[res.id] += probe_rate_temp
            penalties = [((z_temp[r.id] + probe_per_res[r.id]) / r.capacity) ** params['beta_j']
                         for r in resource_list]
            max_penalty = max(penalties)
            current_loss_prob = min(1.0, max_penalty)

        # Scheduler decision (outer loop)
        scheduler.update(t, dt=dt, current_loss=current_loss_prob)
        active_paths_now = scheduler.get_active_paths()

        # Abrupt state reset on active path change (state invalidation)
        prev_ids = {p.id for p in network['active_paths']}
        new_ids = {p.id for p in active_paths_now}
        if new_ids != prev_ids:
            current_state = np.array([0.1] * len(active_paths_now))

        network['active_paths'] = active_paths_now

        # Inner loop dynamics (primal congestion control)
        derivatives = modified_pan_primal_equations(t, current_state, network, params)
        new_state = current_state + np.array(derivatives) * dt
        current_state = np.maximum(0, new_state)

        # Record for plotting
        history['time'].append(t)
        history['total_throughput'].append(np.sum(current_state))
        history['active_path_ids'].append(tuple(p.id for p in active_paths_now))

    history['decision_epochs'] = scheduler.decision_times

    # --- Compute metrics ---
    times = np.array(history['time'])
    throughputs = np.array(history['total_throughput'])

    pre_mask = (times >= 180) & (times <= 200)
    X_pre = np.mean(throughputs[pre_mask]) if np.sum(pre_mask) > 0 else 0.0

    post_mask = (times >= 250) & (times <= t_max)
    if np.sum(post_mask) > 0:
        X_post = np.mean(throughputs[post_mask])
        V = np.std(throughputs[post_mask]) / X_post if X_post > 0 else 0.0
    else:
        X_post = 0.0
        V = 0.0

    R = X_post / X_pre if X_pre > 0 else 0.0

    switches = sum(1 for i in range(1, len(history['active_path_ids']))
                   if history['active_path_ids'][i] != history['active_path_ids'][i-1])
    decisions = len(history['decision_epochs'])
    switch_per_decision = switches / decisions if decisions > 0 else 0.0

    metrics = {
        'N': N,
        'DeltaT': DeltaT,
        'X_pre': X_pre,
        'X_post': X_post,
        'R': R,
        'V': V,
        'switches': switches,
        'decisions': decisions,
        'switch_per_decision': switch_per_decision
    }

    return metrics, history

def plot_trace(history, filename):
    plt.figure(figsize=(12, 7))
    plt.plot(history['time'], history['total_throughput'], color='black', label='Total Throughput $X(t)$')

    epochs = history['decision_epochs']
    alpha = 0.8 if len(epochs) < 20 else 0.3
    lw = 1.0 if len(epochs) < 20 else 0.5
    for t in epochs:
        plt.axvline(x=t, color='red', linestyle='--', linewidth=lw, alpha=alpha,
                    label='Scheduler decision epoch' if t == epochs[0] else "")

    plt.title('Throughput Trace')
    plt.xlabel('Time $t$')
    plt.ylabel('Aggregate Throughput $X(t)$')
    plt.ylim(0, 16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved {filename}")

def sweep():
    N_list = [8, 16, 32, 64, 100, 200]
    DeltaT_list = [2, 5, 10, 20, 40, 80, 160, 320]
    seed = 42
    results = []

    for N in N_list:
        for DeltaT in DeltaT_list:
            metrics, history = run_one_trial(N, DeltaT, seed)
            results.append(metrics)

            # Save example traces for N=8 and N=100 at DeltaT=5
            if N == 8 and DeltaT == 5:
                plot_trace(history, f'trace_N8_DeltaT5.png')
            if N == 100 and DeltaT == 5:
                plot_trace(history, f'trace_N100_DeltaT5.png')

    df = pd.DataFrame(results)
    df.to_csv('stability_sweep_results.csv', index=False)
    print("Saved stability_sweep_results.csv")

    # Heatmap of R
    pivot = df.pivot(index='DeltaT', columns='N', values='R')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt=".2f")
    plt.title('Throughput Ratio R vs N and ΔT')
    plt.savefig('stability_R_heatmap.png', dpi=200)
    plt.close()
    print("Saved stability_R_heatmap.png")

    # DeltaT_crit vs N
    R_min = 0.7
    dt_crit = {}
    for n in N_list:
        sub = df[df['N'] == n].sort_values('DeltaT')
        for _, row in sub.iterrows():
            if row['R'] >= R_min:
                dt_crit[n] = row['DeltaT']
                break
        else:
            dt_crit[n] = np.nan

    if all(np.isnan(v) for v in dt_crit.values()):
        print("Warning: No DeltaT meets R_min for any N. DTcrit plot will be empty.")
    else:
        ns = [k for k, v in dt_crit.items() if not np.isnan(v)]
        dts = [v for v in dt_crit.values() if not np.isnan(v)]
        plt.figure(figsize=(8, 5))
        plt.plot(ns, dts, marker='o')
        plt.xlabel('N')
        plt.ylabel('DeltaT_crit')
        plt.title('Critical Decision Interval vs Path Diversity N')
        plt.grid(True)
        plt.savefig('stability_DTcrit.png', dpi=200)
        plt.close()
        print("Saved stability_DTcrit.png")

if __name__ == '__main__':
    sweep()