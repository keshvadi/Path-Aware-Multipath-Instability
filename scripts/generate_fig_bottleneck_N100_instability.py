# main_pan.py
import random
import matplotlib.pyplot as plt
import numpy as np

from algorithms_pan import pan_primal_equations
from network_components import Resource, Path, Source
from scheduler import DynamicNaiveScheduler

def run_pan_simulation_dynamic():
    random.seed(42)
    np.random.seed(42)
    print("Seed: 42 | N=100 | ΔT=5.0 | dt=0.1 | t_max=400 | active_set=5 | probe_rate=0.1")

    # --- Shared-bottleneck topology (high-diversity setup) ---
    NUM_PATHS = 100
    bottleneck = Resource(id='bottleneck', capacity=25.0)
    resources = {'bottleneck': bottleneck}
    for i in range(9):
        resources[f'j{i}'] = Resource(id=f'j{i}', capacity=30.0)
    resource_list = list(resources.values())

    s1 = Source(id='s1')
    paths = {}
    for i in range(NUM_PATHS):
        path_id = f'r{i}'  # internal ID (r0, r1, ..., r99)
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
        probe_rate=0.1,
        decision_interval=5.0,
        decision_start_time=100.0
    )

    network = {
        'sources': {'s1': s1},
        'resources': resources,
        'paths': paths,
        'path_list': path_list,
        'scheduler': scheduler,
        'active_paths': None
    }
    params = {'a_bar': 0.05, 'b_r': 0.875, 'beta_j': 10}

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
            max_penalty = max(((z_temp[r.id] + probe_per_res[r.id]) / r.capacity) ** params['beta_j']
                              for r in resource_list)
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

        # Enable probing in equations when active
        network['probing_active'] = probing_active

        # Inner loop dynamics (primal congestion control)
        derivatives = pan_primal_equations(t, current_state, network, params)
        new_state = current_state + np.array(derivatives) * dt
        current_state = np.maximum(0, new_state)

        # Record for plotting
        history['time'].append(t)
        history['total_throughput'].append(np.sum(current_state))
        history['active_path_ids'].append(tuple(p.id for p in active_paths_now))

    history['decision_epochs'] = scheduler.decision_times

    # --- Basic stats ---
    switch_count = sum(1 for i in range(1, len(history['active_path_ids']))
                       if history['active_path_ids'][i] != history['active_path_ids'][i-1])
    print(f"Decision epochs: {len(history['decision_epochs'])}")
    print(f"Actual switches: {switch_count}")

    plot_dynamic_pan_results(history)
    return history


def plot_dynamic_pan_results(history):
    plt.figure(figsize=(12, 7))
    plt.plot(history['time'], history['total_throughput'], color='black', label='Total Throughput $X(t)$')

    epochs = history['decision_epochs']
    alpha = 0.8 if len(epochs) < 20 else 0.3
    lw = 1.0 if len(epochs) < 20 else 0.5
    for t in epochs:
        plt.axvline(x=t, color='red', linestyle='--', linewidth=lw, alpha=alpha,
                    label='Scheduler decision epoch' if t == epochs[0] else "")

    plt.title('Thundering Herd under Naive Scheduling\n(PAN Primal + Dynamic Naive Scheduler)')
    plt.xlabel('Time $t$')
    plt.ylabel('Aggregate Throughput $X(t)$')
    plt.ylim(0, 16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ring_N100_instability_trace.png', dpi=200)
    print("Saved ring_N100_instability_trace.png")
    plt.show()


if __name__ == '__main__':
    run_pan_simulation_dynamic()