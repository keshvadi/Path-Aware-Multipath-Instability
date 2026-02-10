# main_ring_N8_instability.py
import random
import numpy as np
import matplotlib.pyplot as plt

from network_components import Resource, Path, Source
from scheduler import DynamicNaiveScheduler
from algorithms_pan import pan_primal_equations   # same inner-loop style as main_pan.py


def build_ring_N8_one_source():
    """
    Build an N=8 candidate-path set using the ring testbed RTT classes:
      - fast:   0.015
      - medium: 0.15
      - slow:   1.5
    We model ONE sender with 8 alternative paths (N=8) to keep the experiment
    directly aligned with "path diversity N".

    Each path traverses exactly ONE link resource, like in your ring testbed.
    """
    # Four ring links (same as experiments 1/2, capacity 15)
    l1 = Resource(id='l1', capacity=15.0)
    l2 = Resource(id='l2', capacity=15.0)
    l3 = Resource(id='l3', capacity=15.0)
    l4 = Resource(id='l4', capacity=15.0)

    resources = {r.id: r for r in [l1, l2, l3, l4]}
    resource_list = list(resources.values())

    # One source with N=8 candidate paths
    s = Source(id='s1')

    # 8 paths: two per link class to mimic your original mapping
    # fast (l1): 2 paths with RTT 0.015
    # medium (l2,l4): 4 paths with RTT 0.15
    # slow (l3): 2 paths with RTT 1.5
    paths = {
        'p1': Path(id='p1', source_id='s1', resources=[l1], rtt=0.015),
        'p8': Path(id='p8', source_id='s1', resources=[l1], rtt=0.015),

        'p2': Path(id='p2', source_id='s1', resources=[l2], rtt=0.15),
        'p3': Path(id='p3', source_id='s1', resources=[l2], rtt=0.15),

        'p6': Path(id='p6', source_id='s1', resources=[l4], rtt=0.15),
        'p7': Path(id='p7', source_id='s1', resources=[l4], rtt=0.15),

        'p4': Path(id='p4', source_id='s1', resources=[l3], rtt=1.5),
        'p5': Path(id='p5', source_id='s1', resources=[l3], rtt=1.5),
    }
    path_list = list(paths.values())

    network = {
        'sources': {'s1': s},
        'resources': resources,
        'paths': paths,
        'path_list': path_list,
        'resource_list': resource_list,
    }
    return network


def run_ring_N8_instability(
    seed=42,
    dt=0.1,
    t_max=400.0,
    decision_start_time=100.0,
    decision_interval=5.0,
    init_rate=0.1,
):
    """
    Hybrid simulation:
      - inner loop: pan_primal_equations (Euler step)
      - outer loop: DynamicNaiveScheduler (switching after decision_start_time)
      - N=8 candidate paths, but only 1 active at a time (all-to-one switching)

    Output:
      - ring_N8_instability_trace.png
    """
    random.seed(seed)
    np.random.seed(seed)

    network = build_ring_N8_one_source()
    path_list = network['path_list']
    resource_list = network['resource_list']

    # Scheduler: choose ONE active path (active_set_size=1)
    scheduler = DynamicNaiveScheduler(
        source_id='s1',
        all_paths=path_list,
        active_set_size=1,
        probe_rate=0.0,                 # start with no probing (cleaner)
        decision_interval=decision_interval,
        decision_start_time=decision_start_time
    )

    # Attach scheduler to network (same pattern as main_pan.py)
    network['scheduler'] = scheduler
    network['active_paths'] = scheduler.get_active_paths()

    # Inner-loop params (keep consistent with your other runs unless justified)
    params = {'a_bar': 0.05, 'b_r': 0.875, 'beta_j': 10}

    # State: rate for currently active path set (size 1)
    current_state = np.array([init_rate] * len(network['active_paths']))

    time_points = np.arange(0, t_max + dt / 2, dt)

    history = {
        'time': [],
        'total_throughput': [],
        'active_path_ids': [],
        'decision_epochs': []
    }

    for t in time_points:
        # Scheduler OFF before decision_start_time
        probing_active = (t >= scheduler.decision_start_time)
        network['probing_active'] = probing_active

        # Compute a simple congestion proxy (like main_pan.py):
        # penalty = max_over_resources ( (load/capacity)^beta )
        # Here load is sum of active path rates over resources it uses.
        z_temp = {r.id: 0.0 for r in resource_list}
        for i, p in enumerate(network['active_paths']):
            if i < len(current_state):
                for res in p.resources:
                    z_temp[res.id] += current_state[i]

        max_penalty = max(((z_temp[r.id] / r.capacity) ** params['beta_j']) for r in resource_list)
        current_loss_proxy = min(1.0, max_penalty)

        # Outer-loop scheduler update
        scheduler.update(t, dt=dt, current_loss=current_loss_proxy)
        active_paths_now = scheduler.get_active_paths()

        # If active path changed, enforce state invalidation (hard reset)
        prev_ids = tuple(p.id for p in network['active_paths'])
        new_ids  = tuple(p.id for p in active_paths_now)
        if new_ids != prev_ids:
            current_state = np.array([init_rate] * len(active_paths_now))
            network['active_paths'] = active_paths_now

        # Inner-loop update (Euler)
        derivatives = pan_primal_equations(t, current_state, network, params)
        current_state = np.maximum(0.0, current_state + np.array(derivatives) * dt)

        # Record
        history['time'].append(t)
        history['total_throughput'].append(float(np.sum(current_state)))
        history['active_path_ids'].append(tuple(p.id for p in active_paths_now))
        history['decision_epochs'] = scheduler.decision_times

    # Print quick stats
    switches = sum(
        1 for i in range(1, len(history['active_path_ids']))
        if history['active_path_ids'][i] != history['active_path_ids'][i - 1]
    )
    print(f"N=8 ring-style candidate paths")
    print(f"Decision epochs: {len(history['decision_epochs'])}")
    print(f"Actual switches: {switches}")

    plot_ring_N8_results(history, out_file='ring_N8_instability_trace.png')
    return history


def plot_ring_N8_results(history, out_file='ring_N8_instability_trace.png'):
    plt.figure(figsize=(12, 7))
    plt.plot(history['time'], history['total_throughput'],
             color='black', label='Total Throughput $X(t)$')

    epochs = history['decision_epochs']
    alpha = 0.8 if len(epochs) < 20 else 0.25
    lw = 1.0 if len(epochs) < 20 else 0.5

    for tt in epochs:
        plt.axvline(x=tt, color='red', linestyle='--', linewidth=lw, alpha=alpha,
                    label='Scheduler decision epoch' if tt == epochs[0] else "")

    plt.title('N=8 Ring Candidate Paths: Thundering Herd under Naive Scheduling')
    plt.xlabel('Time $t$')
    plt.ylabel('Aggregate Throughput $X(t)$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Saved {out_file}")
    plt.show()


if __name__ == '__main__':
    run_ring_N8_instability()