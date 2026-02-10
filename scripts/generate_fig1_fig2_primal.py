from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

from network_components import Resource, Path, Source
from algorithms import primal_model_equations


def run_primal_simulation():
    """
    Run validation for a 4-resource, 4-source RING topology.
    - Pre-failure equilibrium (t < 0)
    - Fail resource ℓ₂ at t = 0
    - Simulate recovery (t > 0)
    """
    print("VALIDATION: Primal-style fluid model on RING topology with failure at t=0")

    # --- Network and Parameter Setup ---
    l1 = Resource(id='l1', capacity=15)
    l2 = Resource(id='l2', capacity=15)
    l3 = Resource(id='l3', capacity=15)
    l4 = Resource(id='l4', capacity=15)

    s1, s2, s3, s4 = Source(id='s1'), Source(id='s2'), Source(id='s3'), Source(id='s4')

    # Ring Topology Configuration:
    paths = {
        'p1': Path(id='p1', source_id='s1', resources=[l1], rtt=0.015),
        'p2': Path(id='p2', source_id='s1', resources=[l2], rtt=0.15),
        'p3': Path(id='p3', source_id='s2', resources=[l2], rtt=0.15),
        'p4': Path(id='p4', source_id='s2', resources=[l3], rtt=1.5),
        'p5': Path(id='p5', source_id='s3', resources=[l3], rtt=1.5),
        'p6': Path(id='p6', source_id='s3', resources=[l4], rtt=0.15),
        'p7': Path(id='p7', source_id='s4', resources=[l4], rtt=0.15),
        'p8': Path(id='p8', source_id='s4', resources=[l1], rtt=0.015),
    }

    path_list = list(paths.values())

    network = {
        'sources': {'s1': s1, 's2': s2, 's3': s3, 's4': s4},
        'resources': {'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4},
        'paths': paths,
        'path_list': path_list
    }

    # Standard primal parameters
    params = {'a_bar': 0.1, 'b_r': 0.875, 'beta_j': 10}

    # --- Phase 1: Pre-failure (t < 0) ---
    print("1) Solving pre-failure dynamics (t in [-100, 0])...")
    y0 = [1.0] * len(path_list)
    pre_span = (-100.0, 0.0)

    pre = solve_ivp(
        fun=primal_model_equations,
        t_span=pre_span,
        y0=y0,
        args=(network, params),
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )

    if not pre.success:
        raise RuntimeError(f"Pre-failure solve failed: {pre.message}")

    y_at_0_minus = pre.y[:, -1]

    # --- Phase 2: Apply failure at t=0 and simulate post-failure recovery ---
    print("2) Applying failure: set capacity of ℓ₂ = 0 at t=0, then solve post-failure...")
    
    network['resources']['l2'].capacity = 0

    post_span = (0.0, 400.0)
    post = solve_ivp(
        fun=primal_model_equations,
        t_span=post_span,
        y0=y_at_0_minus,
        args=(network, params),
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )

    if not post.success:
        raise RuntimeError(f"Post-failure solve failed: {post.message}")

    print("3) Generating plots for Ring Topology...")
    plot_validation_figures(pre, post, network)
    print("Done.")


def plot_validation_figures(pre, post, network):
    """Create (i) individual path rates and (ii) aggregate per-source rates for 8 paths."""

    path_list = network['path_list']
    ids = [p.id for p in path_list]  # ['p1', 'p2', ..., 'p8']

    # Smooth timelines
    t_pre = np.linspace(pre.t.min(), pre.t.max(), 600)    # [-100, 0]
    t_post = np.linspace(post.t.min(), post.t.max(), 1400)  # [0, 400]

    x_pre = pre.sol(t_pre)
    x_post = post.sol(t_post)

    # Map to named arrays
    pre_map = {pid: x_pre[i] for i, pid in enumerate(ids)}
    post_map = {pid: x_post[i] for i, pid in enumerate(ids)}

    # Concatenate timelines
    t_all = np.concatenate([t_pre, t_post])
    x_all = {
        pid: np.concatenate([pre_map[pid], post_map[pid]])
        for pid in ids
    }

    # ---------- Figure: Individual path rates (8 Paths) ----------
    styles = {
        'p1': dict(color='black', linestyle='-'),     
        'p2': dict(color='red', linestyle='--'),      
        'p3': dict(color='red', linestyle=':'),       
        'p4': dict(color='blue', linestyle='-'),      
        'p5': dict(color='blue', linestyle='--'),     
        'p6': dict(color='green', linestyle='-'),     
        'p7': dict(color='green', linestyle='--'),    
        'p8': dict(color='black', linestyle='--'),    
    }

    plt.figure(figsize=(8, 6))
    for idx, pid in enumerate(ids):
        st = styles[pid]
        label = f"$x_{{\\pi_{{{idx+1}}}}}$"
        plt.plot(t_all, x_all[pid], label=label, **st, linewidth=1.5)

    plt.axvline(0.0, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
    plt.xlabel('Time, $t$')
    plt.ylabel('Sending rate, $x_\\pi(t)$')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4, frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig('validation_ring_path_rates.png', dpi=300)
    print("Saved validation_ring_path_rates.png")

    # ---------- Figure: Aggregate per-source rates (4 Sources) ----------
    s1_agg = x_all['p1'] + x_all['p2']
    s2_agg = x_all['p3'] + x_all['p4']
    s3_agg = x_all['p5'] + x_all['p6']
    s4_agg = x_all['p7'] + x_all['p8']

    plt.figure(figsize=(8, 6))
    plt.plot(t_all, s1_agg, color='black', linestyle='-',  linewidth=1.8, label=r'$s_1\ (x_{\pi_1}+x_{\pi_2})$')
    plt.plot(t_all, s2_agg, color='red',   linestyle='--', linewidth=1.8, label=r'$s_2\ (x_{\pi_3}+x_{\pi_4})$')
    plt.plot(t_all, s3_agg, color='blue',  linestyle=':',  linewidth=2.0, label=r'$s_3\ (x_{\pi_5}+x_{\pi_6})$')
    plt.plot(t_all, s4_agg, color='green', linestyle='-.', linewidth=1.8, label=r'$s_4\ (x_{\pi_7}+x_{\pi_8})$')

    plt.axvline(0.0, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
    plt.xlabel('Time, $t$')
    plt.ylabel('Aggregate sending rate')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.xlim(t_pre.min(), t_post.max())
    plt.savefig('validation_ring_aggregate_rates.png', dpi=300)
    print("Saved validation_ring_aggregate_rates.png")

    plt.show()


if __name__ == '__main__':
    run_primal_simulation()