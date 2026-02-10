import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class Resource:
    """Represents a resource (e.g., a router or link ℓ) in the network."""
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        
        # This is for the dual algorithm
        self.mu = 0.1  # Start with a small non-zero price

class Path:
    """Represents a route or path π from a source to a destination."""
    def __init__(self, id, source_id, resources, rtt):
        self.id = id
        self.source_id = source_id  # Which source this path belongs to
        # A list of Resource objects that this path uses
        self.resources = resources 
        self.rtt = rtt  

        # This is the variable for the flow rate
        self.x = 1.0  # Start with a flow of 1.0

class Source:
    """Represents a source of traffic."""
    def __init__(self, id):
        self.id = id
        
        # This is for the dual algorithm
        self.y = 1.0  # Start with a rate of 1.0

class DualAlgorithmDDE:
    def __init__(self, network, params, dt=0.0001):
        self.network = network
        self.params = params
        self.dt = dt
        self.p = params['p']
        self.q = params['q'] 
        self.alpha = params['alpha']
        
        # --- Pre-calculate delay steps ---
        for path in self.network['path_list']:
            path.delay_steps_Tr = int(round(path.rtt / dt))
            path.delay_steps_Tjr = {}
            path.delay_steps_Trj = {}
            for res in path.resources:
                # Delays for topology
                if res.id == 'j1': delay = 0.0075
                elif res.id == 'j2': delay = 0.075
                elif res.id == 'j3': delay = 0.75
                elif res.id == 'j4': delay = 0.075
                path.delay_steps_Tjr[res.id] = int(round(delay / dt))
                path.delay_steps_Trj[res.id] = int(round(delay / dt))

    def run(self, warmup_duration, experiment_duration):
        steps_warmup = int(warmup_duration / self.dt)
        steps_exp = int(experiment_duration / self.dt)
        
        # --- INITIALIZATION ---
        x_init = self.network['resource_list'][0].capacity / 2.0
        y_init = (2 * x_init ** self.q) ** (1 / self.q)
        mu_init = self.params['w_s']['s1'] * y_init ** (-self.p / (self.p + 1)) / x_init ** (1 / (self.p + 1))
        
        current_y = {s.id: y_init for s in self.network['source_list']}
        current_mu = {j.id: mu_init for j in self.network['resource_list']}
        
        # Initialize history buffers with these stable values
        maxlen = 25000
        history_x = {p.id: deque([x_init]*maxlen, maxlen=maxlen) for p in self.network['path_list']}
        history_mu = {j.id: deque([mu_init]*maxlen, maxlen=maxlen) for j in self.network['resource_list']}
        
        results = {'time': [], 'x': {p.id: [] for p in self.network['path_list']}}
        
        print(f"Simulation configured: {warmup_duration}s warmup + {experiment_duration}s experiment")
        
        total_steps = steps_warmup + steps_exp
        for step in range(total_steps):
            t = (step * self.dt) - warmup_duration
            
            # Failure Event at t=0
            failed_routes = []
            if t >= 0:
                self.network['resources']['j2'].capacity = 0
                failed_routes = ['r2', 'r3']
            
            # Update Flows
            current_x = {}
            for p in self.network['path_list']:
                if p.id in failed_routes:
                    current_x[p.id] = 0.0
                else:
                    lambda_r = 0.0
                    for res in p.resources:
                        delay_idx = p.delay_steps_Tjr[res.id]
                        lambda_r += history_mu[res.id][-(delay_idx+1)]
                    
                    lambda_r = max(lambda_r, 1e-9)
                    
                    w_s = self.params['w_s'][p.source_id]
                    y_s = current_y[p.source_id]
                    
                    term1 = lambda_r ** -(self.p + 1)
                    term2 = w_s ** (self.p + 1)
                    term3 = y_s ** (1.0 - (self.alpha * (self.p + 1)))
                    
                    current_x[p.id] = term1 * term2 * term3
            
            for pid, val in current_x.items():
                history_x[pid].append(val)

            # Update Source State
            dy = {}
            for s in self.network['source_list']:
                sum_xq = 0.0
                s_routes = [p for p in self.network['path_list'] if p.source_id == s.id]
                for p in s_routes:
                    x_delayed = history_x[p.id][-(p.delay_steps_Tr+1)]
                    sum_xq += x_delayed ** self.q
                
                y_val = current_y[s.id]
                k_s = self.params['kappa_s'][s.id]
                dy[s.id] = k_s * (y_val ** (1.0/(self.p + 1))) * (sum_xq - (y_val ** self.q))

            # Update Prices
            dmu = {}
            for res in self.network['resource_list']:
                z_j = 0.0
                res_routes = [p for p in self.network['path_list'] if res in p.resources]
                for p in res_routes:
                    x_delayed = history_x[p.id][-(p.delay_steps_Trj[res.id]+1)]
                    z_j += x_delayed
                
                mu_val = current_mu[res.id]
                k_j = self.params['kappa_j'][res.id]
                dmu[res.id] = k_j * mu_val * (z_j - res.capacity)

            # Integrate
            for sid in current_y:
                current_y[sid] = max(current_y[sid] + dy[sid] * self.dt, 1e-6)
            for rid in current_mu:
                current_mu[rid] = max(current_mu[rid] + dmu[rid] * self.dt, 1e-9)
                history_mu[rid].append(current_mu[rid])
            
            # Save data
            if t >= -10 and step % 100 == 0:
                results['time'].append(t)
                for pid in current_x:
                    results['x'][pid].append(current_x[pid])
        
        return results

def run_dual_simulation():
    print("PHASE 2: VALIDATION (Dual Algorithm - Fast Mode)")
    
    # 1. Network Setup
    j1 = Resource(id='j1', capacity=15)
    j2 = Resource(id='j2', capacity=15)
    j3 = Resource(id='j3', capacity=15)
    j4 = Resource(id='j4', capacity=15)
    s1, s2, s3, s4 = Source(id='s1'), Source(id='s2'), Source(id='s3'), Source(id='s4')
    
    paths = [
        Path(id='r1', source_id='s1', resources=[j1], rtt=0.015),
        Path(id='r2', source_id='s1', resources=[j2], rtt=0.15), 
        Path(id='r3', source_id='s2', resources=[j2], rtt=0.15), 
        Path(id='r4', source_id='s2', resources=[j3], rtt=1.5),  
        Path(id='r5', source_id='s3', resources=[j3], rtt=1.5),  
        Path(id='r6', source_id='s3', resources=[j4], rtt=0.15), 
        Path(id='r7', source_id='s4', resources=[j4], rtt=0.15), 
        Path(id='r8', source_id='s4', resources=[j1], rtt=0.015),
    ]
    
    network = {
        'sources': {s.id: s for s in [s1, s2, s3, s4]},
        'resources': {r.id: r for r in [j1, j2, j3, j4]},
        'path_list': paths,
        'source_list': [s1, s2, s3, s4],
        'resource_list': [j1, j2, j3, j4]
    }

    # 2. Parameters 
    params = {
        'p': 7,
        'q': 7/8,
        'alpha': 1, 
        'w_s': {'s1': 0.1, 's2': 0.1, 's3': 0.1, 's4': 0.1},
        'kappa_s': {'s1': 0.25, 's2': 0.01, 's3': 0.01, 's4': 0.25}, 
        'kappa_j': {'j1': 0.1, 'j2': 0.01, 'j3': 0.001, 'j4': 0.01}
    }

    # 3. Run Simulation
    simulator = DualAlgorithmDDE(network, params, dt=0.0001)
    
    # Short warmup
    results = simulator.run(warmup_duration=10.0, experiment_duration=400.0)
    
    plot_dual_results(results)

def plot_dual_results(results):
    # Extract data
    t = np.array(results['time'])
    x = {k: np.array(v) for k, v in results['x'].items()}
    
    # --- Add constant prefix for visual consistency (-100 to 0) ---
    t_prefix = np.linspace(-100, t[0], 100, endpoint=False)
    t_combined = np.concatenate([t_prefix, t])
    
    # Extend x values backwards
    x_combined = {}
    for pid, val_array in x.items():
        initial_val = val_array[0]
        prefix_vals = np.full_like(t_prefix, initial_val)
        x_combined[pid] = np.concatenate([prefix_vals, val_array])
        
    # --- FIGURE 4: Individual Paths ---
    plt.figure(figsize=(7, 6))
    
    # Map paths to source colors and styles
    path_map = [
        ('r1', 's1', '-'),   
        ('r2', 's1', '--'),  
        ('r3', 's2', ':'),   
        ('r4', 's2', '-'),   
        ('r5', 's3', '--'),  
        ('r6', 's3', '-'),   
        ('r7', 's4', '--'),  
        ('r8', 's4', '--'),  
    ]
    
    for pid, sid, sty in path_map:
        if pid in x_combined:
            num = int(pid[1:])  # r1 → 1, r2 → 2, etc.
            label = f"$x_{{\\pi_{{{num}}}}}$"
            plt.plot(t_combined, x_combined[pid], color=colors[sid], linestyle=sty, label=label, linewidth=1.5)

    plt.axvline(0, color='red', linestyle=':', linewidth=1.0)
    plt.xlim(-100, 400)
    plt.ylim(0, 15)
    plt.xlabel('Time, $t$')
    plt.ylabel('Sending rate, $x_\\pi(t)$')
    plt.legend(ncol=4, frameon=True, fontsize='small', loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure3_dual_path_rates.png', dpi=300)
    print("Saved figure3_dual_path_rates.png")
    
    # --- FIGURE 5: Aggregate Rates ---
    s1_flow = x_combined['r1'] + x_combined['r2']
    s2_flow = x_combined['r3'] + x_combined['r4']
    s3_flow = x_combined['r5'] + x_combined['r6']
    s4_flow = x_combined['r7'] + x_combined['r8']
    
    plt.figure(figsize=(7, 6))
    plt.plot(t_combined, s1_flow, color='black', linestyle='-',  linewidth=1.8, label=r'$s_1\ (x_{\pi_1}+x_{\pi_2})$')
    plt.plot(t_combined, s2_flow, color='red',   linestyle='--', linewidth=1.8, label=r'$s_2\ (x_{\pi_3}+x_{\pi_4})$')
    plt.plot(t_combined, s3_flow, color='blue',  linestyle=':',  linewidth=2.0, label=r'$s_3\ (x_{\pi_5}+x_{\pi_6})$')
    plt.plot(t_combined, s4_flow, color='green', linestyle='-.', linewidth=1.8, label=r'$s_4\ (x_{\pi_7}+x_{\pi_8})$')

    plt.axvline(0, color='red', linestyle=':', linewidth=1.0)
    plt.xlim(-100, 400)
    plt.ylim(0, 15)
    plt.xlabel('Time, $t$')
    plt.ylabel('Aggregate sending rate')
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure4_dual_agg_rates.png', dpi=300)
    print("Saved figure4_dual_agg_rates.png")
    plt.show()

colors = {
    's1': 'black', 
    's2': 'red',   
    's3': 'blue',  
    's4': 'green'  
}

if __name__ == '__main__':
    run_dual_simulation()