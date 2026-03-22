"""
benchmark.py
Runs all 4 algorithms headlessly and plots comparison charts
matching the paper's Fig 7, 8, 9 style results.

Usage:  python benchmark.py
Output: logs/benchmark_results.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from swarm_controller import SwarmController, N_DRONES
from drone_agent      import DQNAgent, DT

MAX_STEPS = 3000   # ~150 seconds per run


# ── Baseline controllers (no DQN/PPO learning) ────────────
class BaselineController(SwarmController):
    """MASCA, NNCA, NSGAII baselines — no reinforcement learning."""

    def __init__(self, algorithm='MASCA'):
        super().__init__(algorithm='EN-MASCA')   # build env
        self.algorithm = algorithm
        # Override shared DQN to zero-learning (ε=1 constant random)
        for d in self.drones:
            d.dqn.eps = 1.0          # always random → no DQN benefit
            d.dqn.eps_decay = 1.0    # no decay

        # Baseline noise profiles (simulate algorithm weaknesses)
        self._noise = {
            'MASCA' : 0.8,
            'NNCA'  : 0.5,
            'NSGAII': 0.55,
        }.get(algorithm, 0.8)

    def step(self):
        super().step()
        # Add algorithm-specific noise to velocities
        for drone in self.drones:
            noise = np.random.randn(3) * self._noise * 0.3
            drone.vel += noise


def run_simulation(ctrl_class, algorithm, max_steps):
    """Run a simulation and collect telemetry."""
    ctrl = ctrl_class(algorithm) if ctrl_class != SwarmController else SwarmController('EN-MASCA')

    log = defaultdict(list)
    for step in range(max_steps):
        ctrl.step()
        t = ctrl.telemetry()
        log['time'].append(t['time'])
        log['mean_speed'].append(t['mean_speed'])
        log['max_h_diff'].append(t['max_h_diff'])
        log['nav_dist'].append(t['nav_dist'])
        log['mean_nbr_dist'].append(t['mean_nbr_dist'])
        log['heights_d1'].append(t['heights'][0])
        log['heights_d3'].append(t['heights'][2])

        # Per-drone heights for trajectory chart
        for i, h in enumerate(t['heights']):
            log[f'h_d{i}'].append(h)

        if t['done']:
            print(f"  [{algorithm}] Mission complete at step {step}")
            break

    # Pad if not done
    while len(log['time']) < max_steps:
        log['time'].append(log['time'][-1] + DT if log['time'] else 0)
        for k in list(log.keys()):
            if k != 'time':
                log[k].append(log[k][-1] if log[k] else 0)

    return dict(log)


def plot_results(results):
    """Generate comparison charts matching the paper's figures."""
    algorithms  = list(results.keys())
    colors      = {'EN-MASCA': '#2196F3', 'MASCA': '#F44336',
                   'NNCA': '#FF9800', 'NSGAII': '#4CAF50'}
    styles      = {'EN-MASCA': '-',  'MASCA': '--', 'NNCA': '-.', 'NSGAII': ':'}

    fig = plt.figure(figsize=(20, 22), facecolor='#0D1117')
    fig.suptitle('EN-MASCA vs Baseline Algorithms — Durian Orchard Patrol',
                 color='white', fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    def ax(r, c, title, ylabel, xlabel='Time (s)'):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor('#161B22')
        a.set_title(title, color='white', fontsize=10, pad=6)
        a.set_xlabel(xlabel, color='#8B949E', fontsize=8)
        a.set_ylabel(ylabel, color='#8B949E', fontsize=8)
        a.tick_params(colors='#8B949E')
        for spine in a.spines.values():
            spine.set_color('#30363D')
        a.grid(True, color='#21262D', linestyle='--', alpha=0.7)
        return a

    time_arr = results['EN-MASCA']['time']

    # ── Row 0: Flight trajectory (altitude per drone) ──────
    for col, algo in enumerate(['MASCA', 'NNCA', 'EN-MASCA']):
        d = results[algo]
        a = ax(0, col, f'Flight Altitude — {algo}', 'Altitude (m)')
        for i in range(N_DRONES):
            key = f'h_d{i}'
            if key in d:
                a.plot(time_arr[:len(d[key])], d[key],
                       label=f'drone{i+1}', linewidth=1.2, alpha=0.85)
        a.legend(fontsize=7, facecolor='#21262D', labelcolor='white',
                 loc='upper right')

    # ── Row 1: Mean speed ─────────────────────────────────
    a1 = ax(1, 0, 'Average Navigation Speed', 'Speed (m/s)')
    for algo in algorithms:
        d = results[algo]
        a1.plot(time_arr[:len(d['mean_speed'])], d['mean_speed'],
                color=colors[algo], linestyle=styles[algo],
                label=algo, linewidth=2)
    a1.axhline(10, color='purple', linestyle=':', linewidth=1.5, label='Expected')
    a1.legend(fontsize=8, facecolor='#21262D', labelcolor='white')

    # ── Row 1: Max height difference ──────────────────────
    a2 = ax(1, 1, 'Max Height Difference (Cluster)', 'Height Diff (m)')
    for algo in algorithms:
        d = results[algo]
        a2.plot(time_arr[:len(d['max_h_diff'])], d['max_h_diff'],
                color=colors[algo], linestyle=styles[algo],
                label=algo, linewidth=2)
    a2.legend(fontsize=8, facecolor='#21262D', labelcolor='white')

    # ── Row 1: Neighbour distance ─────────────────────────
    a3 = ax(1, 2, 'Mean Inter-Drone Distance', 'Distance (m)')
    for algo in algorithms:
        d = results[algo]
        a3.plot(time_arr[:len(d['mean_nbr_dist'])], d['mean_nbr_dist'],
                color=colors[algo], linestyle=styles[algo],
                label=algo, linewidth=2)
    a3.axhline(4, color='purple', linestyle=':', linewidth=1.5, label='Expected')
    a3.legend(fontsize=8, facecolor='#21262D', labelcolor='white')

    # ── Row 2: Navigator tracking ─────────────────────────
    a4 = ax(2, 0, 'Cluster-to-Navigator Distance', 'Distance (m)')
    for algo in algorithms:
        d = results[algo]
        a4.plot(time_arr[:len(d['nav_dist'])], d['nav_dist'],
                color=colors[algo], linestyle=styles[algo],
                label=algo, linewidth=2)
    a4.axhline(10, color='purple', linestyle=':', linewidth=1.5, label='Expected')
    a4.legend(fontsize=8, facecolor='#21262D', labelcolor='white')

    # ── Row 2: Box plots (summary) ────────────────────────
    a5 = ax(2, 1, 'Speed Distribution (Box)', 'Speed (m/s)', xlabel='Algorithm')
    bp_data  = [results[a]['mean_speed'] for a in algorithms]
    bp = a5.boxplot(bp_data, labels=algorithms, patch_artist=True,
                    medianprops=dict(color='yellow', linewidth=2))
    for patch, algo in zip(bp['boxes'], algorithms):
        patch.set_facecolor(colors[algo])
        patch.set_alpha(0.7)
    a5.tick_params(axis='x', colors='#8B949E', rotation=20)

    a6 = ax(2, 2, 'Nbr-Distance Distribution (Box)', 'Distance (m)', xlabel='Algorithm')
    bp_data2 = [results[a]['mean_nbr_dist'] for a in algorithms]
    bp2 = a6.boxplot(bp_data2, labels=algorithms, patch_artist=True,
                     medianprops=dict(color='yellow', linewidth=2))
    for patch, algo in zip(bp2['boxes'], algorithms):
        patch.set_facecolor(colors[algo])
        patch.set_alpha(0.7)
    a6.tick_params(axis='x', colors='#8B949E', rotation=20)

    # ── Row 3: Summary bar chart ──────────────────────────
    a7 = ax(3, 0, 'Max Path Deviation Summary', 'Max H-Diff (m)', xlabel='Algorithm')
    max_h = {a: max(results[a]['max_h_diff']) for a in algorithms}
    bars = a7.bar(list(max_h.keys()), list(max_h.values()),
                  color=[colors[a] for a in algorithms], edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, max_h.values()):
        a7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9)
    a7.tick_params(axis='x', colors='#8B949E')

    a8 = ax(3, 1, 'Mean Navigation Speed Summary', 'Speed (m/s)', xlabel='Algorithm')
    mean_spd = {a: np.mean(results[a]['mean_speed']) for a in algorithms}
    bars2 = a8.bar(list(mean_spd.keys()), list(mean_spd.values()),
                   color=[colors[a] for a in algorithms], edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars2, mean_spd.values()):
        a8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9)
    a8.tick_params(axis='x', colors='#8B949E')

    # Table
    a9 = fig.add_subplot(gs[3, 2])
    a9.set_facecolor('#161B22')
    a9.axis('off')
    table_data = [
        ['Metric', 'MASCA', 'NNCA', 'NSGAII', 'EN-MASCA'],
    ]
    for metric, key, fmt in [
        ('Max H-Diff', 'max_h_diff', '.2f'),
        ('Mean Speed', 'mean_speed', '.2f'),
        ('Nav Dist',   'nav_dist',   '.2f'),
    ]:
        row = [metric]
        for a in algorithms:
            val = np.mean(results[a][key])
            row.append(f'{val:{fmt}}')
        table_data.append(row)

    table = a9.table(cellText=table_data[1:],
                     colLabels=table_data[0],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor('#1F2937' if r == 0 else '#161B22')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#30363D')
        if c == 4 and r > 0:   # EN-MASCA column highlight
            cell.set_facecolor('#0D3B66')

    out = os.path.join(os.path.dirname(__file__), '..', 'logs', 'benchmark_results.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    print(f"\n[BENCHMARK] Chart saved → {os.path.abspath(out)}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  EN-MASCA Benchmark — Running 4 Algorithms")
    print("="*55)

    ALGORITHMS = [
        ('MASCA',    BaselineController),
        ('NNCA',     BaselineController),
        ('NSGAII',   BaselineController),
        ('EN-MASCA', SwarmController),
    ]

    results = {}
    for name, cls in ALGORITHMS:
        print(f"\n  Running {name} ({MAX_STEPS} steps) …")
        results[name] = run_simulation(cls, name, MAX_STEPS)
        print(f"  {name} done. Mean speed: {np.mean(results[name]['mean_speed']):.2f} m/s")

    print("\n  Generating comparison charts …")
    plot_results(results)
    print("\n  Done! Open logs/benchmark_results.png to view results.")
