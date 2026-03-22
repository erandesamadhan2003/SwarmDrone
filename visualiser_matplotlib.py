"""
visualiser_matplotlib.py
Fallback 3D visualiser using Matplotlib + FuncAnimation.
Use this if PyBullet GUI is not available (headless server, SSH, etc.)

Usage:  python visualiser_matplotlib.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')        # change to 'Qt5Agg' if TkAgg missing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches

from swarm_controller import SwarmController, N_DRONES
from orchard_env import DurianOrchard

STEPS_PER_FRAME = 3
TRAIL_LEN       = 60

# ── Colour map ─────────────────────────────────────────────
DRONE_COLORS = plt.cm.tab10(np.linspace(0, 0.6, N_DRONES))
NAV_COLOR    = '#FFD700'
TARGET_COLOR = '#FF4444'
OBS_COLOR    = '#4CAF50'


class MatplotlibSim:
    def __init__(self):
        self.controller = SwarmController('EN-MASCA')
        self.env = self.controller.env

        self.fig = plt.figure(figsize=(18, 10), facecolor='#0D1117')
        self.fig.suptitle('EN-MASCA Drone Swarm — Durian Orchard Patrol',
                          color='white', fontsize=14, fontweight='bold')

        # ── 3D Axes ─────────────────────────────────────────
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax3d.set_facecolor('#0D1117')
        self.ax3d.set_xlim(0, 50); self.ax3d.set_ylim(0, 50); self.ax3d.set_zlim(0, 20)
        self.ax3d.set_xlabel('X (m)', color='white'); self.ax3d.set_ylabel('Y (m)', color='white')
        self.ax3d.set_zlabel('Z (m)', color='white')
        self.ax3d.tick_params(colors='#555')
        for pane in [self.ax3d.xaxis.pane, self.ax3d.yaxis.pane, self.ax3d.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#222')

        # ── 2D telemetry axes ────────────────────────────────
        self.ax_spd = self.fig.add_subplot(322, facecolor='#161B22')
        self.ax_hdiff= self.fig.add_subplot(324, facecolor='#161B22')
        self.ax_nav  = self.fig.add_subplot(326, facecolor='#161B22')

        for a in [self.ax_spd, self.ax_hdiff, self.ax_nav]:
            a.tick_params(colors='#8B949E')
            for sp in a.spines.values(): sp.set_color('#30363D')
            a.grid(True, color='#21262D', alpha=0.6)

        self.ax_spd.set_title('Mean Speed (m/s)', color='white', fontsize=9)
        self.ax_hdiff.set_title('Max Height Diff (m)', color='white', fontsize=9)
        self.ax_nav.set_title('Cluster→Nav Distance (m)', color='white', fontsize=9)

        # ── History buffers ──────────────────────────────────
        self.time_hist  = []
        self.spd_hist   = []
        self.hdiff_hist = []
        self.nav_hist   = []

        # ── Draw static environment ──────────────────────────
        self._draw_static_env()

        # ── Drone & navigator scatter placeholders ───────────
        self.drone_scatters = []
        self.drone_trails   = []
        for i in range(N_DRONES):
            sc = self.ax3d.scatter([], [], [], c=[DRONE_COLORS[i][:3]],
                                   s=120, marker='o', depthshade=True)
            self.drone_scatters.append(sc)
            tl, = self.ax3d.plot([], [], [], '-',
                                 color=DRONE_COLORS[i][:3], linewidth=0.8, alpha=0.5)
            self.drone_trails.append(tl)

        self.nav_scatter = self.ax3d.scatter([], [], [], c=[NAV_COLOR],
                                              s=200, marker='*', depthshade=False, zorder=10)
        self.nav_trail,  = self.ax3d.plot([], [], [], '--', color=NAV_COLOR,
                                           linewidth=1.5, alpha=0.7)

        # Time and mission text
        self.info_text = self.ax3d.text2D(0.02, 0.97, '', transform=self.ax3d.transAxes,
                                           color='white', fontsize=8, va='top')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Draw trees / rocks ─────────────────────────────────
    def _draw_static_env(self):
        for obs in self.env.static_obstacles:
            c = obs['center']
            r = obs['radius']
            col = obs['color']
            u = np.linspace(0, 2*np.pi, 12)
            v = np.linspace(0, np.pi, 8)
            x = c[0] + r * np.outer(np.cos(u), np.sin(v))
            y = c[1] + r * np.outer(np.sin(u), np.sin(v))
            z = c[2] + r * np.outer(np.ones(12), np.cos(v))
            self.ax3d.plot_surface(x, y, z, alpha=0.25, color=col, linewidth=0)

        # Ground plane
        gx = np.array([[0, 50], [0, 50]])
        gy = np.array([[0, 0], [50, 50]])
        gz = np.zeros_like(gx)
        self.ax3d.plot_surface(gx, gy, gz, alpha=0.15, color='#4CAF50')

        # Target markers
        for t in self.env.targets:
            self.ax3d.scatter(*t, c=TARGET_COLOR, s=300, marker='^',
                              zorder=15, edgecolors='white', linewidths=1.5)

    # ── Animation update ───────────────────────────────────
    def update(self, frame):
        if not self.controller.done:
            for _ in range(STEPS_PER_FRAME):
                self.controller.step()

        telem = self.controller.telemetry()

        # ── Update drone positions ─────────────────────────
        for i, drone in enumerate(self.controller.drones):
            p = drone.pos
            self.drone_scatters[i]._offsets3d = ([p[0]], [p[1]], [p[2]])
            trail = np.array(drone.trail[-TRAIL_LEN:])
            if len(trail) > 1:
                self.drone_trails[i].set_data(trail[:, 0], trail[:, 1])
                self.drone_trails[i].set_3d_properties(trail[:, 2])

        # ── Navigator ─────────────────────────────────────
        np_ = self.controller.navigator.pos
        self.nav_scatter._offsets3d = ([np_[0]], [np_[1]], [np_[2]])
        nt = np.array(self.controller.navigator.trail[-TRAIL_LEN:])
        if len(nt) > 1:
            self.nav_trail.set_data(nt[:, 0], nt[:, 1])
            self.nav_trail.set_3d_properties(nt[:, 2])

        # ── Dynamic obstacles ──────────────────────────────
        # (static plots; dynamic updates would need scatter updates)

        # ── Telemetry plots ────────────────────────────────
        self.time_hist.append(telem['time'])
        self.spd_hist.append(telem['mean_speed'])
        self.hdiff_hist.append(telem['max_h_diff'])
        self.nav_hist.append(telem['nav_dist'])

        TH = self.time_hist[-500:]
        for ax, data, col in [
            (self.ax_spd,  self.spd_hist[-500:],   '#2196F3'),
            (self.ax_hdiff,self.hdiff_hist[-500:],  '#F44336'),
            (self.ax_nav,  self.nav_hist[-500:],    '#FFD700'),
        ]:
            ax.clear()
            ax.set_facecolor('#161B22')
            ax.plot(TH, data, color=col, linewidth=1.5)
            ax.tick_params(colors='#8B949E')
            ax.grid(True, color='#21262D', alpha=0.6)
            for sp in ax.spines.values(): sp.set_color('#30363D')

        self.ax_spd.set_title('Mean Speed (m/s)', color='white', fontsize=9)
        self.ax_hdiff.set_title('Max Height Diff (m)', color='white', fontsize=9)
        self.ax_nav.set_title('Cluster→Nav Dist (m)', color='white', fontsize=9)
        self.ax_spd.axhline(10, color='purple', linestyle=':', linewidth=1)
        self.ax_nav.axhline(10, color='purple', linestyle=':', linewidth=1)

        # ── Info text ──────────────────────────────────────
        status = "MISSION COMPLETE!" if telem['done'] else "PATROLLING…"
        self.info_text.set_text(
            f"T={telem['time']:.1f}s | Targets:{telem['targets_done']}/3 | "
            f"Spd:{telem['mean_speed']:.1f}m/s | H-diff:{telem['max_h_diff']:.2f}m | {status}"
        )

        return []

    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.update,
            interval=50, blit=False, cache_frame_data=False)
        plt.show()


if __name__ == '__main__':
    print("\n  EN-MASCA Matplotlib 3D Visualiser")
    print("  Close window to exit\n")
    sim = MatplotlibSim()
    sim.run()
