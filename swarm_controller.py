"""
swarm_controller.py
EN-MASCA Swarm Controller
  - Spawns N drones + 1 virtual navigator
  - Runs the simulation loop
  - Collects telemetry for the visualiser
  - Compares EN-MASCA vs baseline MASCA (no DQN/PPO)
"""

import numpy as np
from drone_agent   import DroneAgent, DQNAgent, DT
from virtual_navigator import PPONavigator
from orchard_env   import DurianOrchard


N_DRONES    = 6
SPAWN_RADIUS = 3.0   # metres around start


class SwarmController:
    def __init__(self, algorithm='EN-MASCA'):
        self.algorithm = algorithm
        self.env = DurianOrchard(seed=7)

        start = np.array([2.0, 2.0, 3.0])
        target = self.env.current_target

        # Shared DQN (parameter sharing across drones, as in MAPPO/MASCA)
        self.shared_dqn = DQNAgent() if algorithm == 'EN-MASCA' else None

        # Virtual navigator (PPO)
        self.navigator  = PPONavigator(start.copy(), target.copy())

        # Spawn drones in a circle around start
        self.drones = []
        for i in range(N_DRONES):
            angle = 2 * np.pi * i / N_DRONES
            offset = np.array([
                SPAWN_RADIUS * np.cos(angle),
                SPAWN_RADIUS * np.sin(angle),
                0.0
            ])
            pos = start + offset
            pos[2] = 3.0
            d = DroneAgent(
                drone_id=i,
                init_pos=pos,
                dqn=self.shared_dqn if algorithm == 'EN-MASCA'
                    else DQNAgent(eps=0.0)   # baseline: random/no learning
            )
            self.drones.append(d)

        self.t           = 0.0
        self.step_count  = 0
        self.done        = False
        self.task_log    = []   # (time, dist_to_target)
        self.collision_count = 0
        self.targets_reached = 0

    # ── Cluster centre ──────────────────────────────────────
    @property
    def cluster_center(self):
        return np.mean([d.pos for d in self.drones], axis=0)

    # ── One simulation tick ─────────────────────────────────
    def step(self):
        if self.done:
            return

        self.env.update(DT)
        wind = self.env.get_wind(DT)
        obstacles = self.env.get_all_obstacle_centers()
        target    = self.env.current_target
        cc        = self.cluster_center

        # ── Navigator step ──────────────────────────────────
        nav_done = self.navigator.step_nav(obstacles, cc, DT)

        # ── Drone steps ─────────────────────────────────────
        for drone in self.drones:
            neighbors = [d for d in self.drones if d.id != drone.id]
            drone.update(self.navigator, neighbors, obstacles, target)

            # Apply wind disturbance
            drone.vel += wind * 0.02

            # Collision detection
            if self.env.check_collision(drone.pos):
                self.collision_count += 1

        # ── Target reached? ─────────────────────────────────
        dist_to_tgt = np.linalg.norm(cc - target)
        self.task_log.append((self.t, dist_to_tgt))

        if dist_to_tgt < 2.5:
            self.targets_reached += 1
            if self.targets_reached < len(self.env.targets):
                self.env.advance_target()
                self.navigator.target = self.env.current_target.copy()
            else:
                self.done = True

        self.t           += DT
        self.step_count  += 1

    # ── Telemetry snapshot ─────────────────────────────────
    def telemetry(self):
        positions = [d.pos.copy() for d in self.drones]
        velocities= [d.vel.copy() for d in self.drones]
        speeds    = [np.linalg.norm(v) for v in velocities]
        heights   = [p[2] for p in positions]
        max_h_diff= max(heights) - min(heights)

        nbr_dists = []
        for i, d1 in enumerate(self.drones):
            for j, d2 in enumerate(self.drones):
                if j > i:
                    nbr_dists.append(np.linalg.norm(d1.pos - d2.pos))

        return {
            'time'         : self.t,
            'positions'    : positions,
            'velocities'   : velocities,
            'speeds'       : speeds,
            'heights'      : heights,
            'max_h_diff'   : max_h_diff,
            'mean_speed'   : float(np.mean(speeds)),
            'cluster_center': self.cluster_center.copy(),
            'nav_pos'      : self.navigator.pos.copy(),
            'target'       : self.env.current_target.copy(),
            'targets_done' : self.targets_reached,
            'collisions'   : self.collision_count,
            'nav_dist'     : np.linalg.norm(self.cluster_center - self.navigator.pos),
            'mean_nbr_dist': float(np.mean(nbr_dists)) if nbr_dists else 0,
            'done'         : self.done,
        }
