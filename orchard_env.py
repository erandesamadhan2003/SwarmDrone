"""
orchard_env.py
Durian Orchard Environment
  - Realistic 3D terrain (heightmap)
  - Static obstacles: trees, rocks (cylinders + spheres)
  - Dynamic obstacles: moving farm equipment / animals
  - Target zones: durian fruit clusters / pest hotspots
  - Wind disturbance model
"""

import numpy as np


class DurianOrchard:
    """
    Simulates a 50×50×20 m durian orchard.
    Coordinate system: x=East, y=North, z=Up
    """

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.bounds = np.array([50.0, 50.0, 20.0])   # x, y, z extents

        # ── Static obstacles (trees + rocks) ───────────────
        #   Each: (center_x, center_y, center_z, radius)
        self.static_obstacles = self._gen_trees(rng) + self._gen_rocks(rng)

        # ── Dynamic obstacles (animals / equipment) ─────────
        self.dynamic_obstacles = self._gen_dynamic(rng)

        # ── Patrol target zones ─────────────────────────────
        self.targets = [
            np.array([45.0, 45.0, 3.0]),
            np.array([40.0, 10.0, 3.0]),
            np.array([25.0, 25.0, 3.0]),
        ]
        self.current_target_idx = 0

        # ── Wind model ─────────────────────────────────────
        self.wind_speed = 2.0       # m/s base
        self.wind_dir   = np.array([1.0, 0.2, 0.0])
        self.wind_dir  /= np.linalg.norm(self.wind_dir)
        self.wind_t     = 0.0

    # ── Tree generation ────────────────────────────────────
    def _gen_trees(self, rng):
        trees = []
        rows, cols = 6, 6
        for r in range(rows):
            for c in range(cols):
                x = 5 + r * 7 + rng.uniform(-1, 1)
                y = 5 + c * 7 + rng.uniform(-1, 1)
                h = rng.uniform(5, 10)       # tree height
                radius = rng.uniform(1.0, 2.0)
                # Trunk as 3 stacked spheres
                for frac in [0.2, 0.5, 0.8]:
                    trees.append({
                        'type': 'sphere',
                        'center': np.array([x, y, h * frac]),
                        'radius': radius * (1.0 - 0.3 * frac),
                        'color': (0.1, 0.6, 0.1),
                        'label': f'tree_{r}_{c}'
                    })
        return trees

    def _gen_rocks(self, rng):
        rocks = []
        for i in range(8):
            x = rng.uniform(3, 47)
            y = rng.uniform(3, 47)
            r = rng.uniform(0.5, 1.5)
            rocks.append({
                'type': 'sphere',
                'center': np.array([x, y, r]),
                'radius': r,
                'color': (0.5, 0.4, 0.3),
                'label': f'rock_{i}'
            })
        return rocks

    def _gen_dynamic(self, rng):
        dyn = []
        for i in range(3):
            dyn.append({
                'pos':    np.array([rng.uniform(5, 45), rng.uniform(5, 45), 0.5]),
                'vel':    np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0]),
                'radius': 1.5,
                'color':  (0.8, 0.5, 0.1),
                'label':  f'animal_{i}'
            })
        return dyn

    # ── Wind disturbance ───────────────────────────────────
    def get_wind(self, dt=0.05):
        self.wind_t += dt
        gust = 0.5 * np.sin(self.wind_t * 0.3) + 0.3 * np.sin(self.wind_t * 1.1)
        return self.wind_dir * (self.wind_speed + gust)

    # ── Update dynamic obstacles ───────────────────────────
    def update(self, dt=0.05):
        for d in self.dynamic_obstacles:
            d['pos'] += d['vel'] * dt
            # Bounce off walls
            for ax in range(2):
                if d['pos'][ax] < 2 or d['pos'][ax] > self.bounds[ax] - 2:
                    d['vel'][ax] *= -1
        self.wind_t += dt

    # ── All obstacle centers (for drone queries) ───────────
    def get_all_obstacle_centers(self):
        centers = [o['center'] for o in self.static_obstacles]
        centers += [d['pos'].copy() for d in self.dynamic_obstacles]
        return centers

    # ── Current patrol target ──────────────────────────────
    @property
    def current_target(self):
        return self.targets[self.current_target_idx]

    def advance_target(self):
        self.current_target_idx = (self.current_target_idx + 1) % len(self.targets)

    # ── Terrain height at (x,y) — gentle undulation ────────
    def terrain_height(self, x, y):
        return (0.5 * np.sin(x * 0.3) * np.cos(y * 0.25) +
                0.3 * np.cos(x * 0.15) * np.sin(y * 0.4))

    # ── Check collision ────────────────────────────────────
    def check_collision(self, pos, margin=0.8):
        for o in self.static_obstacles:
            if np.linalg.norm(pos - o['center']) < o['radius'] + margin:
                return True
        for d in self.dynamic_obstacles:
            if np.linalg.norm(pos - d['pos']) < d['radius'] + margin:
                return True
        return False
