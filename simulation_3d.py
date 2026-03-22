"""
simulation_3d.py  ─ EN-MASCA Drone Swarm Simulation (v2 - FIXED)
=================================================================
PyBullet 3D real-time simulation with:
  • Fast-moving drones (proper physics, high speed)
  • Proper spawn spread so drones are visible from start
  • Realistic city/agricultural mixed environment
  • Buildings, trees, towers, roads
  • Drone trails rendered as debug lines
  • Real-time HUD overlay
  • Follow-cam that tracks swarm center

Controls:
  SPACE  — Pause / Resume
  R      — Reset
  C      — Cycle camera view
  T      — Toggle HUD
  ESC    — Quit
"""

import pybullet as pb
import pybullet_data
import numpy as np
import time, sys, os, math

sys.path.insert(0, os.path.dirname(__file__))

# ─── Simulation parameters ────────────────────────────────────
N_DRONES        = 6
WORLD_SIZE      = 80       # 80x80 m environment
DRONE_HEIGHT    = 8.0      # cruise altitude in metres
SPAWN_SPREAD    = 6.0      # drones spawn this far apart
TARGET_RADIUS   = 3.0      # mission complete when swarm within this distance
SIM_STEPS_PER_FRAME = 8    # logic steps per rendered frame (speed multiplier)
PHYSICS_DT      = 0.02     # physics timestep
MAX_SPEED       = 12.0     # m/s — fast movement
TRAIL_HISTORY   = 80       # number of trail segments to draw

# ─── Colours ──────────────────────────────────────────────────
C_DRONE = [
    [0.0, 0.6, 1.0, 1.0],
    [0.0, 1.0, 0.5, 1.0],
    [1.0, 0.4, 0.0, 1.0],
    [0.8, 0.0, 0.8, 1.0],
    [1.0, 0.9, 0.0, 1.0],
    [1.0, 0.2, 0.3, 1.0],
]
C_NAV      = [1.0, 0.95, 0.3, 1.0]
C_TARGET   = [1.0, 0.15, 0.15, 0.9]
C_ROAD     = [0.25, 0.25, 0.25, 1.0]
C_GROUND   = [0.35, 0.50, 0.25, 1.0]
C_TRUNK    = [0.4, 0.25, 0.1, 1.0]


# ════════════════════════════════════════════════════════════════
#  Standalone physics (no external dependency)
# ════════════════════════════════════════════════════════════════

class Drone:
    ACTIONS = []
    for _dx in [-1, 0, 1]:
        for _dy in [-1, 0, 1]:
            for _dz in [-1, 0, 1]:
                if not (_dx == _dy == _dz == 0):
                    _v = np.array([_dx, _dy, _dz], dtype=float)
                    ACTIONS.append(_v / np.linalg.norm(_v))
    ACTIONS = np.array(ACTIONS)

    def __init__(self, drone_id, init_pos):
        self.id    = drone_id
        self.pos   = np.array(init_pos, dtype=float)
        self.vel   = np.random.randn(3) * 0.5
        self.vel[2] = 0.0
        self.trail = [self.pos.copy()]
        self.Q     = np.zeros(26)
        self.eps   = 0.7

    def act(self, target_dir):
        if np.random.rand() < self.eps:
            dots = self.ACTIONS @ target_dir
            return self.ACTIONS[np.argmax(dots + np.random.randn(26) * 0.2)]
        return self.ACTIONS[np.argmax(self.Q)]

    def update(self, nav_pos, nav_vel, neighbors, obstacles, target, dt):
        self.eps = max(0.1, self.eps * 0.9998)

        # Pz: navigator tracking
        to_nav  = nav_pos - self.pos
        d_nav   = np.linalg.norm(to_nav)
        pz = 2.8 * to_nav / (d_nav + 1e-6) + 0.45 * (nav_vel - self.vel)

        # Px: cohesion + separation
        px = np.zeros(3)
        for nb in neighbors:
            diff = nb.pos - self.pos
            dist = np.linalg.norm(diff)
            if 0.1 < dist < 22.0:
                desired = 5.0
                force_mag = (dist - desired) / (dist + 1e-6)
                px += force_mag * diff / (dist + 1e-6)
                px += 0.25 * (nb.vel - self.vel) / (dist + 1e-6)

        # Py: obstacle avoidance
        py = np.zeros(3)
        for obs_pos, obs_r in obstacles:
            diff = self.pos - obs_pos
            dist = np.linalg.norm(diff)
            margin = obs_r + 2.5
            if 0 < dist < margin + 4.0:
                strength = max(0.0, margin + 4.0 - dist) / (margin + 4.0)
                py += diff / (dist + 1e-6) * strength * 9.0

        # DQN bias
        to_tgt    = target - self.pos
        d_tgt     = np.linalg.norm(to_tgt)
        tgt_dir   = to_tgt / (d_tgt + 1e-6)
        dqn_force = self.act(tgt_dir) * 3.0

        # Altitude hold
        alt_err   = DRONE_HEIGHT - self.pos[2]
        alt_force = np.array([0.0, 0.0, alt_err * 3.5])

        accel = pz + px * 0.6 + py + dqn_force * 0.8 + alt_force
        self.vel += accel * dt
        self.vel[:2] *= 0.96
        self.vel[2]  *= 0.93

        spd = np.linalg.norm(self.vel)
        if spd > MAX_SPEED:
            self.vel = self.vel / spd * MAX_SPEED
        elif spd < 0.3 and spd > 1e-9:
            self.vel = self.vel / spd * 0.3

        self.pos += self.vel * dt
        self.pos[2] = np.clip(self.pos[2], 2.0, 28.0)
        self.pos[0] = np.clip(self.pos[0], 1.0, WORLD_SIZE - 1.0)
        self.pos[1] = np.clip(self.pos[1], 1.0, WORLD_SIZE - 1.0)

        self.trail.append(self.pos.copy())
        if len(self.trail) > TRAIL_HISTORY + 5:
            self.trail.pop(0)

        return d_tgt < TARGET_RADIUS


class Navigator:
    def __init__(self, start, target):
        self.pos    = np.array(start, dtype=float)
        self.vel    = np.zeros(3)
        self.target = np.array(target, dtype=float)
        self.trail  = [self.pos.copy()]

    def step(self, obstacles, dt):
        to_tgt = self.target - self.pos
        dist   = np.linalg.norm(to_tgt)
        if dist < 2.0:
            self.vel *= 0.7
            self.trail.append(self.pos.copy())
            return True

        tgt_force = to_tgt / (dist + 1e-6) * 7.0
        avoid = np.zeros(3)
        for obs_pos, obs_r in obstacles:
            d = np.linalg.norm(self.pos - obs_pos)
            if 0 < d < obs_r + 6.0:
                avoid += (self.pos - obs_pos) / (d + 1e-6) * (obs_r + 6.0 - d) * 1.8

        self.vel += (tgt_force + avoid) * dt
        self.vel[:2] *= 0.93
        self.vel[2]  += (DRONE_HEIGHT - self.pos[2]) * 2.5 * dt
        self.vel[2]  *= 0.88

        spd = np.linalg.norm(self.vel)
        nav_max = MAX_SPEED * 0.85
        if spd > nav_max:
            self.vel = self.vel / spd * nav_max

        self.pos += self.vel * dt
        self.pos[2] = np.clip(self.pos[2], 2.0, 28.0)
        self.pos[0] = np.clip(self.pos[0], 1.0, WORLD_SIZE - 1.0)
        self.pos[1] = np.clip(self.pos[1], 1.0, WORLD_SIZE - 1.0)

        self.trail.append(self.pos.copy())
        if len(self.trail) > TRAIL_HISTORY + 5:
            self.trail.pop(0)
        return False


PATROL_TARGETS = [
    [65.0, 65.0, DRONE_HEIGHT],
    [65.0, 15.0, DRONE_HEIGHT],
    [40.0, 40.0, DRONE_HEIGHT],
    [15.0, 65.0, DRONE_HEIGHT],
    [15.0, 15.0, DRONE_HEIGHT],
]


class Swarm:
    def __init__(self):
        self.target_idx   = 0
        self.targets_done = 0
        self.t            = 0.0

        spawn = np.array([8.0, 8.0, DRONE_HEIGHT])
        self.drones = []
        for i in range(N_DRONES):
            angle  = 2 * math.pi * i / N_DRONES
            offset = np.array([SPAWN_SPREAD * math.cos(angle),
                                SPAWN_SPREAD * math.sin(angle),
                                (i % 3) * 0.6])
            self.drones.append(Drone(i, (spawn + offset).tolist()))

        self.navigator = Navigator(spawn.tolist(),
                                   PATROL_TARGETS[self.target_idx])

    @property
    def cluster_center(self):
        return np.mean([d.pos for d in self.drones], axis=0)

    def step(self, obstacles, dt=PHYSICS_DT):
        nav_done = self.navigator.step(obstacles, dt)
        target   = np.array(PATROL_TARGETS[self.target_idx])

        for drone in self.drones:
            nbrs = [d for d in self.drones if d.id != drone.id]
            drone.update(self.navigator.pos, self.navigator.vel,
                         nbrs, obstacles, target, dt)

        cc   = self.cluster_center
        dist = np.linalg.norm(cc - target)
        if dist < TARGET_RADIUS * 2.0 or nav_done:
            self.targets_done += 1
            self.target_idx    = (self.target_idx + 1) % len(PATROL_TARGETS)
            self.navigator.target = np.array(PATROL_TARGETS[self.target_idx],
                                             dtype=float)
            print(f"  ✓ Target reached! Total: {self.targets_done} | "
                  f"Next → {PATROL_TARGETS[self.target_idx][:2]}")

        self.t += dt
        speeds  = [np.linalg.norm(d.vel) for d in self.drones]
        heights = [d.pos[2] for d in self.drones]
        nbr_ds  = [np.linalg.norm(self.drones[i].pos - self.drones[j].pos)
                   for i in range(N_DRONES) for j in range(i+1, N_DRONES)]
        return {
            'time'         : self.t,
            'mean_speed'   : float(np.mean(speeds)),
            'max_h_diff'   : float(max(heights) - min(heights)),
            'nav_dist'     : float(np.linalg.norm(cc - self.navigator.pos)),
            'mean_nbr_dist': float(np.mean(nbr_ds)) if nbr_ds else 0,
            'targets_done' : self.targets_done,
            'cluster_center': cc,
        }


# ════════════════════════════════════════════════════════════════
#  Scene Builder
# ════════════════════════════════════════════════════════════════

class SceneBuilder:
    def __init__(self):
        self.obstacles = []
        self.dyn_bodies = []
        self.dyn_data   = []
        self._rng = np.random.default_rng(42)

    def build(self):
        self._ground()
        self._roads()
        self._buildings()
        self._trees()
        self._towers()
        self._dynamic_vehicles()
        print(f"  [SCENE] {len(self.obstacles)} obstacles | "
              f"{len(self.dyn_data)} moving vehicles")

    def _ground(self):
        half = WORLD_SIZE / 2
        vis = pb.createVisualShape(pb.GEOM_BOX,
                                   halfExtents=[half, half, 0.2],
                                   rgbaColor=C_GROUND)
        col = pb.createCollisionShape(pb.GEOM_BOX,
                                      halfExtents=[half, half, 0.2])
        pb.createMultiBody(0, col, vis, [half, half, -0.2])

    def _roads(self):
        half = WORLD_SIZE / 2
        for pos in range(20, WORLD_SIZE, 20):
            for halfExtents, center in [
                ([half, 2.0, 0.1], [half, pos, 0.08]),
                ([2.0, half, 0.1], [pos, half, 0.08]),
            ]:
                vis = pb.createVisualShape(pb.GEOM_BOX,
                                           halfExtents=halfExtents,
                                           rgbaColor=C_ROAD)
                pb.createMultiBody(0, -1, vis, center)

    def _buildings(self):
        spots = [
            (15,15), (15,35), (15,55), (15,70),
            (35,15), (35,65),
            (50,20), (50,50), (50,70),
            (65,15), (65,40), (65,65),
            (30,45), (45,30),
        ]
        for bx, by in spots:
            h  = float(self._rng.uniform(6, 22))
            wx = float(self._rng.uniform(4, 9))
            wy = float(self._rng.uniform(4, 9))

            grey = [float(x) for x in self._rng.uniform([0.4,0.4,0.5],[0.72,0.72,0.78])] + [1.0]
            vis = pb.createVisualShape(pb.GEOM_BOX,
                                       halfExtents=[wx/2, wy/2, h/2],
                                       rgbaColor=grey)
            col = pb.createCollisionShape(pb.GEOM_BOX,
                                          halfExtents=[wx/2, wy/2, h/2])
            pb.createMultiBody(0, col, vis, [bx, by, h/2])
            self.obstacles.append((np.array([bx, by, h/2]), max(wx, wy)/2 + 1.2))

            # Windows
            win_col = [1.0, 0.95, 0.6, 0.9]
            for wi in range(int(h / 3)):
                for side in [-1, 1]:
                    wvis = pb.createVisualShape(pb.GEOM_BOX,
                                               halfExtents=[0.12, wx/2 + 0.02, 0.28],
                                               rgbaColor=win_col)
                    pb.createMultiBody(0, -1, wvis,
                                       [bx + side * wx/2, by, 2.0 + wi * 3.0])

            # Roof antenna
            if self._rng.random() > 0.4:
                ah = float(self._rng.uniform(1.5, 5))
                avis = pb.createVisualShape(pb.GEOM_CYLINDER,
                                            radius=0.35, length=ah,
                                            rgbaColor=[0.3, 0.3, 0.35, 1.0])
                pb.createMultiBody(0, -1, avis, [bx, by, h + ah/2])

    def _trees(self):
        spots = [(5,5),(5,40),(5,75),(40,5),(40,75),
                 (75,5),(75,40),(75,75),(22,22),(58,58),(28,68),(68,28)]
        for tx, ty in spots:
            n = int(self._rng.integers(3, 7))
            for _ in range(n):
                ox = float(self._rng.uniform(-5, 5))
                oy = float(self._rng.uniform(-5, 5))
                px, py = tx + ox, ty + oy
                h_trunk = float(self._rng.uniform(3, 6))
                r_crown = float(self._rng.uniform(1.8, 3.2))

                # Trunk
                pb.createMultiBody(0, -1,
                    pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.4, length=h_trunk,
                                         rgbaColor=C_TRUNK),
                    [px, py, h_trunk / 2])
                # Crown — 2 layers
                for si in range(2):
                    cr = r_crown * (0.9 - si * 0.15)
                    cz = h_trunk + r_crown * (0.4 + si * 0.55)
                    green = [float(self._rng.uniform(0.05, 0.18)),
                             float(self._rng.uniform(0.48, 0.72)),
                             float(self._rng.uniform(0.05, 0.15)), 1.0]
                    pb.createMultiBody(0, -1,
                        pb.createVisualShape(pb.GEOM_SPHERE, radius=cr, rgbaColor=green),
                        [px, py, cz])
                self.obstacles.append((np.array([px, py, h_trunk + r_crown]),
                                       r_crown + 1.0))

    def _towers(self):
        tower_spots = [(20,60),(60,20),(10,42),(70,68)]
        for tx, ty in tower_spots:
            h = float(self._rng.uniform(22, 32))
            pb.createMultiBody(0, -1,
                pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.55, length=h,
                                     rgbaColor=[0.6, 0.6, 0.65, 1.0]),
                [tx, ty, h/2])
            pb.createMultiBody(0, -1,
                pb.createVisualShape(pb.GEOM_SPHERE, radius=2.8,
                                     rgbaColor=[0.7, 0.3, 0.1, 1.0]),
                [tx, ty, h])
            self.obstacles.append((np.array([tx, ty, h/2]), 3.5))

    def _dynamic_vehicles(self):
        for i in range(6):
            px = float(self._rng.uniform(12, 68))
            py = float(self._rng.uniform(12, 68))
            spd = float(self._rng.uniform(0.4, 1.2))
            ang = float(self._rng.uniform(0, 2 * math.pi))
            vel = np.array([spd * math.cos(ang), spd * math.sin(ang), 0.0])
            r   = float(self._rng.uniform(1.0, 2.0))
            col = [float(x) for x in self._rng.uniform([0.6,0.3,0.0],[1.0,0.6,0.2])] + [1.0]

            vis = pb.createVisualShape(pb.GEOM_BOX,
                                       halfExtents=[r, r * 0.55, 0.55],
                                       rgbaColor=col)
            col_s = pb.createCollisionShape(pb.GEOM_BOX,
                                            halfExtents=[r, r * 0.55, 0.55])
            bid = pb.createMultiBody(0, col_s, vis, [px, py, 0.55])
            self.dyn_bodies.append(bid)
            self.dyn_data.append({'pos': np.array([px, py, 0.55]),
                                   'vel': vel, 'radius': r + 0.5})

    def update_dynamic(self, dt):
        for i, d in enumerate(self.dyn_data):
            d['pos'] += d['vel'] * dt
            for ax in range(2):
                if d['pos'][ax] < 5 or d['pos'][ax] > WORLD_SIZE - 5:
                    d['vel'][ax] *= -1
            pb.resetBasePositionAndOrientation(
                self.dyn_bodies[i], d['pos'].tolist(),
                pb.getQuaternionFromEuler([0, 0, math.atan2(d['vel'][1], d['vel'][0] + 1e-9)]))

    def all_obstacles(self):
        return self.obstacles + [(d['pos'].copy(), d['radius'])
                                  for d in self.dyn_data]


# ════════════════════════════════════════════════════════════════
#  Drone Visual Factory
# ════════════════════════════════════════════════════════════════

def make_drone_body(pos, color):
    body_vis = pb.createVisualShape(pb.GEOM_BOX,
                                    halfExtents=[0.28, 0.28, 0.09],
                                    rgbaColor=color)
    body_col = pb.createCollisionShape(pb.GEOM_BOX,
                                       halfExtents=[0.28, 0.28, 0.09])
    bid = pb.createMultiBody(0.001, body_col, body_vis, pos,
                             pb.getQuaternionFromEuler([0, 0, 0]))
    # Arms + rotors
    for ox, oy in [(0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)]:
        arm_q = pb.getQuaternionFromEuler([0, 0, math.atan2(oy, ox + 1e-9)])
        arm_v = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.05, length=0.4,
                                     rgbaColor=[0.2, 0.2, 0.2, 1.0])
        pb.createMultiBody(0, -1, arm_v,
                           [pos[0] + ox*0.5, pos[1] + oy*0.5, pos[2]], arm_q)
        rotor_v = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.24, length=0.022,
                                       rgbaColor=[0.85, 0.85, 0.95, 0.65])
        pb.createMultiBody(0, -1, rotor_v,
                           [pos[0] + ox, pos[1] + oy, pos[2] + 0.06])
    return bid


# ════════════════════════════════════════════════════════════════
#  Main Simulation Class
# ════════════════════════════════════════════════════════════════

class Simulation3D:

    def __init__(self):
        self.client = pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, 0)
        pb.setRealTimeSimulation(0)
        pb.resetDebugVisualizerCamera(60, 42, -36, [40, 40, 5])
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)

        self.cam_mode  = 0
        self.paused    = False
        self.show_hud  = True
        self.frame     = 0
        self._prev_t   = time.time()
        self.last_tel  = None

        # Build scene
        self.scene = SceneBuilder()
        self.scene.build()

        # Swarm logic
        self.swarm = Swarm()

        # Drone visuals
        self.drone_bodies = [make_drone_body(d.pos.tolist(), C_DRONE[i % len(C_DRONE)])
                              for i, d in enumerate(self.swarm.drones)]

        # Navigator visual
        nav_v = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.65, rgbaColor=C_NAV)
        self.nav_body = pb.createMultiBody(0, -1, nav_v,
                                           self.swarm.navigator.pos.tolist())

        # Target markers + vertical poles
        self.target_markers = []
        for tgt in PATROL_TARGETS:
            tv = pb.createVisualShape(pb.GEOM_SPHERE, radius=2.0, rgbaColor=C_TARGET)
            bid = pb.createMultiBody(0, -1, tv, [tgt[0], tgt[1], 1.8])
            self.target_markers.append(bid)
            pv = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.12,
                                      length=DRONE_HEIGHT + 2,
                                      rgbaColor=[1.0, 0.3, 0.3, 0.35])
            pb.createMultiBody(0, -1, pv, [tgt[0], tgt[1], (DRONE_HEIGHT + 2)/2])

        self.trail_ids = []
        self.hud_ids   = {}

    # ── Sync visual bodies ────────────────────────────────────
    def _sync(self):
        for drone, bid in zip(self.swarm.drones, self.drone_bodies):
            spd = np.linalg.norm(drone.vel)
            if spd > 0.5:
                pitch = -math.atan2(drone.vel[2],
                                    math.sqrt(drone.vel[0]**2 + drone.vel[1]**2))
                yaw   = math.atan2(drone.vel[1], drone.vel[0])
                q = pb.getQuaternionFromEuler([0, pitch * 0.35, yaw])
            else:
                q = pb.getQuaternionFromEuler([0, 0, 0])
            pb.resetBasePositionAndOrientation(bid, drone.pos.tolist(), q)

        pb.resetBasePositionAndOrientation(self.nav_body,
                                           self.swarm.navigator.pos.tolist(),
                                           pb.getQuaternionFromEuler([0, 0, 0]))

        done = self.swarm.targets_done % len(PATROL_TARGETS)
        cur  = self.swarm.target_idx
        for i, bid in enumerate(self.target_markers):
            if i == cur:
                pb.changeVisualShape(bid, -1, rgbaColor=[1.0, 0.15, 0.15, 0.95])
            else:
                pb.changeVisualShape(bid, -1, rgbaColor=[0.15, 0.9, 0.15, 0.35])

    # ── Trails ────────────────────────────────────────────────
    def _trails(self):
        for lid in self.trail_ids:
            pb.removeUserDebugItem(lid)
        self.trail_ids.clear()

        TCOLS = [[0.0,0.7,1.0],[0.0,1.0,0.5],[1.0,0.5,0.0],
                 [0.9,0.0,0.9],[1.0,0.9,0.0],[1.0,0.2,0.2]]

        for i, d in enumerate(self.swarm.drones):
            trail = d.trail[-TRAIL_HISTORY:]
            n = len(trail)
            if n < 2:
                continue
            col = TCOLS[i % len(TCOLS)]
            for j in range(n - 1):
                a = j / n
                lid = pb.addUserDebugLine(
                    trail[j].tolist(), trail[j+1].tolist(),
                    lineColorRGB=[col[0]*a, col[1]*a, col[2]*a],
                    lineWidth=2.2)
                self.trail_ids.append(lid)

        # Navigator gold trail
        nt = self.swarm.navigator.trail[-TRAIL_HISTORY:]
        for j in range(len(nt) - 1):
            a = j / max(len(nt), 1)
            lid = pb.addUserDebugLine(nt[j].tolist(), nt[j+1].tolist(),
                                      lineColorRGB=[1.0, 0.85*a, 0.0],
                                      lineWidth=3.2)
            self.trail_ids.append(lid)

        # Red arrow to current target
        cc  = self.swarm.cluster_center
        tgt = PATROL_TARGETS[self.swarm.target_idx]
        lid = pb.addUserDebugLine(cc.tolist(), tgt,
                                  lineColorRGB=[1.0, 0.3, 0.3], lineWidth=2.0)
        self.trail_ids.append(lid)

    # ── HUD ───────────────────────────────────────────────────
    def _hud(self, t):
        for v in self.hud_ids.values():
            pb.removeUserDebugItem(v)
        self.hud_ids.clear()
        if not self.show_hud or t is None:
            return

        lines = [
            ("Algorithm  : EN-MASCA",                     [1.0,1.0,0.0]),
            (f"Time       : {t['time']:.1f}s",             [1.0,1.0,1.0]),
            (f"Targets    : {t['targets_done']}/{len(PATROL_TARGETS)}",
                                                           [0.2,1.0,0.2]),
            (f"Mean Speed : {t['mean_speed']:.1f} m/s",    [0.4,0.8,1.0]),
            (f"Max H-diff : {t['max_h_diff']:.2f} m",      [0.4,0.8,1.0]),
            (f"Nav Dist   : {t['nav_dist']:.1f} m",        [0.4,0.8,1.0]),
            (f"Nbr Dist   : {t['mean_nbr_dist']:.1f} m",   [0.4,0.8,1.0]),
            ("SPACE=Pause  C=Cam  T=HUD",                  [0.5,0.5,0.5]),
        ]
        for i, (txt, col) in enumerate(lines):
            tid = pb.addUserDebugText(txt,
                                      [1.5, WORLD_SIZE - 2 - i * 3.2, 26],
                                      textColorRGB=col, textSize=1.35)
            self.hud_ids[i] = tid

        for i, d in enumerate(self.swarm.drones):
            spd = np.linalg.norm(d.vel)
            hot = [min(1.0, spd/MAX_SPEED * 2), max(0.0, 1 - spd/MAX_SPEED), 0.2]
            tid = pb.addUserDebugText(f"D{i} {spd:.1f}",
                                      (d.pos + [0, 0, 1.3]).tolist(),
                                      textColorRGB=hot, textSize=0.88)
            self.hud_ids[f'd{i}'] = tid

        tid = pb.addUserDebugText("◆ NAV",
                                  (self.swarm.navigator.pos + [0, 0, 1.6]).tolist(),
                                  textColorRGB=[1.0, 0.95, 0.2], textSize=1.15)
        self.hud_ids['nav'] = tid

        tgt = PATROL_TARGETS[self.swarm.target_idx]
        tid = pb.addUserDebugText(f"▶ TARGET {self.swarm.target_idx+1}",
                                  [tgt[0], tgt[1], DRONE_HEIGHT + 4],
                                  textColorRGB=[1.0, 0.4, 0.4], textSize=1.5)
        self.hud_ids['tgt'] = tid

    # ── Camera ────────────────────────────────────────────────
    def _cam(self):
        cc = self.swarm.cluster_center
        if   self.cam_mode == 0:
            pb.resetDebugVisualizerCamera(38, 50, -30, cc.tolist())
        elif self.cam_mode == 1:
            pb.resetDebugVisualizerCamera(75, 0, -89, [40, 40, 0])
        elif self.cam_mode == 2:
            yaw = 40 + 22 * math.sin(self.frame * 0.002)
            pb.resetDebugVisualizerCamera(55, yaw, -28, cc.tolist())
        # mode 3 = free

    # ── Keys ──────────────────────────────────────────────────
    def _keys(self):
        keys = pb.getKeyboardEvents()
        if ord(' ') in keys and keys[ord(' ')] & pb.KEY_WAS_TRIGGERED:
            self.paused = not self.paused
            print(f"  {'⏸  PAUSED' if self.paused else '▶  RESUMED'}")
        if ord('r') in keys and keys[ord('r')] & pb.KEY_WAS_TRIGGERED:
            pb.disconnect()
            self.__init__()
            return True
        if ord('c') in keys and keys[ord('c')] & pb.KEY_WAS_TRIGGERED:
            self.cam_mode = (self.cam_mode + 1) % 4
            print(f"  Camera: {['Follow','Top-Down','Cinematic','Free'][self.cam_mode]}")
        if ord('t') in keys and keys[ord('t')] & pb.KEY_WAS_TRIGGERED:
            self.show_hud = not self.show_hud
        return False

    # ── Run ───────────────────────────────────────────────────
    def run(self):
        print("\n" + "═"*58)
        print("  EN-MASCA Drone Swarm — City Environment Simulation")
        print("═"*58)
        print(f"  {N_DRONES} drones | {len(PATROL_TARGETS)} patrol targets")
        print("  Controls: SPACE=Pause  R=Reset  C=Camera  T=HUD")
        print("  Drones spawn bottom-left and patrol across city")
        print("═"*58 + "\n")

        obs = self.scene.all_obstacles()

        while pb.isConnected(self.client):
            if self._keys():
                obs = self.scene.all_obstacles()
                continue

            if not self.paused:
                for _ in range(SIM_STEPS_PER_FRAME):
                    self.scene.update_dynamic(PHYSICS_DT)
                obs = self.scene.all_obstacles()
                for _ in range(SIM_STEPS_PER_FRAME):
                    self.last_tel = self.swarm.step(obs, PHYSICS_DT)

            self._sync()
            self._trails()
            self._hud(self.last_tel)
            if self.cam_mode != 3:
                self._cam()

            pb.stepSimulation()
            self.frame += 1

            now = time.time()
            dt  = now - self._prev_t
            if dt < 0.016:
                time.sleep(0.016 - dt)
            self._prev_t = time.time()

            if self.frame % 300 == 0 and self.last_tel:
                t = self.last_tel
                print(f"  t={t['time']:.1f}s | "
                      f"spd={t['mean_speed']:.1f}m/s | "
                      f"targets={t['targets_done']}/{len(PATROL_TARGETS)} | "
                      f"H-diff={t['max_h_diff']:.2f}m")


if __name__ == '__main__':
    sim = Simulation3D()
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n  Exiting …")
    finally:
        try:
            pb.disconnect()
        except Exception:
            pass