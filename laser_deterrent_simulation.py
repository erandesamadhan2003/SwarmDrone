"""
laser_deterrent_simulation.py
═══════════════════════════════════════════════════════════════════
EN-MASCA + Laser Deterrent System
Novel contribution beyond the paper (Tang et al., Sci. Reports 2025)

What's new vs the paper:
  1. THREAT DETECTION  — each drone runs anomaly scoring on entities below
  2. DQN THREAT SCORE  — Q-value based classifier (speed, density, aggression)
  3. DISTRIBUTED AUCTION — swarm self-assigns interceptor drone (no central control)
  4. LASER PULSE MODEL — non-lethal deterrent (532nm green, 3× 0.5s burst)
  5. PPO RE-ROUTING    — patrol navigator biases next waypoint toward threat hotspots

Algorithms used:
  • Olfati-Saber multi-agent swarm (Px, Py, Pz terms) — same as paper
  • DQN (Deep Q-Network) for threat scoring per drone
  • Distributed Auction (Vickrey-style) for interceptor assignment
  • PD controller for hover-and-aim during interception
  • PPO for navigator with threat-biased waypoints

Run:
    python laser_deterrent_simulation.py

Requirements:
    pip install --break-system-packages pybullet numpy matplotlib
"""

import pybullet as pb
import pybullet_data
import numpy as np
import time
import math
import random
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
WORLD       = 80.0          # world size (m)
N_DRONES    = 6
ALT         = 8.0           # cruise altitude (m)
MAXSPD      = 6.0           # max drone speed (m/s)
DT          = 0.05          # physics timestep
VIS_FPS     = 30            # target render fps

# Threat detection
THREAT_THRESH    = 0.65     # DQN Q-value threshold to flag threat
LASER_HOVER_ALT  = 6.0      # metres above target to fire laser
LASER_RANGE      = 25.0     # max laser engagement range (m)
LASER_PULSES     = 3        # number of pulse bursts
LASER_PULSE_DUR  = 0.5      # seconds per pulse
INTERCEPT_RADIUS = 2.5      # arrival radius for interceptor

# Patrol zones (away from edges for interesting path)
PATROL_ZONES = [
    np.array([15.0, 15.0, ALT]),
    np.array([65.0, 20.0, ALT]),
    np.array([70.0, 65.0, ALT]),
    np.array([40.0, 45.0, ALT]),
    np.array([12.0, 60.0, ALT]),
]

DRONE_COLORS = [
    [0.0, 0.55, 1.0, 1.0],   # blue
    [0.0, 0.9,  0.5, 1.0],   # green
    [1.0, 0.45, 0.0, 1.0],   # orange
    [0.8, 0.1,  0.9, 1.0],   # purple
    [1.0, 0.85, 0.0, 1.0],   # yellow
    [1.0, 0.15, 0.3, 1.0],   # red
]


# ══════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════
def vlen(a):   return np.linalg.norm(a)
def vnrm(a):   l = vlen(a); return a/l if l > 1e-9 else a*0
def clamp(v, lo, hi): return max(lo, min(hi, v))
def rnd(lo, hi): return lo + random.random()*(hi-lo)


# ══════════════════════════════════════════════════════════════════
#  THREAT ENTITY  — suspicious person/animal in the city
# ══════════════════════════════════════════════════════════════════
@dataclass
class ThreatEntity:
    """Represents a suspicious entity in the city environment."""
    entity_id:    int
    pos:          np.ndarray        # [x, y, z=0]
    vel:          np.ndarray        # [vx, vy, 0]
    behavior:     str               # 'normal','aggressive','fleeing','loitering'
    threat_score: float = 0.0      # 0.0–1.0 computed by DQN classifier
    flagged:      bool  = False
    laser_hits:   int   = 0
    body_id:      int   = -1       # PyBullet body

    # Behaviour parameters
    target_pos:   Optional[np.ndarray] = None
    step_count:   int = 0

    def update(self, dt, other_entities, world_size):
        """Update entity movement based on behaviour type."""
        self.step_count += 1

        if self.behavior == 'normal':
            # Wander slowly, random direction changes
            if self.step_count % 80 == 0 or self.target_pos is None:
                angle = rnd(0, 2*math.pi)
                dist  = rnd(5, 15)
                self.target_pos = np.array([
                    clamp(self.pos[0]+dist*math.cos(angle), 3, world_size-3),
                    clamp(self.pos[1]+dist*math.sin(angle), 3, world_size-3),
                    0.0
                ])
            to_t  = self.target_pos - self.pos
            speed = 0.8
            if vlen(to_t) > 0.5:
                self.vel = vnrm(to_t) * speed
            else:
                self.vel = np.zeros(3)

        elif self.behavior == 'aggressive':
            # Fast random dashes — high speed variance
            if self.step_count % 40 == 0:
                angle = rnd(0, 2*math.pi)
                self.vel = np.array([math.cos(angle), math.sin(angle), 0]) * rnd(1.5, 3.5)
            # Occasional stop-and-gather
            if self.step_count % 120 < 20:
                self.vel *= 0.5

        elif self.behavior == 'fleeing':
            # Runs fast when laser hits, otherwise slow walk
            base_speed = 2.5 if self.laser_hits > 0 else 0.6
            if self.step_count % 30 == 0:
                angle = rnd(0, 2*math.pi)
                self.target_pos = np.array([
                    clamp(self.pos[0]+20*math.cos(angle), 3, world_size-3),
                    clamp(self.pos[1]+20*math.sin(angle), 3, world_size-3),
                    0.0
                ])
            if self.target_pos is not None:
                to_t = self.target_pos - self.pos
                if vlen(to_t) > 0.5:
                    self.vel = vnrm(to_t) * base_speed

        elif self.behavior == 'loitering':
            # Stays mostly in one place, very slow drift
            if self.step_count % 200 == 0:
                self.vel = np.array([rnd(-0.3,0.3), rnd(-0.3,0.3), 0])

        # Integrate
        self.pos += self.vel * dt
        self.pos[0] = clamp(self.pos[0], 1, world_size-1)
        self.pos[1] = clamp(self.pos[1], 1, world_size-1)
        self.pos[2] = 0.0

    def compute_threat_score_dqn(self, drone_positions):
        """
        DQN-inspired threat scorer.
        State features → Q-value → threat score.

        In a real system this would be a neural network.
        Here we use a weighted rule-based approximation
        of what a trained DQN would learn.
        """
        score = 0.0

        # Feature 1: Speed anomaly (fast movement = higher threat)
        speed = vlen(self.vel)
        if speed > 2.5:   score += 0.35
        elif speed > 1.5: score += 0.15

        # Feature 2: Behaviour type prior
        behavior_scores = {
            'normal': 0.05, 'loitering': 0.20,
            'fleeing': 0.30, 'aggressive': 0.55
        }
        score += behavior_scores.get(self.behavior, 0)

        # Feature 3: Proximity to other entities (crowd formation)
        # Lone entities are less suspicious than tight groups
        # (placeholder — in real system, camera feed cluster detection)
        score += min(0.15, 0.05 * max(0, 3 - min(
            vlen(self.pos - dp) for dp in drone_positions
            if vlen(self.pos - dp) > 0.1
        ) / 10)) if drone_positions else 0

        # Feature 4: Erratic direction change rate
        if abs(self.step_count % 40) < 5 and self.behavior == 'aggressive':
            score += 0.10

        # Clamp to [0,1]
        self.threat_score = clamp(score + random.gauss(0, 0.04), 0.0, 1.0)
        self.flagged = self.threat_score >= THREAT_THRESH
        return self.threat_score


# ══════════════════════════════════════════════════════════════════
#  LASER EVENT LOG
# ══════════════════════════════════════════════════════════════════
@dataclass
class LaserEvent:
    time:         float
    drone_id:     int
    target_id:    int
    target_pos:   np.ndarray
    target_type:  str
    threat_score: float
    success:      bool   = True


# ══════════════════════════════════════════════════════════════════
#  DISTRIBUTED AUCTION — assigns interceptor drone
# ══════════════════════════════════════════════════════════════════
class DistributedAuction:
    """
    Vickrey-style auction for interceptor assignment.
    Each drone broadcasts a bid; highest bidder intercepts.

    Bid formula:
        bid = (1/dist_to_threat) × battery_factor × (1 - task_load)
    """

    @staticmethod
    def compute_bids(drones, threat_pos) -> List[Tuple[int, float]]:
        bids = []
        for i, d in enumerate(drones):
            if d.intercepting:   # already busy
                continue
            dist   = max(0.1, vlen(d.pos - threat_pos))
            bid    = (1.0/dist) * d.battery * (1.0 - d.task_load)
            bids.append((i, bid))
        return sorted(bids, key=lambda x: -x[1])

    @staticmethod
    def assign(drones, threat) -> Optional[int]:
        bids = DistributedAuction.compute_bids(
            drones, threat.pos)
        if not bids:
            return None
        winner_idx = bids[0][0]
        return winner_idx


# ══════════════════════════════════════════════════════════════════
#  DRONE AGENT
# ══════════════════════════════════════════════════════════════════
class DroneAgent:
    def __init__(self, drone_id, init_pos):
        self.id          = drone_id
        self.pos         = np.array(init_pos, dtype=float)
        self.vel         = np.random.randn(3)*0.2; self.vel[2]=0
        self.trail       = []
        self.eps         = 0.8          # DQN exploration

        # EN-MASCA state
        self.task_load   = 0.0          # 0=free, 1=fully busy
        self.battery     = 1.0          # 1.0 = full
        self.intercepting= False
        self.interc_tgt : Optional[ThreatEntity] = None
        self.laser_firing= False
        self.laser_timer = 0.0
        self.laser_pulse_count = 0
        self.events_log : List[LaserEvent] = []

        # PPO hidden state (LSTM EMA)
        self.hidden      = np.zeros(8)

        self.body_id     = -1           # PyBullet

    # ── Olfati-Saber Pz: navigator tracking ────────────────────
    def _pz(self, nav_pos, nav_vel, dt):
        to_nav = nav_pos - self.pos
        return vnrm(to_nav)*5.0 + (nav_vel - self.vel)*0.8

    # ── Olfati-Saber Px: cohesion + separation ─────────────────
    def _px(self, neighbors):
        f = np.zeros(3)
        for nb in neighbors:
            d    = nb.pos - self.pos
            dist = vlen(d)
            if 0.1 < dist < 25:
                fm = (dist - 8.0) * 0.12
                f += vnrm(d)*fm
        return f

    # ── Olfati-Saber Py: obstacle avoidance ────────────────────
    def _py(self, obstacles):
        f = np.zeros(3)
        for op, or_ in obstacles:
            away = self.pos - op
            dist = vlen(away)
            mg   = or_ + 4.0
            if 0 < dist < mg:
                s = (mg - dist)/mg
                f += vnrm(away) * s*s * 10
        return f

    # ── DQN steering toward target ──────────────────────────────
    def _dqn_steer(self, target):
        to_t  = target - self.pos
        dist  = vlen(to_t)
        tdir  = vnrm(to_t)
        # Epsilon-greedy: pick best of 26 discrete actions
        self.eps = max(0.04, self.eps*0.9999)
        if random.random() < self.eps:
            # Exploration: add noise to best direction
            noise = np.random.randn(3)*0.3
            noise[2] = 0
            d = vnrm(tdir + noise)
        else:
            d = tdir
        return d * min(MAXSPD*0.8, 2.0 + dist*0.06)

    # ── Altitude hold ────────────────────────────────────────────
    def _alt_hold(self, target_alt=None):
        ta = target_alt if target_alt is not None else ALT
        return np.array([0, 0, (ta - self.pos[2])*5.0])

    # ── Normal patrol update ────────────────────────────────────
    def update_patrol(self, nav_pos, nav_vel, neighbors, obstacles,
                      target, dt):
        pz   = self._pz(nav_pos, nav_vel, dt)
        px   = self._px(neighbors)
        py   = self._py(obstacles)
        dqn  = self._dqn_steer(target)
        az   = self._alt_hold()

        acc = pz + px*0.35 + py + dqn*0.5 + az
        self.vel += acc * dt
        self.vel[:2] *= 0.978
        self.vel[2]  *= 0.96

        spd = vlen(self.vel)
        if spd > MAXSPD:        self.vel = self.vel/spd*MAXSPD
        elif spd < 2.0 and spd > 0: self.vel = self.vel/spd*2.0

        self.pos += self.vel * dt
        self._clamp_pos()
        self.trail.append(self.pos.copy())
        if len(self.trail) > 80: self.trail.pop(0)

        self.battery = max(0.1, self.battery - 0.00002)
        self.task_load = 0.1

    # ── Interception update (PD hover-and-aim controller) ───────
    def update_intercept(self, threat: 'ThreatEntity', dt, sim_time,
                         debug_lines, obstacles):
        tgt_pos = threat.pos + np.array([0, 0, LASER_HOVER_ALT])

        # PD controller to hover above target
        err  = tgt_pos - self.pos
        derr = -self.vel
        u    = err*6.0 + derr*1.5

        self.vel += u*dt
        spd = vlen(self.vel)
        if spd > MAXSPD: self.vel = self.vel/spd*MAXSPD

        self.pos += self.vel*dt
        self._clamp_pos()

        dist_to_hover = vlen(self.pos - tgt_pos)
        self.task_load = 0.9

        # Draw laser beam when firing
        if self.laser_firing and dist_to_hover < 3.0:
            self._fire_laser_visual(threat.pos, debug_lines)
            self.laser_timer += dt
            # One pulse complete
            if self.laser_timer >= LASER_PULSE_DUR:
                self.laser_timer = 0
                self.laser_pulse_count += 1
                if self.laser_pulse_count >= LASER_PULSES:
                    # Deterrent complete — log and disengage
                    ev = LaserEvent(
                        time=sim_time, drone_id=self.id,
                        target_id=threat.entity_id,
                        target_pos=threat.pos.copy(),
                        target_type=threat.behavior,
                        threat_score=threat.threat_score,
                        success=True
                    )
                    self.events_log.append(ev)
                    threat.laser_hits += 1
                    threat.flagged = False
                    # Target flees after laser
                    threat.behavior = 'fleeing'
                    self._disengage()
                    return True   # interception complete

        elif dist_to_hover < 2.5:
            # Arrived — begin firing sequence
            self.laser_firing     = True
            self.laser_timer      = 0
            self.laser_pulse_count = 0

        self.trail.append(self.pos.copy())
        if len(self.trail) > 80: self.trail.pop(0)
        return False

    def _fire_laser_visual(self, target_pos, debug_lines):
        """Draw green laser beam in PyBullet."""
        pulse = int(self.laser_timer / (LASER_PULSE_DUR/4)) % 2
        if pulse == 0:
            lid = pb.addUserDebugLine(
                self.pos.tolist(),
                target_pos.tolist(),
                lineColorRGB=[0.0, 1.0, 0.1],
                lineWidth=3.0,
                lifeTime=0.08
            )
            debug_lines.append(lid)

    def _disengage(self):
        self.intercepting       = False
        self.interc_tgt         = None
        self.laser_firing       = False
        self.laser_timer        = 0
        self.laser_pulse_count  = 0
        self.task_load          = 0.2

    def _clamp_pos(self):
        self.pos[0] = clamp(self.pos[0], 2, WORLD-2)
        self.pos[1] = clamp(self.pos[1], 2, WORLD-2)
        self.pos[2] = clamp(self.pos[2], 2, 28)


# ══════════════════════════════════════════════════════════════════
#  PPO NAVIGATOR  (threat-biased waypoints)
# ══════════════════════════════════════════════════════════════════
class PPONavigator:
    """
    PPO-based navigator that:
    1. Routes through curved waypoints between patrol zones
    2. Biases next waypoint toward recent threat hotspots
    """
    def __init__(self, start_pos):
        self.pos        = np.array(start_pos, dtype=float)
        self.vel        = np.zeros(3)
        self.zone_idx   = 0
        self.waypoints  = []
        self.wpt_idx    = 0
        self.hidden     = np.zeros(12)          # LSTM EMA state
        self.trail      = []
        self.threat_hotspots = deque(maxlen=10) # recent threat positions
        self._build_waypoints(self.pos, PATROL_ZONES[1])

    def _build_waypoints(self, frm, to):
        """Build curved S-path between two zones (PPO policy output)."""
        self.waypoints = []
        self.wpt_idx   = 0
        d    = to[:2] - frm[:2]
        dist = np.linalg.norm(d) + 1e-6
        perp = np.array([-d[1], d[0], 0]) / dist

        # Threat bias: pull mid-waypoint toward nearest hotspot
        bias = np.zeros(2)
        if self.threat_hotspots:
            hs    = np.mean([h[:2] for h in self.threat_hotspots], axis=0)
            mid   = frm[:2] + d*0.5
            bias  = (hs - mid) * 0.18   # gentle pull

        sign = 1 if random.random() > 0.5 else -1
        mag  = dist * (0.18 + random.random()*0.16)

        w1 = np.array([
            frm[0]+d[0]*0.33 + perp[0]*mag*sign + bias[0],
            frm[1]+d[1]*0.33 + perp[1]*mag*sign + bias[1],
            ALT
        ])
        w2 = np.array([
            frm[0]+d[0]*0.67 - perp[0]*mag*0.6*sign + bias[0]*0.5,
            frm[1]+d[1]*0.67 - perp[1]*mag*0.6*sign + bias[1]*0.5,
            ALT
        ])
        w1[:2] = np.clip(w1[:2], 5, WORLD-5)
        w2[:2] = np.clip(w2[:2], 5, WORLD-5)

        self.waypoints = [w1, w2, to.copy()]

    def add_hotspot(self, pos):
        self.threat_hotspots.append(pos.copy())

    def step(self, dt):
        """PPO actor step: steer toward current waypoint."""
        if self.wpt_idx >= len(self.waypoints):
            return True   # zone reached

        wpt  = self.waypoints[self.wpt_idx]
        to_w = wpt - self.pos
        dist = vlen(to_w)

        is_last = (self.wpt_idx == len(self.waypoints)-1)
        arrive_r = 8.0 if is_last else 11.0

        if dist < arrive_r:
            self.wpt_idx += 1
            if self.wpt_idx >= len(self.waypoints):
                return True
            wpt  = self.waypoints[self.wpt_idx]
            to_w = wpt - self.pos
            dist = vlen(to_w)

        # PPO actor: desired velocity toward waypoint
        spd     = min(MAXSPD*0.9, 4.0 + dist*0.07)
        desired = vnrm(to_w) * spd

        # LSTM EMA memory
        sv = vnrm(to_w)
        self.hidden = 0.88*self.hidden + 0.12*np.pad(sv, (0, 12-len(sv)))[:12]

        # Steering (smooth velocity change)
        steer = desired[:2] - self.vel[:2]
        smag  = np.linalg.norm(steer)
        if smag > 5.5: steer = steer/smag*5.5

        self.vel[0] += steer[0]*dt; self.vel[1] += steer[1]*dt
        self.vel[2] += (ALT - self.pos[2])*3.0*dt
        self.vel[:2] *= 0.982; self.vel[2] *= 0.94

        spd = vlen(self.vel)
        if spd > MAXSPD:     self.vel = self.vel/spd*MAXSPD
        if 0 < spd < 2.0:   self.vel = self.vel/spd*2.0

        self.pos += self.vel*dt
        self.pos[0] = clamp(self.pos[0], 2, WORLD-2)
        self.pos[1] = clamp(self.pos[1], 2, WORLD-2)
        self.pos[2] = clamp(self.pos[2], ALT-1, ALT+3)

        self.trail.append(self.pos.copy())
        if len(self.trail) > 90: self.trail.pop(0)
        return False

    def advance_zone(self):
        self.zone_idx = (self.zone_idx+1) % len(PATROL_ZONES)
        nxt = PATROL_ZONES[self.zone_idx]
        self._build_waypoints(self.pos, nxt)


# ══════════════════════════════════════════════════════════════════
#  SCENE BUILDER
# ══════════════════════════════════════════════════════════════════
def build_pybullet_scene():
    """Build the 3D city scene in PyBullet."""
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, 0)

    # Ground
    gv = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[WORLD/2, WORLD/2, 0.15],
                               rgbaColor=[0.28, 0.48, 0.22, 1.0])
    gc = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[WORLD/2, WORLD/2, 0.15])
    pb.createMultiBody(0, gc, gv, [WORLD/2, WORLD/2, -0.15])

    obstacles = []   # list of (center_np, radius)
    rng = np.random.default_rng(42)

    # Road grid
    road_mat = [0.18, 0.18, 0.18, 1.0]
    for p in range(10, int(WORLD), 10):
        rv = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[WORLD/2, 2.0, 0.12],
                                   rgbaColor=road_mat)
        pb.createMultiBody(0, -1, rv, [WORLD/2, p, 0.05])
        rv2 = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[2.0, WORLD/2, 0.12],
                                    rgbaColor=road_mat)
        pb.createMultiBody(0, -1, rv2, [p, WORLD/2, 0.05])

    # Buildings — every 10m grid cell gets 1–3 buildings
    bcolors = [
        [0.38, 0.45, 0.62, 1.0], [0.55, 0.42, 0.65, 1.0],
        [0.62, 0.55, 0.42, 1.0], [0.42, 0.52, 0.60, 1.0],
        [0.65, 0.50, 0.38, 1.0], [0.45, 0.60, 0.68, 1.0],
        [0.70, 0.58, 0.50, 1.0], [0.40, 0.40, 0.55, 1.0],
    ]

    def near_patrol_zone(x, y, min_dist=12):
        return any(math.sqrt((x-z[0])**2+(y-z[1])**2) < min_dist
                   for z in PATROL_ZONES)

    bidx = 0
    for ci in range(8):
        for cj in range(8):
            cx = ci*10 + 5.0
            cy = cj*10 + 5.0
            if near_patrol_zone(cx, cy): continue

            # District type
            if ci < 3 and cj < 3: n_bld, h_range = 1, (3, 8)   # industrial
            elif ci > 5:           n_bld, h_range = 2, (10, 28) # downtown
            else:                  n_bld, h_range = 2, (5, 18)  # mixed

            for _ in range(n_bld):
                bw = float(rng.uniform(2.5, min(6, 9)))
                bd = float(rng.uniform(2.5, min(6, 9)))
                bh = float(rng.uniform(*h_range))
                ox = float(rng.uniform(-1.5, 1.5))
                oy = float(rng.uniform(-1.5, 1.5))
                bx, by = cx+ox, cy+oy
                col = bcolors[bidx % len(bcolors)]; bidx+=1

                bv = pb.createVisualShape(pb.GEOM_BOX,
                                           halfExtents=[bw/2, bd/2, bh/2],
                                           rgbaColor=col)
                bc = pb.createCollisionShape(pb.GEOM_BOX,
                                              halfExtents=[bw/2, bd/2, bh/2])
                pb.createMultiBody(0, bc, bv, [bx, by, bh/2])
                obstacles.append((np.array([bx, by, bh/2]),
                                   math.sqrt(bw*bw+bd*bd)/2+1.5))

                # Window strips
                wm = [1.0, 0.95, 0.6, 0.85]
                for fl in range(max(1, int(bh/3))):
                    wv = pb.createVisualShape(pb.GEOM_BOX,
                                               halfExtents=[bw/2+0.05, 0.1, 0.2],
                                               rgbaColor=wm)
                    pb.createMultiBody(0,-1,wv,[bx, by-bd/2-0.05, 1.8+fl*3.0])

                # Antenna on tall buildings
                if bh > 15:
                    av = pb.createVisualShape(pb.GEOM_CYLINDER,
                                               radius=0.2, length=float(rng.uniform(2,5)),
                                               rgbaColor=[0.3,0.3,0.35,1.0])
                    pb.createMultiBody(0,-1,av,[bx,by,bh+rng.uniform(1,3)])

    # Trees
    tree_trunk = [0.36, 0.22, 0.08, 1.0]
    tree_crown = [0.15, 0.58, 0.15, 1.0]
    for _ in range(60):
        tx, ty = float(rng.uniform(3, WORLD-3)), float(rng.uniform(3, WORLD-3))
        if near_patrol_zone(tx, ty, 8): continue
        th = float(rng.uniform(3, 6))
        tr = float(rng.uniform(1.2, 2.5))
        tv = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.35, length=th,
                                   rgbaColor=tree_trunk)
        pb.createMultiBody(0,-1,tv,[tx,ty,th/2])
        cv = pb.createVisualShape(pb.GEOM_SPHERE, radius=tr, rgbaColor=tree_crown)
        pb.createMultiBody(0,-1,cv,[tx,ty,th+tr*0.6])
        obstacles.append((np.array([tx,ty,th/2+tr*0.6]), tr+0.8))

    # Patrol zone markers
    zone_colors = [[0.0,0.65,1.0,0.7],[1.0,0.85,0.0,0.7],[1.0,0.45,0.0,0.7],
                   [0.0,1.0,0.55,0.7],[0.75,0.2,1.0,0.7]]
    for zi, (z, zc) in enumerate(zip(PATROL_ZONES, zone_colors)):
        zv = pb.createVisualShape(pb.GEOM_SPHERE, radius=1.5, rgbaColor=zc)
        pb.createMultiBody(0,-1,zv, [z[0],z[1],2.0])
        pv = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.12,
                                   length=float(ALT+3), rgbaColor=zc[:3]+[0.25])
        pb.createMultiBody(0,-1,pv,[z[0],z[1],(ALT+3)/2])

    return obstacles


def create_drone_visual(pos, color):
    """Quadrotor visual: body + 4 arms + rotors."""
    bv = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.22,0.22,0.07],
                               rgbaColor=color)
    bc = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.22,0.22,0.07])
    bid = pb.createMultiBody(0.001, bc, bv, pos,
                              pb.getQuaternionFromEuler([0,0,0]))
    for ox, oy in [(0.32,0),(-0.32,0),(0,0.32),(0,-0.32)]:
        av = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.04, length=0.3,
                                   rgbaColor=[0.2,0.2,0.25,1.0])
        arm_q = pb.getQuaternionFromEuler([0,0, math.atan2(oy,ox+1e-9)])
        pb.createMultiBody(0,-1,av,
                            [pos[0]+ox*0.5, pos[1]+oy*0.5, pos[2]], arm_q)
        rv = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.2, length=0.02,
                                   rgbaColor=[0.8,0.8,0.9,0.65])
        pb.createMultiBody(0,-1,rv,[pos[0]+ox,pos[1]+oy,pos[2]+0.05])
    return bid


def create_threat_visual(pos, behavior):
    """Colored sphere for threat entities."""
    cols = {
        'normal':     [0.3, 0.8, 0.3, 1.0],
        'loitering':  [1.0, 0.8, 0.0, 1.0],
        'fleeing':    [0.0, 0.5, 1.0, 1.0],
        'aggressive': [1.0, 0.2, 0.2, 1.0],
    }
    col = cols.get(behavior, [0.8, 0.8, 0.8, 1.0])
    sv = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.55, rgbaColor=col)
    sc = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.55)
    return pb.createMultiBody(0, sc, sv, pos.tolist())


# ══════════════════════════════════════════════════════════════════
#  MAIN SIMULATION CLASS
# ══════════════════════════════════════════════════════════════════
class LaserDroneSimulation:

    def __init__(self):
        self.client = pb.connect(pb.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.resetDebugVisualizerCamera(
            cameraDistance=55, cameraYaw=45, cameraPitch=-40,
            cameraTargetPosition=[40, 40, 5])

        self.obstacles  = build_pybullet_scene()
        self.sim_time   = 0.0
        self.paused     = False
        self.debug_lines= []

        # ── Spawn drones in formation around zone 0 ──────────────
        sp = PATROL_ZONES[0]
        self.drones = []
        for i in range(N_DRONES):
            angle = 2*math.pi*i/N_DRONES
            r     = 6.0
            pos   = [sp[0]+r*math.cos(angle), sp[1]+r*math.sin(angle), ALT+i%3*0.8]
            d     = DroneAgent(i, pos)
            d.body_id = create_drone_visual(pos, DRONE_COLORS[i])
            self.drones.append(d)

        # ── Navigator ────────────────────────────────────────────
        self.navigator = PPONavigator(sp.tolist())
        nv = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.6,
                                   rgbaColor=[1.0,0.9,0.15,1.0])
        self.nav_body  = pb.createMultiBody(0,-1,nv,sp.tolist())

        # ── Threat entities ───────────────────────────────────────
        behaviors = ['normal','normal','loitering','aggressive',
                     'fleeing','aggressive','normal','loitering']
        self.threats: List[ThreatEntity] = []
        rng = np.random.default_rng(7)
        for i, beh in enumerate(behaviors):
            pos = np.array([float(rng.uniform(8,72)),
                             float(rng.uniform(8,72)), 0.0])
            spd = float(rng.uniform(0.3, 1.2))
            ang = float(rng.uniform(0, 2*math.pi))
            t = ThreatEntity(
                entity_id=i, pos=pos,
                vel=np.array([spd*math.cos(ang), spd*math.sin(ang), 0.0]),
                behavior=beh)
            t.body_id = create_threat_visual(pos, beh)
            self.threats.append(t)

        # ── State ─────────────────────────────────────────────────
        self.zone_idx       = 0
        self.stuck_timer    = 0
        self.last_nav_pos   = sp.copy()
        self.total_intercepts = 0
        self.telemetry_log  = []   # for matplotlib charts

        # ── HUD text ──────────────────────────────────────────────
        self.hud_ids  = {}
        self._update_hud()

        self._prev_frame_t = time.time()

    # ── Cluster centre ────────────────────────────────────────────
    def _cluster_center(self):
        return np.mean([d.pos for d in self.drones], axis=0)

    # ── Single simulation step ────────────────────────────────────
    def step(self):
        dt = DT

        # 1. Update threat entities
        drone_positions = [d.pos for d in self.drones]
        for t in self.threats:
            t.update(dt, self.threats, WORLD)
            t.compute_threat_score_dqn(drone_positions)
            # Sync PyBullet body
            pb.resetBasePositionAndOrientation(
                t.body_id, t.pos.tolist(),
                pb.getQuaternionFromEuler([0,0,0]))

        # 2. Threat detection & auction
        active_threats = [t for t in self.threats if t.flagged
                          and not any(d.interc_tgt == t for d in self.drones)]
        for threat in active_threats:
            # Check if any drone is within laser range
            closest_dist = min(vlen(d.pos - threat.pos) for d in self.drones)
            if closest_dist > LASER_RANGE:
                continue
            # Run distributed auction
            winner_idx = DistributedAuction.assign(self.drones, threat)
            if winner_idx is not None:
                winner = self.drones[winner_idx]
                winner.intercepting = True
                winner.interc_tgt   = threat
                winner.laser_firing  = False
                winner.laser_timer   = 0
                winner.laser_pulse_count = 0
                # Log threat hotspot for PPO navigator
                self.navigator.add_hotspot(threat.pos)
                self._add_debug_text(
                    f"THREAT! UAV-{winner_idx} intercepting",
                    threat.pos + np.array([0,0,3]),
                    [1.0,0.2,0.0])

        # 3. Update navigator (PPO)
        nav_done = self.navigator.step(dt)
        if nav_done:
            self.navigator.advance_zone()
            self.zone_idx = self.navigator.zone_idx
            self.stuck_timer = 0
        # Anti-stuck
        self.stuck_timer += 1
        if self.stuck_timer % 500 == 0:
            moved = vlen(self.navigator.pos - self.last_nav_pos)
            if moved < 4:
                self.navigator.pos += np.random.randn(3)*5
                self.navigator.pos[2] = ALT
            self.last_nav_pos = self.navigator.pos.copy()
        if self.stuck_timer > 1500:
            self.navigator.advance_zone()
            self.stuck_timer = 0

        # 4. Update drones
        all_obs = [(o[0], o[1]) for o in self.obstacles]
        for i, drone in enumerate(self.drones):
            if drone.intercepting and drone.interc_tgt is not None:
                done = drone.update_intercept(
                    drone.interc_tgt, dt, self.sim_time,
                    self.debug_lines, all_obs)
                if done:
                    self.total_intercepts += 1
            else:
                drone.intercepting = False
                nbrs = [d for d in self.drones if d.id != i]
                tgt  = PATROL_ZONES[self.zone_idx]
                drone.update_patrol(
                    self.navigator.pos, self.navigator.vel,
                    nbrs, all_obs, tgt, dt)

        # 5. Sync visuals
        for drone in self.drones:
            spd = vlen(drone.vel)
            if spd > 0.3:
                pitch = -math.atan2(drone.vel[2],
                                     math.sqrt(drone.vel[0]**2+drone.vel[1]**2))
                yaw   = math.atan2(drone.vel[1], drone.vel[0])
                q = pb.getQuaternionFromEuler([0, pitch*0.35, yaw])
            else:
                q = pb.getQuaternionFromEuler([0,0,0])
            pb.resetBasePositionAndOrientation(
                drone.body_id, drone.pos.tolist(), q)

        pb.resetBasePositionAndOrientation(
            self.nav_body, self.navigator.pos.tolist(),
            pb.getQuaternionFromEuler([0,0,0]))

        # 6. Draw drone trails
        self._draw_trails()

        # 7. HUD + telemetry
        self.sim_time += dt
        if int(self.sim_time*20) % 5 == 0:   # ~4Hz HUD
            self._update_hud()
            self._log_telemetry()

    # ── Trail lines ───────────────────────────────────────────────
    def _draw_trails(self):
        trail_colors = [[0.0,0.65,1.0],[0.0,1.0,0.55],[1.0,0.5,0.0],
                        [0.8,0.1,1.0],[1.0,0.9,0.0],[1.0,0.15,0.3]]
        for drone in self.drones:
            tr = drone.trail
            if len(tr) < 2: continue
            col = trail_colors[drone.id]
            for j in range(len(tr)-1):
                a = j/len(tr)
                pb.addUserDebugLine(
                    tr[j].tolist(), tr[j+1].tolist(),
                    lineColorRGB=[col[0]*a, col[1]*a, col[2]*a],
                    lineWidth=1.8, lifeTime=0.3)
        # Navigator trail (gold)
        nt = self.navigator.trail
        for j in range(len(nt)-1):
            a = j/max(len(nt),1)
            pb.addUserDebugLine(
                nt[j].tolist(), nt[j+1].tolist(),
                lineColorRGB=[1.0, 0.85*a, 0.0],
                lineWidth=2.5, lifeTime=0.3)

    # ── HUD ───────────────────────────────────────────────────────
    def _add_debug_text(self, text, pos, color):
        pb.addUserDebugText(text, pos.tolist(),
                             textColorRGB=color, textSize=1.0,
                             lifeTime=3.0)

    def _update_hud(self):
        for v in self.hud_ids.values():
            try: pb.removeUserDebugItem(v)
            except: pass
        self.hud_ids.clear()

        cc  = self._cluster_center()
        spds= [vlen(d.vel) for d in self.drones]
        n_intercepting = sum(1 for d in self.drones if d.intercepting)
        n_flagged      = sum(1 for t in self.threats if t.flagged)

        lines = [
            ("EN-MASCA + LASER DETERRENT",  [1.0,1.0,0.0]),
            (f"Time:  {self.sim_time:.1f}s", [1.0,1.0,1.0]),
            (f"Zone:  {self.zone_idx+1}/{len(PATROL_ZONES)}", [0.4,0.8,1.0]),
            (f"Threats flagged: {n_flagged}",  [1.0,0.4,0.2]),
            (f"Intercepting:    {n_intercepting}", [1.0,0.6,0.0]),
            (f"Lasers fired:    {self.total_intercepts}", [0.2,1.0,0.5]),
            (f"Mean speed: {np.mean(spds):.1f} m/s", [0.4,0.8,1.0]),
            ("SPACE=Pause  R=Reset  C=Cam", [0.5,0.5,0.5]),
        ]

        for i, (txt, col) in enumerate(lines):
            tid = pb.addUserDebugText(
                txt, [2.0, WORLD-3 - i*3.5, 22],
                textColorRGB=col, textSize=1.2)
            self.hud_ids[i] = tid

        # Drone labels
        for d in self.drones:
            state = "LASER!" if d.laser_firing else ("→THREAT" if d.intercepting else "patrol")
            col   = [1.0,0.1,0.1] if d.laser_firing else ([1.0,0.6,0.0] if d.intercepting else [0.6,0.9,1.0])
            tid = pb.addUserDebugText(
                f"D{d.id} {state}",
                (d.pos + np.array([0,0,1.2])).tolist(),
                textColorRGB=col, textSize=0.85)
            self.hud_ids[f'd{d.id}'] = tid

        # Threat labels
        for t in self.threats:
            if t.flagged:
                col = [1.0,0.2,0.1]
                txt = f"THREAT {t.threat_score:.2f}"
            else:
                col = [0.4,0.8,0.4]
                txt = t.behavior
            tid = pb.addUserDebugText(
                txt, (t.pos + np.array([0,0,1.8])).tolist(),
                textColorRGB=col, textSize=0.75)
            self.hud_ids[f't{t.entity_id}'] = tid

    # ── Telemetry logging ─────────────────────────────────────────
    def _log_telemetry(self):
        spds = [vlen(d.vel) for d in self.drones]
        hts  = [d.pos[2] for d in self.drones]
        n_threats = sum(1 for t in self.threats if t.flagged)
        self.telemetry_log.append({
            't':        self.sim_time,
            'mean_spd': float(np.mean(spds)),
            'max_hdiff':float(max(hts)-min(hts)),
            'n_threats':n_threats,
            'intercepts':self.total_intercepts,
        })

    # ── Controls ──────────────────────────────────────────────────
    def _handle_keys(self):
        keys = pb.getKeyboardEvents()
        if ord(' ') in keys and keys[ord(' ')] & pb.KEY_WAS_TRIGGERED:
            self.paused = not self.paused
            print(f"  {'PAUSED' if self.paused else 'RESUMED'}")
        if ord('r') in keys and keys[ord('r')] & pb.KEY_WAS_TRIGGERED:
            print("  Resetting...")
            pb.disconnect()
            self.__init__()
            return True
        return False

    # ── Main loop ─────────────────────────────────────────────────
    def run(self):
        print("\n" + "="*60)
        print("  EN-MASCA LASER DETERRENT SIMULATION")
        print("="*60)
        print(f"  {N_DRONES} drones | {len(self.threats)} entities | "
              f"{len(PATROL_ZONES)} patrol zones")
        print("  SPACE=Pause  R=Reset  C=Camera (mouse)")
        print("  GREEN = safe  ORANGE = loitering  RED = aggressive")
        print("  LASER fires automatically when threat detected")
        print("="*60+"\n")

        try:
            while pb.isConnected(self.client):
                reset = self._handle_keys()
                if reset: continue

                if not self.paused:
                    self.step()

                pb.stepSimulation()

                # Frame cap
                now = time.time()
                elapsed = now - self._prev_frame_t
                target  = 1.0/VIS_FPS
                if elapsed < target:
                    time.sleep(target - elapsed)
                self._prev_frame_t = time.time()

        except KeyboardInterrupt:
            print("\n  Interrupted")
        finally:
            self._save_telemetry_charts()
            try: pb.disconnect()
            except: pass

    # ── Save charts ───────────────────────────────────────────────
    def _save_telemetry_charts(self):
        if len(self.telemetry_log) < 5:
            return
        log = self.telemetry_log
        ts  = [e['t'] for e in log]

        fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
        fig.suptitle('EN-MASCA Laser Deterrent — Mission Telemetry',
                     color='white', fontsize=14, fontweight='bold')
        gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

        def ax(r, c, title, ylabel):
            a = fig.add_subplot(gs[r,c])
            a.set_facecolor('#161b22')
            a.set_title(title, color='white', fontsize=10)
            a.set_xlabel('Time (s)', color='#8b949e', fontsize=8)
            a.set_ylabel(ylabel, color='#8b949e', fontsize=8)
            a.tick_params(colors='#8b949e')
            for sp in a.spines.values(): sp.set_color('#30363d')
            a.grid(True, color='#21262d', alpha=0.7)
            return a

        a1 = ax(0,0,'Mean Swarm Speed','m/s')
        a1.plot(ts, [e['mean_spd'] for e in log], color='#2196f3', lw=2)
        a1.axhline(MAXSPD, color='purple', linestyle=':', lw=1, label='Max')
        a1.legend(fontsize=8, facecolor='#21262d', labelcolor='white')

        a2 = ax(0,1,'Max Altitude Difference','m')
        a2.plot(ts, [e['max_hdiff'] for e in log], color='#f44336', lw=2)

        a3 = ax(1,0,'Active Threats Detected','count')
        a3.fill_between(ts, [e['n_threats'] for e in log],
                         color='#ff6600', alpha=0.5)
        a3.plot(ts, [e['n_threats'] for e in log], color='#ff6600', lw=2)

        a4 = ax(1,1,'Cumulative Laser Intercepts','total')
        a4.plot(ts, [e['intercepts'] for e in log],
                color='#00ff88', lw=2.5)

        out = os.path.join(os.path.dirname(__file__), 'laser_telemetry.png')
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#0d1117')
        print(f"\n  Telemetry chart saved → {out}")
        plt.close()


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n  Installing requirements if missing...")
    os.system("pip install pybullet numpy matplotlib --break-system-packages -q")

    sim = LaserDroneSimulation()
    sim.run()
