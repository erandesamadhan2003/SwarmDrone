"""
drone_agent.py
EN-MASCA Drone Agent — models a single UAV in the swarm.
Implements:
  - Second-order dynamics (position + velocity)
  - Cohesion / Separation / Alignment (Olfati-Saber)
  - Obstacle avoidance via virtual beta-agents
  - DQN-based action selection
  - Navigator tracking (Pz term)
"""

import numpy as np
from collections import deque
import random


# ──────────────────────────────────────────────────
# Hyper-parameters (tuned for durian-orchard scale)
# ──────────────────────────────────────────────────
ALPHA       = 5.0    # Sα sharpness
BETA        = 2.0    # minimum allowed inter-agent distance τ
OMEGA       = 8.0    # influence radius
MU          = 0.5    # coupling constant
K_CONST     = 10.0   # normalisation constant

EX = 1.0;  EY = 1.2;  EZ = 0.8   # gain constants
AX = 2.0;  AY = 2.0;  AZ = 1.5   # α constants for step-size

DETECT_DIST = 6.0    # obstacle detection radius δ
DT          = 0.05   # simulation timestep (seconds)
MAX_SPEED   = 5.0    # m/s
MIN_SPEED   = 0.5

# DQN action space: 26 discrete 3-D heading directions
ACTIONS = []
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if not (dx == dy == dz == 0):
                v = np.array([dx, dy, dz], dtype=float)
                ACTIONS.append(v / np.linalg.norm(v))
ACTIONS = np.array(ACTIONS)   # shape (26, 3)
N_ACTIONS = len(ACTIONS)
STATE_DIM  = 15   # [pos(3), vel(3), rel_to_nav(3), nearest_obs_dir(3), nearest_nbr_dir(3)]


# ──────────────────────────────────────────────────
#  Utility maths
# ──────────────────────────────────────────────────
def sigma_norm(z, eps=0.1):
    return (1/eps) * (np.sqrt(1 + eps * np.dot(z, z)) - 1)

def grad_sigma(z, eps=0.1):
    return z / np.sqrt(1 + eps * np.dot(z, z))

def bump_function(s, h=0.2):
    """Smooth bump ∈ [0,1] used as Olfati-Saber H."""
    if   s < h:          return 1.0
    elif s < 1.0:        return 0.5 * (1 + np.cos(np.pi * (s - h) / (1 - h)))
    else:                return 0.0

def phi_alpha(s, d, r, h=0.2, a=5, b=5):
    """Olfati-Saber φα action function."""
    c   = abs(a - b) / np.sqrt(4 * a * b)
    z   = np.array([s - d])
    sig_val = float((1/0.1) * (np.sqrt(1 + 0.1 * np.dot(z, z)) - 1))
    phi = 0.5 * ((a + b) * sigma1(sig_val + c) + (a - b))
    rho = bump_function(s / r, h)
    return rho * phi

def sigma1(z):
    return z / np.sqrt(1 + z**2)


# ──────────────────────────────────────────────────
#  Simple DQN (tabular Q-table approximation)
#  (No deep network dep — pure numpy so it runs
#   immediately without PyTorch install issues)
# ──────────────────────────────────────────────────
class DQNAgent:
    """Lightweight Q-table DQN for action selection."""

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS,
                 lr=0.001, gamma=0.99, eps=0.9, eps_min=0.05, eps_decay=0.995):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.eps        = eps
        self.eps_min    = eps_min
        self.eps_decay  = eps_decay
        self.lr         = lr
        self.memory     = deque(maxlen=5000)

        # Simple linear approximator  Q(s,a) ≈ s · W[:,a]
        np.random.seed(42)
        self.W       = np.random.randn(state_dim, n_actions) * 0.01
        self.W_tgt   = self.W.copy()
        self.step    = 0
        self.update_freq = 100

    def predict(self, state):
        return state @ self.W   # shape (n_actions,)

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.predict(state)))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self, batch=64):
        if len(self.memory) < batch:
            return
        batch_data = random.sample(self.memory, batch)
        for s, a, r, s2, done in batch_data:
            target = r if done else r + self.gamma * np.max(self.W_tgt.T @ s2)
            current = self.W[:, a] @ s
            grad = (current - target) * s
            self.W[:, a] -= self.lr * grad
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self.step += 1
        if self.step % self.update_freq == 0:
            self.W_tgt = self.W.copy()


# ──────────────────────────────────────────────────
#  Drone Agent
# ──────────────────────────────────────────────────
class DroneAgent:
    def __init__(self, drone_id, init_pos, dqn: DQNAgent):
        self.id       = drone_id
        self.pos      = np.array(init_pos, dtype=float)
        self.vel      = np.zeros(3)
        self.dqn      = dqn
        self.prev_state = None
        self.prev_action = None
        self.total_reward = 0.0
        self.trail    = [self.pos.copy()]   # for visualisation
        self.alive    = True

    # ── state vector for DQN ──────────────────────
    def get_state(self, navigator_pos, neighbors, obstacles):
        rel_nav = navigator_pos - self.pos
        rel_nav_n = rel_nav / (np.linalg.norm(rel_nav) + 1e-6)

        # nearest obstacle direction
        if obstacles:
            dists = [np.linalg.norm(o - self.pos) for o in obstacles]
            nearest_o = obstacles[np.argmin(dists)]
            obs_dir = (nearest_o - self.pos)
            obs_dir = obs_dir / (np.linalg.norm(obs_dir) + 1e-6)
        else:
            obs_dir = np.zeros(3)

        # nearest neighbour direction
        if neighbors:
            nbr_dirs = [(n.pos - self.pos) for n in neighbors if n.id != self.id]
            if nbr_dirs:
                nd = nbr_dirs[0] / (np.linalg.norm(nbr_dirs[0]) + 1e-6)
            else:
                nd = np.zeros(3)
        else:
            nd = np.zeros(3)

        state = np.concatenate([
            self.pos / 50.0,        # normalised position
            self.vel / MAX_SPEED,   # normalised velocity
            rel_nav_n,
            obs_dir,
            nd
        ])
        return state.astype(np.float32)

    # ── Olfati-Saber alpha term (cohesion + separation) ──
    def compute_px(self, neighbors):
        force = np.zeros(3)
        for n in neighbors:
            if n.id == self.id:
                continue
            diff = n.pos - self.pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6 or dist > OMEGA:
                continue
            # gradient of Vα
            s_norm = sigma_norm(diff)
            phi    = phi_alpha(dist, BETA, OMEGA)
            grad   = grad_sigma(diff)
            force += phi * grad                          # attraction/repulsion
            # velocity alignment
            bump = bump_function(dist / OMEGA)
            force += MU * bump * (n.vel - self.vel)
        return EX * force

    # ── Beta term (obstacle avoidance) ───────────────────
    def compute_py(self, obstacles):
        force = np.zeros(3)
        for obs_pos in obstacles:
            diff = obs_pos - self.pos
            dist = np.linalg.norm(diff)
            if dist > DETECT_DIST or dist < 1e-6:
                continue
            # virtual beta agent on obstacle surface
            beta_pos = self.pos + (diff / dist) * (dist - 1.0)
            beta_diff = beta_pos - self.pos
            beta_dist = np.linalg.norm(beta_diff) + 1e-6
            bump  = bump_function(beta_dist / DETECT_DIST)
            grad  = grad_sigma(beta_diff)
            force -= EY * bump * grad     # repel
        return force

    # ── Gamma / Pz term (navigator tracking) ─────────────
    def compute_pz(self, nav_pos, nav_vel):
        dp = self.pos - nav_pos
        dv = self.vel - nav_vel
        return -(EZ * grad_sigma(dp) + EZ * dv)

    # ── Main update step ──────────────────────────────────
    def update(self, navigator, neighbors, obstacles, target_pos):
        nav_pos = navigator.pos
        nav_vel = navigator.vel

        # ── Build state & pick DQN action ─────────────────
        state = self.get_state(nav_pos, neighbors, obstacles)

        if self.prev_state is not None:
            # reward: approach target, avoid obs, stay in cluster
            d_target  = np.linalg.norm(self.pos - target_pos)
            d_obs_min = min([np.linalg.norm(self.pos - o) for o in obstacles], default=99)
            d_nav     = np.linalg.norm(self.pos - nav_pos)
            reward = -0.1 * d_target + 0.5 * min(d_obs_min, 5.0) - 0.05 * d_nav
            self.total_reward += reward
            done = d_target < 1.5
            self.dqn.remember(self.prev_state, self.prev_action, reward, state, done)
            self.dqn.replay(batch=32)

        action_idx = self.dqn.act(state)
        action_dir = ACTIONS[action_idx]

        # ── Physics forces ─────────────────────────────────
        px = self.compute_px(neighbors)
        py = self.compute_py(obstacles)
        pz = self.compute_pz(nav_pos, nav_vel)

        # DQN contributes a small steering bias
        dqn_force = action_dir * 1.5

        accel = px + py + pz + dqn_force

        # ── Integrate ──────────────────────────────────────
        self.vel += accel * DT
        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = self.vel / speed * MAX_SPEED
        if speed < MIN_SPEED and speed > 0:
            self.vel = self.vel / speed * MIN_SPEED

        self.pos += self.vel * DT
        # Keep altitude positive
        self.pos[2] = max(self.pos[2], 0.5)

        self.trail.append(self.pos.copy())
        if len(self.trail) > 300:
            self.trail.pop(0)

        self.prev_state  = state
        self.prev_action = action_idx
