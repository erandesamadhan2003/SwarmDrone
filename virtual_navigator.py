"""
virtual_navigator.py
Virtual Navigator — the 'leader' that the swarm follows.
Implements a PPO-inspired policy (lightweight, numpy-only):
  - LSTM state memory simulated via exponential smoothing
  - Actor outputs velocity direction toward goal
  - Reward: approach target + avoid obstacles + keep cluster close
  - Gradient clipping (threshold 0.2 as in the paper)
"""

import numpy as np

# PPO hyper-parameters (from paper §Enhanced model construction)
LR_PPO          = 0.0003
GAMMA           = 0.99
CLIP_EPSILON    = 0.2       # gradient clipping threshold
NAV_SPEED_MIN   = 1.0
NAV_SPEED_MAX   = 4.0
NAV_STATE_DIM   = 12        # [pos(3), vel(3), to_target(3), nearest_obs(3)]
NAV_N_ACTIONS   = 26


class PPONavigator:
    """
    Lightweight PPO Actor-Critic navigator.
    Uses a linear approximation + LSTM-like hidden state (EMA).
    """

    def __init__(self, start_pos, target_pos):
        self.pos     = np.array(start_pos, dtype=float)
        self.vel     = np.zeros(3)
        self.target  = np.array(target_pos, dtype=float)
        self.hidden  = np.zeros(NAV_STATE_DIM)   # EMA "LSTM" state

        # Actor weights: state → action logits
        np.random.seed(0)
        self.W_actor  = np.random.randn(NAV_STATE_DIM, NAV_N_ACTIONS) * 0.01
        self.W_critic = np.random.randn(NAV_STATE_DIM, 1) * 0.01

        self.log_std = np.ones(3) * np.log(0.5)   # variance σ²=0.05I as in paper
        self.memory  = []
        self.step    = 0

        # Build action table (same as drone_agent)
        acts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if not (dx == dy == dz == 0):
                        v = np.array([dx, dy, dz], dtype=float)
                        acts.append(v / np.linalg.norm(v))
        self.ACTIONS  = np.array(acts)
        self.trail    = [self.pos.copy()]

    # ── State builder ──────────────────────────────────────
    def get_state(self, obstacles, cluster_center):
        to_target = (self.target - self.pos)
        to_target_n = to_target / (np.linalg.norm(to_target) + 1e-6)

        if obstacles:
            dists = [np.linalg.norm(o - self.pos) for o in obstacles]
            nearest = obstacles[np.argmin(dists)]
            obs_vec = (self.pos - nearest)
            obs_vec = obs_vec / (np.linalg.norm(obs_vec) + 1e-6)
        else:
            obs_vec = np.zeros(3)

        raw = np.concatenate([
            self.pos / 50.0,
            self.vel / NAV_SPEED_MAX,
            to_target_n,
            obs_vec
        ]).astype(np.float32)

        # EMA hidden state (simulates LSTM memory)
        self.hidden = 0.8 * self.hidden + 0.2 * raw
        return self.hidden.copy()

    # ── PPO Actor: softmax policy ──────────────────────────
    def _policy(self, state):
        logits = state @ self.W_actor
        logits -= logits.max()            # numerical stability
        probs  = np.exp(logits)
        probs /= probs.sum()
        return probs

    def act(self, state):
        probs = self._policy(state)
        action_idx = np.random.choice(len(self.ACTIONS), p=probs)
        return action_idx, probs[action_idx]

    # ── Critic: value estimate ─────────────────────────────
    def value(self, state):
        return float(np.dot(state, self.W_critic.squeeze()))

    # ── PPO update (clip ratio) ────────────────────────────
    def update(self, states, actions, old_probs, rewards, dones):
        """Mini-batch PPO update with gradient clipping."""
        if len(states) < 4:
            return

        # Compute discounted returns
        G = []
        g = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + GAMMA * g * (1 - d)
            G.insert(0, g)
        G = np.array(G)
        G = (G - G.mean()) / (G.std() + 1e-8)

        for s, a, op, g_val in zip(states, actions, old_probs, G):
            probs = self._policy(s)
            new_prob = probs[a]
            ratio = new_prob / (op + 1e-8)

            # Clipped surrogate loss
            clipped_ratio = np.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            advantage = g_val - self.value(s)

            # Actor gradient (simple policy gradient step)
            pg = min(ratio, clipped_ratio) * advantage
            grad_actor = np.outer(s, np.eye(NAV_N_ACTIONS)[a]) * (-pg)
            grad_actor = np.clip(grad_actor, -CLIP_EPSILON, CLIP_EPSILON)
            self.W_actor -= LR_PPO * grad_actor

            # Critic gradient
            v_error = self.value(s) - g_val
            grad_critic = s.reshape(-1, 1) * v_error
            grad_critic = np.clip(grad_critic, -CLIP_EPSILON, CLIP_EPSILON)
            self.W_critic -= LR_PPO * grad_critic

    # ── Main step ─────────────────────────────────────────
    def step_nav(self, obstacles, cluster_center, dt=0.05):
        state = self.get_state(obstacles, cluster_center)

        # Obstacle repulsion
        obs_force = np.zeros(3)
        for obs in obstacles:
            diff = self.pos - obs
            dist = np.linalg.norm(diff)
            if 0 < dist < 8.0:
                obs_force += (diff / dist) * (8.0 / (dist + 0.1)) * 0.5

        # Strong pull toward target
        to_target = self.target - self.pos
        dist_to_target = np.linalg.norm(to_target)
        target_force = (to_target / (dist_to_target + 1e-6)) * 3.0

        # PPO action bias
        action_idx, prob = self.act(state)
        action_dir = self.ACTIONS[action_idx]

        total_force = target_force + obs_force + action_dir * 1.0
        self.vel += total_force * dt
        speed = np.linalg.norm(self.vel)
        if speed > NAV_SPEED_MAX:
            self.vel = self.vel / speed * NAV_SPEED_MAX
        if speed < NAV_SPEED_MIN and speed > 1e-6:
            self.vel = self.vel / speed * NAV_SPEED_MIN

        self.pos += self.vel * dt
        self.pos[2] = max(self.pos[2], 0.5)

        # Reward for PPO memory
        d_target  = np.linalg.norm(self.pos - self.target)
        d_obs_min = min([np.linalg.norm(self.pos - o) for o in obstacles], default=99)
        d_cluster = np.linalg.norm(self.pos - cluster_center)
        reward = -0.2 * d_target + 0.3 * min(d_obs_min, 8.0) - 0.05 * d_cluster

        done = d_target < 1.5
        self.memory.append((state, action_idx, prob, reward, done))

        # PPO batch update every 64 steps
        if len(self.memory) >= 64:
            ss, aa, pp, rr, dd = zip(*self.memory)
            self.update(list(ss), list(aa), list(pp), list(rr), list(dd))
            self.memory.clear()

        self.trail.append(self.pos.copy())
        if len(self.trail) > 300:
            self.trail.pop(0)

        self.step += 1
        return done
