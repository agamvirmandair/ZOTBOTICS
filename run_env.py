"""
Training loop for SpdrBot using the PPO agent.

The env runs one step at a time. When an episode ends (robot fell, reached
the goal, or hit the MAX_STEPS time limit), the agent learns from everything
it experienced, then a new episode starts automatically.
"""

import os
import numpy as np
from spider_env_new import SpiderEnv
from agent import Agent

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 1_000_000   # how many env steps to train for in total
SAVE_INTERVAL = 10          # save model weights every N episodes
LOG_FILE      = "reward_log.txt"
train = False # set to True to train, False to just run with saved weights (if they exist)


# ── Setup ─────────────────────────────────────────────────────────────────────
env   = SpiderEnv(render_mode="human")
obs, _= env.reset()

agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0])
if not train:
    agent.load_models()   # loads saved weights if they exist, otherwise starts fresh

# ── Training loop ─────────────────────────────────────────────────────────────
episode        = 0
episode_reward = 0.0
episode_length = 0
reward_history = []
length_history = []
collision_history = []

for step in range(TOTAL_STEPS):

    action, raw_action, log_prob, value = agent.choose_action(obs)
    current_obs = obs   # save obs_t before stepping — log_prob/value were computed here
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    episode_length += 1
    done = terminated or truncated

    # store current_obs (obs_t) so learn() evaluates new policy on the same
    # observation that produced log_prob and value — fixes obs_t vs obs_t+1 mismatch
    agent.remember(current_obs, raw_action, log_prob, value, reward, done)

    if done:
        episode    += 1
        end_reason  = info.get("end_reason", "?")
        per_step    = episode_reward / episode_length

        reward_history.append(episode_reward)
        length_history.append(episode_length)
        collision_history.append(1 if end_reason == "collision" else 0)

        mean_reward    = np.mean(reward_history[-100:])
        mean_length    = np.mean(length_history[-100:])
        collision_rate = np.mean(collision_history[-100:]) * 100.0

        actor_loss, critic_loss = agent.learn()

        with open(LOG_FILE, "a") as f:
            f.write(
                f"ep={episode}"
                f"  total={episode_reward:.2f}"
                f"  per_step={per_step:.3f}"
                f"  steps={episode_length}"
                f"  end={end_reason}"
                f"  mean100={mean_reward:.2f}"
                f"  mean_len={mean_length:.0f}"
                f"  coll%={collision_rate:.1f}"
                f"  a_loss={actor_loss:.4f}"
                f"  c_loss={critic_loss:.4f}"
                f"\n"
            )

        if episode % SAVE_INTERVAL == 0:
            print(
                f"ep {episode:>5} | step {step:>7} | "
                f"total {episode_reward:>8.2f} | steps {episode_length:>5} | "
                f"end={end_reason:<9} | mean100 {mean_reward:>8.2f} | "
                f"coll% {collision_rate:>5.1f} | "
                f"a_loss {actor_loss:.4f} | c_loss {critic_loss:.4f}"
            )
            agent.save_models()

        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

env.close()
