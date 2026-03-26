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
train = True

# ── Setup ─────────────────────────────────────────────────────────────────────
env   = SpiderEnv(render_mode="human")
obs, _= env.reset()

agent = Agent(n_inputs=env.observation_space.shape[0],
              n_actions=env.action_space.shape[0])
if not train:
    agent.load_models()   # loads saved weights if they exist, otherwise starts fresh

# ── Training loop ─────────────────────────────────────────────────────────────
episode        = 0
episode_reward = 0.0
reward_history = []

for step in range(TOTAL_STEPS):

    # Agent picks an action based on the current observation
    action, log_prob, value = agent.choose_action(obs)

    # Environment executes the action and returns the new state
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    done = terminated or truncated

    # Store this transition in the agent's memory
    agent.remember(obs, action, log_prob, value, reward, done)

    if done:
        episode += 1
        reward_history.append(episode_reward)
        mean_reward = np.mean(reward_history[-100:])  # rolling average over last 100 eps

        # Log to file
        with open(LOG_FILE, "a") as f:
            f.write(f"Episode {episode}: reward={episode_reward:.2f}  mean={mean_reward:.2f}\n")

        # Learn from the episode
        agent.learn()

        # Print progress and save every SAVE_INTERVAL episodes
        if episode % SAVE_INTERVAL == 0:
            print(f"Episode {episode:>5} | step {step:>7} | "
                  f"reward {episode_reward:>8.2f} | mean(100) {mean_reward:>8.2f}")
            agent.save_models()

        # Start a new episode
        obs, _ = env.reset()
        episode_reward = 0.0

env.close()
