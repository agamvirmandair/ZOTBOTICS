"""Essentialy, this is like the main.py or main file
for the PPO Agent, we will basically call all of our functions and classes from here,
We will puzle the pieces together here, and this is where we will train our agent.
This will be in charge of steps 6 and onwards. """

import torch
import numpy as np
from Actor import Actor
from critic import Critic
from buffer import Buffer
from episode import episode

class PPOAgent:
    def __init__(self, environment, gamma=0.99, clip_ratio=0.2, epochs=4, batch_size=64, episodes_per_update=10):
        self.environment = environment
        self.actor = Actor()
        self.critic = Critic()
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.episodes_per_update = episodes_per_update
        # Use your enhanced buffer with all features
        self.buffer = Buffer(gamma=gamma, batch_size=batch_size)

    def collect_experience(self):
        """Collect experience using your modular episode function"""
        total_rewards = []
        
        for _ in range(self.episodes_per_update):
            episode_reward = episode(self.environment, self.actor, self.critic, self.buffer)
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)

    def update_policy(self):
        """PPO policy update with clipping using your buffer's features"""
        # Get all collected data
        states, actions, old_log_probs, advantages, returns = self.buffer.get_all_data()
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of optimization (PPO's key feature)
        for epoch in range(self.epochs):
            #Use mini-batches (your batch_size feature)
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in self.buffer.get_batches():
                
                # Update Actor (separate forward pass)
                dist = self.actor(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1) if batch_actions.dim() > 1 else dist.log_prob(batch_actions)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                actor_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                self.actor.update(actor_loss)
                
                # Update Critic (separate forward pass)
                values = self.critic(batch_states).squeeze()
                critic_loss = torch.mean((batch_returns - values) ** 2)
                
                self.critic.update(critic_loss)


    def train(self, num_updates=1000):
        """Train the PPO agent using your modular design"""
        for update in range(num_updates):
            # Collect experience using your episode function
            avg_episode_reward = self.collect_experience()
            
            # Update policy with PPO clipping
            self.update_policy()
            
            # Log progress
            if update % 10 == 0:
                print(f"Update {update}, Avg Episode Reward: {avg_episode_reward:.2f}")
            
            # Clear buffer for next collection (your buffer's feature)
            self.buffer.clear()
        
        return self.actor, self.critic