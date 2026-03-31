import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


"""
Memory function for PPO that stores
1) state
2) reward
3) action
4) log probability of action as per policy
5) value as per critic
"""
class PPOMemory:
    def __init__(self, batch_size):
        self.rewards = []
        self.states = []
        self.actions = []
        self.values = []
        self.probs = []   # log probability of action under old policy
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.values), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, reward, value, prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(prob)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []


"""
Actor Network with 2 hidden layers with 256 neurons.
REMOVED softmax.
"""
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_2')

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)   # outputs mean vector (size 12)
        )

        # learned log_std parameter for each action dimension
        self.log_std = nn.Parameter(T.zeros(n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mean = self.actor(state)
        std = T.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo_2')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10, entropy_coef=0.01):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        # Entropy bonus — prevents the policy collapsing to a single deterministic
        # action (e.g. always turning left).  Adds -entropy_coef * H(π) to actor
        # loss, keeping the action distribution spread out during early training.
        self.entropy_coef = entropy_coef

        # accept either an int or a tuple/list for input_dims
        if isinstance(input_dims, int):
            input_dims = [input_dims]

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def save_memory(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, reward, vals, probs, done)

    def remember(self, state, action, probs, vals, reward, done):
        self.save_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float32).to(self.actor.device)

        dist  = self.actor(state)
        value = self.critic(state)

        raw_action = dist.sample()

        # log_prob in pre-tanh space — must stay consistent with learn()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # squash to (-1, 1) for the environment
        action = T.tanh(raw_action)

        # raw_action goes to memory; tanh action goes to env
        raw_action_np = T.squeeze(raw_action).cpu().detach().numpy()
        action_np     = T.squeeze(action).cpu().detach().numpy()
        log_prob      = T.squeeze(log_prob).item()
        value         = T.squeeze(value).item()

        return action_np, raw_action_np, log_prob, value

    def learn(self):
        # FIX: Pull all data once — GAE is computed once, not per epoch
        state_arr, action_arr, old_prob_arr, vals_arr, \
        reward_arr, dones_arr, _ = self.memory.generate_batches()
        # _ discards the pre-shuffled batches; we reshuffle each epoch below

        N = len(reward_arr)

        # FIX: O(N) backward GAE replacing the previous O(N²) nested loop.
        # Iterates from last timestep to first, accumulating the discounted
        # TD-error (delta) into the advantage estimate.
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        # A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        advantage = np.zeros(N, dtype=np.float32)
        last_adv  = 0.0
        for t in reversed(range(N)):
            next_non_terminal = 1.0 - float(dones_arr[t])
            next_value        = float(vals_arr[t + 1]) if t < N - 1 else 0.0
            delta             = (reward_arr[t]
                                 + self.gamma * next_value * next_non_terminal
                                 - vals_arr[t])
            last_adv          = delta + self.gamma * self.gae_lambda * next_non_terminal * last_adv
            advantage[t]      = last_adv

        # FIX: Compute returns BEFORE normalizing advantage.
        # returns = A + V (the target for the critic).
        # Must use the raw advantage here — normalizing first would shift the target.
        returns = advantage + vals_arr  # shape (N,)

        # FIX: Normalize advantages for stable actor updates.
        # FIX: epsilon = 1e-8 prevents division by zero.
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # FIX: Convert to float32 tensors on the correct device once,
        # so every epoch's batch indexing is cheap and type-safe.
        advantage_t = T.tensor(advantage, dtype=T.float32).to(self.actor.device)
        returns_t   = T.tensor(returns,   dtype=T.float32).to(self.actor.device)

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        n_batches         = 0

        for _ in range(self.n_epochs):
            # FIX: Re-shuffle indices each epoch so every epoch sees a
            # different mini-batch composition (standard PPO practice).
            indices = np.arange(N, dtype=np.int64)
            np.random.shuffle(indices)
            batches = [indices[i:i + self.memory.batch_size]
                       for i in range(0, N, self.memory.batch_size)]

            for batch in batches:
                # FIX: explicit float32 dtype on all tensors for consistency
                states    = T.tensor(state_arr[batch],    dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device)
                # FIX: actions are RAW (pre-tanh) values stored in memory —
                # consistent with how log_prob was computed in choose_action()
                actions   = T.tensor(action_arr[batch],   dtype=T.float32).to(self.actor.device)

                dist         = self.actor(states)
                critic_value = self.critic(states).squeeze()

                # FIX: log_prob of RAW actions — same space as old_probs
                new_probs = dist.log_prob(actions).sum(dim=-1)

                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantage_t[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio,
                    1 - self.policy_clip,
                    1 + self.policy_clip
                ) * advantage_t[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Entropy bonus: subtract entropy from actor loss so maximising
                # the loss also maximises entropy — keeps action distribution
                # spread out and prevents collapsing to always-left / always-right.
                entropy    = dist.entropy().sum(dim=-1).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                # FIX: Critic loss uses returns derived from UNNORMALIZED advantage.
                # returns_t was computed before advantage normalization above.
                critic_loss = ((returns_t[batch] - critic_value) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # FIX: clip gradients to prevent large critic updates from
                # destabilising training when returns have high variance
                nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_batches         += 1

        self.memory.clear_memory()
        avg_actor_loss  = total_actor_loss  / max(n_batches, 1)
        avg_critic_loss = total_critic_loss / max(n_batches, 1)
        return avg_actor_loss, avg_critic_loss
