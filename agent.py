import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
import gymnasium as gym

class PPOMemory:
    """
    stores data collected from algorithm until there is enough to make an 
    update/take an action.
    
    """
    def __init__(self, mem_size, state_dim, action_dim, batch_size):
        self.mem_size = mem_size
        self.batch_size = batch_size
        # pre-allocating everything as float32 numpy arrays
        self.states = np.zeros((mem_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((mem_size, action_dim), dtype=np.float32)
        self.probs = np.zeros((mem_size), dtype=np.float32)
        self.vals = np.zeros((mem_size), dtype=np.float32)
        self.rewards = np.zeros((mem_size), dtype=np.float32)
        self.dones = np.zeros((mem_size), dtype=np.float32)
        # using a pointer to tell computer which row in the numpy arrays 
        # should recieve the next piece of data
        # replaces append from lists
        self.ptr = 0

    def generate_batches(self):
        """
        breaks the full data into smaller chunks in order to make 
        it easier for the algorithm to determine next action.
        
        """
        n_states = self.ptr  # only use data up to where pointer stopped to prevent training on empty zeroes
        indices = np.arange(n_states)  # creates numpy array with elements 0...n_states
        np.random.shuffle(indices) # randomize the memories, to avoid memorization

        batches = [indices[i:i + self.batch_size] for i in range(0, n_states, self.batch_size)]

        # returning batched data and numpy arrays containing our memory data
        return self.states[:self.ptr], \
            self.actions[:self.ptr], \
            self.probs[:self.ptr], \
            self.vals[:self.ptr], \
            self.rewards[:self.ptr], \
            self.dones[:self.ptr], \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        stores memory at the location of current numpy array row

        """
        index = self.ptr
        self.states[index] = state
        self.actions[index] = action
        self.probs[index] = probs
        self.vals[index] = vals
        self.rewards[index] = reward
        self.dones[index] = done

        self.ptr = (self.ptr + 1) % self.mem_size

    def clear_memory(self):
        """
        resets pointer, with old data remaining in arrays.
        any new steps after this will just over-write the old data. 
        
        """
        self.ptr = 0
    

class Actor(nn.Module):
    """
    neural network that takes action based on the advantages each potential action could give.
    
    """
    def __init__(self, n_inputs=372, n_actions=12, alpha=0.0003, chkpt_dir='models'):
        """
        - actor network is implemented as a feed-forward
        - (sequential) neural network
        - 1st layer: maps input to 256 "features" to find patterns
        - 2nd layer: combines the features to understand more
        - 3rd layer: output layer, maps the 256 features to mean target positions for each servo
        - Tanh: keeps continous values between -1 and 1
        - ReLU: allows for non-linear boundaries by giving negative signals a zero value
        """
        super(Actor, self).__init__()
        # stores the neural network weights
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo.pth')
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        # tanh used here because servos have physical limits
        self.actor = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        )
        # nn.Softmax(dim=-1) - for discrete actions?

        # defines learning algorithm to update weights
        # using Adam which auto-adjusts learning rate 
        # for each individual weight
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # detects GPU and sets device to it if available
        # else falls back to CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        given actor's best guess of action based on the state and the exploration factor, create a probability Normal distribution where the center is mu
        defines data flow of neural network/how data is transformed by the layers defined in __init__
        
        """
        mu = self.actor(state)
        sigma = torch.exp(self.log_std)
        dist = Normal(mu, sigma)
        return dist
    
    def save_checkpoint(self):
        """
        saves the information in the state dictionary provided via inheriting from nn.Module into a checkpoint file
        """
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """
        loads state dictionary from the checkpoint file
        
        """
        self.load_state_dict(torch.load(self.checkpoint_file))



class Critic(nn.Module):
    """
    neural network that takes the input and predicts a single number -> state value. this is an estimate of the total amount of reward the robot is expected to collect from its current position until end of simulation.

    """
    def __init__(self, n_inputs=372, alpha=0.0003, chkpt_dir='models'):
        super(Critic, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo.pth')
        # no Tanh since we want linear output
        self.critic = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        """
        saves the information in the state dictionary provided via inheriting from nn.Module into a checkpoint file
        """
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """
        loads state dictionary from the checkpoint file
        
        """
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_inputs=372, n_actions=12, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.horizon = N
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = Actor(n_inputs, n_actions, alpha)
        self.critic = Critic(n_inputs, alpha)
        # n_inputs = state_dims
        self.memory = PPOMemory(mem_size=self.horizon, state_dim=n_inputs, action_dim=n_actions, batch_size=self.batch_size)
    
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    
    def save_models(self):
        print("...saving models...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print("...loading models...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # convert our observation to tensor and move to GPU
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        
        # get actor probability distribution and critic state value
        dist = self.actor(state)
        value = self.critic(state)
        # sample the action (12 servo angles)
        action = dist.sample()

        # sum log_probs across 12 actions to treat 12 servo moves as one joint step
        probs = torch.sum(dist.log_prob(action), dim=1)
        # extract scalar values memory 
        probs_item = probs.item()
        value_item = torch.squeeze(value).item()
        
        # prepare action for robot by squeezing to numpy array, removes batch dimension, moves tp cpu and converts.
        action_numpy = torch.squeeze(action).cpu().detach().numpy()

        return action_numpy, probs_item, value_item
    
    def learn(self):

        # Get memory
        state_arr, action_arr, old_probs_arr, vals_arr, \
            reward_arr, done_arr, batches = \
            self.memory.generate_batches()

        values = vals_arr
        n_steps = len(reward_arr)

        # ---------------------
        # GAE Advantage Compute
        # ---------------------
        advantage = np.zeros(n_steps, dtype=np.float32)
        last_gae_lam = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = reward_arr[t] + self.gamma * next_value * (1 - int(done_arr[t])) - values[t]

            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - int(done_arr[t])) * last_gae_lam
            advantage[t] = last_gae_lam

        # Normalize advantages (VERY important)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        advantage = torch.tensor(advantage, dtype=torch.float32).to(self.actor.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.actor.device)

        # ---------------------
        # PPO Updates
        # ---------------------
        for epoch in range(self.n_epochs):

            for batch in batches:

                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch], dtype=torch.float32).to(self.actor.device)

                # ---- Critic forward ----
                critic_value = self.critic(states).squeeze()

                # ---- Actor forward ----
                dist = self.actor(states)
                new_probs = dist.log_prob(actions).sum(dim=-1)

                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio,
                    1 - self.policy_clip,
                    1 + self.policy_clip
                ) * advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # ---- Critic loss ----
                returns = advantage[batch] + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                # ---- Backprop ----
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

            print(
                f"Epoch {epoch} | "
                f"Actor Loss: {actor_loss.item():.4f} | "
                f"Critic Loss: {critic_loss.item():.4f}"
            )

        self.memory.clear_memory()

def run_test_env():
    """
    Local Tester Function: Runs Pendulum-v1 on your PC.
    Use this to verify PPO math before moving to the robot.
    """
    env = gym.make('Pendulum-v1', render_mode='human')
    input_dims = env.observation_space.shape[0] # [3]
    n_actions = env.action_space.shape[0]    # 1
    
    agent = Agent(n_inputs=input_dims, n_actions=n_actions, alpha=0.0003)
    
    # Simple training loop for the pendulum
    for i in range(100):
        observation, _ = env.reset()
        score = 0
        done = False
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, prob, val, reward, done)
            
            if len(agent.memory) >= 2048:
                agent.learn()
            
            observation = observation_
            score += reward
        print(f"Test Episode {i} | Score: {score:.2f}")




def run_spider_bot():
    """
    Real Deployment Function: Runs on the Jetson Nano.
    Connects to the 372 sensors and 12 servos.
    """
    # env = MyRobotEnv() 
    # agent = Agent(input_dims=[372], n_actions=12...)
    pass




        







