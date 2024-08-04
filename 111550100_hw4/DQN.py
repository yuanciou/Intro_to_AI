import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import os
from tqdm import tqdm
total_rewards = []


class replay_buffer():
    '''
    A deque storing trajectories
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.
        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.
        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''
    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.        
        Parameter:
            states: a batch size of states
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.95, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done this for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            None (Don't need to return anything)
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Begin your code
        # TODO
        """
        Sample trajectories of batch size from the replay buffer.
        1. Use sample() to get the data, and change these data to tensor.
        
        Forward the data to the evaluate net and the target net.
        1. 'q_value' is the predicted value from evaluate net.
        2. 'next_q_values' is the actual value from target net.
        3. 'target_q_values' is the expected value from the formula: "reward + gamma * max(next_q_values)"

        Compute the loss with MSE.
        1. Use nn.MSELoss() to compute the loss of 'q_values' and 'target_q_values'

        Zero-out the gradients, Backpropagation, and Optimize the loss function.
        1. Use zero_grad(), backward(), and self.optimizer.step() to finish the requirement.
        """
        observations, actions, rewards, next_observations, done = self.buffer.sample(self.batch_size)
        observations = torch.tensor(np.array(observations), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        q_values = self.evaluate_net(observations).gather(1, actions)
        next_q_values = self.target_net(next_observations).detach() * (~done).unsqueeze(-1)
        target_q_values = rewards.unsqueeze(-1) + self.gamma * torch.max(next_q_values, dim=1)[0].unsqueeze(1)

        loss_function = nn.MSELoss()
        loss = loss_function(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # End your code
        torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            action: the chosen action.
        """
        with torch.no_grad():
            # Begin your code
            # TODO
            """
            1. Randomly generate number in the interval of [0, 1]. 
            2. If the number is smaller than epsilon, 
               choose the action with maximum value of the given state in Q table.
            3. Otherwise, choose the action which is randomly chosen in the possible action.
            """
            tmp_epsilon = np.random.uniform(0, 1)
            if tmp_epsilon < self.epsilon:
                action = torch.argmax(self.evaluate_net(torch.tensor(state, dtype=torch.float))).item()
            else:
                action = env.action_space.sample()
            # End your code
        return action

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state        
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        # TODO
        """
        1. Use self.env.reset() to define the initial state.
        2. Forward the tensor, which is convert by initial state
           in the target neural network.
        3. Return the maximum q value.
        """
        initial_state = self.env.reset()
        q_values = self.target_net(torch.FloatTensor(initial_state))
        max_q = torch.max(q_values).item()
        return max_q
        # End your code
    


def train(env):
    """
    Train the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    agent = Agent(env)
    episode = 1000
    rewards = []
    for _ in tqdm(range(episode)):
        state = env.reset()
        count = 0
        while True:
            count += 1
            agent.count += 1
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.insert(state, int(action), reward, next_state, int(done))
            
            if len(agent.buffer) >= 1000:
                agent.learn()
            if done:
                rewards.append(count)
                break
            state = next_state
    total_rewards.append(rewards)


def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(100):
        state = env.reset()
        count = 0
        while True:
            # env.render() #Uncomment for visualizing the process
            count += 1
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, _, done, _ = env.step(action)
            if done:
                rewards.append(count)
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


if __name__ == "__main__":
    env = gym.make('CartPole-v0')        
    os.makedirs("./Tables", exist_ok=True)

    # training section:
    for i in range(5):
        print(f"#{i + 1} training progress")
        train(env)
        
    # testing section:
    test(env)
    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))