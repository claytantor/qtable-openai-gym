import sys, os
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# Create the environment. Make can take a number of values including:
# MountainCar-v0, CartPole-v0, MsPacman-v0, or Hopper-v0, BeamRider-v0, Berzerk-v0
env = gym.make('Berzerk-v0')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.get_device_name(0), torch.cuda.is_available())

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


# Reset the environment before begining
env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action_random(state):
    action = env.action_space.sample() # take a random action
    return action

def select_action_drl(state):
    # n_actions_b = env.action_space.n
    # print(n_actions_b)

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # print(device)
        if(str(device)=='cuda'):
            # print('cuda', n_actions_b)
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long).cuda()
        
        if(str(device)=='cpu'):
            # print('cpu select actions', n_actions_b)        
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long).cpu()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # print(target_net(non_final_next_states).max(1)[0], str(device)=='cuda')
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    if(str(device)=='cuda'):
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach().cuda()
    
    if(str(device)=='cpu'):      
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()


    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

episode_durations = []
episode_scores = []

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_scores():
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    plt.plot(moving_average(episode_scores,6))
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())      

def loop_test1(argv):
    try:
        print("starting qlearning app.")
        num_episodes = 10000
        for i_episode in range(num_episodes):
            env.reset()


            last_screen = get_screen()
            current_screen = get_screen()
            state = current_screen - last_screen
            score = 0
            for t in count():
                env.render()
                # Select and perform an action
                action = select_action_drl(state)
                # print("action", action)

                _, reward, done, _ = env.step(action.item())

                reward = torch.tensor([reward], device=device)
                score += reward.data[0]
                
                # Observe new state
                last_screen = current_screen
                current_screen = get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # # Perform one step of the optimization (on the target network)
                optimize_model()
                if done:
                    # episode_durations.append(t + 1)
                    episode_scores.append(score)
                    print("episode:{} score: {} time: {}".format(i_episode, score, t))
                    # plot_durations()
                    # plot_scores()
                    break
                # else:
                #     plt.figure(3)
                #     plt.imshow(last_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
                #             interpolation='none')
                #     plt.title('extracted screen')
                #     # plt.show()

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
    


    except KeyboardInterrupt:
        print('exit')

    print('Complete')

# def loop_random(argv):

#     # training loop
#     score = 0

#     # Requires 100 episodes for evaluation
#     for i_episodes in range(100):
#         # Reset the environment for the current episode
#         observation = env.reset()
#         # Set up a loop to perform 1000 steps
#         for t in range(100):
#             env.render()
#             # Step returns 4 values: observation (object)
#             #            reward (float)
#             #			 done (boolean)
#             # 			 info (dictionary)
#             # print(observation)
#             action = env.action_space.sample() # take a random action
    
#             print("action: {}".format(action))
#             observation, reward, done, info = env.step(action)
#             score += reward
#             print(score)
#             if done:
#                 print("Episode finished after {} timepsteps. score: {}".format(t+1, score))
#                 break



def main(argv):   
    loop_test1(argv)
    #loop_random(argv)



if __name__ == "__main__":
    main(sys.argv[1:])
