from typing import Tuple
import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

import conf
from controller import Agent
from model import CmdRecogNetwork
import train, test, evaluation
from dataloader import training_dataloader, testing_dataloader, validation_dataloader


class PolicyGradient:
    def __init__(self):

        self.EPOCH_NUM = conf.RL_EPOCH_NUM
        self.ALPHA = conf.RL_ALPHA
        # number of models to generate for each action
        self.BATCH_SIZE = conf.RL_BATCH_SIZE
        self.HIDDEN_SIZE = conf.RL_HIDDEN_SIZE
        self.BETA = conf.RL_BETA
        self.GAMMA = conf.RL_GAMMA
        self.INPUT_SIZE = conf.RL_INPUT_SIZE
        self.NUM_STEPS = conf.RL_NUM_STEPS
        self.ACTION_SPACE = conf.RL_ACTION_SPACE
        self.MODEL_PATH = './agent_ckpt'
        if not os.path.exists(self.MODEL_PATH): os.makedirs(self.MODEL_PATH)

        # instantiate the tensorboard writer
        self.writer = SummaryWriter(comment=f'_PG_CP_Gamma={self.GAMMA},'
                                            f'LR={self.ALPHA},'
                                            f'BS={self.BATCH_SIZE},'
                                            f'NH={self.HIDDEN_SIZE},'
                                            f'BETA={self.BETA}')

        # the agent driven by a neural network architecture
        self.agent = Agent(
            input_size=self.INPUT_SIZE, 
            hidden_size=self.HIDDEN_SIZE,
            num_steps=self.NUM_STEPS
        ).to(conf.device)
        self.optim = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        self.total_rewards = deque([], maxlen=100)

    def solve_environment(self):
        """
            The main interface for the Policy Gradient solver
        """
        # init the episode and the epoch
        epoch = 0

        for epoch in range(self.EPOCH_NUM):
            # init the epoch arrays
            # used for entropy calculation
            epoch_logits = torch.empty(size=(0, self.ACTION_SPACE)) \
                .to(device=conf.device)
            epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float) \
                .to(device=conf.device)

            # Sample BATCH_SIZE models and do average
            for i in range(self.BATCH_SIZE):
                # play an episode of the environment
                (episode_weighted_log_prob_trajectory,
                 episode_logits,
                 sum_of_episode_rewards) = self.play_episode()

                # after each episode append the sum of total rewards to the deque
                self.total_rewards.append(sum_of_episode_rewards)

                # append the weighted log-probabilities of actions
                epoch_weighted_log_probs = torch.cat(
                    (epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                    dim=0
                )
                # append the logits - needed for the entropy bonus calculation
                epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # calculate the loss
            loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                                weighted_log_probs=epoch_weighted_log_probs)

            # zero the gradient
            self.optim.zero_grad()

            # backprop
            loss.backward()

            # update the parameters
            self.optim.step()

            # feedback
            print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                  end="",
                  flush=True)

            self.writer.add_scalar(tag='Average Return over 100 episodes',
                                   scalar_value=np.mean(self.total_rewards),
                                   global_step=epoch)

            self.writer.add_scalar(tag='Entropy',
                                   scalar_value=entropy,
                                   global_step=epoch)
            # check if solved
            # if np.mean(self.total_rewards) > 200:
            #     print('\nSolved!')
            #     break
        # close the writer
        self.writer.close()

    def play_episode(self):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # Init state
        init_state = [[(random.choice(conf.WIDTH_SPACE), random.choice(conf.CONTEXT_SPACE)) for i in range(conf.DEPTH)]]
        res = []
        for tp in init_state[0]:
            res.append(tp[0])
            res.append(tp[1])
        init_state = [res]

        # get the action logits from the agent - (preferences)
        episode_logits = self.agent(
            torch.tensor(init_state) \
                .float() \
                .to(conf.device)
        )

        # sample an action according to the action distribution
        action_index = Categorical(logits=episode_logits).sample().unsqueeze(1)

        mask = one_hot(action_index, num_classes=self.ACTION_SPACE)

        episode_log_probs = torch.sum(
            mask.float() * log_softmax(episode_logits, dim=1), dim=1)

        # append the action to the episode action list to obtain the trajectory
        # we need to store the actions and logits so we could calculate the gradient of the performance
        # episode_actions = torch.cat((episode_actions, action_index), dim=0)

        # Get action actions
        action_space = torch.tensor([
            conf.WIDTH_SPACE,
            conf.CONTEXT_SPACE,
            conf.WIDTH_SPACE,
            conf.CONTEXT_SPACE,
            conf.WIDTH_SPACE,
            conf.CONTEXT_SPACE,
            conf.WIDTH_SPACE,
            conf.CONTEXT_SPACE,
        ]) \
            .to(conf.device)
        action = torch.gather(action_space, 1, action_index).squeeze(1)
        print(action)

        # generate a submodel given predicted actions
        # net = NASModel(action)
        # net = Net()
        model = CmdRecogNetwork(
            num_class=conf.NUM_CLASS,
            network_depth=conf.DEPTH,
            feature_dim=conf.FEATURE_DIM,
            network_width=action.tolist()[0::2],
            network_context_size=action.tolist()[1::2],
        ).to(conf.device)

        losses = []
        acc = []
        pbar_update = 1 / (len(training_dataloader) + len(testing_dataloader))
        with tqdm(total=conf.EPOCH_NUM) as pbar:
            for epoch in range(1, conf.EPOCH_NUM + 1):
                training_losses = \
                    train.train(pbar, pbar_update, model, epoch, conf.LOG_INTERVAL)
                test_acc = \
                    test.test(pbar, pbar_update, model, epoch)
                losses += training_losses
                acc += [test_acc]

        # load best performance epoch in this training session
        # model.load_weights('weights/temp_network.h5')

        # evaluate the model
        vali_acc = evaluation.evaluation(model=model)
        print('Accuracy of the network on the 10000 test images: {}'.format(vali_acc))

        # compute the reward
        reward = vali_acc

        episode_weighted_log_probs = episode_log_probs * reward
        sum_weighted_log_probs = torch.sum(
            episode_weighted_log_probs).unsqueeze(dim=0)

        return sum_weighted_log_probs, episode_logits, reward

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * self.BETA * entropy

        return policy_loss + entropy_bonus, entropy

if __name__ == '__main__':
    instance = PolicyGradient()
    instance.solve_environment()