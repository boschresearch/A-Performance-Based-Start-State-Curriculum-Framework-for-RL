# A Performance-Based Start State Curriculum Framework for Reinforcement Learning
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import copy
import math
import random
import time
import numpy as np
from random import randint
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic_sg_direct import TRPO

from lib.environments.mujoco_key import MujocoKeyEnv


class TRPOTrainer:

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_sizes,
                 max_kl,
                 damping,
                 batch_size,
                 inner_episodes,
                 max_iter,
                 use_fim=False,
                 use_gpu=False
                 ):

        self.grid = MujocoKeyEnv()
        self.policy = ContinuousMLP(state_dim,
                                    action_dim,
                                    hidden_sizes=hidden_sizes,
                                    activation=F.relu)
        self.optimizer = TRPO(policy=self.policy,
                              use_gpu=use_gpu,
                              max_kl=max_kl,
                              damping=damping,
                              use_fim=use_fim,
                              discount=0.998,
                              imp_weight=False)
        self.batch_size = batch_size
        self.inner_episodes = inner_episodes
        self.max_iter = max_iter
        self.state_dim = state_dim
        self.exp_b_param = 0.005
        self.exp_a_param = 1.
        self.lin_inc_param = 1.

    def roll_out_in_env(self, start, goal):
        roll_out = Batch()
        s = self.grid.env.reset(start)
        # print("START")
        s_list = []
        a_list = []
        r_list = []
        info = True
        for step_i in range(self.max_iter):
            s_save = copy.deepcopy(s)
            s_list.append(s_save)
            s = Variable(torch.from_numpy(np.float32(s_save))).unsqueeze(0)
            a, _ = self.policy.select_action(s, t=step_i)
            a_list.append(copy.deepcopy(a))
            s_new, r, d, info = self.grid.env.step(action=a, goal=goal, env_col=self.grid)
            r_list.append(r)
            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new.astype(np.float32)],
                      [0 if (d or (step_i + 1 == self.max_iter)) else 1],
                      [info],
                      [1.0]))
            s = s_new
            if d:
                break
        return roll_out, s_list, a_list, r_list, not info['no_goal_reached']

    def simulate_env(self, start, goal, mode, probs, smode):
        batch = Batch()
        state_list = []
        action_list = []
        reward_list = []
        start_list = []

        num_roll_outs = 0
        num_steps = 0
        total_success = 0
        j = 0.

        if mode == 'train':
            len_list = []
            while num_steps < self.batch_size:
                if smode:
                    start_pos = random.choices(population=start, weights=probs)[0]
                else:
                    if not start:
                        start_pos = self.grid.sample_uniform(number=1)[0]
                    else:
                        start_idx = randint(0, len(start) - 1)
                        start_pos = tuple(start[start_idx])

                start_list.append(start_pos)
                roll_out, states, actions, rewards, success = self.roll_out_in_env(start=start_pos, goal=goal)
                batch.append(roll_out)

                num_roll_outs += 1
                num_steps += roll_out.length()
                len_list.append(roll_out.length())
                total_success += success
                j += 1

                state_list.append(states)
                action_list.append(actions)
                reward_list.append(rewards)
        elif mode == 'test':
            start_pos = start

            roll_out, states, actions, rewards, success = self.roll_out_in_env(start=start_pos, goal=goal)
            batch.append(roll_out)

            num_roll_outs += 1
            num_steps += roll_out.length()
            total_success += success
            j += 1

            state_list.append(states)
            action_list.append(actions)
            reward_list.append(rewards)

        batch["num_episodes"] = num_roll_outs

        return batch, num_steps, total_success / j, state_list, action_list, reward_list, start_list

    def train(self, start_p_list, goal_p, sampling_probs, sampling_mode):
        acc_steps = 0
        acc_successes = 0
        s_list = []
        a_list = []
        r_list = []
        st_list = []

        for iter_loop in range(self.inner_episodes):
            batch, steps, successes, s_list_part, a_list_part, r_list_part, start_list = \
                self.simulate_env(start=start_p_list,
                                  goal=goal_p,
                                  mode='train',
                                  probs=sampling_probs,
                                  smode=sampling_mode)
            acc_steps += steps
            acc_successes += successes
            s_list = s_list + s_list_part
            a_list = a_list + a_list_part
            r_list = r_list + r_list_part
            st_list = st_list + start_list
            self.optimizer.process_batch(self.policy, batch, [])

        return acc_steps, acc_successes/self.inner_episodes, s_list, a_list, r_list, st_list, []

    def test(self, start_p_list, goal_p):
        batch, steps, successes, s_list, a_list, r_list, _ = self.simulate_env(start=start_p_list,
                                                                               goal=goal_p,
                                                                               mode='test',
                                                                               probs=[],
                                                                               smode=False
                                                                               )
        return steps, successes, s_list, a_list, r_list
