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


import math
import copy
import random
import numpy as np
from random import randint
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.gridworld import GridWorldContinuous


class TRPOTrainer:

    def __init__(self,
                 grid_type,
                 state_dim,
                 action_dim,
                 hidden_sizes,
                 max_kl,
                 damping,
                 batch_size,
                 inner_episodes,
                 max_iter,
                 use_fim=False,
                 use_gpu=False,
                 usym_var=False,
                 outer_iterations=5000
                 ):

        self.grid = GridWorldContinuous(grid_type=grid_type, usym=usym_var)
        self.policy = ContinuousMLP(state_dim,
                                    action_dim,
                                    hidden_sizes=hidden_sizes,
                                    activation=F.relu)
        self.optimizer = TRPO(policy=self.policy,
                              use_gpu=use_gpu,
                              max_kl=max_kl,
                              damping=damping,
                              use_fim=use_fim,
                              discount=1.0,
                              imp_weight=False)
        self.batch_size = batch_size
        self.inner_episodes = inner_episodes
        self.max_iter = max_iter
        self.default_starts = []
        self.state_dim = state_dim
        for x_coord in range(self.grid.x_size):
            for y_coord in range(self.grid.y_size):
                if self.grid.occupancy_map[x_coord][y_coord] == 0:
                    self.default_starts.append((x_coord, y_coord))
        self.state_cache = np.zeros((self.grid.x_size, self.grid.y_size, 8))
        for start in self.default_starts:
            self.state_cache[start[0], start[1], :] = list(self.grid.get_neighborhood(start))
        self.exp_b_param = 1. / (outer_iterations - 1) * math.log(
            -100. / ((1. / (self.grid.x_size * self.grid.y_size)) *
                     math.log(1. / (self.grid.x_size * self.grid.y_size))))
        self.exp_a_param = 1. / (math.exp(self.exp_b_param))
        self.lin_inc_param = (-100. / ((1. / (self.grid.x_size * self.grid.y_size)) *
                                       math.log(1. / (self.grid.x_size * self.grid.y_size))) - 1) / (
                                     outer_iterations - 1)

    def roll_out_in_env(self, start, goal):
        roll_out = Batch()
        s = self.grid.reset(start, goal)
        s_list = []
        a_list = []
        r_list = []
        d = 0
        for step_i in range(self.max_iter):
            s_save = copy.deepcopy(np.array(list(s) +
                                            list(self.state_cache[max(min(int(round(s[2])), self.grid.x_size-1), 0),
                                                                  max(min(int(round(s[3])), self.grid.y_size-1), 0)])))
            s_list.append(s_save)
            s = Variable(torch.from_numpy(np.float32(s_save))).unsqueeze(0)
            a, _ = self.policy.select_action(s, t=step_i)
            s_new, r, d, info, _ = self.grid.step(action=100 * a)
            a_list.append(a)
            r_list.append(r)
            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new.astype(np.float32)],
                      [0 if ((not info) or (step_i + 1 == self.max_iter)) else 1],
                      [info]))
            s = s_new
            if not info:
                break
        return roll_out, s_list, a_list, r_list, d

    def simulate_env(self, start, goal, mode, probs, smode, direct):
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
                    if direct:
                        start_pos = random.choices(population=start, weights=probs)[0]
                    else:
                        start_pos_pre = random.choices(population=start, weights=probs)
                        start_pos = self.grid.sample_random_pos_tile(tile_x=start_pos_pre[0][0], tile_y=start_pos_pre[0][1])
                else:
                    if not start:
                        start_pos = self.grid.sample_random_pos(number=1)[0]
                    else:
                        start_idx = randint(0, len(start) - 1)
                        start_pos = start[start_idx]

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

        return batch, num_steps, total_success / j, state_list, action_list, reward_list, start_list

    def train(self, start_p_list, goal_p, sampling_probs, sampling_mode, direct=False):
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
                                  smode=sampling_mode,
                                  direct=direct)
            acc_steps += steps
            acc_successes += successes
            s_list = s_list + s_list_part
            a_list = a_list + a_list_part
            r_list = r_list + r_list_part
            st_list = st_list + start_list
            self.optimizer.process_batch(self.policy, batch, [])

        return acc_steps, acc_successes, s_list, a_list, r_list, st_list, []

    def test(self, start_p_list, goal_p):
        batch, steps, successes, s_list, a_list, r_list, _ = self.simulate_env(start=start_p_list,
                                                                               goal=goal_p,
                                                                               mode='test',
                                                                               probs=[],
                                                                               smode=False,
                                                                               direct=False
                                                                               )
        return steps, successes, s_list, a_list, r_list
