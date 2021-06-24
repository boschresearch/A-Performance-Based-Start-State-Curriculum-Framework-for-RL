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


import os
import gym
import random
import numpy as np

from gym.spaces import Box
from random import randint
from lib.utility.state_list import sample_list


class MujocoKeyEnv:
    def __init__(self):
        self.env = gym.make('Arm3dKey-v1')
        s_init = self.env.reset(start=[1.55, 0.4, -3.75, -1.15, 1.81, -2.09, 0.05])
        # print(s_init)
        self.uniform_starts = np.load(os.getcwd() + '/key_starts_230k_kill_outside_M100k.npy')
        self.uniform_starts_in = np.load(os.getcwd() + '/key_starts_230k_kill_outside_M100k_states_inside.npy')
        self.uniform_starts_out = np.load(os.getcwd() + '/key_starts_230k_kill_outside_M100k_states_outside.npy')
        self.num_uniform_starts = self.uniform_starts.shape[0]
        self.num_uniform_starts_in = self.uniform_starts_in.shape[0]
        self.num_uniform_starts_out = self.uniform_starts_out.shape[0]
        self.goal = np.array([1.55, 0.4, -3.75, -1.15, 1.81, -2.09, 0.05])

    def sample_uniform(self, number, mode='all'):
        state_list_uniform = []
        while len(state_list_uniform) < number:
            if mode == 'all':
                idx = random.randint(0, self.num_uniform_starts-1)
                state_list_uniform.append(tuple(self.uniform_starts[idx, :]))
            elif mode == 'in':
                idx = random.randint(0, self.num_uniform_starts_in - 1)
                state_list_uniform.append(tuple(self.uniform_starts_in[idx, :]))
            elif mode == 'out':
                idx = random.randint(0, self.num_uniform_starts_out - 1)
                state_list_uniform.append(tuple(self.uniform_starts_out[idx, :]))
        return state_list_uniform

    def sample_nearby(self, starts, N_new, T_B, M, all=False):
        space = Box(self.env.action_space.low, self.env.action_space.high)
        starts_out = []
        i = 0
        while len(starts_out) < M:
            rand_idx = randint(0, len(starts) - 1)
            # rand_idx = i % len(starts)
            # print("\n")
            # print(rand_idx)
            # print("\n")
            start = self.env.reset(list(starts[rand_idx]))
            # print("START")
            # print(start)
            for tk in range(0, T_B):
                a_t = space.sample()
                # print(a_t)
                s_t, _, d, info = self.env.step(a_t, goal=None, env_col=None)
                # print(s_t)
                # print(d)
                if d and info['no_goal_reached']:
                    break
                starts_out.append(np.array(s_t[:7]))
        random.shuffle(starts_out)
        if all:
            return starts_out
        else:
            return sample_list(starts_out, N_new)

    def sample_nearby_gen(self, starts, T_B, M):
        space = Box(self.env.action_space.low, self.env.action_space.high)
        starts_out = []
        i = 0
        while len(starts_out) < M:
            rand_idx = randint(0, len(starts) - 1)
            # rand_idx = i % len(starts)
            i += 1
            _ = self.env.reset(list(starts[rand_idx]))
            for tk in range(0, T_B):
                a_t = space.sample()
                s_t, _, d, info = self.env.step(a_t, goal=None, env_col=None)
                if d and info['no_goal_reached']:
                    break
                starts_out.append(np.array(s_t))
        random.shuffle(starts_out)
        return starts_out
