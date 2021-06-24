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


import numpy as np
from scipy.spatial.distance import cdist
from random import randint
from lib.utility.state_list import sample_list
from gym.spaces import Box


class OldStarts:
    def __init__(self, dist_threshold, init, no_threshold=False):
        self.dist_threshold = dist_threshold
        self.start_list = [init]
        self.threshold = no_threshold

    def add_starts(self, starts2add):
        for start2add in starts2add:
            if self.threshold:
                self.start_list.append(start2add)
            else:
                if (cdist(XA=[start2add], XB=self.start_list, metric='euclidean').squeeze() > self.dist_threshold).all():
                    self.start_list.append(start2add)


class OldStartsKey:
    def __init__(self, dist_threshold, init):
        self.dist_threshold = dist_threshold
        self.start_list = [init]
        self.pos_list = [(0.0, 0.3, -0.7, 0.0, 0.3, -0.4, - 0.15, 0.3, -0.55)]

    def add_starts(self, starts2add, positions2add):
        for (start2add, position2add) in zip(starts2add, positions2add):
            # print("add starts")
            # print(start2add)
            # print(position2add)
            # print(np.expand_dims(np.array(position2add), axis=0).shape)
            # print(np.array(self.pos_list).shape)
            if (cdist(XA=np.expand_dims(np.array(position2add), axis=0),
                      XB=np.array(self.pos_list), metric='euclidean').squeeze() > self.dist_threshold).all():
                self.start_list.append(start2add)
                self.pos_list.append(position2add)


def label_states_from_paths(path_trajectories, path_successes):
    label_dict = {}
    traj_num = 0
    for path in path_trajectories:
        state = list(path[0].data)
        state = state[:2]
        state = tuple([element for element in state])
        if state in label_dict:
            label_dict[state][0] += 1
            label_dict[state][1] += path_successes[traj_num][-1]
        else:
            label_dict[state] = [1, path_successes[traj_num][-1]]
        traj_num += 1
    states = []
    labels = []
    for element in label_dict:
        if label_dict[element][0] > 1:
            states.append(element)
            labels.append(label_dict[element][1]/label_dict[element][0])
    return states, labels


def label_states_from_paths_key(path_trajectories, path_successes):
    label_dict = {}
    traj_num = 0
    for path in path_trajectories:
        state = list(path[0].data)
        state = state[:7]
        state = tuple([element for element in state])
        if state in label_dict:
            label_dict[state][0] += 1
            label_dict[state][1] += path_successes[traj_num][-1]
        else:
            label_dict[state] = [1, path_successes[traj_num][-1]]
        traj_num += 1
    states = []
    labels = []
    for element in label_dict:
        if label_dict[element][0] > 1:
            states.append(element)
            labels.append(label_dict[element][1]/label_dict[element][0])
    return states, labels


def sample_nearby(grid, starts, N_new, T_B, M, rand_mean=0.0, rand_std=1.0, action_dim=2):
    space = Box(grid.env.action_space.low, grid.env.action_space.high)
    starts_out = []
    while len(starts_out) < M:
        rand_idx = randint(0, len(starts) - 1)
        _ = grid.env.reset(starts[rand_idx])
        for _ in range(0, T_B):
            a_t = np.random.normal(loc=rand_mean, scale=rand_std, size=(action_dim,))
            # a_t = space.sample()
            s_t, _, _, _ = grid.env.roll_out(a_t)
            starts_out.append((s_t[0], s_t[1]))
    return sample_list(starts_out, N_new)


def select_starts(starts, rews, R_min, R_max):
    del_idx = []
    starts_out = []
    for i in range(len(rews)):
        if not (R_min < rews[i] < R_max):
            del_idx.append(i)
    for i in range(len(starts)):
        if i not in del_idx:
            starts_out.append(starts[i])
    return starts_out


def select_starts_key(starts, rews, R_min, R_max):
    del_idx = []
    starts_out = []
    pos_out = []
    for i in range(len(rews)):
        if not (R_min < rews[i] < R_max):
            del_idx.append(i)
    for i in range(len(starts)):
        if i not in del_idx:
            starts_out.append(starts[i][:7])
            pos_out.append(starts[i][14:23])
    return starts_out, pos_out
