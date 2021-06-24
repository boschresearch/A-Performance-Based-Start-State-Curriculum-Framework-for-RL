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


def uniquify_list(l_in):
    l_out = []
    for element in l_in:
        if element not in l_out:
            l_out.append(element)
    return l_out


def sample_nearby(grid, starts, N_new, T_B, M, unique=False):
    starts_out = []
    if unique:
        i = 0
        starts_idx = list(range(len(starts)))
        while len(starts_out) < M and i < len(starts):
            rand_idx = randint(0, len(starts) - 1)
            s_t = grid.reset(starts[starts_idx[rand_idx]], grid.goal)
            starts_idx.remove(starts_idx[rand_idx])
            i += 1
            for _ in range(0, T_B):
                a_t = randint(0, 3)
                s_t = grid.roll_out(s_t, a_t)
                starts_out.append(tuple(s_t))
    else:
        while len(starts_out) < M:
            rand_idx = randint(0, len(starts) - 1)
            s_t = grid.reset(starts[rand_idx], grid.goal)
            for _ in range(0, T_B):
                a_t = randint(0, 3)
                s_t = grid.roll_out(s_t, a_t)
                starts_out.append(tuple(s_t))
    return sample_list(starts_out, N_new)


def label_states_from_paths(path_trajectories, path_successes):
    label_dict = {}
    traj_num = 0
    for path in path_trajectories:
        state = list(path[0].data)
        state = tuple([int(state[0]), int(state[1])])
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


class OldStarts:
    def __init__(self, dist_threshold, init):
        self.dist_threshold = dist_threshold
        self.start_list = [init]

    def add_starts(self, starts2add):
        for start2add in starts2add:
            if (cdist(XA=[start2add], XB=self.start_list, metric='euclidean').squeeze() > self.dist_threshold).all():
                self.start_list.append(start2add)


def label_states_from_paths_continuous(path_trajectories, path_successes):
    label_dict = {}
    traj_num = 0
    for path in path_trajectories:
        state = list(path[0].data)
        state = state[2:4]
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


def sample_nearby_continuous(grid, starts, N_new, T_B, M, unique=False, rand_mean=0.0, rand_std=1.0):
    starts_out = []
    if unique:
        i = 0
        starts_idx = list(range(len(starts)))
        while len(starts_out) < M and i < len(starts):
            rand_idx = randint(0, len(starts) - 1)
            s_t = grid.reset(starts[starts_idx[rand_idx]], grid.goal)
            starts_idx.remove(starts_idx[rand_idx])
            s_t = s_t[2:4]
            i += 1
            for _ in range(0, T_B):
                a_t = np.random.normal(loc=rand_mean, scale=rand_std, size=(2,))
                s_t = grid.roll_out(s_t, 100 * a_t)
                starts_out.append(tuple(s_t))
    else:
        while len(starts_out) < M:
            rand_idx = randint(0, len(starts) - 1)
            s_t = grid.reset(starts[rand_idx], grid.goal)
            s_t = s_t[2:4]
            for _ in range(0, T_B):
                a_t = np.random.normal(loc=rand_mean, scale=rand_std, size=(2,))
                s_t = grid.roll_out(s_t, 100 * a_t)
                starts_out.append(tuple(s_t))
    return sample_list(starts_out, N_new)
