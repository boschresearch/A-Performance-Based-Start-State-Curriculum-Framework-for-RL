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
import numpy as np


def update_reach_prob_map_fine(reach_prob_map, path_trajectories, path_successes):
    traj_num = 0
    reach_prob_map = np.zeros(reach_prob_map.shape)
    count_map = np.zeros(reach_prob_map.shape)
    for path in path_trajectories:
        for state in path:
            state = state.data.tolist()
            state = tuple([int(np.clip(np.round(4*(state[0] + 0.875)), 0, 55)),
                           int(np.clip(np.round(4*(state[1] + 0.875)), 0, 55))])
            count_map[state[0], state[1]] += 1
            if path_successes[traj_num][-1] == 1:
                reach_prob_map[state[0], state[1]] += 1
        traj_num += 1
    reach_prob_map = np.divide(reach_prob_map, count_map, out=np.zeros_like(reach_prob_map), where=count_map != 0)
    return reach_prob_map


def update_reach_prob_map_ant(reach_prob_map, path_trajectories, path_successes):
    traj_num = 0
    reach_prob_map = np.zeros(reach_prob_map.shape)
    count_map = np.zeros(reach_prob_map.shape)
    for path in path_trajectories:
        for state in path:
            state = state.data.tolist()
            state = tuple([int(np.clip(np.round(4*(state[0] + 1.375)), 0, 59)),
                           int(np.clip(np.round(4*(state[1] + 1.375)), 0, 59))])
            count_map[state[0], state[1]] += 1
            if path_successes[traj_num][-1] == 1:
                reach_prob_map[state[0], state[1]] += 1
        traj_num += 1
    reach_prob_map = np.divide(reach_prob_map, count_map, out=np.zeros_like(reach_prob_map), where=count_map != 0)
    return reach_prob_map


def calc_spatial_reach_prob_grads(grad_states, occupancy_map, reach_prob_map, square_mode=False):
    dim = occupancy_map.shape
    reach_prob_grad_map = np.zeros(dim)
    state_list = []
    rp_grad_list = []
    for state in grad_states:
        if state[0] - 1 >= 0 and state[0] + 1 < dim[0]:
            if occupancy_map[state[0] - 1, state[1]] == 0 and occupancy_map[state[0] + 1, state[1]] == 0:
                grad_x = 0.5 * (reach_prob_map[state[0] + 1, state[1]] - reach_prob_map[state[0] - 1, state[1]])
            elif occupancy_map[state[0] - 1, state[1]] == 0:
                grad_x = 0.5 * (reach_prob_map[state[0], state[1]] - reach_prob_map[state[0] - 1, state[1]])
            elif occupancy_map[state[0] + 1, state[1]] == 0:
                grad_x = 0.5 * (reach_prob_map[state[0] + 1, state[1]] - reach_prob_map[state[0], state[1]])
            else:
                grad_x = 0
        elif state[0] - 1 >= 0:
            if occupancy_map[state[0] - 1, state[1]] == 0:
                grad_x = 0.5 * (reach_prob_map[state[0], state[1]] - reach_prob_map[state[0] - 1, state[1]])
            else:
                grad_x = 0
        elif state[0] + 1 < dim[0]:
            if occupancy_map[state[0] + 1, state[1]] == 0:
                grad_x = 0.5 * (reach_prob_map[state[0] + 1, state[1]] - reach_prob_map[state[0], state[1]])
            else:
                grad_x = 0
        else:
            grad_x = 0

        if state[1] - 1 >= 0 and state[1] + 1 < dim[1]:
            if occupancy_map[state[0], state[1] - 1] == 0 and occupancy_map[state[0], state[1] + 1] == 0:
                grad_y = 0.5 * (reach_prob_map[state[0], state[1] + 1] - reach_prob_map[state[0], state[1] - 1])
            elif occupancy_map[state[0], state[1] - 1] == 0:
                grad_y = 0.5 * (reach_prob_map[state[0], state[1]] - reach_prob_map[state[0], state[1] - 1])
            elif occupancy_map[state[0], state[1] + 1] == 0:
                grad_y = 0.5 * (reach_prob_map[state[0], state[1] + 1] - reach_prob_map[state[0], state[1]])
            else:
                grad_y = 0
        elif state[1] - 1 >= 0:
            if occupancy_map[state[0], state[1] - 1] == 0:
                grad_y = 0.5 * (reach_prob_map[state[0], state[1]] - reach_prob_map[state[0], state[1] - 1])
            else:
                grad_y = 0
        elif state[1] + 1 < dim[1]:
            if occupancy_map[state[0], state[1] + 1] == 0:
                grad_y = 0.5 * (reach_prob_map[state[0], state[1] + 1] - reach_prob_map[state[0], state[1]])
            else:
                grad_y = 0
        else:
            grad_y = 0

        state_list.append((state[0], state[1]))
        if square_mode:
            full_grad = grad_x**2 + grad_y**2
        else:
            full_grad = math.sqrt(grad_x ** 2 + grad_y ** 2)
        rp_grad_list.append(full_grad)
        reach_prob_grad_map[state[0], state[1]] = full_grad

    return state_list, rp_grad_list, reach_prob_grad_map
