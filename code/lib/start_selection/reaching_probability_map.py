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

import torch
from torch.autograd import Variable


def generate_reach_prob_map_oracle(trpo, state_list, full_mode=True, num_eval=10, continuous=False):
    reach_prob_map = np.zeros(trpo.grid.rp_map.shape)
    count_map = np.zeros(reach_prob_map.shape)
    if continuous:
        for state in state_list:
            for _ in range(num_eval):
                state_sampled = trpo.grid.sample_random_pos_tile(tile_x=state[0], tile_y=state[1])
                _, traj, _, _, success = trpo.roll_out_in_env(start=state_sampled, goal=trpo.grid.goal)
                if full_mode:
                    for traj_state in traj:
                        traj_state = traj_state.squeeze().tolist()
                        traj_state = [int(round(element)) for element in traj_state]
                        traj_state[2] = max(min(traj_state[2], trpo.grid.x_size-1), 0)
                        traj_state[3] = max(min(traj_state[3], trpo.grid.y_size-1), 0)
                        traj_state = tuple(traj_state)
                        count_map[traj_state[2], traj_state[3]] += 1
                        if success:
                            reach_prob_map[traj_state[2], traj_state[3]] += 1
                else:
                    count_map[state[2], state[3]] += 1
                    if success:
                        reach_prob_map[state[2], state[3]] += 1
    else:
        for state in state_list:
            for _ in range(num_eval):
                _, traj, _, _, success = trpo.roll_out_in_env(start=state, goal=trpo.grid.goal)
                if full_mode:
                    for traj_state in traj:
                        traj_state = traj_state.squeeze().tolist()
                        traj_state = tuple([int(element) for element in traj_state])
                        count_map[traj_state[0], traj_state[1]] += 1
                        if success:
                            reach_prob_map[traj_state[0], traj_state[1]] += 1
                else:
                    count_map[state[0], state[1]] += 1
                    if success:
                        reach_prob_map[state[0], state[1]] += 1
    reach_prob_map = np.divide(reach_prob_map, count_map, out=np.zeros_like(reach_prob_map), where=count_map != 0)
    return reach_prob_map


def update_reach_prob_map(reach_prob_map, path_trajectories, path_successes):
    traj_num = 0
    reach_prob_map = np.zeros(reach_prob_map.shape)
    count_map = np.zeros(reach_prob_map.shape)
    for path in path_trajectories:
        for state in path:
            state = state.data.tolist()
            state = tuple([max(min(int(round(state[0])), 30 - 1), 0), max(min(int(round(state[1])), 20 - 1), 0)])
            count_map[state[0], state[1]] += 1
            if path_successes[traj_num][-1] == 1:
                reach_prob_map[state[0], state[1]] += 1
        traj_num += 1
    reach_prob_map = np.divide(reach_prob_map, count_map, out=np.zeros_like(reach_prob_map), where=count_map != 0)
    return reach_prob_map


def update_reach_prob_map_continuous(reach_prob_map, path_trajectories, path_successes):
    traj_num = 0
    reach_prob_map = np.zeros(reach_prob_map.shape)
    count_map = np.zeros(reach_prob_map.shape)
    for path in path_trajectories:
        for state in path:
            state = state.data.tolist()
            state = tuple([max(min(int(round(state[2])), 30 - 1), 0), max(min(int(round(state[3])), 20 - 1), 0)])
            count_map[state[0], state[1]] += 1
            if path_successes[traj_num][-1] == 1:
                reach_prob_map[state[0], state[1]] += 1
        traj_num += 1
    reach_prob_map = np.divide(reach_prob_map, count_map, out=np.zeros_like(reach_prob_map), where=count_map != 0)
    return reach_prob_map


def states_reaches_array(path_trajectories, path_successes):
    traj_num = 0
    pos_list = []
    reach_list = []
    for path in path_trajectories:
        for state in path:
            state = state.data.tolist()
            state = np.array([state[2], state[3]])
            pos_list.append(state)
            if path_successes[traj_num][-1] == 1:
                reach_list.append(1)
            else:
                reach_list.append(0)
        traj_num += 1

    return np.array(pos_list), np.array(reach_list)


def states_reaches_array_mj_point(path_trajectories, path_successes):
    traj_num = 0
    pos_list = []
    reach_list = []
    for path in path_trajectories:
        for state in path:
            state = state.data.tolist()
            state = np.array([state[0], state[1]])
            pos_list.append(state)
            if path_successes[traj_num][-1] == 1:
                reach_list.append(1)
            else:
                reach_list.append(0)
        traj_num += 1

    return np.array(pos_list), np.array(reach_list)


def states_reaches_array_mj_key(path_trajectories, path_successes):
    traj_num = 0
    pos_list = []
    reach_list = []
    for path in path_trajectories:
        for state in path:
            state = state.data.tolist()
            state = np.array(state[14:23])
            pos_list.append(state)
            if path_successes[traj_num][-1] == 1:
                reach_list.append(1)
            else:
                reach_list.append(0)
        traj_num += 1

    return np.array(pos_list), np.array(reach_list)


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


def rpm2gradlist(grad_states, reach_prob_map):
    dim = reach_prob_map.shape
    reach_prob_grad_map = np.zeros(dim)
    state_list = []
    rp_grad_list = []
    for state in grad_states:
        state_list.append((state[0], state[1]))
        full_grad = np.sqrt(reach_prob_map[state[0], state[1]] - (reach_prob_map[state[0], state[1]] ** 2))
        rp_grad_list.append(full_grad)
        reach_prob_grad_map[state[0], state[1]] = full_grad

    return state_list, rp_grad_list, reach_prob_grad_map


def calc_temporal_reach_prob_grads(grad_states, reach_prob_map_old, reach_prob_map_new):
    reach_prob_grad_map = np.zeros(reach_prob_map_new.shape)
    rp_grad_list_full = []
    state_list_pos = []
    rp_grad_list_pos = []
    for state in grad_states:
        temp_grad = reach_prob_map_new[state[0], state[1]] - reach_prob_map_old[state[0], state[1]]
        rp_grad_list_full.append(temp_grad)
        reach_prob_grad_map[state[0], state[1]] = temp_grad
        if temp_grad > 0:
            state_list_pos.append(state)
            rp_grad_list_pos.append(temp_grad)
    return grad_states, rp_grad_list_full, state_list_pos, rp_grad_list_pos, reach_prob_grad_map


def compute_competences(path_trajectories, path_successes, goal):
    state_list = []
    competence_list = []
    goal_np = np.array(goal)
    for (path, successes) in zip(path_trajectories, path_successes):
        start_np = np.array(path[0].data.tolist())
        final_np = np.array(path[-1].data.tolist())
        d_final = np.linalg.norm(goal_np - final_np[2:4])
        d_initial = np.linalg.norm(start_np[2:4] - goal_np)
        if d_final > d_initial:
            competence = -1
        elif successes[-1]:
            competence = 0
        else:
            competence = - d_final / d_initial

        state_list.append(start_np[2:4].tolist())
        competence_list.append(competence)
    return state_list, competence_list


def compute_competences_mj(path_trajectories, path_successes, goal):
    state_list = []
    competence_list = []
    goal_np = np.array(goal)
    for (path, successes) in zip(path_trajectories, path_successes):
        start_np = np.array(path[0].data.tolist())
        final_np = np.array(path[-1].data.tolist())
        d_final = np.linalg.norm(goal_np - final_np[0:2])
        d_initial = np.linalg.norm(start_np[0:2] - goal_np)
        if d_final > d_initial:
            competence = -1
        elif successes[-1]:
            competence = 0
        else:
            competence = - d_final / d_initial

        state_list.append(start_np[0:2].tolist())
        competence_list.append(competence)
    return state_list, competence_list


def update_reach_prob_map_vf_fine(reach_prob_map, trpo):
    reach_prob_map = np.zeros(reach_prob_map.shape)
    for i in range(56):
        for j in range(56):
            x_coord = i / 4 - 0.875
            y_coord = j / 4 - 0.875
            s = Variable(torch.from_numpy(
                np.float32(np.array([x_coord, y_coord])))).unsqueeze(0)
            values = trpo.optimizer.networks["critic_nd"](s)
            reach_prob_map[i, j] = values.data.numpy().squeeze()
    return reach_prob_map
