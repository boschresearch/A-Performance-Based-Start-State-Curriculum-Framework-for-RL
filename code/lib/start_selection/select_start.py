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


def select_start_states_oracle(reach_prob_map, r_min=0.1, r_max=0.9):
    start_list = []
    x_size, y_size = reach_prob_map.shape
    for i in range(x_size):
        for j in range(y_size):
            if r_min < reach_prob_map[i, j] < r_max:
                start_list.append((i, j))
    return start_list


def start_state_sampling_probabilities_from_grads(start_candidates, rp_grads, boltzmann=False, temp=1.):
    if boltzmann:
        grad_dict = {}
        for start_candidate, grad in zip(start_candidates, rp_grads):
            # start_candidate = (start_candidate[0], start_candidate[1])
            if start_candidate not in grad_dict.keys():
                grad_dict[start_candidate] = grad
            elif start_candidate in grad_dict.keys() and grad > grad_dict[start_candidate]:
                print("START CANDIDATE TWICE !!!")
                grad_dict[start_candidate] = grad
        state_list = list(grad_dict.keys())
        grad_list = list(grad_dict.values())
        return state_list, (np.exp(np.array(grad_list)/temp)/sum(np.exp(np.array(grad_list)/temp))).tolist()
    else:
        if True:
            idx = 0
            start_candidates_final = []
            sampling_probs_final = []
            for grad in rp_grads:
                if grad > 0:
                    start_candidates_final.append(start_candidates[idx])
                    sampling_probs_final.append(rp_grads[idx])
                idx += 1
            return start_candidates_final, (np.array(sampling_probs_final)/sum(np.array(sampling_probs_final))).tolist()
        else:
            sampling_dict = {}
            idx = 0
            start_candidates_final = []
            sampling_probs_final = []
            for grad in rp_grads:
                if grad > 0:
                    start_candidates_final.append(start_candidates[idx])
                    sampling_probs_final.append(rp_grads[idx])
                idx += 1
            sampling_probs_final = (np.array(sampling_probs_final) / sum(np.array(sampling_probs_final))).tolist()
            idx = 0
            for state in start_candidates_final:
                if (state[0], state[1]) not in sampling_dict.keys():
                    sampling_dict[(state[0], state[1])] = sampling_probs_final[idx]
                else:
                    print("TWICE")
                    sampling_dict[(state[0], state[1])] += sampling_probs_final[idx]
                idx += 1
            return start_candidates_final, sampling_probs_final, sampling_dict


def start_state_sampling_probabilities_from_grads_cont(start_candidates, rp_grads):
    if sum(np.array(rp_grads)) < 1e-12:
        return start_candidates, np.ones_like(np.array(rp_grads))
    else:
        return start_candidates, (np.array(rp_grads)/sum(np.array(rp_grads))).tolist()


def start_state_sampling_probabilities_from_rp(trpo, start_candidates, rpm, step, temp_param_mode='exp'):
    if temp_param_mode == 'exp':
        temp_param = trpo.exp_a_param * math.exp(trpo.exp_b_param * step)
    elif temp_param_mode == 'lin':
        temp_param = 1 + trpo.lin_inc_param * (step - 1)
    elif temp_param_mode == 'fix':
        temp_param = 0.1

    rp_list = []
    for start_candidate in start_candidates:
        rp_list.append(rpm[start_candidate[0], start_candidate[1]])

    return start_candidates, (np.exp(np.array(rp_list)/temp_param)/sum(np.exp(np.array(rp_list)/temp_param))).tolist()


def start_state_sampling_probabilities_from_temp_grads(start_candidates_full, start_candidates_pos, grad_list_full,
                                                       grad_list_pos, boltzmann=False, temp=1., pos=False):
    if boltzmann:
        if pos:
            return start_candidates_pos, (np.exp(np.array(grad_list_pos) / temp) /
                                          sum(np.exp(np.array(grad_list_pos) / temp))).tolist()
        else:
            return start_candidates_full, (np.exp(np.array(grad_list_full) / temp) /
                                          sum(np.exp(np.array(grad_list_full) / temp))).tolist()
    else:
        return start_candidates_pos, (np.array(grad_list_pos)/sum(np.array(grad_list_pos))).tolist()
