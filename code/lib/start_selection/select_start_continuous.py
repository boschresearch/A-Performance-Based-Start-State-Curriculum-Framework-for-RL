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


def sg_direct_mj_key(trpo, state_num=1000, eps=0.03):
    start_candidate_list = []
    grad_list = []

    for i in range(state_num):

        sampled_state = trpo.grid.sample_uniform(number=1)[0]

        grad_dim_1 = 0.5 * (trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] + np.array([eps, 0., 0., 0., 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] - np.array([eps, 0., 0., 0., 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze())
        grad_dim_2 = 0.5 * (trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] + np.array([0., eps, 0., 0., 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] - np.array([0., eps, 0., 0., 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze())
        grad_dim_3 = 0.5 * (trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] + np.array([0., 0., eps, 0., 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] - np.array([0., 0., eps, 0., 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze())
        grad_dim_4 = 0.5 * (trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] + np.array([0., 0., 0., eps, 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] - np.array([0., 0., 0., eps, 0., 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze())
        grad_dim_5 = 0.5 * (trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] + np.array([0., 0., 0., 0., eps, 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] - np.array([0., 0., 0., 0., eps, 0., 0.])))).unsqueeze(
            0)).data.numpy().squeeze())
        grad_dim_6 = 0.5 * (trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] + np.array([0., 0., 0., 0., 0., eps, 0.])))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] - np.array([0., 0., 0., 0., 0., eps, 0.])))).unsqueeze(
            0)).data.numpy().squeeze())
        grad_dim_7 = 0.5 * (trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] + np.array([0., 0., 0., 0., 0., 0., eps])))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic_nd"](Variable(
            torch.from_numpy(np.float32(sampled_state[:7] - np.array([0., 0., 0., 0., 0., 0., eps])))).unsqueeze(
            0)).data.numpy().squeeze())

        start_candidate_list.append(sampled_state)
        grad_list.append(np.sqrt(grad_dim_1 ** 2 + grad_dim_2 ** 2 + grad_dim_3 ** 2 + grad_dim_4 ** 2 +
                                 grad_dim_5 ** 2 + grad_dim_6 ** 2 + grad_dim_7 ** 2))

    return start_candidate_list, grad_list


def spcrl_direct_mj_key(trpo, step, state_num=1000):
    temp_param = trpo.exp_a_param * math.exp(trpo.exp_b_param * step)

    start_candidate_list = []
    grad_list = []

    for i in range(state_num):

        sampled_state = trpo.grid.sample_uniform(number=1)[0]
        val = trpo.optimizer.networks["critic_nd"](
            Variable(torch.from_numpy(np.float32(sampled_state[:7]))).unsqueeze(0)).data.numpy().squeeze()

        start_candidate_list.append(sampled_state)
        grad_list.append(val)

    return start_candidate_list, (np.exp(np.array(grad_list)/temp_param)/sum(np.exp(np.array(grad_list)/temp_param))).tolist()


def sample_starts_tpg_direct_mj_key(trpo, state_num=1000):
    start_candidate_list = []
    val_list = []

    for i in range(state_num):

        sampled_state = trpo.grid.sample_uniform(number=1)[0]
        val = trpo.optimizer.networks["critic_nd"](
            Variable(torch.from_numpy(np.float32(sampled_state[:7]))).unsqueeze(0)).data.numpy().squeeze()

        start_candidate_list.append(sampled_state)
        val_list.append(val)

    return start_candidate_list, np.array(val_list)


def tpg_direct_mj_key(trpo, states, values, temp):
    grads = np.zeros((len(states), ))

    for i in range(len(states)):

        val = trpo.optimizer.networks["critic_nd"](
            Variable(torch.from_numpy(np.float32(states[i][:7]))).unsqueeze(0)).data.numpy().squeeze()
        grads[i] = val - values[i]

    return states, (np.exp(grads / temp) / sum(np.exp(grads / temp))).tolist()


def sg_direct_grid_c(trpo, state_num=100, eps=1.0):
    start_candidate_list = []
    grad_list = []

    for i in range(state_num):

        sampled_state = trpo.grid.sample_random_pos(number=1)[0]

        grad_dim_1 = 0.5 * (trpo.optimizer.networks["critic"](Variable(
            torch.from_numpy(np.float32(np.array([0., 0., sampled_state[0] + 1.0, sampled_state[1]] +
                                                 list(trpo.state_cache[max(
                                                     min(int(round(sampled_state[0] + 1.0)), trpo.grid.x_size - 1), 0),
                                                                       max(min(int(round(sampled_state[1])),
                                                                               trpo.grid.y_size - 1),
                                                                           0)]))))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic"](Variable(
            torch.from_numpy(np.float32(np.array([0., 0., sampled_state[0] - 1.0, sampled_state[1]] +
                                                 list(trpo.state_cache[max(
                                                     min(int(round(sampled_state[0] - 1.0)), trpo.grid.x_size - 1), 0),
                                                                       max(min(int(round(sampled_state[1])),
                                                                               trpo.grid.y_size - 1),
                                                                           0)]))))).unsqueeze(
            0)).data.numpy().squeeze())

        grad_dim_2 = 0.5 * (trpo.optimizer.networks["critic"](Variable(
            torch.from_numpy(np.float32(np.array([0., 0., sampled_state[0], sampled_state[1] + 1.0] +
                                                 list(trpo.state_cache[max(
                                                     min(int(round(sampled_state[0])), trpo.grid.x_size - 1), 0),
                                                                       max(min(int(round(sampled_state[1] + 1.0)),
                                                                               trpo.grid.y_size - 1),
                                                                           0)]))))).unsqueeze(
            0)).data.numpy().squeeze() - trpo.optimizer.networks["critic"](Variable(
            torch.from_numpy(np.float32(np.array([0., 0., sampled_state[0], sampled_state[1] - 1.0] +
                                                 list(trpo.state_cache[max(
                                                     min(int(round(sampled_state[0])), trpo.grid.x_size - 1), 0),
                                                                       max(min(int(round(sampled_state[1] - 1.0)),
                                                                               trpo.grid.y_size - 1),
                                                                           0)]))))).unsqueeze(
            0)).data.numpy().squeeze())

        start_candidate_list.append(sampled_state)
        grad_list.append(np.sqrt(grad_dim_1 ** 2 + grad_dim_2 ** 2))

    return start_candidate_list, grad_list
