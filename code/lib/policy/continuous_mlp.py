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
#
# This source code is derived from PyTorch-RL
#   (https://github.com/Khrylx/PyTorch-RL)
# Copyright (c) 2020 Ye Yuan, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree,
# which derived it from pytorch-trpo
#   (https://github.com/ikostrikov/pytorch-trpo)
# Copyright (c) 2017 Ilya Kostrikov, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.policy.policy import PGSupportingPolicy


class ContinuousMLP(PGSupportingPolicy):
    def __init__(self, state_dim, action_dim, hidden_sizes=None, embedding=None,
                 activation=F.elu, log_std=0):
        """
        :param input_dim: Input dimension for each element in sequence
        :param output_dim: Output dimension for embedding
        :param hidden_sizes: List, Size of hidden layers
        :param embedding: Embedding module of type nn.Module
        :param activation: PyTorch functional activation (e.g. F.elu)
        :param log_std: initial value of log std
        """

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        super().__init__([state_dim, action_dim], is_disc_action=False)
        self.embedding = embedding

        self.activation = activation

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for hidden_size in hidden_sizes:
            self.affine_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def _init_input_output_dims(self, state_dim, action_dim):
        return state_dim, action_dim

    def forward(self, x, embedding_features=None):
        """
        :param x: State of Shape Batch x Features
        :param embedding_features: Either None or Batch x Sequence x Features
        :return: Mean, log of standard deviation and standard deviation of
        action distribution and additional embedding information if available
        """

        embedding_info = None
        if self.embedding is not None:
            assert embedding_features is not None, \
                "Can't run forward pass: Missing additional_features"
            embedding_output = self.embedding(embedding_features)

            # Some feature representations return additional information
            # about the embeddings (e.g. PointNet)
            if type(embedding_output) == tuple:
                x = torch.cat((x, embedding_output[0]), 1)
                embedding_info = embedding_output[1]
            else:
                x = torch.cat((x, embedding_output), 1)

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std, embedding_info

    def select_action(self, x, t):
        action_mean, _, action_std, _ = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action.data.cpu().numpy()[0], \
            dict(mean=action_mean, std=action_std)

    def get_kl(self, x):
        mean1, log_std1, std1, _ = self.forward(x)

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std, _ = self.forward(x)
        return self._normal_log_density(
            actions, action_mean, action_log_std, action_std)

    def get_entropy(self, x, actions):
        action_prob, _, _, _ = self.forward(x)
        action_prob_log = self.get_log_prob(x, actions)
        entropy = action_prob * action_prob_log
        # print("ENT")
        # print(entropy)
        return torch.sum(entropy, dim=1)

    @staticmethod
    def _normal_log_density(x, mean, log_std, std):
        # assert np.array(x)!=np.array(mean), print("SAME")
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var) - \
            0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)