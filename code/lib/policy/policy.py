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


from abc import abstractmethod
from typing import Union

import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class Policy(object):
    @abstractmethod
    def select_action(self, obs: Union[np.ndarray, Variable], t: int) -> \
            (int, dict):
        """
        Applies the policy for a given observation and time step
        :param obs: an observation vector of the environment
        :param t: the current time step
        :return: the chosen action of the policy as well as an info dictionary
        """


class PytorchPolicy(Policy, nn.Module):
    pass


class PGSupportingPolicy(PytorchPolicy):
    def __init__(self, init_io_args, is_disc_action):
        super().__init__()

        self.is_disc_action = is_disc_action
        self.input_dim, self.output_dim = \
            self._init_input_output_dims(*init_io_args)

    @abstractmethod
    def _init_input_output_dims(self, *args):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def select_action(self, obs: Union[np.ndarray, Variable], t: int) -> \
            (int, dict):
        raise NotImplementedError

    @abstractmethod
    def get_kl(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_log_prob(self, x, actions):
        raise NotImplementedError

    @abstractmethod
    def get_fim(self, x):
        raise NotImplementedError
