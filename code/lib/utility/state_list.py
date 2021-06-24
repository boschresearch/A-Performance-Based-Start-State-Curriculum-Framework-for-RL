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


from random import randint
from scipy.spatial.distance import cdist


def sample_list(input_list, num_samples, unique=True):
    len_l = len(input_list)
    l_idx = []
    l_out = []
    if len_l <= num_samples:
        return input_list
    else:
        while len(l_idx) < num_samples:
            idx = randint(0, len_l - 1)
            if unique:
                if idx not in l_idx:
                    l_idx.append(idx)
            else:
                l_idx.append(idx)
        for idx in l_idx:
            l_out.append(input_list[idx])
        return l_out


def uniquify_list(l_in):
    l_out = []
    for element in l_in:
        if element not in l_out:
            l_out.append(element)
    return l_out


class OldStarts:
    def __init__(self, dist_threshold, init):
        self.dist_threshold = dist_threshold
        self.start_list = [init]

    def add_starts(self, starts2add):
        for start2add in starts2add:
            if (cdist(XA=[start2add], XB=self.start_list, metric='euclidean').squeeze() > self.dist_threshold).all():
                self.start_list.append(start2add)
