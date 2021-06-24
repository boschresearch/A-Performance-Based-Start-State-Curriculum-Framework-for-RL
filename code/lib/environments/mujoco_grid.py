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


import copy
import math
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lib.environments.helpers.intersection import check_intersection_object


class AntU:
    def __init__(self):
        self.env = gym.make('AntU-v2')
        self.start = (3.0, 3.0)
        self.goal = (3.0, 9.0)
        self.x_size = 60
        self.y_size = 60

        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        self.obstacle_list = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    self.obstacle_list.append((j * 3, i * 3))

        structure_occ = [
            [1,  1,   1,  1,   1],
            [1,  'r', 1,  'g', 1],
            [1,  0,   1,  0,   1],
            [1,  0,   0,  0,   1],
            [1,  1,   1,  1,   1],
        ]

        self.occupancy_map = np.zeros((self.x_size, self.y_size))
        self.occupancy_map_coarse = np.zeros((10, 10))
        for i in range(5):
            for j in range(5):
                if structure_occ[i][j] == 1:
                    for inner_i in range(max(i * 12 - 3, 0), min(i * 12 + 12 + 3, 60)):
                        for inner_j in range(max(j * 12 - 3, 0), min(j * 12 + 12 + 3, 60)):
                            self.occupancy_map[inner_i, inner_j] = 1
                    for inner_i in range(max(i * 2, 0), min(i * 2 + 2, 10)):
                        for inner_j in range(max(j * 2, 0), min(j * 2 + 2, 10)):
                            self.occupancy_map_coarse[inner_i, inner_j] = 1
        print(self.occupancy_map_coarse)
        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.rp_map = np.zeros((self.x_size, self.y_size))
        self.iter_count = 0
        self.default_starts = []
        for x_coord in range(self.x_size):
            for y_coord in range(self.y_size):
                if self.occupancy_map[x_coord][y_coord] == 0:
                    self.default_starts.append((x_coord, y_coord))

    def sample_random_pos(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-1.5, 13.5)
            y_coord = random.uniform(-1.5, 13.5)
            admissible = True
            for obstacle in self.obstacle_list:
                if obstacle[0] - 2.25 < x_coord < obstacle[0] + 2.25 and \
                        obstacle[1] - 2.25 < y_coord < obstacle[1] + 2.25:
                    admissible = False
            if admissible:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_full(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-1.5, 13.5)
            y_coord = random.uniform(-1.5, 13.5)
            admissible = True
            for obstacle in self.obstacle_list:
                if obstacle[0] - 2.25 < x_coord < obstacle[0] + 2.25 and \
                        obstacle[1] - 2.25 < y_coord < obstacle[1] + 2.25:
                    admissible = False
            if admissible:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_tile(self, tile, number):
        start_list = []
        counter = 0
        while len(start_list) < number:
            x_coord = -1.375 + tile[0]*0.25 + random.uniform(-0.125, 0.125)
            y_coord = -1.375 + tile[1]*0.25 + random.uniform(-0.125, 0.125)
            if self.occupancy_map[tile[0], tile[1]] == 0:
                start_list.append((x_coord, y_coord))
            counter += 1
        return start_list

    def check_collision(self, state):
        collision = False
        for obstacle in self.obstacle_list:
            if obstacle[0] - 2.25 < state[0] < obstacle[0] + 2.25 and obstacle[1] - 2.25 < state[1] < obstacle[1] + 2.25:
                collision = True
        return collision


class PointMassSpiralFine:
    def __init__(self):
        self.env = gym.make('Point-v2')
        self.start = (6.0, 6.0)
        self.goal = (10.0, 10.0)
        self.x_size = 56
        self.y_size = 56

        structure = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 'r', 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
        self.obstacle_list = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    self.obstacle_list.append((j * 2, i * 2))

        structure_occ = [
            [1,   1,   1,   1,   1,   1,   1],
            [1,   0,   0,   0,   0,   0,   1],
            [1,   0,   1,   1,   1,   0,   1],
            [1,   0,   1,   'r', 1,   0,   1],
            [1,   0,   1,   0,   1,   0,   1],
            [1,   0,   0,   0,   1,   'g', 1],
            [1,   1,   1,   1,   1,   1,   1],
        ]

        self.occupancy_map = np.zeros((self.x_size, self.y_size))
        self.occupancy_map_coarse = np.zeros((14, 14))
        for i in range(7):
            for j in range(7):
                if structure_occ[i][j] == 1:
                    for inner_i in range(max(i * 8 - 2, 0), min(i * 8 + 8 + 2, 56)):
                        for inner_j in range(max(j * 8 - 2, 0), min(j * 8 + 8 + 2, 56)):
                            self.occupancy_map[inner_i, inner_j] = 1
                    for inner_i in range(max(i * 2, 0), min(i * 2 + 2, 14)):
                        for inner_j in range(max(j * 2, 0), min(j * 2 + 2, 14)):
                            self.occupancy_map_coarse[inner_i, inner_j] = 1
        print(self.occupancy_map_coarse)
        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.rp_map = np.zeros((self.x_size, self.y_size))
        self.iter_count = 0
        self.default_starts = []
        for x_coord in range(self.x_size):
            for y_coord in range(self.y_size):
                if self.occupancy_map[x_coord][y_coord] == 0:
                    self.default_starts.append((x_coord, y_coord))
        self.obstacle_x = np.array([[1., 1., 11.],
                                    [3., 3., 9.],
                                    [5., 5., 7.],
                                    [9., 3., 5.],
                                    [11., 1., 11.]])
        self.obstacle_y = np.array([[1., 1., 11.],
                                    [3., 3., 9.],
                                    [5., 5., 9.],
                                    [7., 5., 11.],
                                    [9., 3., 11.],
                                    [11., 1., 11.]])
        self.objects = [(3.5, 7.5, 8, 2),
                        (3.5, 5.5, 2, 2),
                        (3.5, 3.5, 6, 2)]

    def check_collision_prm(self, line_seg):
        collision_total = False
        obj = 0
        coll_obj = []
        coll_seg_list = []
        for obstacles in self.objects:
            collision, coll_seg = check_intersection_object(geom_object=obstacles, line_seg=line_seg)
            if collision:
                coll_obj.append(obj)
                coll_seg_list.append(coll_seg)
            collision_total = collision_total or collision
            obj += 1
        return not collision_total, coll_obj, coll_seg_list

    def sample_random_pos(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-1., 13.)
            y_coord = random.uniform(-1., 13.)
            admissible = True
            for obstacle in self.obstacle_list:
                if obstacle[0] - 1.5 < x_coord < obstacle[0] + 1.5 and \
                        obstacle[1] - 1.5 < y_coord < obstacle[1] + 1.5:
                    admissible = False
            if admissible:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_full(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-1., 13.)
            y_coord = random.uniform(-1., 13.)
            admissible = True
            for obstacle in self.obstacle_list:
                if obstacle[0] - 1.5 < x_coord < obstacle[0] + 1.5 and \
                        obstacle[1] - 1.5 < y_coord < obstacle[1] + 1.5:
                    admissible = False
            if admissible:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_full_dist(self, number, max_dist, state):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(state[0] - max_dist, state[0] + max_dist)
            y_coord = random.uniform(state[1] - max_dist, state[1] + max_dist)
            admissible = True
            for obstacle in self.obstacle_list:
                if obstacle[0] - 1.5 < x_coord < obstacle[0] + 1.5 and \
                        obstacle[1] - 1.5 < y_coord < obstacle[1] + 1.5:
                    admissible = False
            if admissible:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_tile(self, tile, number):
        start_list = []
        counter = 0
        while len(start_list) < number:
            x_coord = -0.875 + tile[0]*0.25 + random.uniform(-0.125, 0.125)
            y_coord = -0.875 + tile[1]*0.25 + random.uniform(-0.125, 0.125)
            if self.occupancy_map[tile[0], tile[1]] == 0:
                start_list.append((x_coord, y_coord))
            counter += 1
        return start_list

    def check_collision(self, state):
        collision = False
        for obstacle in self.obstacle_list:
            if obstacle[0] - 1.5 < state[0] < obstacle[0] + 1.5 and obstacle[1] - 1.5 < state[1] < obstacle[1] + 1.5:
                collision = True
        return collision

    def get_occupancy(self, s_in):
        if 0 <= s_in[0] < 14 and 0 <= s_in[1] < 14:
            return self.occupancy_map_coarse[s_in[0], s_in[1]]
        else:
            return 1

    def get_neighborhood(self, state):
        return \
            self.get_occupancy(s_in=(state[0]-1, state[1]+1)), \
            self.get_occupancy(s_in=(state[0],   state[1]+1)), \
            self.get_occupancy(s_in=(state[0]+1, state[1]+1)), \
            self.get_occupancy(s_in=(state[0]-1, state[1])), \
            self.get_occupancy(s_in=(state[0]+1, state[1])), \
            self.get_occupancy(s_in=(state[0]-1, state[1]-1)), \
            self.get_occupancy(s_in=(state[0],   state[1]-1)), \
            self.get_occupancy(s_in=(state[0]+1, state[1]-1))

    def get_measurement(self, state):
        x_pos_list = []
        x_neg_list = []
        y_pos_list = []
        y_neg_list = []
        cross_1_list = []
        cross_2_list = []
        cross_3_list = []
        cross_4_list = []
        for obstacle in self.obstacle_x:
            if obstacle[1] <= state[1] <= obstacle[2]:
                x_dist = obstacle[0] - state[0]
                if x_dist > 0:
                    x_pos_list.append(copy.deepcopy(x_dist))
                    if obstacle[1] <= state[1] + x_dist <= obstacle[2]:
                        cross_4_list.append(copy.deepcopy(x_dist*math.sqrt(2)))
                    if obstacle[1] <= state[1] - x_dist <= obstacle[2]:
                        cross_1_list.append(copy.deepcopy(x_dist*math.sqrt(2)))
                else:
                    x_neg_list.append(copy.deepcopy(-x_dist))
                    if obstacle[1] <= state[1] - x_dist <= obstacle[2]:
                        cross_3_list.append(copy.deepcopy(-x_dist*math.sqrt(2)))
                    if obstacle[1] <= state[1] + x_dist <= obstacle[2]:
                        cross_2_list.append(copy.deepcopy(-x_dist*math.sqrt(2)))
        for obstacle in self.obstacle_y:
            if obstacle[1] <= state[0] <= obstacle[2]:
                y_dist = obstacle[0] - state[1]
                if y_dist > 0:
                    y_pos_list.append(copy.deepcopy(y_dist))
                    if obstacle[1] <= state[0] + y_dist <= obstacle[2]:
                        cross_4_list.append(copy.deepcopy(y_dist*math.sqrt(2)))
                    if obstacle[1] <= state[0] - y_dist <= obstacle[2]:
                        cross_3_list.append(copy.deepcopy(y_dist*math.sqrt(2)))
                else:
                    y_neg_list.append(copy.deepcopy(-y_dist))
                    if obstacle[1] <= state[0] - y_dist <= obstacle[2]:
                        cross_1_list.append(copy.deepcopy(-y_dist*math.sqrt(2)))
                    if obstacle[1] <= state[0] + y_dist <= obstacle[2]:
                        cross_2_list.append(copy.deepcopy(-y_dist*math.sqrt(2)))
        if not x_pos_list:
            x_pos_list.append(0)
        if not x_neg_list:
            x_neg_list.append(0)
        if not y_pos_list:
            y_pos_list.append(0)
        if not y_neg_list:
            y_neg_list.append(0)
        if not cross_1_list:
            cross_1_list.append(0)
        if not cross_2_list:
            cross_2_list.append(0)
        if not cross_3_list:
            cross_3_list.append(0)
        if not cross_4_list:
            cross_4_list.append(0)
        return min(x_pos_list) < 0.75, min(x_neg_list) < 0.75, min(y_pos_list) < 0.75, min(y_neg_list) < 0.75, \
               min(cross_1_list) < 0.75, min(cross_2_list) < 0.75, min(cross_3_list) < 0.75, min(cross_4_list) < 0.75

    def plot_graph(self, save_path, filename, v, e):

        color_dict = {0: 'gray', 1: 'black', 2: 'green', 3: 'red'}

        def plot_tile(x_coord, y_coord, plot_color):
            tile = matplotlib.patches.Rectangle((x_coord - 1, y_coord - 1), 1, 1, color=plot_color)
            ax.add_patch(tile)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i_p in range(14):
            if i_p > 0:
                ax.vlines(x=i_p - 1, ymin=-1, ymax=13, color='black')
            for j_p in range(14):
                plot_tile(x_coord=i_p, y_coord=j_p, plot_color=color_dict.get(self.occupancy_map_coarse[i_p][j_p]))
                if j_p > 0:
                    ax.hlines(y=j_p - 1, xmin=-1, xmax=13, color='black')
        ax.hlines(y=-1, xmin=-1, xmax=13, color='black')
        ax.hlines(y=13, xmin=-1, xmax=13, color='black')
        ax.vlines(x=-1, ymin=-1, ymax=13, color='black')
        ax.vlines(x=13, ymin=-1, ymax=13, color='black')
        plt.plot()
        plt.xlim([-1, 13])
        plt.ylim([-1, 13])

        for node in v:
            plt.plot(node[0], node[1], 'ro')
        for edge in e:
            plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'b-')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(filename)
        fig.savefig(save_path + filename)
        plt.close(fig)
