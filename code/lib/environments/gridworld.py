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


import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

from lib.environments.helpers.intersection import *


# GRID WORLD DEFINITION #
class GridWorldContinuous:

    def __init__(self,
                 grid_type,
                 delta_t=0.1,
                 col_break=False,
                 coll_mode='elastic',
                 goal_rew=1,
                 col_rew=0,
                 time_rew=0,
                 a_lim=100,
                 v_lim=10,
                 usym=False
                 ):

        def add_object(object_params):
            for i_g in range(object_params[0], object_params[0] + object_params[2]):
                for j_g in range(object_params[1], object_params[1] + object_params[3]):
                    self.occupancy_map[i_g][j_g] = 1

        if grid_type == 0:
            self.x_size = 7
            self.y_size = 6
            self.start = np.array([1., 4.])
            self.goal = np.array([5., 4.])
            self.objects = []
        elif grid_type == 1:
            self.x_size = 7
            self.y_size = 6
            self.start = np.array([1., 4.])
            self.goal = np.array([5., 4.])
            self.objects = [(3, 2, 1, 4)]
        elif grid_type == 2:
            self.x_size = 10
            self.y_size = 6
            self.start = np.array([1., 4.])
            self.goal = np.array([8., 1.])
            self.objects = [(3, 2, 1, 4),
                            (6, 0, 1, 4)
                            ]
        elif grid_type == 3:
            self.x_size = 12
            self.y_size = 6
            self.start = np.array([1., 4.])
            self.goal = np.array([10., 1.])
            self.objects = [(3, 2, 1, 4),
                            (8, 0, 1, 4)
                            ]
        elif grid_type == 4:
            self.x_size = 20
            self.y_size = 20
            self.start = np.array([1., 1.])
            self.goal = np.array([18., 18.])
            self.objects = [(3, 3, 3, 3),
                            (4, 11, 6, 6),
                            (10, 2, 2, 1),
                            (11, 7, 1, 1),
                            (16, 2, 2, 4),
                            (14, 10, 2, 2),
                            (13, 15, 2, 2),
                            (18, 13, 1, 1)
                            ]
        elif grid_type == 5:
            self.x_size = 20
            self.y_size = 20
            self.start = np.array([1., 1.])
            self.goal = np.array([18., 18.])
            self.objects = [(3, 0, 1, 6),
                            (4, 5, 4, 1),
                            (4, 1, 2, 1),
                            (8, 2, 3, 1),
                            (11, 5, 9, 1),
                            (14, 2, 1, 3),
                            (17, 0, 1, 3),
                            (13, 8, 7, 1),
                            (15, 9, 1, 3),
                            (11, 10, 1, 2),
                            (3, 8, 1, 12),
                            (4, 11, 4, 1),
                            (4, 17, 3, 1),
                            (2, 13, 1, 1),
                            (0, 16, 1, 1),
                            (9, 14, 3, 1),
                            (11, 15, 1, 5),
                            (14, 17, 1, 3),
                            (15, 14, 5, 1),
                            (17, 15, 1, 2)
                            ]
        elif grid_type == 6:
            self.x_size = 30
            self.y_size = 20
            self.start = np.array([1., 1.])
            self.goal = np.array([25., 1.])
            self.objects = [(3, 3, 1, 4),
                            (4, 6, 3, 1),
                            (7, 6, 1, 3),
                            (7, 0, 1, 3),
                            (8, 2, 3, 1),
                            (10, 3, 1, 3),
                            (14, 5, 1, 7),
                            (13, 2, 3, 1),
                            (14, 0, 1, 2),
                            (19, 2, 1, 3),
                            (20, 4, 5, 1),
                            (23, 0, 1, 4),
                            (27, 0, 1, 5),
                            (24, 5, 1, 3),
                            (21, 6, 1, 1),
                            (18, 4, 1, 7),
                            (15, 11, 4, 1),
                            (8, 8, 6, 1),
                            (4, 10, 2, 1),
                            (5, 11, 9, 1),
                            (3, 14, 1, 6),
                            (23, 8, 4, 1),
                            (26, 9, 1, 3),
                            (22, 8, 1, 7),
                            (27, 15, 3, 1),
                            (26, 15, 1, 2),
                            (7, 15, 16, 1),
                            (16, 16, 1, 2),
                            (10, 18, 1, 2)
                            ]

        self.occupancy_map = np.zeros((self.x_size, self.y_size))
        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.rp_map = np.zeros((self.x_size, self.y_size))
        self.state = None
        self.start = None
        self.dt = delta_t
        self.iter_count = 0
        self.col_break = col_break
        self.coll_mode = coll_mode
        self.goal_rew = goal_rew
        self.col_rew = col_rew
        self.time_rew = time_rew
        self.a_lim = a_lim
        self.v_lim = v_lim
        self.usym = usym
        self.state_cache = np.zeros((self.x_size, self.y_size, 8))
        self.default_starts = []
        for item in self.objects:
            add_object(item)
        for x_coord in range(self.x_size):
            for y_coord in range(self.y_size):
                if self.occupancy_map[x_coord][y_coord] == 0:
                    self.default_starts.append((x_coord, y_coord))
        for start in self.default_starts:
            self.state_cache[start[0], start[1], :] = list(self.get_neighborhood(start))
        np.save("occupancy_map", np.array(self.occupancy_map))
        self.occupancy_map_padded = \
            np.logical_not(np.pad(np.array(self.occupancy_map), 1, 'constant', constant_values=1)).astype(int)
        self.occupancy_map_un_padded = \
            np.logical_not(np.array(self.occupancy_map)).astype(int)

    def check_occupancy(self, position):
        if self.occupancy_map[min(max(0, int(round(position[0]))), self.x_size)][min(max(0, int(round(position[1]))), self.y_size)] == 0:
            return False
        else:
            return True

    def reset(self, start_p, goal_p):
        self.iter_count = 0
        self.start = start_p
        self.goal = goal_p
        # self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.state = np.array([0, 0] + list(self.start))
        return self.state

    def reset_v(self, start_p, goal_p):
        self.iter_count = 0
        self.start = start_p
        self.goal = goal_p
        # self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.state = np.array(list(self.start))
        return self.state

    def step(self, action, eps_val=1e-3):

        pos_list = []

        action = np.clip(action, a_min=-self.a_lim, a_max=self.a_lim)

        if self.usym:
            if action[0] > 0:
                a0 = 1.
            else:
                a0 = 0.25
            if action[1] > 0:
                a1 = 1.
            else:
                a1 = 0.25

        if self.usym:
            self.state[0:2] = np.clip(self.state[0:2] + action * np.array([a0, a1]) * self.dt, a_min=-self.v_lim, a_max=self.v_lim)
        else:
            self.state[0:2] = np.clip(self.state[0:2] + action * self.dt, a_min=-self.v_lim, a_max=self.v_lim)

        p_new = self.state[2:4] + self.state[0:2] * self.dt
        p_new_old = p_new[:]

        pos_list.append(tuple(self.state[2:4]))

        collision_free, causing_obj, causing_seg_list = \
            self.check_collision(line_seg=(self.state[2], self.state[3], p_new[0], p_new[1]))
        collision_var_tot = \
            (not (-0.5 < p_new[0] < self.x_size - 0.5)) or \
            (not (-0.5 < p_new[1] < self.y_size - 0.5)) or \
            not collision_free
        collision_indicator = collision_var_tot

        if collision_var_tot:
            if self.coll_mode == 'inelastic':
                while collision_var_tot:
                    if not collision_free:
                        coll_tuples = []
                        for obj_i in range(len(causing_obj)):
                            seg_i = 0
                            for seg in causing_seg_list[obj_i]:
                                if seg:
                                    coll_tuples.append((causing_obj[obj_i], seg_i))
                                seg_i += 1
                        coll_point_list = []
                        for coll_tuple in coll_tuples:
                            coll_point_temp, normal_vec = \
                                calc_line_object_border_intersection_point(p1=self.state[2:4],
                                                                           p2=p_new,
                                                                           obj_def=self.objects[coll_tuple[0]],
                                                                           seg_num=coll_tuple[1])
                            coll_point_list.append((coll_point_temp, normal_vec))
                        coll_dist_list = []
                        for coll_point_element in coll_point_list:
                            coll_dist_list.append(calc_point_dist(p1=self.state[2:4], p2=coll_point_element[0]))
                        coll_idx = coll_dist_list.index(min(coll_dist_list))
                        coll_point_final, normal_vec = coll_point_list[coll_idx]
                        coll_vec_in, coll_vec_in_len = calc_dir_vec_norm(p1=self.state[2:4], p2=coll_point_final)
                        coll_vec_out = np.array(coll_vec_in) - 2*np.dot(np.array(coll_vec_in),
                                                                        np.array(normal_vec))*np.array(normal_vec)
                        p_new = tuple(np.array(coll_point_final) + (calc_point_dist(p1=self.state[2:4], p2=p_new)
                                                                    - coll_vec_in_len - eps_val)*coll_vec_out)
                    else:
                        bound_seg_list = [(-0.5, -0.5, -0.5, self.y_size-0.5),
                                          (self.x_size - 0.5, -0.5, self.x_size - 0.5, self.y_size - 0.5),
                                          (-0.5, self.y_size - 0.5, self.x_size - 0.5, self.y_size - 0.5),
                                          (-0.5, -0.5, self.x_size - 0.5, -0.5)]
                        left_bound_col = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[0],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        right_bound_col = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[1],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        up_bound_col = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[2],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        down_bound_col = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[3],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        coll_idx = 0
                        for bound_col in [left_bound_col, right_bound_col, up_bound_col, down_bound_col]:
                            if bound_col:
                                coll_bound_seg = bound_seg_list[coll_idx]
                            coll_idx += 1
                        coll_point_final = calc_line_intersection_point(p1_l1=self.state[2:4],
                                                                        p2_l1=p_new,
                                                                        p1_l2=coll_bound_seg[0:2],
                                                                        p2_l2=coll_bound_seg[2:4])
                        coll_vec_in, coll_vec_in_len = calc_dir_vec_norm(p1=self.state[2:4], p2=coll_point_final)
                        if coll_bound_seg[0] == coll_bound_seg[2]:
                            normal_vec = (1, 0)
                        elif coll_bound_seg[1] == coll_bound_seg[3]:
                            normal_vec = (0, 1)
                        coll_vec_out = np.array(coll_vec_in) - 2 * np.dot(np.array(coll_vec_in),
                                                                          np.array(normal_vec)) * np.array(normal_vec)
                        p_new = tuple(np.array(coll_point_final) + (calc_point_dist(p1=self.state[2:4], p2=p_new) -
                                                                    coll_vec_in_len - eps_val) * coll_vec_out)

                    if tuple(p_new) == tuple(p_new_old):
                        print("ERROR")
                        break
                    p_new_old = p_new[:]
                    pos_list.append(tuple(coll_point_final))
                    coll_point_final = list(coll_point_final)
                    coll_point_final[0] = self.state[2] + (1 - 2 * eps_val) * (coll_point_final[0] - self.state[2])
                    coll_point_final[1] = self.state[3] + (1 - 2 * eps_val) * (coll_point_final[1] - self.state[3])
                    collision_free, causing_obj, causing_seg_list = \
                        self.check_collision(line_seg=(coll_point_final[0], coll_point_final[1], p_new[0], p_new[1]))
                    collision_var_tot = \
                        (not (-0.5 < p_new[0] < self.x_size - 0.5)) or \
                        (not (-0.5 < p_new[1] < self.y_size - 0.5)) or \
                        not collision_free
                    if collision_var_tot:
                        self.state[2] = coll_point_final[0]
                        self.state[3] = coll_point_final[1]

                self.state[2] = p_new[0]
                self.state[3] = p_new[1]
                pos_list.append(tuple(self.state[2:4]))
            elif self.coll_mode == 'elastic':
                if collision_var_tot:
                    collision_not_found = False
                    if not collision_free:
                        # print("here")
                        # print(causing_obj)
                        # print(causing_seg_list)
                        coll_tuples = []
                        for obj_i in range(len(causing_obj)):
                            seg_i = 0
                            for seg in causing_seg_list[obj_i]:
                                if seg:
                                    coll_tuples.append((causing_obj[obj_i], seg_i))
                                seg_i += 1
                        coll_point_list = []
                        for coll_tuple in coll_tuples:
                            coll_point_temp, _ = calc_line_object_border_intersection_point(
                                p1=self.state[2:4], p2=p_new, obj_def=self.objects[coll_tuple[0]], seg_num=coll_tuple[1])
                            coll_point_list.append(coll_point_temp)
                        coll_dist_list = []
                        for coll_point_element in coll_point_list:
                            coll_dist_list.append(calc_point_dist(p1=self.state[2:4], p2=coll_point_element))
                        coll_idx = coll_dist_list.index(min(coll_dist_list))
                        coll_point_final = coll_point_list[coll_idx]
                    else:
                        bound_seg_list = [(-0.5, -0.5, -0.5, self.y_size - 0.5),
                                          (self.x_size - 0.5, -0.5, self.x_size - 0.5, self.y_size - 0.5),
                                          (-0.5, self.y_size - 0.5, self.x_size - 0.5, self.y_size - 0.5),
                                          (-0.5, -0.5, self.x_size - 0.5, -0.5)]
                        # print("HHH")
                        left_bound_col, save1 = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[0],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        right_bound_col, save2 = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[1],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        up_bound_col, save3 = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[2],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        down_bound_col, save4 = check_intersection_line_seg(
                            line_seg_1=bound_seg_list[3],
                            line_seg_2=(self.state[2], self.state[3], p_new[0], p_new[1]))
                        coll_idx = 0
                        # print([left_bound_col, right_bound_col, up_bound_col, down_bound_col])
                        # print("\n")
                        coll_bound_seg = []
                        for bound_col in [left_bound_col, right_bound_col, up_bound_col, down_bound_col]:
                            if bound_col:
                                coll_bound_seg = bound_seg_list[coll_idx]
                            coll_idx += 1
                        if not coll_bound_seg:
                            with open('error_log.txt', 'a') as f:
                                f.write(str(p_new))
                                f.write(str(save1))
                                f.write(str(save2))
                                f.write(str(save3))
                                f.write(str(save4))
                                print("COLLISION NOT FOUND")
                                collision_not_found = True
                        else:
                            coll_point_final = calc_line_intersection_point(p1_l1=self.state[2:4],
                                                                            p2_l1=p_new,
                                                                            p1_l2=coll_bound_seg[0:2],
                                                                            p2_l2=coll_bound_seg[2:4])

                    if collision_not_found:
                        self.state[0] = 0
                        self.state[1] = 0
                        self.state[2] = p_new[0]
                        self.state[3] = p_new[1]
                        pos_list.append(tuple(self.state[2:4]))
                        self.state[2] = np.clip(self.state[2], a_min=-0.5, a_max=self.x_size - 0.5)
                        self.state[3] = np.clip(self.state[3], a_min=-0.5, a_max=self.y_size - 0.5)
                    else:
                        self.state[0] = 0
                        self.state[1] = 0
                        self.state[2] = self.state[2] + (1 - 2 * eps_val) * (coll_point_final[0] - self.state[2])
                        self.state[3] = self.state[3] + (1 - 2 * eps_val) * (coll_point_final[1] - self.state[3])
                        pos_list.append(tuple(self.state[2:4]))
        else:
            self.state[2:4] = p_new

        self.visitation_map[max(min(int(round(self.state[2])), 19), 0), max(min(int(round(self.state[3])), 19), 0)] += 1
        done = (np.round(self.state[2:4]) == self.goal).all()
        reward = 0
        if done:
            reward = self.goal_rew
        if self.col_break:
            break_var = done or collision_indicator
        else:
            break_var = done
        self.iter_count += 1

        return self.state, reward, done, not break_var, pos_list

    def roll_out(self, curr_state, action):
        # print("\n")
        # print(curr_state)
        """
        s_full = np.array([0, 0] + list(curr_state) + list(self.state_cache[min(int(round(curr_state[0])), self.x_size-1),
                                                                            min(int(round(curr_state[1])), self.y_size-1)])).squeeze()
        """
        s_full = np.array([0, 0] + list(curr_state))
        self.state = s_full
        full_state, _, _, _, _ = self.step(action)
        return full_state[2:4]

    def roll_out_full(self, curr_state, action):
        self.state = curr_state[:4]
        full_state, _, _, _, _ = self.step(action)
        return full_state

    def check_collision(self, line_seg):
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

    def plot(self, pos, filename):

        color_dict = {0: 'gray', 1: 'black', 2: 'green', 3: 'red'}

        def plot_tile(x_coord, y_coord, plot_color):
            tile = matplotlib.patches.Rectangle((x_coord - 0.5, y_coord - 0.5), 1, 1, color=plot_color)
            ax.add_patch(tile)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i_p in range(self.x_size):
            if i_p > 0:
                ax.vlines(x=i_p - 0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
            for j_p in range(self.y_size):
                plot_tile(x_coord=i_p, y_coord=j_p, plot_color=color_dict.get(self.occupancy_map[i_p][j_p]))
                if j_p > 0:
                    ax.hlines(y=j_p - 0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.hlines(y=-0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.hlines(y=self.y_size - 0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.vlines(x=-0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
        ax.vlines(x=self.x_size - 0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
        plot_tile(x_coord=self.goal[0], y_coord=self.goal[1], plot_color=color_dict.get(3))
        plt.plot()
        plt.xlim([-1, self.x_size])
        plt.ylim([-1, self.y_size])
        for ij in range(len(pos) - 1):
            plt.plot([pos[ij][0], pos[ij + 1][0]], [pos[ij][1], pos[ij + 1][1]], color="orange")
        plt.xlim([-1, self.x_size])
        plt.ylim([-1, self.y_size])
        plt.gca().set_aspect('equal', adjustable='box')
        fig.savefig(filename)
        plt.close(fig)

    def plot_grid(self, save_path, filename, trajectory):

        color_dict = {0: 'gray', 1: 'black', 2: 'green', 3: 'red'}

        def plot_tile(x_coord, y_coord, plot_color):
            tile = matplotlib.patches.Rectangle((x_coord - 0.5, y_coord - 0.5), 1, 1, color=plot_color)
            ax.add_patch(tile)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i_p in range(self.x_size):
            if i_p > 0:
                ax.vlines(x=i_p - 0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
            for j_p in range(self.y_size):
                plot_tile(x_coord=i_p, y_coord=j_p, plot_color=color_dict.get(self.occupancy_map[i_p][j_p]))
                if j_p > 0:
                    ax.hlines(y=j_p - 0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.hlines(y=-0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.hlines(y=self.y_size - 0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.vlines(x=-0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
        ax.vlines(x=self.x_size - 0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
        plot_tile(x_coord=1, y_coord=1, plot_color=color_dict.get(2))
        plot_tile(x_coord=25, y_coord=1, plot_color=color_dict.get(3))
        plt.plot()
        plt.xlim([-1, self.x_size])
        plt.ylim([-1, self.y_size])
        for path_pair in trajectory:
            plt.plot([path_pair[0][2], path_pair[1][2]],
                     [path_pair[0][3], path_pair[1][3]], color="orange")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(filename)
        fig.savefig(save_path + filename)
        plt.close(fig)

    def sample_random_pos(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-0.5, self.x_size - 0.5)
            y_coord = random.uniform(-0.5, self.y_size - 0.5)
            if self.occupancy_map[round(x_coord)][round(y_coord)] == 0 and (x_coord, y_coord) not in start_list:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_tile(self, tile_x, tile_y):
        x_coord = tile_x + random.uniform(-0.5, 0.5)
        y_coord = tile_y + random.uniform(-0.5, 0.5)
        start_c = (x_coord, y_coord)
        return start_c

    def sample_random_pos_full(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-0.5, self.x_size - 0.5)
            y_coord = random.uniform(-0.5, self.y_size - 0.5)
            if self.occupancy_map[round(x_coord)][round(y_coord)] == 0 and (x_coord, y_coord) not in start_list:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_full_v(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-0.5, self.x_size - 0.5)
            y_coord = random.uniform(-0.5, self.y_size - 0.5)
            x_vel = random.uniform(-10., 10.)
            y_vel = random.uniform(-10., 10.)
            if self.occupancy_map[round(x_coord)][round(y_coord)] == 0 and (x_coord, y_coord) not in start_list:
                start_list.append((x_vel, y_vel, x_coord, y_coord))
        return start_list

    def sample_random_pos_full_dist(self, number, max_dist, state):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(state[0] - max_dist, state[0] + max_dist)
            y_coord = random.uniform(state[1] - max_dist, state[1] + max_dist)
            if ((x_coord, y_coord) not in start_list) and -0.5 < x_coord < self.x_size - 0.5 and \
                    -0.5 < y_coord < self.y_size - 0.5 and self.occupancy_map[round(x_coord)][round(y_coord)] == 0:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_pos_full_dist_circle(self, number, state, radius=1):
        start_list = []
        while len(start_list) < number:
            angle = random.uniform(0, 2*math.pi)
            x_coord = state[0] + radius * math.cos(angle)
            y_coord = state[1] + radius * math.sin(angle)
            if ((x_coord, y_coord) not in start_list) and -0.5 < x_coord < self.x_size - 0.5 and \
                    -0.5 < y_coord < self.y_size - 0.5 and self.occupancy_map[round(x_coord)][round(y_coord)] == 0:
                start_list.append((x_coord, y_coord))
        return start_list

    def get_occupancy(self, s_in):
        if 0 <= s_in[0] < self.x_size and 0 <= s_in[1] < self.y_size:
            return self.occupancy_map[s_in[0], s_in[1]]
        else:
            return 1

    def get_neighborhood(self, state):

        return self.get_occupancy(s_in=(state[0]-1, state[1]+1)), self.get_occupancy(s_in=(state[0], state[1]+1)), \
               self.get_occupancy(s_in=(state[0]+1, state[1]+1)), self.get_occupancy(s_in=(state[0]-1, state[1])), \
               self.get_occupancy(s_in=(state[0]+1, state[1])), self.get_occupancy(s_in=(state[0]-1, state[1]-1)), \
               self.get_occupancy(s_in=(state[0], state[1]-1)), self.get_occupancy(s_in=(state[0]+1, state[1]-1))


class GridWorldDiscrete:

    def __init__(self, grid_type, goal_rew=1, col_rew=0, time_rew=0, col_break=False):

        def add_object(object_params):
            for i_g in range(object_params[0], object_params[0] + object_params[2]):
                for j_g in range(object_params[1], object_params[1] + object_params[3]):
                    self.occupancy_map[i_g][j_g] = 1

        self.state = None
        self.start = None
        self.iter_count = 0
        self.col_break = col_break
        self.goal_rew = goal_rew
        self.col_rew = col_rew
        self.time_rew = time_rew
        if grid_type == 0:
            self.x_size = 7
            self.y_size = 6
            self.start = np.array([1, 4])
            self.goal = np.array([5, 4])
            self.objects = []
        elif grid_type == 1:
            self.x_size = 7
            self.y_size = 6
            self.start = np.array([1, 4])
            self.goal = np.array([5, 4])
            self.objects = [(3, 2, 1, 4)]
        elif grid_type == 2:
            self.x_size = 10
            self.y_size = 6
            self.start = np.array([1, 4])
            self.goal = np.array([8, 1])
            self.objects = [(3, 2, 1, 4),
                            (6, 0, 1, 4)
                            ]
        elif grid_type == 3:
            self.x_size = 12
            self.y_size = 6
            self.start = np.array([1, 4])
            self.goal = np.array([10, 1])
            self.objects = [(3, 2, 1, 4),
                            (8, 0, 1, 4)
                            ]
        elif grid_type == 4:
            self.x_size = 20
            self.y_size = 20
            self.start = np.array([1, 1])
            self.goal = np.array([18, 18])
            self.objects = [(3, 3, 3, 3),
                            (4, 11, 6, 6),
                            (10, 2, 2, 1),
                            (11, 7, 1, 1),
                            (16, 2, 2, 4),
                            (14, 10, 2, 2),
                            (13, 15, 2, 2),
                            (18, 13, 1, 1)
                            ]
        elif grid_type == 5:
            self.x_size = 20
            self.y_size = 20
            self.start = np.array([1, 1])
            self.goal = np.array([18, 18])
            self.objects = [(3, 0, 1, 6),
                            (4, 5, 4, 1),
                            (4, 1, 2, 1),
                            (8, 2, 3, 1),
                            (11, 5, 9, 1),
                            (14, 2, 1, 3),
                            (17, 0, 1, 3),
                            (13, 8, 7, 1),
                            (15, 9, 1, 3),
                            (11, 10, 1, 2),
                            (3, 8, 1, 12),
                            (4, 11, 4, 1),
                            (4, 17, 3, 1),
                            (2, 13, 1, 1),
                            (0, 16, 1, 1),
                            (9, 14, 3, 1),
                            (11, 15, 1, 5),
                            (14, 17, 1, 3),
                            (15, 14, 5, 1),
                            (17, 15, 1, 2)
                            ]
        elif grid_type == 6:
            self.x_size = 30
            self.y_size = 20
            self.start = np.array([1, 1])
            self.start2 = np.array([2, 10])
            self.start3 = np.array([10, 13])
            self.start4 = np.array([5, 16])
            self.start5 = np.array([13, 17])
            self.goal = np.array([25, 1])
            self.objects = [(3, 3, 1, 4),
                            (4, 6, 3, 1),
                            (7, 6, 1, 3),
                            (7, 0, 1, 3),
                            (8, 2, 3, 1),
                            (10, 3, 1, 3),
                            (14, 5, 1, 7),
                            (13, 2, 3, 1),
                            (14, 0, 1, 2),
                            (19, 2, 1, 3),
                            (20, 4, 5, 1),
                            (23, 0, 1, 4),
                            (27, 0, 1, 5),
                            (24, 5, 1, 3),
                            (21, 6, 1, 1),
                            (18, 4, 1, 7),
                            (15, 11, 4, 1),
                            (8, 8, 6, 1),
                            (4, 10, 2, 1),
                            (5, 11, 9, 1),
                            (3, 14, 1, 6),
                            (23, 8, 4, 1),
                            (26, 9, 1, 3),
                            (22, 8, 1, 7),
                            (27, 15, 3, 1),
                            (26, 15, 1, 2),
                            (7, 15, 16, 1),
                            (16, 16, 1, 2),
                            (10, 18, 1, 2)
                            ]

        self.occupancy_map = np.zeros((self.x_size, self.y_size))
        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.rp_map = np.zeros((self.x_size, self.y_size))

        for item in self.objects:
            add_object(item)

    def get_occupancy(self, s_in):
        if 0 <= s_in[0] < self.x_size and 0 <= s_in[1] < self.y_size:
            return self.occupancy_map[s_in[0], s_in[1]]
        else:
            return 1

    def get_neighborhood(self, state):

        return self.get_occupancy(s_in=(state[0]-1, state[1]+1)), self.get_occupancy(s_in=(state[0], state[1]+1)), \
               self.get_occupancy(s_in=(state[0]+1, state[1]+1)), self.get_occupancy(s_in=(state[0]-1, state[1])), \
               self.get_occupancy(s_in=(state[0]+1, state[1])), self.get_occupancy(s_in=(state[0]-1, state[1]-1)), \
               self.get_occupancy(s_in=(state[0], state[1]-1)), self.get_occupancy(s_in=(state[0]+1, state[1]-1))
        """
        return self.get_occupancy(s_in=(state[0], state[1] + 1)), self.get_occupancy(s_in=(state[0] - 1, state[1])), \
               self.get_occupancy(s_in=(state[0] + 1, state[1])), self.get_occupancy(s_in=(state[0], state[1] - 1))
        """
    def step(self, action):

        def up():
            if self.state[1] < self.y_size - 1 and self.occupancy_map[self.state[0]][self.state[1] + 1] == 0:
                return np.array([self.state[0], self.state[1] + 1]), self.time_rew, False
            else:
                return np.array([self.state[0], self.state[1]]), self.col_rew, True

        def right():
            if self.state[0] < self.x_size - 1 and self.occupancy_map[self.state[0] + 1][self.state[1]] == 0:
                return np.array([self.state[0] + 1, self.state[1]]), self.time_rew, False
            else:
                return np.array([self.state[0], self.state[1]]), self.col_rew, True

        def down():
            if self.state[1] > 0 and self.occupancy_map[self.state[0]][self.state[1] - 1] == 0:
                return np.array([self.state[0], self.state[1] - 1]), self.time_rew, False
            else:
                return np.array([self.state[0], self.state[1]]), self.col_rew, True

        def left():
            if self.state[0] > 0 and self.occupancy_map[self.state[0] - 1][self.state[1]] == 0:
                return np.array([self.state[0] - 1, self.state[1]]), self.time_rew, False
            else:
                return np.array([self.state[0], self.state[1]]), self.col_rew, True

        def action_switch_case(a):
            switch = {
                0: up,
                1: right,
                2: down,
                3: left
            }
            func = switch.get(a, lambda: "nothing")
            return func()

        old_state = copy.deepcopy(self.state)
        self.state, reward, collision = action_switch_case(action)
        done = (self.state == self.goal).all()
        if done:
            reward = self.goal_rew
        """
        else:
            rp_diff = self.rp_map[self.state[0], self.state[1]] - self.rp_map[old_state[0], old_state[1]]
            if rp_diff > 0.05:
                reward = 0.01
            elif rp_diff < -0.05:
                reward = -0.01
        """
        if self.col_break:
            break_var = done or collision
        else:
            break_var = done
        self.visitation_map[self.state[0], self.state[1]] += 1
        self.iter_count += 1

        return np.array(self.state), reward, done, not break_var

    def roll_out(self, curr_state, action):

        def up():
            if curr_state[1] < self.y_size - 1 and self.occupancy_map[curr_state[0]][curr_state[1] + 1] == 0:
                return np.array([curr_state[0], curr_state[1] + 1])
            else:
                return np.array([curr_state[0], curr_state[1]])

        def right():
            if curr_state[0] < self.x_size - 1 and self.occupancy_map[curr_state[0] + 1][curr_state[1]] == 0:
                return np.array([curr_state[0] + 1, curr_state[1]])
            else:
                return np.array([curr_state[0], curr_state[1]])

        def down():
            if curr_state[1] > 0 and self.occupancy_map[curr_state[0]][curr_state[1] - 1] == 0:
                return np.array([curr_state[0], curr_state[1] - 1])
            else:
                return np.array([curr_state[0], curr_state[1]])

        def left():
            if curr_state[0] > 0 and self.occupancy_map[curr_state[0] - 1][curr_state[1]] == 0:
                return np.array([curr_state[0] - 1, curr_state[1]])
            else:
                return np.array([curr_state[0], curr_state[1]])

        def action_switch_case(a):
            switch = {
                0: up,
                1: right,
                2: down,
                3: left
            }
            func = switch.get(a, lambda: "nothing")
            return func()

        return action_switch_case(action)

    def reset(self, start_t, goal_t):
        self.start = start_t
        self.goal = goal_t
        # self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.visitation_map[self.start[0], self.start[1]] += 1
        self.iter_count = 0
        self.state = np.array(self.start)
        return self.state

    def sample_random_pos(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.randint(0, self.x_size - 1)
            y_coord = random.randint(0, self.y_size - 1)
            if self.occupancy_map[x_coord][y_coord] == 0 and (x_coord, y_coord) not in start_list:
                start_list.append((x_coord, y_coord))
        return start_list

    def check_collision(self, line_seg):
        collision_total = False
        for obstacles in self.objects:
            collision, _ = check_intersection_object(geom_object=obstacles, line_seg=line_seg)
            collision_total = collision_total or collision
        return not collision_total

    def plot_grid(self):

        color_dict = {0: 'gray', 1: 'black', 2: 'green', 3: 'red'}

        def plot_tile(x_coord, y_coord, plot_color):
            tile = matplotlib.patches.Rectangle((x_coord - 0.5, y_coord - 0.5), 1, 1, color=plot_color)
            ax.add_patch(tile)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i_p in range(self.x_size):
            if i_p > 0:
                ax.vlines(x=i_p - 0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
            for j_p in range(self.y_size):
                plot_tile(x_coord=i_p, y_coord=j_p, plot_color=color_dict.get(self.occupancy_map[i_p][j_p]))
                if j_p > 0:
                    ax.hlines(y=j_p - 0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.hlines(y=-0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.hlines(y=self.y_size - 0.5, xmin=-0.5, xmax=self.x_size - 0.5, color='black')
        ax.vlines(x=-0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
        ax.vlines(x=self.x_size - 0.5, ymin=-0.5, ymax=self.y_size - 0.5, color='black')
        plot_tile(x_coord=1, y_coord=1, plot_color=color_dict.get(2))
        plot_tile(x_coord=25, y_coord=1, plot_color=color_dict.get(3))
        plt.plot()
        plt.xlim([-1, self.x_size])
        plt.ylim([-1, self.y_size])
        plt.gca().set_aspect('equal', adjustable='box')
        fig.savefig('grid_6.png')
        plt.close(fig)
