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
import random
import numpy as np
from random import randint
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.mujoco_grid import PointMassSpiralFine


class TRPOTrainer:

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_sizes,
                 max_kl,
                 damping,
                 batch_size,
                 inner_episodes,
                 max_iter,
                 use_fim=False,
                 use_gpu=False
                 ):

        self.grid = PointMassSpiralFine()
        self.policy = ContinuousMLP(state_dim,
                                    action_dim,
                                    hidden_sizes=hidden_sizes,
                                    activation=F.relu)
        self.alice = ContinuousMLP(state_dim,
                                   action_dim + 1,
                                   hidden_sizes=hidden_sizes,
                                   activation=F.relu)
        self.optimizer = TRPO(policy=self.policy,
                              use_gpu=use_gpu,
                              max_kl=max_kl,
                              damping=damping,
                              use_fim=use_fim)
        self.optimizer_alice = TRPO(policy=self.policy,
                                    use_gpu=use_gpu,
                                    max_kl=max_kl,
                                    damping=damping,
                                    use_fim=use_fim)
        self.batch_size = batch_size
        self.inner_episodes = inner_episodes
        self.max_iter = max_iter
        self.state_dim = state_dim
        self.val_eval_states = []

    def roll_out_in_env(self, start, goal):
        roll_out = Batch()
        s = self.grid.env.reset(start)
        s_list = []
        a_list = []
        r_list = []
        info = True
        for step_i in range(self.max_iter):
            s_save = copy.deepcopy(np.array(list(s) + list(self.grid.get_measurement(state=(s[0], s[1])))))
            s_list.append(s_save)
            s = Variable(torch.from_numpy(np.float32(s_save))).unsqueeze(0)
            a, _ = self.policy.select_action(s, t=step_i)
            a_list.append(copy.deepcopy(a))
            s_new, r, d, info = self.grid.env.step(action=a, goal=goal, env_col=self.grid)
            r_list.append(r)
            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new.astype(np.float32)],
                      [0 if ((not info['no_goal_reached']) or (step_i + 1 == self.max_iter)) else 1],
                      [info]))
            s = s_new
            if not info['no_goal_reached']:
                break
        return roll_out, s_list, a_list, r_list, not info['no_goal_reached']

    def roll_out_in_env_asym_selfplay(self, goal, gamma=1.0):
        roll_out = Batch()
        roll_out_alice = Batch()
        s_alice = self.grid.env.reset(goal)
        s_list = []
        a_list = []
        r_list = []
        s_list_alice = []
        a_list_alice = []
        r_list_alice = []
        d = 0
        t_alice = 2*self.max_iter

        # TURN OF ALICE
        for step_i in range(2*self.max_iter):
            s_alice_save = copy.deepcopy(np.array(list(s_alice) + list(self.grid.get_measurement(state=(s_alice[0], s_alice[1])))))
            s_list_alice.append(s_alice_save)
            s_alice = Variable(torch.from_numpy(np.float32(s_alice_save))).unsqueeze(0)
            a_alice, _ = self.alice.select_action(s_alice, t=step_i)
            s_new_alice, r_alice, d_alice, info_alice = self.grid.env.step(action=a_alice[:2],
                                                                           goal=[1000, 1000],
                                                                           env_col=self.grid)
            if np.tanh(a_alice[-1]) > 0.99:
                info_alice['no_goal_reached'] = False
            a_list_alice.append(a_alice)
            r_list_alice.append(r_alice)
            roll_out_alice.append(
                Batch([a_alice.astype(np.float32)],
                      [s_alice_save.astype(np.float32)],
                      [r_alice],
                      [s_new_alice.astype(np.float32)],
                      [0 if ((not info_alice['no_goal_reached']) or (step_i + 1 == 2*self.max_iter)) else 1],
                      [info_alice]))
            s_alice = s_new_alice
            if not info_alice['no_goal_reached']:
                t_alice = step_i + 1
                break

        # TURN OF BOB
        s = self.grid.env.reset((s_alice[0], s_alice[1]))
        t_bob = 2*self.max_iter - t_alice
        t_bob = max(t_bob, 5)
        for step_j in range(t_bob):
            s_save = copy.deepcopy(np.array(list(s) + list(self.grid.get_measurement(state=(s[0], s[1])))))
            s_list.append(s_save)
            s = Variable(torch.from_numpy(np.float32(s_save))).unsqueeze(0)
            a, _ = self.policy.select_action(s, t=step_j)
            s_new, r, d, info = self.grid.env.step(action=a, goal=goal, env_col=self.grid)
            a_list.append(a)
            r_list.append(r)
            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new.astype(np.float32)],
                      [0 if ((not info['no_goal_reached']) or (step_j + 1 == t_bob)) else 1],
                      [info]))
            s = s_new
            if not info['no_goal_reached']:
                t_bob = step_j + 1
                break
        roll_out_alice['rewards'][-1] = gamma * max(0, t_bob - t_alice)
        # print(roll_out_alice['rewards'])
        if roll_out.length() > 0:
            roll_out['rewards'][-1] = -gamma * t_bob
            # print(roll_out['rewards'])

        return roll_out_alice, roll_out, s_list, a_list, r_list, d

    def simulate_env(self, start, goal, mode, probs, smode):
        batch = Batch()
        state_list = []
        action_list = []
        reward_list = []
        start_list = []

        num_roll_outs = 0
        num_steps = 0
        total_success = 0
        j = 0.

        if mode == 'train':
            len_list = []
            while num_steps < self.batch_size:
                if smode:
                    start_pos_pre = random.choices(population=start, weights=probs)
                    start_pos = self.grid.sample_random_pos_tile(tile=(int(start_pos_pre[0][0]), int(start_pos_pre[0][1])), number=1)[0]
                else:
                    if not start:
                        start_pos = self.grid.sample_random_pos(number=1)[0]
                    else:
                        start_idx = randint(0, len(start) - 1)
                        start_pos = start[start_idx]

                start_list.append(start_pos)
                roll_out, states, actions, rewards, success = self.roll_out_in_env(start=start_pos, goal=goal)
                batch.append(roll_out)

                num_roll_outs += 1
                num_steps += roll_out.length()
                len_list.append(roll_out.length())
                total_success += success
                j += 1

                state_list.append(states)
                action_list.append(actions)
                reward_list.append(rewards)
        elif mode == 'test':
            start_pos = start

            roll_out, states, actions, rewards, success = self.roll_out_in_env(start=start_pos, goal=goal)
            batch.append(roll_out)

            num_roll_outs += 1
            num_steps += roll_out.length()
            total_success += success
            j += 1

            state_list.append(states)
            action_list.append(actions)
            reward_list.append(rewards)

        batch["num_episodes"] = num_roll_outs

        return batch, num_steps, total_success / j, state_list, action_list, reward_list, start_list

    def simulate_env_asym_selfplay(self, goal):
        batch = Batch()
        batch_alice = Batch()
        state_list = []
        action_list = []
        reward_list = []
        start_list = []

        num_roll_outs = 0
        num_steps_alice = 0
        num_steps_bob = 0
        total_success = 0
        j = 0.

        while max(num_steps_alice, num_steps_bob) < self.batch_size:
            roll_out_alice, roll_out, states, actions, rewards, success = self.roll_out_in_env_asym_selfplay(goal=goal)
            batch.append(roll_out)
            batch_alice.append(roll_out_alice)

            num_roll_outs += 1
            num_steps_alice += roll_out_alice.length()
            num_steps_bob += roll_out.length()
            # num_steps += roll_out.length()
            total_success += success
            j += 1

            state_list.append(states)
            action_list.append(actions)
            reward_list.append(rewards)

        print(batch_alice.length())
        print(batch.length())

        return batch_alice, batch, num_steps_alice+num_steps_bob, total_success / j, state_list, action_list, reward_list, start_list

    def asym_self_play_start_generation(self, goal, num_starts):
        batch_alice = Batch()
        start_list = []

        for i in range(num_starts):
            roll_out_alice, _, states, _, _, _ = self.roll_out_in_env_asym_selfplay(goal=goal)
            # print("here")
            # print(states[0])
            start_list.append((states[0][0], states[0][1]))
            batch_alice.append(roll_out_alice)

        print(batch_alice.length())

        self.optimizer_alice.process_batch(self.alice, batch_alice, [])

        return start_list

    def train(self, start_p_list, goal_p, sampling_probs, sampling_mode, asp_rc=False):
        acc_steps = 0
        acc_successes = 0
        s_list = []
        a_list = []
        r_list = []
        st_list = []

        if not asp_rc:

            # SELF-SUPERVISED TRAINING
            batch_alice, batch, steps, successes, s_list_part, a_list_part, r_list_part, start_list = \
                self.simulate_env_asym_selfplay(goal=goal_p)

            self.optimizer_alice.process_batch(self.alice, batch_alice, [])
            acc_steps += steps
            acc_successes += successes
            s_list = s_list + s_list_part
            a_list = a_list + a_list_part
            r_list = r_list + r_list_part
            st_list = st_list + start_list
            self.optimizer.process_batch(self.policy, batch, [])

        print("\n")

        # NORMAL TRAINING
        if asp_rc:
            episodes = self.inner_episodes
        else:
            episodes = self.inner_episodes - 1
        for iter_loop in range(episodes):
            batch, steps, successes, s_list_part, a_list_part, r_list_part, start_list = \
                self.simulate_env(start=start_p_list,
                                  goal=goal_p,
                                  mode='train',
                                  probs=sampling_probs,
                                  smode=sampling_mode)
            acc_steps += steps
            acc_successes += successes
            s_list = s_list + s_list_part
            a_list = a_list + a_list_part
            r_list = r_list + r_list_part
            st_list = st_list + start_list
            self.optimizer.process_batch(self.policy, batch, [])

        return acc_steps, acc_successes, s_list, a_list, r_list, st_list, []

    def test(self, start_p_list, goal_p):
        batch, steps, successes, s_list, a_list, r_list, _ = self.simulate_env(start=start_p_list,
                                                                               goal=goal_p,
                                                                               mode='test',
                                                                               probs=[],
                                                                               smode=False
                                                                               )
        return steps, successes, s_list, a_list, r_list
