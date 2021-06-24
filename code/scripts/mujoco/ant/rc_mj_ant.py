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


# IMPORT #
import os
import argparse
import random
import time
import numpy as np
import torch

from lib.trpo.trpo_mujoco_ant import TRPOTrainer
from lib.mujoco.reverse_curriculum import sample_nearby, OldStarts, label_states_from_paths, select_starts
from lib.utility.state_list import sample_list


def run_fun(seed_list,
            num_inner_episodes,
            num_outer_steps,
            policy_hidden_sizes_list,
            max_kl_val,
            damp_val,
            max_iter_val,
            batch_size_val,
            start_mode,
            lr_arg,
            N_new_val,
            N_old_val,
            T_B_val,
            M_val,
            R_min_val,
            R_max_val,
            cluster=False,
            num_eval=10
            ):

    # INITIALIZATION
    start_time = time.time()
    prefix = 'rc_mj_ant'
    suffix = '%s' % lr_arg

    config_dict = {
        'prefix': prefix,
        'suffix': suffix,
        'barcode': int(start_time),
        'seed_list': seed_list,
        'num_inner_episodes': num_inner_episodes,
        'num_outer_steps': num_outer_steps,
        'policy_hidden_sizes_list': policy_hidden_sizes_list,
        'max_kl_val': max_kl_val,
        'damp_val': damp_val,
        'max_iter_val': max_iter_val,
        'batch_size_val': batch_size_val,
        'start_mode': start_mode,
        'N_new_val': N_new_val,
        'N_old_val': N_old_val,
        'T_B_val': T_B_val,
        'M_val': M_val,
        'R_min_val': R_min_val,
        'R_max_val': R_max_val
    }
    if cluster:
        save_path = os.environ.get('SSD') + "/" + os.environ.get('EXP') + "/"
    else:
        save_path = ""

    f = open(save_path + "config_" + prefix + "_" + str(config_dict['barcode']) + "_" + suffix + '.txt', 'w')
    f.write(str(config_dict))
    f.close()

    print("CONFIG")
    print(config_dict)

    # LOOP OVER SEEDS #
    for seed_val in seed_list:

        # SEED #
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)

        # INITIALIZATION #
        myTRPO = TRPOTrainer(state_dim=41,
                             action_dim=8,
                             hidden_sizes=policy_hidden_sizes_list,
                             max_kl=max_kl_val,
                             damping=damp_val,
                             batch_size=batch_size_val,
                             inner_episodes=num_inner_episodes,
                             max_iter=max_iter_val
                             )

        eps_tot = 0
        acc_eps_list = [0]
        acc_result_list_ust = [0]
        start_save = []

        print("\n")
        print("SEED")
        print(seed_val)
        print("\n")

        initial_start = [myTRPO.grid.goal]
        starts = sample_nearby(grid=myTRPO.grid, starts=initial_start, N_new=N_new_val, T_B=T_B_val, M=M_val,
                               action_dim=8)
        old_starts = OldStarts(dist_threshold=0.02, init=tuple(myTRPO.grid.goal))

        i = 0
        while eps_tot < num_outer_steps:
            i += 1
            print(eps_tot)

            starts = sample_nearby(grid=myTRPO.grid, starts=starts, N_new=N_new_val, T_B=T_B_val, M=M_val, action_dim=8)
            print("NEW STARTS")
            # print(starts)

            sampled_old_starts = sample_list(input_list=old_starts.start_list, num_samples=N_old_val)
            starts = starts + sampled_old_starts
            print("OLD STARTS")
            # print(sampled_old_starts)

            print("STARTS TO TRAIN")
            # print(starts)
            rho = starts[:]

            _, result, s_list, a_list, r_list, st_list, _ = myTRPO.train(start_p_list=starts,
                                                                         goal_p=myTRPO.grid.goal,
                                                                         sampling_probs=[],
                                                                         sampling_mode=False)

            start_save.append(st_list)

            print("RESULT")
            print(result)
            eps_tot += num_inner_episodes

            states, labels = label_states_from_paths(path_trajectories=s_list, path_successes=r_list)
            print("STATES")
            # print(states)
            print("LABELS")
            # print(labels)

            starts = select_starts(starts=states, rews=labels, R_min=R_min_val, R_max=R_max_val)
            print("STARTS SELECTED")
            # print(starts)

            old_starts.add_starts(starts2add=starts)
            print("OLD STARTS")
            # print(old_starts.start_list)

            train_success = result  # / num_inner_episodes
            print("TRAIN SUCCESS")
            print(train_success)

            if not starts:
                if train_success < 0.5:
                    print("BAD")
                    starts = sample_list(input_list=old_starts.start_list, num_samples=(N_new_val + N_old_val))
                    print(starts)
                else:
                    print("GOOD")
                    starts = sample_nearby(grid=myTRPO.grid,
                                           starts=rho,
                                           N_new=N_new_val,
                                           T_B=2 * T_B_val,
                                           M=M_val,
                                           rand_std=10.0,
                                           action_dim=8
                                           )

            print("TEST UNIFORM")
            test_starts = myTRPO.grid.sample_random_pos(number=num_eval)
            print("TEST STARTS")
            print(test_starts)
            reach_prob_ust = 0
            for start_state in test_starts:
                _, result_ust, _, _, _ = myTRPO.test(start_p_list=start_state, goal_p=myTRPO.grid.goal)
                reach_prob_ust += result_ust
                # print(result_ust)
            reach_prob_ust = reach_prob_ust / len(test_starts)

            print("Iteration: %i" % i)
            print("Time passed (in min): %.2f" % ((time.time() - start_time) / 60))
            print("Reach Prob UST: %.2f" % reach_prob_ust)

            acc_eps_list.append(eps_tot)
            acc_result_list_ust.append(reach_prob_ust)

        reach_prob_list_ust = acc_result_list_ust

        # SAVE RESULTS #
        episodes_np = np.array(acc_eps_list)
        np.save(save_path + "episodes_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), episodes_np)

        results_ust_np = np.array(reach_prob_list_ust)
        np.save(save_path + "results_ust_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), results_ust_np)

        st_np = np.array(start_save)
        np.save(save_path + "st_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), st_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="s1")
    parser.add_argument("--cluster", type=bool, default=False)
    args = parser.parse_args()
    if args.seed == "s1":
        seed_l = [1535719580]
    elif args.seed == "s2":
        seed_l = [1535720536]
    elif args.seed == "s3":
        seed_l = [1535721129]
    elif args.seed == "s4":
        seed_l = [1535721985]
    elif args.seed == "s5":
        seed_l = [1535723522]
    elif args.seed == "s6":
        seed_l = [1535724275]
    elif args.seed == "s7":
        seed_l = [1535726291]
    elif args.seed == "s8":
        seed_l = [1535954757]
    elif args.seed == "s9":
        seed_l = [1535957367]
    elif args.seed == "s10":
        seed_l = [1535953242]
    elif args.seed == "ps1":
        seed_l = [1535719580, 1535720536, 1535721129]
    elif args.seed == "ps2":
        seed_l = [1535721985, 1535723522, 1535724275]
    elif args.seed == "ps3":
        seed_l = [1535726291, 1535954757]
    elif args.seed == "ps4":
        seed_l = [1535957367, 1535953242]
    elif args.seed == "p1":
        seed_l = [1535719580, 1535720536, 1535721129, 1535721985, 1535723522]
    elif args.seed == "p2":
        seed_l = [1535724275, 1535726291, 1535954757, 1535957367, 1535953242]
    elif args.seed == "all":
        seed_l = [1535719580, 1535720536, 1535721129, 1535721985, 1535723522,
                  1535724275, 1535726291, 1535954757, 1535957367, 1535953242]
    else:
        seed_l = [0]

    run_fun(seed_list=seed_l,
            num_inner_episodes=5,
            num_outer_steps=2500,
            policy_hidden_sizes_list=[64, 64, 64],
            max_kl_val=1e-2,
            damp_val=1e-3,
            max_iter_val=2000,
            batch_size_val=80000,
            start_mode="uniform",
            lr_arg="hlr",
            N_new_val=200,
            N_old_val=100,
            T_B_val=50,
            M_val=10000,
            R_min_val=0.1,
            R_max_val=0.9,
            cluster=args.cluster
            )
