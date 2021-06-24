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
import copy
import argparse
import random
import time
import numpy as np
import torch
import os

from lib.trpo.trpo_mujoco_key import TRPOTrainer
from lib.mujoco.reverse_curriculum import OldStarts, label_states_from_paths_key, select_starts
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
            num_eval=100
            ):

    # INITIALIZATION
    start_time = time.time()
    prefix = 'rc_mj_key'
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
        myTRPO = TRPOTrainer(state_dim=23,
                             action_dim=7,
                             hidden_sizes=policy_hidden_sizes_list,
                             max_kl=max_kl_val,
                             damping=damp_val,
                             batch_size=batch_size_val,
                             inner_episodes=num_inner_episodes,
                             max_iter=max_iter_val,
                             discount_val=0.995
                             )

        eps_tot = 0
        acc_eps_list = [0]
        acc_result_list_ust = [0]
        acc_result_list_ust_in = [0]
        acc_result_list_ust_out = [0]
        start_save = []
        s_new_save = []
        s_old_save = []
        s_sel_save = []
        s_tot_save = []

        print("\n")
        print("SEED")
        print(seed_val)
        print("\n")

        initial_start = [tuple(myTRPO.grid.goal)]
        starts = myTRPO.grid.sample_nearby(starts=initial_start, N_new=N_new_val, T_B=10, M=M_val)
        old_starts = OldStarts(dist_threshold=0.01, init=tuple(myTRPO.grid.goal))

        i = 0
        while eps_tot < num_outer_steps:
            i += 1
            print(eps_tot)

            starts = myTRPO.grid.sample_nearby(starts=starts, N_new=N_new_val, T_B=T_B_val, M=M_val)
            snew2save = copy.deepcopy(starts)
            s_new_save.append(snew2save)
            print("NEW STARTS")
            # print(starts)

            sampled_old_starts = sample_list(input_list=old_starts.start_list, num_samples=N_old_val)
            sold2save = copy.deepcopy(sampled_old_starts)
            s_old_save.append(sold2save)
            starts = starts + sampled_old_starts
            print("OLD STARTS")
            # print(sampled_old_starts)

            # starts = uniquify_list(starts)
            print("STARTS TO TRAIN")
            # print(starts)
            stot2save = copy.deepcopy(starts)
            s_tot_save.append(stot2save)
            rho = starts[:]

            _, result, s_list, a_list, r_list, st_list, _ = myTRPO.train(start_p_list=starts,
                                                                         goal_p=[],
                                                                         sampling_probs=[],
                                                                         sampling_mode=False)

            start_save.append(st_list)

            print("RESULT")
            print(result)
            eps_tot += num_inner_episodes

            states, labels = label_states_from_paths_key(path_trajectories=s_list, path_successes=r_list)
            print("STATES")
            print(states)
            print("LABELS")
            print(labels)

            starts = select_starts(starts=states, rews=labels, R_min=R_min_val, R_max=R_max_val)
            sel2save = copy.deepcopy(starts)
            s_sel_save.append(sel2save)
            print("STARTS SELECTED")
            print(starts)

            old_starts.add_starts(starts2add=starts)
            print("OLD STARTS")
            print(len(old_starts.start_list))
            # print(old_starts.start_list)
            # print(old_starts.pos_list)

            train_success = result # / num_inner_episodes
            print("TRAIN SUCCESS")
            print(train_success)

            if not starts:
                if train_success < 0.5:
                    print("BAD")
                    starts = sample_list(input_list=old_starts.start_list, num_samples=N_old_val)
                    # print(starts)
                else:
                    print("GOOD")
                    starts = myTRPO.grid.sample_nearby(starts=rho, N_new=N_new_val, T_B=10 * T_B_val, M=M_val)
                    # print(starts)

            print("TEST UNIFORM")
            test_starts = myTRPO.grid.sample_uniform(number=num_eval)
            print("TEST STARTS")
            print(test_starts)
            reach_prob_ust = 0
            for start_state in test_starts:
                _, result_ust, _, _, _ = myTRPO.test(start_p_list=start_state, goal_p=myTRPO.grid.goal)
                reach_prob_ust += result_ust
                # print(result_ust)
            reach_prob_ust = reach_prob_ust / len(test_starts)

            test_starts = myTRPO.grid.sample_uniform(number=num_eval, mode='in')
            reach_prob_ust_in = 0
            for start_state in test_starts:
                _, result_ust, _, _, _ = myTRPO.test(start_p_list=start_state, goal_p=myTRPO.grid.goal)
                reach_prob_ust_in += result_ust
                # print(result_ust)
            reach_prob_ust_in = reach_prob_ust_in / len(test_starts)

            test_starts = myTRPO.grid.sample_uniform(number=num_eval, mode='out')
            reach_prob_ust_out = 0
            for start_state in test_starts:
                _, result_ust, _, _, _ = myTRPO.test(start_p_list=start_state, goal_p=myTRPO.grid.goal)
                reach_prob_ust_out += result_ust
                # print(result_ust)
            reach_prob_ust_out = reach_prob_ust_out / len(test_starts)

            print("Iteration: %i" % i)
            print("Time passed (in min): %.2f" % ((time.time() - start_time) / 60))
            print("Reach Prob UST: %.2f" % reach_prob_ust)
            print("Reach Prob UST IN: %.2f" % reach_prob_ust_in)
            print("Reach Prob UST OUT: %.2f" % reach_prob_ust_out)

            acc_eps_list.append(eps_tot)
            acc_result_list_ust.append(reach_prob_ust)
            acc_result_list_ust_in.append(reach_prob_ust_in)
            acc_result_list_ust_out.append(reach_prob_ust_out)

        reach_prob_list_ust = acc_result_list_ust
        reach_prob_list_ust_in = acc_result_list_ust_in
        reach_prob_list_ust_out = acc_result_list_ust_out

        # SAVE RESULTS #
        episodes_np = np.array(acc_eps_list)
        np.save(save_path + "episodes_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), episodes_np)

        results_ust_np = np.array(reach_prob_list_ust)
        np.save(save_path + "results_ust_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), results_ust_np)

        results_ust_in_np = np.array(reach_prob_list_ust_in)
        np.save(save_path + "results_ust_in_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), results_ust_in_np)

        results_ust_out_np = np.array(reach_prob_list_ust_out)
        np.save(save_path + "results_ust_out_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), results_ust_out_np)

        st_np = np.array(start_save)
        np.save(save_path + "st_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), st_np)

        sold_np = np.array(start_save)
        np.save(save_path + "sold_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), sold_np)

        snew_np = np.array(start_save)
        np.save(save_path + "snew_" + prefix + "_" + str(seed_val) + "_" + "_" + suffix + "_" +
                str(config_dict['barcode']), snew_np)

        stot_np = np.array(start_save)
        np.save(save_path + "stot_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), stot_np)

        ssel_np = np.array(start_save)
        np.save(save_path + "ssel_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), ssel_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="s1")
    parser.add_argument("--lr", type=str, default="llr")
    parser.add_argument("--bs", type=int, default=50000)
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
    if args.lr == "llr":
        klv = 5e-4
        dpv = 5e-3
    elif args.lr == "hlr":
        klv = 1e-2
        dpv = 1e-3
    else:
        klv = 1e-1
        dpv = 1e-2

    run_fun(seed_list=seed_l,
            num_inner_episodes=2,
            num_outer_steps=4000,
            policy_hidden_sizes_list=[64, 64, 64],
            max_kl_val=klv,
            damp_val=dpv,
            max_iter_val=500,
            batch_size_val=args.bs,
            start_mode="uniform",
            lr_arg=args.lr,
            N_new_val=600,
            N_old_val=300,
            T_B_val=50,
            M_val=10000,
            R_min_val=0.1,
            R_max_val=0.9,
            cluster=args.cluster
            )
