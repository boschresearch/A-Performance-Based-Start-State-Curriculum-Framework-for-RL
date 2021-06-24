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
import copy
import argparse
import random
import time
import numpy as np
import torch

from lib.trpo.trpo_mujoco_point_fine import TRPOTrainer
from lib.mujoco.spatial_gradient import update_reach_prob_map_fine


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
            cluster=False,
            num_eval=10
            ):

    # INITIALIZATION
    start_time = time.time()
    prefix = 'ust_mj_point'
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
        'start_mode': start_mode
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
        myTRPO = TRPOTrainer(state_dim=12,
                             action_dim=2,
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
        rpm_est_save = []

        print("\n")
        print("SEED")
        print(seed_val)
        print("\n")

        i = 0
        while eps_tot < num_outer_steps:
            i += 1
            print(eps_tot)

            if start_mode == 'uniform':
                _, result, s_list, a_list, r_list, st_list = myTRPO.train(start_p_list=[],
                                                                          goal_p=myTRPO.grid.goal,
                                                                          sampling_probs=[],
                                                                          sampling_mode=False)
            else:
                _, result, s_list, a_list, r_list, st_list = myTRPO.train(start_p_list=[myTRPO.grid.start],
                                                                          goal_p=myTRPO.grid.goal,
                                                                          sampling_probs=[],
                                                                          sampling_mode=False)

            start_save.append(st_list)

            rpm_est = update_reach_prob_map_fine(reach_prob_map=myTRPO.grid.rp_map,
                                                 path_trajectories=s_list,
                                                 path_successes=r_list)

            rpmest2save = copy.deepcopy(rpm_est)
            rpm_est_save.append(rpmest2save)

            print("RESULT")
            print(result)
            eps_tot += num_inner_episodes

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

        rpm_np = np.array(rpm_est_save)
        np.save(save_path + "rpm_" + prefix + "_" + str(seed_val) + "_" + suffix + "_" +
                str(config_dict['barcode']), rpm_np)


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
    elif args.seed == "s11":
        seed_l = [148222]
    elif args.seed == "s12":
        seed_l = [386772]
    elif args.seed == "s13":
        seed_l = [99394]
    elif args.seed == "s14":
        seed_l = [199662]
    elif args.seed == "s15":
        seed_l = [179852]
    elif args.seed == "s16":
        seed_l = [106425]
    elif args.seed == "s17":
        seed_l = [289031]
    elif args.seed == "s18":
        seed_l = [231042]
    elif args.seed == "s19":
        seed_l = [386786]
    elif args.seed == "s20":
        seed_l = [914963]
    elif args.seed == "s21":
        seed_l = [976132]
    elif args.seed == "s22":
        seed_l = [387836]
    elif args.seed == "s23":
        seed_l = [82692]
    elif args.seed == "s24":
        seed_l = [556318]
    elif args.seed == "s25":
        seed_l = [590610]
    elif args.seed == "s26":
        seed_l = [162444]
    elif args.seed == "s27":
        seed_l = [747881]
    elif args.seed == "s28":
        seed_l = [460457]
    elif args.seed == "s29":
        seed_l = [311988]
    elif args.seed == "s30":
        seed_l = [215173]
    elif args.seed == "s31":
        seed_l = [635399]
    elif args.seed == "s32":
        seed_l = [178712]
    elif args.seed == "s33":
        seed_l = [750458]
    elif args.seed == "s34":
        seed_l = [306262]
    elif args.seed == "s35":
        seed_l = [488445]
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
    elif args.seed == "p3":
        seed_l = [148222, 386772, 99394, 199662, 179852]
    elif args.seed == "p4":
        seed_l = [106425, 289031, 231042, 386786, 914963]
    elif args.seed == "p5":
        seed_l = [976132, 387836, 82692, 556318, 590610]
    elif args.seed == "p6":
        seed_l = [162444, 747881, 460457, 311988, 215173]
    elif args.seed == "p7":
        seed_l = [635399, 178712, 750458, 306262, 488445]
    elif args.seed == "all":
        seed_l = [1535719580, 1535720536, 1535721129, 1535721985, 1535723522,
                  1535724275, 1535726291, 1535954757, 1535957367, 1535953242]
    else:
        seed_l = [0]

    run_fun(seed_list=seed_l,
            num_inner_episodes=5,
            num_outer_steps=1500,
            policy_hidden_sizes_list=[64, 64, 64],
            max_kl_val=5e-4,
            damp_val=5e-3,
            max_iter_val=500,
            batch_size_val=20000,
            start_mode="uniform",
            lr_arg="llr",
            cluster=args.cluster
            )
