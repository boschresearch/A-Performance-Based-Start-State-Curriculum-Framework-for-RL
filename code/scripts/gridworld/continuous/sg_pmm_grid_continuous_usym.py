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
import random
import time
import numpy as np
import torch
import argparse

from lib.trpo.trpo_continuous import TRPOTrainer
from lib.start_selection.reaching_probability_map import update_reach_prob_map_continuous, \
    calc_spatial_reach_prob_grads
from lib.start_selection.select_start import start_state_sampling_probabilities_from_grads


def run_fun(seed_list,
            grid_var,
            num_inner_episodes,
            num_outer_steps,
            policy_hidden_sizes_list,
            max_kl_val,
            damp_val,
            max_iter_val,
            batch_size_val,
            grad_square,
            boltzmann_prob,
            boltzmann_temp,
            seed_arg,
            lr_arg,
            hidden_arg,
            cluster=False,
            num_eval=10
            ):

    # INITIALIZATION
    start_time = time.time()
    prefix = 'sg_pmm_grid_continuous_usym'
    suffix = '%s_%s_%i_%s' % (hidden_arg, lr_arg, grid_var, seed_arg)

    config_dict = {
        'prefix': prefix,
        'suffix': suffix,
        'barcode': int(start_time),
        'seed_list': seed_list,
        'grid_var': grid_var,
        'num_inner_episodes': num_inner_episodes,
        'num_outer_steps': num_outer_steps,
        'policy_hidden_sizes_list': policy_hidden_sizes_list,
        'max_kl_val': max_kl_val,
        'damp_val': damp_val,
        'max_iter_val': max_iter_val,
        'batch_size_val': batch_size_val,
        'grad_square': grad_square,
        'boltzmann_prob': boltzmann_prob,
        'boltzmann_temp': boltzmann_temp,
        'hidden_arg': hidden_arg
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
        myTRPO = TRPOTrainer(grid_type=grid_var,
                             state_dim=12,
                             action_dim=2,
                             hidden_sizes=policy_hidden_sizes_list,
                             max_kl=max_kl_val,
                             damping=damp_val,
                             batch_size=batch_size_val,
                             inner_episodes=num_inner_episodes,
                             max_iter=max_iter_val,
                             usym_var=True
                             )

        eps_tot = 0
        acc_eps_list = [0]
        acc_result_list_ust = [0]
        start_save = []
        rpm_save = []
        rp_grad_map_save = []

        print("\n")
        print("SEED")
        print(seed_val)
        print("\n")

        train_starts = [myTRPO.grid.goal]
        sampling_prob_list = [1]

        i = 0
        while eps_tot < num_outer_steps:
            i += 1
            print(eps_tot)

            print("TRAIN STARTS")
            # print(train_starts)
            # print(sampling_prob_list)

            print(not train_starts)
            if not train_starts:
                _, result, s_list, a_list, r_list, st_list, _ = myTRPO.train(start_p_list=[],
                                                                             goal_p=myTRPO.grid.goal,
                                                                             sampling_probs=[],
                                                                             sampling_mode=False)
            else:
                _, result, s_list, a_list, r_list, st_list, _ = myTRPO.train(start_p_list=train_starts,
                                                                             goal_p=myTRPO.grid.goal,
                                                                             sampling_probs=sampling_prob_list,
                                                                             sampling_mode=True)

            start_save.append(st_list)

            rpm_est = update_reach_prob_map_continuous(reach_prob_map=myTRPO.grid.rp_map,
                                                       path_trajectories=s_list,
                                                       path_successes=r_list)

            rpm2save = copy.deepcopy(rpm_est)
            rpm_save.append(rpm2save)

            start_candidates, rp_grad_list, rp_grad_map = calc_spatial_reach_prob_grads(
                grad_states=myTRPO.default_starts,
                occupancy_map=myTRPO.grid.occupancy_map,
                reach_prob_map=rpm_est,
                square_mode=grad_square)

            rp_grad_map2save = copy.deepcopy(rp_grad_map)
            rp_grad_map_save.append(rp_grad_map2save)

            train_starts, sampling_prob_list = start_state_sampling_probabilities_from_grads(
                start_candidates=start_candidates,
                rp_grads=rp_grad_list,
                boltzmann=boltzmann_prob,
                temp=boltzmann_temp)

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
            reach_prob_ust = reach_prob_ust/len(test_starts)

            print("Iteration: %i" % i)
            print("Reach Prob UST: %.2f" % reach_prob_ust)
            print("Time passed (in min): %.2f" % ((time.time() - start_time)/60))

            acc_eps_list.append(eps_tot)
            acc_result_list_ust.append(reach_prob_ust)

        reach_prob_list_ust = acc_result_list_ust

        # SAVE RESULTS #
        episodes_np = np.array(acc_eps_list)
        np.save(save_path + "episodes_" + prefix + "_" + str(seed_val) + "_" + str(grid_var) + "_" + suffix + "_" +
                str(config_dict['barcode']), episodes_np)

        results_ust_np = np.array(reach_prob_list_ust)
        np.save(save_path + "results_ust_" + prefix + "_" + str(seed_val) + "_" + str(grid_var) + "_" + suffix + "_" +
                str(config_dict['barcode']), results_ust_np)

        st_np = np.array(start_save)
        np.save(save_path + "st_" + prefix + "_" + str(seed_val) + "_" + str(grid_var) + "_" + suffix + "_" +
                str(config_dict['barcode']), st_np)

        rpm_np = np.array(rpm_save)
        np.save(save_path + "rpm_" + prefix + "_" + str(seed_val) + "_" + str(grid_var) + "_" + suffix + "_" +
                str(config_dict['barcode']), rpm_np)

        rpgm_np = np.array(rp_grad_map_save)
        np.save(save_path + "rpgm_" + prefix + "_" + str(seed_val) + "_" + str(grid_var) + "_" + suffix + "_" +
                str(config_dict['barcode']), rpgm_np)


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
    elif args.seed == "s36":
        seed_l = [332410]
    elif args.seed == "s37":
        seed_l = [270257]
    elif args.seed == "s38":
        seed_l = [623498]
    elif args.seed == "s39":
        seed_l = [569763]
    elif args.seed == "s40":
        seed_l = [429388]
    elif args.seed == "s41":
        seed_l = [176858]
    elif args.seed == "s42":
        seed_l = [637798]
    elif args.seed == "s43":
        seed_l = [964795]
    elif args.seed == "s44":
        seed_l = [47111]
    elif args.seed == "s45":
        seed_l = [870749]
    elif args.seed == "s46":
        seed_l = [109299]
    elif args.seed == "s47":
        seed_l = [566294]
    elif args.seed == "s48":
        seed_l = [442375]
    elif args.seed == "s49":
        seed_l = [39062]
    elif args.seed == "s50":
        seed_l = [789412]
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
    elif args.seed == "p8":
        seed_l = [332410, 270257, 623498, 569763, 429388]
    elif args.seed == "p9":
        seed_l = [176858, 637798, 964795, 47111, 870749]
    elif args.seed == "p10":
        seed_l = [109299, 566294, 442375, 39062, 789412]
    elif args.seed == "all":
        seed_l = [1535719580, 1535720536, 1535721129, 1535721985, 1535723522,
                  1535724275, 1535726291, 1535954757, 1535957367, 1535953242]
    else:
        seed_l = [0]

    run_fun(seed_list=seed_l,
            grid_var=6,
            num_inner_episodes=5,
            num_outer_steps=5000,
            policy_hidden_sizes_list=[64, 64, 64],
            max_kl_val=5e-4,
            damp_val=5e-3,
            max_iter_val=100,
            batch_size_val=3200,
            grad_square=False,
            boltzmann_prob=False,
            boltzmann_temp=0.2,
            seed_arg=args.seed,
            lr_arg="llr",
            hidden_arg="3x64",
            cluster=args.cluster
            )
# 1535719580, 1535720536, 1535721129, 1535721985, 1535723522, 1535724275, 1535726291, 1535954757, 1535957367, 1535953242
