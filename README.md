# A-Performance-Based-Start-State-Curriculum-Framework-for-RL

This is the official code, implemented by Jan Wöhlke, accompanying the AAMAS 2020 paper A Performance-Based Start State Curriculum Framework for Reinforcement Learning by Jan Wöhlke, Felix Schmitt, and Herke
van Hoof. The paper can be found here:
http://www.ifaamas.org/Proceedings/aamas2020/pdfs/p1503.pdf. The code allows 
the users to reproduce and extend the results reported in the paper. Please 
cite the above paper when reporting, reproducing or extending the results:
```
@inproceedings{wohlke2020performance,
  title={A Performance-Based Start State Curriculum Framework for Reinforcement Learning},
  author={W{\"o}hlke, Jan and Schmitt, Felix and van Hoof, Herke},
  booktitle={Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems},
  pages={1503--1511},
  year={2020}
}
```

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor 
monitored in any way.

## Requirements, how to build, test, install, use, etc.

Place the folder [code](code) where you want to execute the code.

Add the path of the folder [code](code) to your $PYTHONPATH environment variable.


#### GRIDWORLD EXPERIMENTS

You need a Python Set-Up with the following packages:
* python>=3.6
* gym==0.10.9 (install in editable mode: pip install -e, in case you want to run mujoco experiments)
* numpy>=1.15.2
* scipy>=1.3.1
* torch==0.3.1
* matplotlib>=3.0.1

The start scripts for the experiments can be found 
[here](code/scripts/gridworld).

For the experiments in Section 5.1 / Figure 2 you find the start scripts
[here](code/scripts/gridworld/discrete):

* [UST](code/scripts/gridworld/discrete/ust_grid_discrete.py)
* [GS](code/scripts/gridworld/discrete/gs_grid_discrete.py)
* [SPCRL](code/scripts/gridworld/discrete/spcrl_grid_discrete.py)
* [TPG](code/scripts/gridworld/discrete/tpg_grid_discrete.py)
* [SG](code/scripts/gridworld/discrete/sg_grid_discrete.py)
* [SG PMM](code/scripts/gridworld/discrete/sg_pmm_grid_discrete.py)
* [SG VISITED](code/scripts/gridworld/discrete/sg_visited_grid_discrete.py)

For the experiments in Sections 5.2.1, 5.2.4, and 5.2.7 / Figures 3 and 5 you 
find the start scripts [here](code/scripts/gridworld/continuous):

* [UST](code/scripts/gridworld/continuous/ust_grid_continuous.py)
* [SG PMM](code/scripts/gridworld/continuous/sg_pmm_grid_continuous.py)
* [SG PMN](code/scripts/gridworld/continuous/sg_pmn_grid_continuous.py)
* [RC](code/scripts/gridworld/continuous/rc_grid_continuous.py)
* [SPCRL PMM](code/scripts/gridworld/continuous/spcrl_pmm_grid_continuous.py)
* [TPG PMM](code/scripts/gridworld/continuous/tpg_pmm_grid_continuous.py)
* [ASP](code/scripts/gridworld/continuous/asp_grid_continuous.py)
* [ASP RC](code/scripts/gridworld/continuous/asp_rc_grid_continuous.py)
* [SAGG-RIAC](code/scripts/gridworld/continuous/sagg_riac_grid_continuous.py)
* [UST USYM](code/scripts/gridworld/continuous/ust_grid_continuous_usym.py)
* [RC USYM](code/scripts/gridworld/continuous/rc_grid_continuous_usym.py)
* [SPCRL PMM USYM](code/scripts/gridworld/continuous/spcrl_pmm_grid_coninuous_usym.py)
* [TPG PMM USYM](code/scripts/gridworld/continuous/tpg_pmm_grid_continuous_usym.py)
* [SG PMM USYM](code/scripts/gridworld/continuous/sg_pmm_grid_continuous_usym.py)

Run the scripts with: _python [scriptname] --seed=[seed option]_

There are different seed options to run the number random seeds used for 
the respective experiment:
* "all" : run all / the first 10 seeds one after one another
* "s1" to "s10 (or s50)" : run an individual random seed
* "p1" to "p2 (or p10)" : run a slice of 5 random seeds one after another
* "ps1" to "ps4" : run first three, second three, second last two, or last 
two random seeds, respectively (for 10 seeds).


#### ROBOTIC EXPERIMENTS

You need to insert the files provided in the [gym](gym) folder in the 
corresponding folders of your (editable) gym installation (same folder 
structure). By this you add some new files and replace some existing ones.

You need to have [Mujoco](http://www.mujoco.org/) (version 1.5 - mujoco/150) [no experience whether it works with newer Mujoco versions] installed and have a valid license key for it.

You need to install the following additional Python package:
* mujoco-py<1.50.2,>=1.50.1 (our version: 1.50.1.68)

Furthermore, you need to get / modify the Mujoco model files for the 
point mass, the ant, and the key insertion. For this purpose, you can
perform, the following steps:
* Get the *arm3d_key_tight.xml* from [here](https://github.com/florensacc/rllab-curriculum/tree/master/vendor/mujoco_models)
and place it in the */gym/gym/envs/mujoco/assets* folder of your 
editable gym installation
+ Run this [script](code/lib/utility/model_setup.py) specifying the location of the 
*assets* folder in line 126 (bottom)
* An *ant_new.xml* and a *point_spiral_2D.xml* will be generated


The start scripts for the experiments can be found [here](code/scripts/mujoco).

For the experiments in Section 5.2.5 / Figure 6a you find the start scripts 
[here](code/scripts/mujoco/point_mass):

* [UST](code/scripts/mujoco/point_mass/ust_mj_point.py)
* [RC](code/scripts/mujoco/point_mass/rc_mj_point.py)
* [SPCRL PMM](code/scripts/mujoco/point_mass/spcrl_pmm_mj_point.py)
* [TPG PMM](code/scripts/mujoco/point_mass/tpg_pmm_mj_point.py)
* [SG PMM](code/scripts/mujoco/point_mass/sg_pmm_mj_point.py)
* [ASP](code/scripts/mujoco/point_mass/asp_mj_point.py)
* [ASP RC](code/scripts/mujoco/point_mass/asp_rc_mj_point.py)
* [SAGG-RIAC](code/scripts/mujoco/point_mass/sagg_riac_mj_point.py)

For the experiments in Section 5.2.6 / Figure 6b you find the start scripts 
[here](code/scripts/mujoco/ant):

* [UST](code/scripts/mujoco/ant/ust_mj_ant.py)
* [RC](code/scripts/mujoco/ant/rc_mj_ant.py)
* [SPCRL PMM](code/scripts/mujoco/ant/spcrl_pmm_mj_ant.py)
* [TPG PMM](code/scripts/mujoco/ant/tpg_pmm_mj_ant.py)
* [SG PMM](code/scripts/mujoco/ant/sg_pmm_mj_ant.py)

For the experiments in Section 5.3 / Figure 8 you find the start scripts 
[here](code/scripts/mujoco/key):

* [UST](code/scripts/mujoco/key/ust_mj_key.py)
* [RC](code/scripts/mujoco/key/rc_mj_key.py)
* [SPCRL PMM](code/scripts/mujoco/key/spcrl_pmn_mj_key.py)
* [TPG PMM](code/scripts/mujoco/key/tpg_pmn_mj_key.py)
* [SG PMM](code/scripts/mujoco/key/sg_pmn_mj_key.py)

Run the scripts with: _python [scriptname] --seed=[seed option]_

There are different seed options to run the number random seeds used for 
the respective experiment:
* "all" : run all / the first 10 seeds one after one another
* "s1" to "s10 (or s35)" : run an individual random seed
* "p1" to "p2 (or p7)" : run a slice of 5 random seeds one after another
* "ps1" to "ps4" : run first three, second three, second last two, or last 
two random seeds, respectively (for 10 seeds).


#### RESULTS

Plot the test goal reaching probabilities in the "results_ust_ ..." files of 
all the random seeds of the respective experiment of interest as mean +- 
standard error to arrive at the curves shown in the paper.

## License

A-Performance-Based-Start-State-Curriculum-Framework-for-RL is open-sourced 
under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in 
A-Performance-Based-Start-State-Curriculum-Framework-for-RL, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
