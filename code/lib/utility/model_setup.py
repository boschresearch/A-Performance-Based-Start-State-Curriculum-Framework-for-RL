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


import xml.etree.ElementTree as ET


def modify_models(asset_path):
    # The following snippet is derived from rllab-curriculum
    #   (https://github.com/florensacc/rllab-curriculum)
    # Copyright (c) 2016 rllab contributors, licensed under the MIT license,
    # cf. 3rd-party-licenses.txt file in the root directory of this source tree.

    ant_path = asset_path + 'ant.xml'
    tree = ET.parse(ant_path)
    tree.write(asset_path + 'ant_new.xml')

    ant_path = asset_path + 'ant_new.xml'
    tree = ET.parse(ant_path)
    worldbody = tree.find(".//worldbody")

    structure = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 'g', 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if str(structure[i][j]) == '1':
                # offset all coordinates so that robot starts at the origin
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (j * 3,
                                      i * 3,
                                      3),
                    size="%f %f %f" % (0.5 * 3,
                                       0.5 * 3,
                                       3),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.4 0.4 0.4 0.5"
                )

    tree.write(asset_path + 'ant_new.xml')

    # End of snippet.

    # The following snippet is derived from rllab-curriculum
    #   (https://github.com/florensacc/rllab-curriculum)
    # Copyright (c) 2016 rllab contributors, licensed under the MIT license,
    # cf. 3rd-party-licenses.txt file in the root directory of this source tree.

    ant_path = asset_path + 'point.xml'
    tree = ET.parse(ant_path)
    tree.write(asset_path + 'point_spiral_2D.xml')

    ant_path = asset_path + 'point_spiral_2D.xml'
    tree = ET.parse(ant_path)
    worldbody = tree.find(".//worldbody")

    structure = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 'r', 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 'g', 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]

    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if str(structure[i][j]) == '1':
                # offset all coordinates so that robot starts at the origin
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (j * 2,
                                      i * 2,
                                      2 / 2 * 1),
                    size="%f %f %f" % (1,
                                       1,
                                       2 / 2 * 1),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.4 0.4 0.4 0.5"
                )

    body = tree.getroot()[4][2]
    body.remove(body[1])
    body.remove(body[3])

    motor1 = tree.getroot()[5][0]
    motor2 = tree.getroot()[5][1]

    motor1.attrib["ctrlrange"] = "-5 5"
    motor2.attrib["ctrlrange"] = "-5 5"
    motor2.attrib["joint"] = "bally"

    tree.write(asset_path + 'point_spiral_2D.xml')

    # End of snippet.


modify_models("<specify-the-path-to-your-gym-installation>/gym/gym/envs/mujoco/assets/")
