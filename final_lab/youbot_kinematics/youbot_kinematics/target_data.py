import numpy as np

# Switch for task1d
default = True

target_joint_pos_1 = [4.622576563079859, 1.4919956782391877, -2.4435069071962348, 2.0936591207981596, 0.9990974803253598]
target_joint_pos_2 = [1.4387632277949232, 0.7054776193039478, -2.512352196859098, 1.3866665704382877, 1.6211613228469561]
target_joint_pos_3 = [4.708484383458099, 1.3752175535418365, -3.214261824370574, 1.7887614596312122, 1.7302036080836274]
target_joint_pos_4 = [4.054210505693871, 0.7992699204278928, -2.288749607899941, 1.2692020100313526, 2.645584347531603]

target_joint_pos_5 = [3, 1, -2.5, 1.4, 2]
target_joint_pos_6 = [2, 1, -3, 2, 1]
# rows are target joint positions, columns are the joints
if (default):
    TARGET_JOINT_POSITIONS = np.array([target_joint_pos_1, target_joint_pos_2, target_joint_pos_3, target_joint_pos_4])
else:  
    TARGET_JOINT_POSITIONS = np.array([target_joint_pos_1, target_joint_pos_2, target_joint_pos_3, target_joint_pos_4, target_joint_pos_5, target_joint_pos_6])