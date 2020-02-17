import numpy as np
import matplotlib.pyplot as plt


loss=np.loadtxt("loss.txt")
reward1=np.loadtxt("reward.txt")
reward2=np.loadtxt("reward2.txt")

loss_show= plt.subplot(131)
loss_x=list(np.arange(len(loss)))
loss_y=list(loss)
plt.plot(loss_x,loss_y)
plt.title('loss')

reward1_show= plt.subplot(132)
reward1_x=list(np.arange(len(reward1)))
reward1_y=list(reward1)
plt.plot(reward1_x,reward1_y)
plt.title('reward1')

reward2_show= plt.subplot(133)
reward2_x=list(np.arange(len(reward2)))
reward2_y=list(reward2)
plt.plot(reward2_x,reward2_y)
plt.title('reward2')



plt.show()

#
# batch_index = np.arange(32, dtype=np.int32)
# eval_act_index = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
#                            1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
#
# q_target1 = np.array([[0.92102736, 0.3146143],
#                       [3.1571445, 2.8655252],
#                       [3.3007102, 3.1710606],
#                       [2.4685621, 1.9288611],
#                       [1.5050876, 1.7771744],
#                       [-0.75561917, -0.28676414],
#                       [-2.008418, -1.4559263],
#                       [1.3916769, 1.6931094],
#                       [1.7944734, 1.11914],
#                       [0.5884355, -0.2106688],
#                       [2.1944041, 1.5721865],
#                       [3.1580424, 3.2227187],
#                       [2.7360075, 2.2426476],
#                       [1.3601353, 1.6719272],
#                       [0.6629566, 1.0209508],
#                       [3.420518, 3.4380732],
#                       [0.90053463, 1.2081465],
#                       [2.997787, 3.232586],
#                       [3.3411412, 3.2651963],
#                       [3.2881722, 3.150045],
#                       [3.0589504, 2.7001104],
#                       [3.0714931, 3.2222538],
#                       [3.1057436, 3.184091],
#                       [3.2743864, 3.2854266],
#                       [2.6554642, 2.1759348],
#                       [3.1580424, 3.2227187],
#                       [1.0741441, 1.4070015],
#                       [2.03817, 2.3035097],
#                       [3.277285, 3.3002725],
#                       [2.8546102, 2.9752202],
#                       [-0.48500776, -1.2391297],
#                       [-1.252623, -0.75097454]])
# print(batch_index),
# print(eval_act_index)
# print(q_target1)
# print(q_target1[batch_index, eval_act_index])
