import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

data = pkl.load(open('checkpointsdata.pkl', 'rb'))

avg_rewards = data['avg_reward'][:-1]
max_rewards = data['max_rewards'][:-1]
min_rewards = data['min_rewards'][:-1]
time_list = data['time_record'][:-1]

plt.title('Episodic Reward')
plt.plot(np.arange(0,len(avg_rewards),1),avg_rewards, color='royalblue')
plt.fill_between(np.arange(0,len(avg_rewards),1),min_rewards,max_rewards,facecolor='lightsteelblue',linewidth=0)
plt.savefig('episodicReward.png')
plt.show()