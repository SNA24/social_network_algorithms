import matplotlib.pyplot as plt
import random
import numpy as np
import math
import copy

class Adversarial_Environment:
    def __init__(self, arms_set, T):
        self.__arms_set = arms_set
        self.__T = T
        self.__t = 1
        #An adversary must build a reward table for each arm and each time step
        #This reward table must be independent on the arm selected by the learner
        #Hence, it must be initialized in the constructor
        self.__reward_table = self.__determine_reward_table()

    #Next we provide an example for building a reward table
    #The idea is that we have T different base values and that each arm has it own scaling factor
    #Each arm is assigned a different permutation of the T values scaled by the corresponding scaling factor
    #Essentially, the scaling factor is a measure of the quality of the arm
    #But the permutation of values allow to hide this quality,
    #since it may be the case that arm of low quality are assigned the highest base values in the initial steps,
    #by fooling the learner in this way
    def __determine_reward_table(self):
        reward_table = dict()
        base_cost = [t/self.__T for t in range(1, self.__T+1)] #The T different base values
        for a in self.__arms_set:
            f = random.uniform(0.2, 1) #The scaling factor of arm a
            reward_table[a]=np.random.permutation([i * f for i in base_cost]) #The sequence of reward for arm a
        return reward_table

    #An alternative choice for the adverary reward table would be
    #to fool the learner by having one arm achieving large reward in the first N steps
    #and then to switch to the optimal one.
    #We let eps and 1 be the small and large reward value
    #Observe that N must be small, otherwise the cumulative reward of the first arm
    #is close to the cumulative reward of the second arm, and hence a good approximation of it
    '''def __determine_reward_table(self, eps, N):
        reward_table = dict()
        a_start = random.choice(self.__arms_set) #The arm that is good for only the first N steps
        a_end = random.choice(self.__arms_set.exclude(a_start)) #The arm that is good for the remaining steps
        reward_table[a_start] = [1 for i in range(1, N+1)].extend([eps for i in range(N+1, T+1)])
        reward_table[a_end] = [eps for i in range(1, N + 1)].extend([1 for i in range(N + 1, T + 1)])
        for a in self.__arms_set:
            if a != a_start and a != a_end:
                reward_table[a] = [eps for i in range(1, T+1)] #For the remaining arms we assume they are always bad

        return reward_table'''

    #It returns the t-th entry of the reward table corresponding to arm a
    def receive_reward(self,a):
        reward = self.__reward_table[a][self.__t-1]
        self.__t += 1
        return reward

    #getter method needed to compute the regret in Adversarial_Bandit.py
    #This method is created only for didactic purposes, in reality the reward table must be private!
    def get_reward_table(self):
        return copy.deepcopy(self.__reward_table)

class Exp_3_Learner:

    def __init__(self, arms_set, T, eps, gamma, environment):
        self.__arms_set = arms_set
        self.__T = T
        self.__eps = eps
        self.__gamma = gamma
        self.__environment = environment
        #It saves Hedge weights, that are initially 1
        self.__weights = {a:1 for a in self.__arms_set}

    #It use the exponential function of Hedge to update the weights based on the received rewards
    def __Hedge_update_weights(self, rewards):
        for a in self.__arms_set:
            self.__weights[a] = self.__weights[a]*((1-self.__eps)**(1-rewards[a]))

    #Compute the Hedge distribution: each arm is chosen with a probability that is proportional to its weight
    def __Hedge_compute_distr(self):
        w_sum = sum(self.__weights.values())
        prob = list()
        for i in range(len(self.__arms_set)):
            prob.append(self.__weights[self.__arms_set[i]]/w_sum)

        return prob

    def play_arm(self):
        p = self.__Hedge_compute_distr()
        r=random.random()
        # We chose a random arm with probability gamma
        if r <= self.__gamma:
            a_t = random.choice(self.__arms_set)
        else: #and an arm according the Hedge distribution otherwise
            a_t = random.choices(self.__arms_set, p)[0]

        reward = self.__environment.receive_reward(a_t)

        # We compute the fake rewards
        fake_rewards = dict()
        for i in range(len(self.__arms_set)):
            a = self.__arms_set[i]
            if a != a_t:
                fake_rewards[a] = 1
            else:
                fake_rewards[a] = 1 - (1-reward)/p[i]
        self.__Hedge_update_weights(fake_rewards)


        return a_t, reward

    #utily function only used in Adversarial_Bandit script for didactic purposes
    def get_p(self):
        return self.__Hedge_compute_distr()

#COMPUTE OPTIMAL ARM
def compute_reward(T, a, table_cost):
    cum_reward = 0
    for i in range(0,T):
        cum_reward += table_cost[a][i]
    return cum_reward

def compute_opt_arm(T, arms_set, table_cost):
    opt_reward = -float('inf')
    for a in arms_set:
        temp_opt_reward = compute_reward(T, a, table_cost)
        if temp_opt_reward >= opt_reward:
            opt_reward = temp_opt_reward
            opt_arm = a
    return opt_arm

#define the arms set
arms_set = [1,2,3,4,5,6]

#define the time horizon
T = 500
#enviromnent
env = Adversarial_Environment(arms_set, T)
#impose espilon and gamma
gamma = 1/(3*T)
eps = math.sqrt((1-gamma)*math.log(len(arms_set))/(3*len(arms_set)*T))
#learner
exp3_learn = Exp_3_Learner(arms_set, T, eps, gamma, env)

#play the game t steps
for t in range(1, T+1):
    #learner plays the arm
    exp3_learn.play_arm()

#i-th element of cum_reward_vec is the cumulated reward of the i-th arm in T step
cum_reward_vec = list()
for a in arms_set:
    cum_reward_vec.append(compute_reward(T, a, env.get_reward_table()))

#compute the optimal_arm
opt_arm = compute_opt_arm(T, arms_set, env.get_reward_table())

#plot
#It shows how many times each arm has been played
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(f'Optimal arm: {opt_arm}; Horizon: {T}')
ax1.bar(arms_set, exp3_learn.get_p(), label = 'Probability Distribution computed by Exp3')
ax1.set_ylabel(f"Probability distribution")

#It shows the cumulated revenue of each arm
ax2.bar(arms_set, cum_reward_vec)
ax2.set_ylabel('Cumulated Revenue')
ax2.set_xlabel('Arms')

#It can be observed that the optimal arm is not necessarily the one arm that is used more often
#However, if other arms are used more often than the optimal one, then it is the case that
#the reward gained by using these arms is a good approximation of the one achieved by the optimal arm.
#(It may be necessary to run the application several times in order to see a case in which the optimal arm is not the most frequent choice)
plt.show()

