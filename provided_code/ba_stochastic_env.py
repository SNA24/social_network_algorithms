from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import math
import random

#A Stochastic environment that throws the reward of arm a from a Bernoulli distribution with a given mean
class Bernoulli_Environment:
    #We initialise the mean of the Bernoulli distribution for each arm
    def __init__(self, means_dict):
        self.__means_dict = means_dict

    #It select the reward for the given arm according to a Bernoulli distribution with the given mean
    def receive_reward(self, arm):
        return bernoulli.rvs(self.__means_dict[arm])

class EpsGreedy_Learner:
    # eps is a function that takes in input a time step t and returns eta_t
    # Use T = None for unknown time horizon
    def __init__(self, arms_set, environment, eps, T=None):
        self.__arms_set = arms_set #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__eps = eps #initialize the sequence of eps_t
        self.__T = T
        #Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in arms_set} #It saves the average reward achieved by arm a until the current time step
        self.__t = 0 #It saves the current time step

    #This function returns the arm chosen by the learner and the corresponding reward returned by the environment
    def play_arm(self):
        r = random.random()
        if r <= self.__eps(self.__t): #With probability eps_t
            a_t = random.choice(self.__arms_set) #We choose an arm uniformly at random
        else:
            a_t = max(self.__avgrew, key=self.__avgrew.get) #We choose the arm that has the highest average revenue
        reward = self.__environment.receive_reward(a_t) #We save the reward assigned by the environment
        # We update the number of times arm a_t has been chosen, its cumulative and its average reward
        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        self.__t += 1 #We are ready for a new time step

        return a_t, reward

class UCB_Learner:

    def __init__(self, arms_set, environment, T = None):
        self.__arms_set = arms_set #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__T = T
        if self.__T is None:
            self.__t = 1 #If the time horizon is unknown, we need to remember the current time step
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a: 0 for a in arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a: 0 for a in arms_set} #It saves the cumulative reward achieved by arm a when selected
        # It saves the ucb value of each arm until the current time step
        # It is initialised to infinity in order to allow that each arm is selected at least once
        self.__ucb = {a: float('inf') for a in arms_set}

    # This function returns the arm chosen by the learner and the corresponding reward returned by the environment
    def play_arm(self):
        a_t = max(self.__ucb, key=self.__ucb.get) #We choose the arm that has the highest average revenue
        reward = self.__environment.receive_reward(a_t) #We save the reward assigned by the environment
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.__num[a_t] +=1
        self.__rew[a_t] += reward
        if self.__T is not None: #If the time horizon is known
            self.__ucb[a_t] = self.__rew[a_t]/self.__num[a_t] + math.sqrt(2*math.log(self.__T)/self.__num[a_t])
        else: #If the time horizon is unknown, each time step can be the last. Hence, we are more conservative and use t in place of T
            self.__ucb[a_t] = self.__rew[a_t] / self.__num[a_t] + math.sqrt(2 * math.log(self.__t) / self.__num[a_t])
            self.__t += 1

        return a_t, reward

if __name__ == '__main__':
    #INITIALIZATION
    #Set of Arms
    arms_set = [1,2,3,4,5,6,7,8,9,10]
    #Means the Bernoulli Stochastic Environment
    means_dict = {1: 0.01, 2: 0.75, 3: 0.1, 4: 0.1, 5: 0.01, 6: 0.2, 7: 0.12, 8: 0.08, 9: 0.7, 10: 0.99}
    #Optimal Arm in Hindsight: I need it for computing the regret
    #For the Bernoulli Stochastic Environment this is the one with larger mean
    opt_a = max(means_dict, key=means_dict.get)
    #Time Horizon
    T = 5000
    #We would like to evaluate the expected regret with respect to t
    #To this aim, we cannot just run a single simulation:
    #the result can be biased by extreme random choices (of both the environment and the learner)
    #For this reason we run N simulations,
    #and we will evaluate against t the average regret over the N simulations
    #To this aim, we define N, and we will record for each learner a matrix containing
    #the regret for each simulation and each time step within the simulation
    N = 50 #number of simulations
    eps_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the eps-greedy learner
    ucb_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the UCB learner

    #INITIALIZATION FOR EPS-GREEDY
    #A common choice for eps_t = (K log t/t)^1/3
    def give_eps(t):
        if t == 0:
            return 1  #for the first step we cannot make exploitation, so eps_1 = 1
        return (len(arms_set)*math.log(t+1)/(t+1))**(1/3)
    eps = give_eps

    #SIMULATION PLAY
    for n in range(N):
        ucb_cum_reward = 0 #it saves the cumulative reward of the UCB learner
        eps_cum_reward = 0 #it saves the cumulative reward of the eps-greedy learner
        cum_opt_reward = 0 #it saves the cumulative reward of the best-arm in hindsight
        #Environment
        env = Bernoulli_Environment(means_dict)
        #Eps-Greedy Learner
        eps_learn = EpsGreedy_Learner(arms_set, env, eps, T)
        #UCB Learner
        ucb_learn = UCB_Learner(arms_set, env, T)
        for t in range(T):
            #reward obtained by the optimal arm
            cum_opt_reward += env.receive_reward(opt_a)
            #reward obtained by the eps_greedy learner
            a, reward = eps_learn.play_arm()
            eps_cum_reward += reward
            #regret of the eps_greedy learner
            eps_regrets[n][t] = cum_opt_reward - eps_cum_reward
            # reward obtained by the ucb learner
            a, reward = ucb_learn.play_arm()
            ucb_cum_reward += reward
            #regret of the ucb learner
            ucb_regrets[n][t] = cum_opt_reward - ucb_cum_reward

    #compute the mean regret of the eps greedy and ucb learner
    eps_mean_regrets = {t:0 for t in range(T)}
    ucb_mean_regrets = {t:0 for t in range(T)}
    for t in range(T):
        eps_mean_regrets[t] = sum(eps_regrets[n][t] for n in range(N))/N
        ucb_mean_regrets[t] = sum(ucb_regrets[n][t] for n in range(N))/N

    #VISUALIZATION OF RESULTS
    #compute t^2/3 (c K log t)^1/3
    ref_eps = list()
    for t in range(1, T+1):
        ref_eps.append((t**(2/3))*(2*len(arms_set)*math.log(t))**(1/3))

    #compute c*sqrt(KtlogT)
    ref_ucb = list()
    for t in range(1, T+1):
        ref_ucb.append(math.sqrt(len(arms_set)*t*math.log(T)))

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    #Plot eps-greedy regret against its reference value
    ax1.plot(range(1,T+1), eps_mean_regrets.values(), label = 'eps_mean_regret')
    ax1.plot(range(1,T+1), ref_eps, label = f't^2/3 (2 K log t)^1/3')
    ax1.set_xlabel('t')
    ax1.set_ylabel('E[R(t)]')
    ax1.legend()

    #Plot ucb regret against its reference value
    ax2.plot(range(1,T+1), ucb_mean_regrets.values(), label = 'ucb_mean_regret')
    ax2.plot(range(1,T+1), ref_ucb, label = f'sqrt(K*t*logT)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('E[R(t)]')
    ax2.legend()

    #Plot ucb regret against eps-greedy regret
    ax3.plot(range(1,T+1), eps_mean_regrets.values(), label = 'eps_mean_regret')
    ax3.plot(range(1,T+1), ucb_mean_regrets.values(), label = 'ucb_mean_regret')
    ax3.set_xlabel('t')
    ax3.set_ylabel('E[R(t)]')
    ax3.legend()

    #Observe that each algorithm performs better than the worst case regret bound
    #Anyway, UCB performs better than Eps-Greedy, as expected
    plt.show()
