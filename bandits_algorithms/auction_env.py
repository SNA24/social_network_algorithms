import networkx as nx
import math, random
import matplotlib.pyplot as plt

class Environment:

    def __init__(self, G : nx.Graph):
        self.__social_graph = G
        self.__alive_edges = {e: random.random() < self.__social_graph.edges[e]['weight'] for e in self.__social_graph.edges}

    def find_alive_edges(self, node):
        return [e for e in self.__social_graph.edges(node) if e in self.__alive_edges and self.__alive_edges[e]]

    def receive_reward(self, a_t):
        return len(self.find_alive_edges(a_t))

class UCB_Learner:

    def __init__(self, nodes, environment, T = None):
        self.arms_set = list(nodes) #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__T = T
        if self.__T is None:
            self.__t = 1 #If the time horizon is unknown, we need to remember the current time step
        # Next initialization will serve for computing the best strategy during exploitation steps
        self.num = {a: 0 for a in self.arms_set} #It saves the number of times arm a has been selected
        self.rew = {a: 0 for a in self.arms_set} #It saves the cumulative reward achieved by arm a when selected
        # It saves the ucb value of each arm until the current time step
        # It is initialised to infinity in order to allow that each arm is selected at least once
        self.__ucb = {a: float('inf') for a in self.arms_set}

    def __choose_arm(self):
        return max(self.__ucb, key=self.__ucb.get) #We choose the arm that has the highest average revenue

    # This function returns the arm chosen by the learner and the corresponding reward returned by the environment
    def play_arm(self):
        a_t = self.__choose_arm() #We choose the arm to play
        reward = self.__environment.receive_reward(a_t) #We save the reward assigned by the environment
        # We update the number of times arm a_t has been chosen, its cumulative reward and its UCB value
        self.num[a_t] +=1
        self.rew[a_t] += reward
        if self.__T is not None: #If the time horizon is known
            self.__ucb[a_t] = self.rew[a_t]/self.num[a_t] + math.sqrt(2*math.log(self.__T)/self.num[a_t])
        else: #If the time horizon is unknown, each time step can be the last. Hence, we are more conservative and use t in place of T
            self.__ucb[a_t] = self.rew[a_t] / self.num[a_t] + math.sqrt(2 * math.log(self.__t) / self.num[a_t])
            self.__t += 1

        return a_t, reward

if __name__ == '__main__':
    # Create a social network graph using NetworkX
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from([1, 2, 3, 4, 5])
    # Add edges with random weights (probability of being alive)
    G.add_edge(1, 2, weight=0.6)
    G.add_edge(1, 3, weight=0.1)
    G.add_edge(1, 4, weight=0.3)
    G.add_edge(2, 4, weight=0.7)
    G.add_edge(2, 5, weight=0.2)
    G.add_edge(2, 3, weight=0.1)
    G.add_edge(3, 5, weight=0.9)
    G.add_edge(3, 4, weight=0.1)
    G.add_edge(4, 5, weight=0.9)
    G.add_edge(5, 1, weight=0.1)
    
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
    ucb_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the UCB learner

    #SIMULATION PLAY
    for n in range(N):
        ucb_cum_reward = 0 #it saves the cumulative reward of the UCB learner
        cum_opt_reward = 0 #it saves the cumulative reward of the best-arm in hindsight
        #Environment
        env = Environment(G)
        alive_edges_count = {n: env.find_alive_edges(n) for n in G.nodes()}
        opt_a = max(alive_edges_count, key=lambda k: len(alive_edges_count[k]))
        #UCB Learner
        ucb_learn = UCB_Learner(G.nodes(), env, T)
        for t in range(T):
            #reward obtained by the optimal arm
            cum_opt_reward += env.receive_reward(opt_a)

            # reward obtained by the ucb learner
            a, reward = ucb_learn.play_arm()
            ucb_cum_reward += reward
            #regret of the ucb learner
            ucb_regrets[n][t] = cum_opt_reward - ucb_cum_reward

    #compute the mean regret of the eps greedy and ucb learner
    ucb_mean_regrets = {t:0 for t in range(T)}
    for t in range(T):
        ucb_mean_regrets[t] = sum(ucb_regrets[n][t] for n in range(N))/N

    #VISUALIZATION OF RESULTS

    #compute c*sqrt(KtlogT)
    ref_ucb = list()
    for t in range(1, T+1):
        ref_ucb.append(math.sqrt(len(G.nodes())*t*math.log(T)))

    fig, (ax2) = plt.subplots(1)

    #Plot ucb regret against its reference value
    ax2.plot(range(1,T+1), ucb_mean_regrets.values(), label = 'ucb_mean_regret')
    ax2.plot(range(1,T+1), ref_ucb, label = f'sqrt(K*t*logT)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('E[R(t)]')
    ax2.legend()

    #Observe that each algorithm performs better than the worst case regret bound
    #Anyway, UCB performs better than Eps-Greedy, as expected
    plt.show()
