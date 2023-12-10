import networkx as nx
import math, random
import matplotlib.pyplot as plt

class Environment:

    def __init__(self, G : nx.Graph):
        self.__social_graph = G
        
    def set_alive_edges(self):
        self.__alive_edges = {e: random.random() < self.__social_graph.edges[e]['weight'] for e in self.__social_graph.edges()}

    def find_reachable_nodes(self, node):
        reachable_nodes = []
        for e in self.__social_graph.edges():
            if self.__alive_edges[e] == True:
                if self.__social_graph.is_directed():
                    if node == e[0]:
                        reachable_nodes.append(e[1])
                else:
                    if node in e:
                        other_node = e[0] if e[0] != node else e[1]
                        reachable_nodes.append(other_node)
                    
        return reachable_nodes

    def receive_reward(self, a_t):
        return len(self.find_reachable_nodes(a_t))
    
class EpsGreedy_Learner:
    # eps is a function that takes in input a time step t and returns eta_t
    # Use T = None for unknown time horizon
    def __init__(self, nodes, environment, eps, T=None):
        self.__arms_set = list(nodes) #initialize the set of arms
        self.__environment = environment #initialize the environment
        self.__eps = eps #initialize the sequence of eps_t
        self.__T = T
        #Next initialization will serve for computing the best strategy during exploitation steps
        self.__num = {a:0 for a in self.__arms_set} #It saves the number of times arm a has been selected
        self.__rew = {a:0 for a in self.__arms_set} #It saves the cumulative reward achieved by arm a when selected
        self.__avgrew = {a:0 for a in self.__arms_set} #It saves the average reward achieved by arm a until the current time step
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
    G = nx.DiGraph()
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
    eps_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the eps-greedy learner
    ucb_regrets = {n: {t: 0 for t in range(T)} for n in range(N)} #regret matrix for the UCB learner

    #INITIALIZATION FOR EPS-GREEDY
    #A common choice for eps_t = (K log t/t)^1/3
    def give_eps(t):
        if t == 0:
            return 1  #for the first step we cannot make exploitation, so eps_1 = 1
        return (len(G.nodes())*math.log(t+1)/(t+1))**(1/3)
    eps = give_eps

    #SIMULATION PLAY
    for n in range(N):
        ucb_cum_reward = 0 #it saves the cumulative reward of the UCB learner
        eps_cum_reward = 0 #it saves the cumulative reward of the eps-greedy learner
        cum_opt_reward = 0 #it saves the cumulative reward of the best-arm in hindsight
        #Environment
        env = Environment(G)
        
        #Eps-Greedy Learner
        eps_learn = EpsGreedy_Learner(G.nodes(), env, eps, T)
        #UCB Learner
        ucb_learn = UCB_Learner(G.nodes(), env, T)
        for t in range(T):

            env.set_alive_edges()
            reached_nodes = {n: env.find_reachable_nodes(n) for n in G.nodes()}
            opt_a = max(reached_nodes, key=lambda k: len(reached_nodes[k]))
            
            #reward obtained by the eps_greedy learner
            a, reward = eps_learn.play_arm()
            eps_cum_reward += reward
            
            # reward obtained by the ucb learner
            a, reward = ucb_learn.play_arm()
            ucb_cum_reward += reward
            
            #reward obtained by the optimal arm
            cum_opt_reward += env.receive_reward(opt_a)
            
            #regret of the eps_greedy learner
            eps_regrets[n][t] = cum_opt_reward - eps_cum_reward
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
        ref_eps.append((t**(2/3))*(2*len(G.nodes())*math.log(t))**(1/3))

    #compute c*sqrt(KtlogT)
    ref_ucb = list()
    for t in range(1, T+1):
        ref_ucb.append(math.sqrt(len(G.nodes())*t*math.log(T)))

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
