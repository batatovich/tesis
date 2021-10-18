import ndlib.models.ModelConfig as mc
import ndlib.models.CompositeModel as gc
import ndlib.models.compartments as cpm
import ndlib.models.DynamicCompositeModel as dgc
import ndlib.models.dynamic.DynSIRModel as DynSIRModel
import random as random
import numpy as np
import dynetx as dn
from tqdm import tqdm
import graphs
import dynetx as dn

def p(x, m, N):
    return np.tanh(m*x/(N-x))

class SIR_bis(gc.CompositeModel):
    def __init__(self, g, N, beta, gamma, m, inc_period, I0):
        # Composite Model instantiation
        self.model = gc.CompositeModel(g)
    
        # Model statuses
        self.model.add_status("Susceptible")
        self.model.add_status("Blocked") #The agent has been infected, but it cannot yet infect other agents.
        self.model.add_status("Infected") #In this context, it means that the agent is capable of transmiting the desease.
        self.model.add_status("Recovered")

        #Model parameters
        self.alfa = p(I0, m, N) #Spontaneous probability of contagion
        self.beta = beta
        self.gamma = gamma
        self.m = m #Modulates the alfa probability
        self.N = N
        
        # Compartment definition
        self.c1 = cpm.NodeStochastic(self.beta, triggering_status = 'Infected')
        self.c2 = cpm.CountDown("Incubation", iterations = inc_period)
        self.c3 = cpm.NodeStochastic(self.gamma)
        self.c4 = cpm.NodeStochastic(self.alfa)

        # Rule definition
        if inc_period != 0:
            self.model.add_rule("Susceptible", "Blocked", self.c1)
            self.model.add_rule("Blocked", "Infected", self.c2)
        else:
            self.model.add_rule("Susceptible", "Infected", self.c1)
        self.model.add_rule("Infected", "Recovered", self.c3)
        if m != 0:
            self.model.add_rule("Susceptible", "Blocked", self.c4)
    
        # Model initial status configuration
        self.iterations = [] 
        config = mc.Configuration()
        infected_nodes = random.sample([i for i in range(0,N)], I0) #Picks I0 initial susceptibles at random and makes them infectious
        config.add_model_initial_configuration("Infected", infected_nodes)

        # Simulation execution
        self.model.set_initial_status(config)
    
    def full_simulation(self, max_it):
        inf_index = self.model.get_status_map()['Infected']
        inc_index = self.model.get_status_map()['Blocked']
        self.iterations = []

        for it in tqdm(range(0, max_it)):
            self.iterations.append(self.model.iteration())
            node_count = self.iterations[-1]['node_count']
            Inf = node_count[inf_index]
            Inc = node_count[inc_index]
            self.alfa = p(Inf, self.m, self.N) #Update alfa value
            if Inf == 0 and Inc == 0: #If there are no more infected/incubating agents
                break

        return self.iterations


class temporal_SIR_bis(gc.CompositeModel):
    def __init__(self, ifilename, beta, gamma, m, inc_period, I0):
        self.dgraph = graphs.dynNetwork(ifilename) #Loads dynamic graph
        self.gamma = gamma
        self.beta = beta
        self.inc_period = inc_period

        #Timestamps set up
        self.timestamps = self.dgraph.temporal_snapshots_ids() #Retrieves timestamps
        time_step = self.timestamps[1] - self.timestamps[0]
        for i in range(2,len(self.timestamps)): #Picks the minimum delta t in the dynamic graph, and set it to be the "time step" between iterations.
            aux_time_step = self.timestamps[i] - self.timestamps[i-1]
            if aux_time_step < time_step:
                time_step = aux_time_step 
        self.norm_timestamps = [int((t - min(self.timestamps))/time_step) for t in self.timestamps] #Polish timestamps, now time step is 1.
        self.current_it = 0 #Current iteration

        #Initialize state lists
        s = self.dgraph.time_slice(t_from = self.timestamps[0], t_to = self.timestamps[1]) #Slices first timestamp from the graph.
        
        self.infected = []
        if len(s.nodes()) > I0:
            self.infected = random.sample(s.nodes(), I0)
            self.Inft = [I0]
        else: 
            self.infected = s.nodes()
            self.Inft = [len(s.nodes())]

        self.recovered = [] #Initial state is empty for recovered and incubating.
        self.incubating_t = [] #This will store the time at with a given node got infected
        self.incubating = [] #This one stores only node index of when the node got infected
        self.iterate_status = True
        self.Rect = [len(self.recovered)]
        self.Inct = [len(self.incubating)]
        self.Sust = [len(self.dgraph.nodes())]
        self.status  = [] # status[i][j] is the status of node at position j at time: timestamps[i]
        self.ordered_status = []
        stat = []
        for node in list(s.nodes()):
            s = 1
            if node in self.incubating: 
                s = 2
            if node in self.infected:
                s = 3
            if node in self.recovered:
                s = 4
            self.ordered_status.append(s)
            stat.append(s)
        self.status.append(stat)
    def iterate(self):
        if self.current_it < len(self.timestamps) - 1: 
            s = self.dgraph.time_slice(t_from = self.timestamps[self.current_it], t_to = self.timestamps[self.current_it + 1])
            active_nodes = list(s.nodes()) #Active nodes at current time
            new_recovered = []
            new_incubating = [] 
            new_incubating_t = []
            new_infected = []       
            dt = self.norm_timestamps[self.current_it+1] - self.norm_timestamps[self.current_it]

            #Susceptible -> Incubating
            for node in active_nodes:
                if node not in self.recovered and node not in self.infected and node not in self.incubating: #If susceptible
                    neig = s.neighbors(node) #Neighbors of node
                    inf_neig = sum(ele in neig for ele in self.infected) #Counts how many neigbors are infected
                    if random.random() < inf_neig*self.beta:
                        if self.inc_period != 0:
                            new_incubating.append(node)
                            new_incubating_t.append(self.norm_timestamps[self.current_it+1])
                        else:
                            new_infected.append(node)
       
            #Incubating -> Infected
            removed_inc_id= []
            for node in self.incubating: 
                id = self.incubating.index(node)
                if self.norm_timestamps[self.current_it + 1] - self.incubating_t[id] > self.inc_period:
                    new_infected.append(node)
                    removed_inc_id.append(id)
            aux_inc = [n for n in self.incubating if n not in new_infected] + new_incubating
            self.incubating = aux_inc
            aux_inc_t = [t for t in self.incubating_t if self.incubating_t.index(t) not in removed_inc_id] + new_incubating_t
            self.incubating_t = aux_inc_t

            #Infected -> Recovered
            for node in self.infected:
                if random.random() < self.gamma*dt:
                    new_recovered.append(node)
                    self.recovered.append(node)

            aux_inf = [elem for elem in self.infected if elem not in new_recovered]
            self.infected = aux_inf + new_infected
            self.Rect.append(len(self.recovered))
            self.Inct.append(len(self.incubating))
            self.Inft.append(len(self.infected))
            self.Sust.append(self.Sust[0] - self.Rect[-1] -self.Inct[-1] -self.Inft[-1])
            stat = [] 
            for node in active_nodes:
                s = 1
                if node in self.incubating: 
                    s = 2
                if node in self.infected:
                    s = 3
                if node in self.recovered:
                    s = 4
                stat.append(s)
                self.ordered_status.append(s)
            self.status.append(stat)
        
        else: 
            print('Simulation completed!')
            self.iterate_status = False

        self.current_it += 1

        if len(self.infected) == 0 and len(self.incubating) == 0:
            self.iterate_status = False
            print('Simulation completed, desease expired.')

    def full_simulation(self):
        while self.iterate_status:
            self.iterate()
