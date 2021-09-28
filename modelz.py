import ndlib.models.ModelConfig as mc
import ndlib.models.CompositeModel as gc
import ndlib.models.compartments as cpm
import random as random
import numpy as np
from tqdm import tqdm

def p(x, m, N):
    return np.tanh(m*x/(N-x))

class delayed_SIR(gc.CompositeModel):
    def __init__(self, g, N, beta, gamma, m, inc_period, I0):
        #super(self.__class__,self).__init__(g, N, beta, gamma, inc_period, I0)
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
            self.model.add_rule("Susceptible", "Blocked", self.c4)
        else:
            self.model.add_rule("Susceptible", "Infected", self.c1)
            self.model.add_rule("Susceptible", "Infected", self.c4)
        self.model.add_rule("Infected", "Recovered", self.c3)
    
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
