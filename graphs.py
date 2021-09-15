import random as random
import networkx as nx
import numpy as np

def klemm_eguiluz(N0, max_step): 
    #N0 Number of generation 0 nodes.
    #Initial network: N0 nodes fully connected
    actives = [] #List of active nodes
    inactives = [] #List of inactive nodes
    inlinks = [] #List of inlinks of each node
    indegree = []
    for i in range(0,N0):
        actives.append(i)
        inlink = [k for k in range(0,N0)] #List of outlinks|
        inlink.remove(i) #Remove itself from outlinks 
        inlinks.append(inlink)
        indegree.append(len(inlink))
    #Now we iterate to grow the network    
    for step in range(0,max_step):
        for i in actives: 
            inlinks[i] =  inlinks[i] + [N0 + step] #Adds the new node to the list of in-links of the node i
            indegree[i] += 1 #Adds +1 to all active nodes degree
        inlinks.append([]) #New node has no inlinks and degree 0
        indegree.append(0)
        
        #Deactivation step
        a = N0 #Model constant.
        r = sum([(a + len(inlinks[i]))**(-1) for i in actives]) #Constant needed to calculate probability of deactivation
        p = [r**(-1)/(a+len(inlinks[i])) for i in actives] #To store probabilities of deactivation
        deactivated_node = random.choices(actives,p,k=1)[0] #Picks weighted node at random
        actives.remove(deactivated_node)
        inactives.append(deactivated_node)
        actives.append(N0 + step)
    #Creates a networkx graph with the inlinks list
    edges = []
    G = nx.Graph() #Initializes empty graph
    for i in range(0,len(inlinks)): #Iterates over all nodes
        for j in inlinks[i]: #Goes over all neighbours of the node i
            G.add_edge(i,j) #Add the edge i-j
    return G

def erdos_renyi(N,p): #Random graph
    #N number of nodes, p probability of 2 nodes being connected. 
    return nx.erdos_renyi_graph(N, p)

def watts_strogatz(N,k,p): #Small world
    #N number of nodes, k the mean degree (even interger), p probability of being rewired.
    return nx.watts_strogatz_graph(N, k, p)

def barabasi_albert(N0,m): #Scale free
    #N0 number of initial nodes, each new node is connected to m existing nodes with probability prop. to node connectivity.
    return nx.barabasi_albert_graph(N0, m)