from tqdm import tqdm

def ind_supra_node(supra_node, t_nodes): #Index supra nodes in tree way order (time->node index)
    t = supra_node[1]
    ind = 0
    for t_node in t_nodes:
        if t_node[0] == t: #If time is the same
            ind += t_node[1].index(supra_node[0])
            break
        else: 
            ind += len(t_node[1])
    return ind


def convert(ifilename, ofilename):
    ifile = open(ifilename) #Opens file
    edges = [] #Time + conventional edge format! --> Time represents a window of [t-20,t] seconds.
    for line in ifile: 
        c = line.split(" ")
        edges.append([int(c[0]),int(c[1]),int(c[2])]) #Store edges
    ifile.close()

    t_nodes = [] #List of nodes for each time t : t_node[0] = [t0, [nodes at t0]]
    t_nodes.append([edges[0][2], [edges[0][0], edges[0][1]]]) 
    for edge in edges:
        node_a = edge[0]
        node_b = edge[1]
        if edge[2] == t_nodes[-1][0]: #If time is the same as previous edge time
            if node_a not in t_nodes[-1][1]:
                t_nodes[-1][1].append(node_a)

            if node_b not in t_nodes[-1][1]:
                t_nodes[-1][1].append(node_b)
        else:
            t_nodes.append([edge[2], [edge[0], edge[1]]])
    ####
    supra_edges = [] #Format: [[node_x, t_a], [node_y, t_b], weight]
    print('Transforming temporal edges to supra edges.')
    for i in tqdm(range(0, len(edges)-1)):
        edge = edges[i]
        find_a = True
        find_b = True
        for j in range(i+1, len(edges)):
            aux_edge = edges[j]
            if aux_edge[2] != edge[2]: #If time is different
                w = 1/abs(aux_edge[2] - edge[2]) #Weight of the edge
                if aux_edge[0] == edge[0] or aux_edge[1] == edge[0]:
                    if find_a == True: #If first node (edge[0]) is found active at time t':
                        supra_edges.append([[edge[1], edge[2]], [edge[0], aux_edge[2]], w]) #Shared edge
                        supra_edges.append([[edge[0], edge[2]], [edge[0], aux_edge[2]], w]) #Self edge
                        find_a = False
                if aux_edge[0] == edge[1] or aux_edge[1] == edge[1]: #If second node (edge[1]) is found active at time t':
                    if find_b == True:
                        supra_edges.append([[edge[0], edge[2]], [edge[1], aux_edge[2]], w])
                        supra_edges.append([[edge[1], edge[2]], [edge[1], aux_edge[2]], w])
                        find_b = False

            if find_a == False and find_b == False: #If both nodes have been found, stop searching
                break
    ofile = open(ofilename, 'w')
    print('Re indexing supra edges and storing.')
    for edge in tqdm(supra_edges):
        ia = ind_supra_node(edge[0], t_nodes)
        ib = ind_supra_node(edge[1], t_nodes)
        s = str(ia) + '\t' + str(ib) + '\t' + str(edge[2]) + '\n'
        ofile.write(s)
    ofile.close()
        