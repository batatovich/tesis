from pecanpy import node2vec
from gensim.models import Word2Vec

class embedder():
    def __init__(self, ifilename, w, d, P, Q):
        self.workers = 4

        # Initialize node2vec object.
        g = node2vec.PreComp(p = P, q = Q, workers = self.workers, verbose = False)
        
        # Load graph from edgelist file
        g.read_edg(ifilename, weighted = w, directed = d)

        # precompute and save 2nd order transition probs (for PreComp only)
        g.preprocess_transition_probs()

        self.g = g

    def embed(self, n_features, n_walks, walk_len, w_size, n_iterations ):
        # Get embeddings
        embeddings = self.g.embed(dim = n_features, num_walks = n_walks, walk_length = walk_len, window_size = w_size, epochs = n_iterations)
        self.embeddings =  embeddings
    
    def w2f(self, ofilename):
        file = open(ofilename, 'w')
        for embedding in self.embeddings:
            for w in embedding:
                file.write(str(w) + '\t')
            file.write('\n')
        file.close()
        return self.embeddings

    def get_embeddings(self):
        return self.embeddings

    def estimate_memory(self, n_features, n_walks, walk_len, window_size, n_iterations):
        # generate random walks
        walks = self.g.simulate_walks(num_walks = n_walks, walk_length = walk_len)
        # build up word2vec model
        w2v_model = Word2Vec(walks, vector_size = n_features, window = window_size, min_count=0, sg=1, workers = self.workers, epochs = n_iterations)
        print('Memory requiered: ' + str(float(w2v_model.estimate_memory()['total'])/10**9) + ' Gigabytes.')