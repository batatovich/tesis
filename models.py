import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random as random

def SIR(G,N,b,g,I0):
    model = ep.SIRModel(G) #Selects the model to be used
    # Model Configuration
    config = mc.Configuration()
    indexes = [i for i in range(0,N)]
    infected_nodes = random.sample(indexes, I0) #Picks I0 initial susceptibles at random and makes them infectious
    config.add_model_initial_configuration("Infected", infected_nodes)
    config.add_model_parameter('beta', b)
    config.add_model_parameter('gamma', g)
    model.set_initial_status(config)
    return model