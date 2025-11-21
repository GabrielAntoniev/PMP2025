from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from hmmlearn import hmm


def subject1():
    #1a)
    print("1a)")
    model = DiscreteBayesianNetwork([('O','H'),('O','W'),('W','R'),('H','R'),('H','E'),('R','C')])
    pos = nx.circular_layout(model)
    nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000, font_weight='bold', node_color='skyblue')
    plt.show()

    #1b)
    print("1b)")
    cpd_o = TabularCPD(variable='O', variable_card=2, values = [[0.7],[0.3]])

    cpd_h = TabularCPD(variable='H', variable_card=2,
                                     values = [[0.8, 0.1],
                                               [0.2, 0.9]],
                                     evidence = ['O'],
                                     evidence_card = [2])
    
    cpd_w = TabularCPD(variable='W', variable_card=2,
                                     values = [[0.4, 0.9],
                                               [0.6, 0.1]],
                                     evidence = ['O'],
                                     evidence_card = [2])
    
    cpd_r = TabularCPD(variable = 'R', variable_card = 2,
                                       values = [[0.5, 0.7, 0.1, 0.4],
                                                 [0.5, 0.3, 0.9, 0.6]],
                                       evidence = ['H', 'W'],
                                       evidence_card= [2,2])
    
    cpd_e = TabularCPD(variable='E', variable_card=2,
                                     values = [[0.8, 0.2],
                                               [0.2, 0.8]],
                                     evidence = ['H'],
                                     evidence_card = [2])
    
    cpd_c = TabularCPD(variable='C', variable_card=2,
                                     values = [[0.6, 0.15],
                                               [0.4, 0.85]],
                                     evidence = ['R'],
                                     evidence_card = [2])
    
    model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)
    assert model.check_model()
    infer = VariableElimination(model)

    print("\nP(H | C=1):")
    result = infer.query(variables=['H'], evidence={'C': 1})
    print(result)

    print("\nP(E | C=1):")
    result = infer.query(variables=['E'], evidence={'C': 1})
    print(result)

    print("\nP(H, W | C=1):")
    result = infer.query(variables=['H','W'], evidence={'C': 1})
    print(result)


    #1c)
    print("1c)")
    print("\nP(W, E | H=1):")
    result = infer.query(variables=['W','E'], evidence={'H': 1})
    print(result)

    print("\nP(W, E | H=0):")
    result = infer.query(variables=['W','E'], evidence={'H': 0})
    print(result)
    



    print("\nP(W | H=1):")
    result = infer.query(variables=['W'], evidence={'H': 1})
    print(result)

    print("\nP(W | H=0):")
    result = infer.query(variables=['W'], evidence={'H': 0})
    print(result)
    
    print("\nP(E | H=1):")
    result = infer.query(variables=['E'], evidence={'H': 1})
    print(result)

    print("\nP(E | H=0):")
    result = infer.query(variables=['E'], evidence={'H': 0})
    print(result)


def subject2():

    #1a)
    print("1a)")
    state_probability = np.array([0.4, 0.3, 0.3])#w-r-s
    n_states = len(state_probability)

    transition_probability = np.array([[0.6, 0.3, 0.1],
                                    [0.2, 0.7, 0.1],
                                    [0.3, 0.2, 0.5]])
    
    emission_probability = np.array([[0.1, 0.7, 0.2],
                                    [0.05, 0.25, 0.7],
                                    [0.8, 0.15, 0.05]])
    
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = state_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    #1b)
    print("1b)")
    observations_sequence = np.array([0,1,2]).reshape(-1, 1)
    prob_obs = model.score(observations_sequence, lengths = len(observations_sequence))
    print('Probabilitatea observarii observatiilor:',np.exp(prob_obs))


    #1c)
    print("1c)")
    hidden_states = model.predict(observations_sequence)
    print("Most likely hidden states:", hidden_states)

    log_probability, hidden_states = model.decode(observations_sequence,
                                                lengths = len(observations_sequence),
                                                algorithm ='viterbi' )

    print('Log Probability :',log_probability)
    print("Most likely hidden states:", hidden_states)


subject1()
subject2()

