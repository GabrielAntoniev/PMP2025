from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt


def ex1a():

    email_model = DiscreteBayesianNetwork([('S','O'),('S','L'),('L','M'),('S','M')])
    pos = nx.circular_layout(email_model)
    nx.draw(email_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000, font_weight='bold', node_color='skyblue')
    plt.show()

                  

def ex1b():

    cpd_s = TabularCPD(variable='S', variable_card=2, values = [[0.6],[0.4]])


    cpd_o = TabularCPD(variable='O', variable_card=2,
                                     values = [[0.9, 0.3],
                                               [0.1, 0.7]],
                                     evidence = ['S'],
                                     evidence_card = [2])

    cpd_l = TabularCPD(variable='L', variable_card = 2,
                                     values = [[0.7, 0.2],
                                               [0.3, 0.8]],
                                     evidence = ['S'],
                                     evidence_card= [2])

    cpd_m = TabularCPD(variable = 'M', variable_card = 2,
                                       values = [[0.8, 0.4, 0.8, 0.1],
                                                 [0.2, 0.6, 0.2, 0.9]],
                                       evidence = ['S', 'L'],
                                       evidence_card= [2,2])

    print(cpd_s, cpd_o, cpd_l, cpd_m)


    
def ex2():  

    cpd_add = TabularCPD(variable='AddBall', variable_card=3, values=[[1/2], [1/6], [1/3]],)

    cpd_extract = TabularCPD(variable='ExtractRed',variable_card=2,
                                                   values=[[1/6, 1.0, 1/5],
                                                           [5/6, 0.0, 4/5]],
                                                    
                                                   evidence=['AddBall'],
                                                   evidence_card=[3])
    
    print(cpd_add, cpd_extract)

ex1a()
ex1b()
ex2()

