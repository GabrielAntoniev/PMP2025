from pgmpy.models import MarkovNetwork
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import numpy as np

def map_number_to_values(n, nrbits):
    
    v = [-1] * nrbits
    contor = nrbits
    while n:
        c = n%2
        if c == 0:
            c = -1
        contor -= 1
        v[contor]=c
        n //=2
    return v

def map_variable_name_to_number(variable_name):
    variable_name = str(variable_name)
    return int(variable_name[1:])

def phi(variables):
    possible_values_for_variables = [-1, 1]

    nr_combinations = 2**len(variables)
    variable_indexes = []
    for j in variables:
            variable_indexes.append(map_variable_name_to_number(j))

    rez = []
    for i in range(nr_combinations):
        
        exponent = 0
        possible_values_combination = map_number_to_values(i, len(variables))
         
        for j in range(len(variables)):
            exponent += variable_indexes[j]*possible_values_combination[j]

        exponent = np.exp(exponent)
        rez.append(exponent)

    print(f"variables: {variables} => phi: {rez}")
    rez = np.array(rez)
    rez = rez / rez.sum()
    return rez.tolist()


#ex1a
model = MarkovNetwork([('A1', 'A2'), ('A1', 'A3'), ('A2', 'A4'), ('A2', 'A5'), ('A3', 'A4'), ('A4', 'A5')])

pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=1, node_size=2000)
plt.show()


#ex1b
factor_a_b = DiscreteFactor(variables=['A2', 'A4', 'A5'],cardinality=[2, 2, 2],values=phi(['A2', 'A4', 'A5']))
factor_b_c = DiscreteFactor(variables=['A3', 'A4'],cardinality=[2, 2],values=phi(['A3', 'A4']))
factor_c_d = DiscreteFactor(variables=['A1', 'A3'],cardinality=[2, 2],values=phi(['A1', 'A3']))
factor_d_a = DiscreteFactor(variables=['A1', 'A2'],cardinality=[2, 2],values=phi(['A1', 'A2']))

model.add_factors(factor_a_b, factor_b_c,factor_c_d, factor_d_a)
model.get_factors()
model.get_local_independencies()

print("\nFactors:")
for f in model.get_factors():
    print(f)

print("\nLocal independencies")
print(model.get_local_independencies())

#ex2

np.random.seed(1245433)
original = np.random.randint(0, 2, size=(5, 5))

noisy = original.copy()
num_noisy = noisy.size//10
coords = np.random.randint(0,5, size = (2,2))
for i, j  in coords:
    noisy[i, j] = 1 - noisy[i, j] 

print("Original image:\n", original)
print("Noisy image:\n", noisy)

model = MarkovNetwork()

rows, cols = noisy.shape
variables = []
lamda = 2
states = [0, 1]

for i in range(rows):
    for j in range(cols):
        variables.append(f'X{i}_{j}')

for i in range(rows):
    for j in range(cols):
        v = f'X{i}_{j}'
        if i > 0: model.add_edge(v, f'X{i-1}_{j}')  #n
        if i < rows - 1: model.add_edge(v, f'X{i+1}_{j}')  #s
        if j > 0: model.add_edge(v, f'X{i}_{j-1}')  #w
        if j < cols - 1: model.add_edge(v, f'X{i}_{j+1}')  #e


for i in range(rows):
    for j in range(cols):
        y = noisy[i, j]
        values = []
        for x in states:
            e = np.exp(-lamda * (x - y) ** 2)
            values.append(e)
        values = np.array(values)
        values = values / np.sum(values)
        f = DiscreteFactor([f'X{i}_{j}'], [2], values)
        model.add_factors(f)

for i in range(rows):
    for j in range(cols):
        v = f'X{i}_{j}'
        if i < rows - 1:  #s
            n = f'X{i+1}_{j}'
            vals = []
            for x1 in states:
                for x2 in states:
                    e = np.exp(-(x1 - x2) ** 2)  
                    vals.append(e)
            vals = np.array(vals)
            vals = vals / np.sum(vals)
            f = DiscreteFactor([v, n], [2, 2], vals)
            model.add_factors(f)
        if j < cols - 1:  #e
            n = f'X{i}_{j+1}'
            vals = []
            for x1 in states:
                for x2 in states:
                    e = np.exp(-(x1 - x2) ** 2)
                    vals.append(e)
            vals = np.array(vals)
            vals = vals / np.sum(vals)
            f = DiscreteFactor([v, n], [2, 2], vals)
            model.add_factors(f)

print("Graph built with", len(model.nodes()), "nodes and", len(model.edges()), "edges.")
pos = {f'X{i}_{j}': (j, -i) for i in range(rows) for j in range(cols)}
plt.figure(figsize=(5, 5))
nx.draw(model, pos=pos, with_labels=True, node_size=600, node_color="skyblue", font_size=8)
plt.title("MRF")
plt.show()



bp = BeliefPropagation(model)
map_result = bp.map_query(variables=variables)
denoised = np.zeros_like(original)
for i in range(rows):
    for j in range(cols):
        denoised[i, j] = map_result[f'X{i}_{j}']

print("Denoised image: \n", denoised)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(original, cmap='gray'); axs[0].set_title("Original")
axs[1].imshow(noisy, cmap='gray'); axs[1].set_title("Noisy")
axs[2].imshow(denoised, cmap='gray'); axs[2].set_title("Denoised (MAP)")
for ax in axs: ax.axis('off')
plt.show()
