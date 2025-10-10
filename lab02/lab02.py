import numpy as np
import matplotlib.pyplot as plt

def isPrime(n):
    if n==2 or n==3 or n==5:
        return 1
    return 0

def event1(l):
    die = np.random.randint(1, 7)

    if isPrime(die):
        l.append(2)
    elif die == 6:
        l.append(0)
    else: 
        l.append(1)

    #pos = np.random.randint(0,len(l)-1,1)
    return l[np.random.randint(0,len(l))]


def ex1():
    #0 -> red
    #1 -> blue
    #2 -> black
    l = [0,0,0,1,1,1,1,2,2]

    #a) - simulare experiment 1 data
    print(event1(l))

    #b) - estimare probabilitate extragere bila rosie
    nr_experimente_independente = 1000000
    nr_red = 0
    for _ in range(nr_experimente_independente):
        if event1(l) == 0:
            nr_red +=1

    print(nr_red/nr_experimente_independente)

    #c)
    exact_probability = 0 #nush


def ex2():

    #a)
    X1 = np.random.poisson(lam=1, size=1000)
    X2 = np.random.poisson(lam=2, size=1000)
    X3 = np.random.poisson(lam=5, size=1000)
    X4 = np.random.poisson(lam=10, size=1000)

    print(X1)
    print("--------")
    print(X2)
    print("--------")
    print(X3)
    print("--------")
    print(X4)
    print("#####################")

    #b)
    l = [1,2,5,10]
    lamda = l[np.random.randint(0,len(l))]
    L = np.random.poisson(lam=lamda, size=1000)

    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    axs = axs.flatten()

    datasets = [X1, X2, X3, X4, L]
    titles = ["Poisson(1)", "Poisson(2)", "Poisson(5)", "Poisson(10)", "Poisson(lambda)"]

    for i in range(5):
        axs[i].hist(datasets[i], bins=range(0, max(datasets[i]) + 2), alpha=0.7, edgecolor='black')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Value")
        axs[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


ex1()
ex2()
