import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

az.style.use('arviz-darkgrid')

data = np.loadtxt('date.csv')
x_1 = data[:, 0]
y_1 = data[:, 1]

#1)

order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

beta_sigmas = [10, 100, np.array([10, 0.1, 0.1, 0.1, 0.1])]
#beta_sigmas = [10, 100, np.array([10, 0.1])]
labels = ['sigma=10', 'sigma=100', 'sigma=[10, 0.1...]']
colors = ['C1', 'C2', 'C3']

plt.figure(figsize=(10, 6))
plt.scatter(x_1s[0], y_1s, c='C0', marker='.', label='Data')

for i, sigma_val in enumerate(beta_sigmas):
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=sigma_val, shape=order)
        ε = pm.HalfNormal('ε', 5)
        
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        
        idata_p = pm.sample(1000, return_inferencedata=True, progressbar=False)

    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    
    plt.plot(x_1s[0][idx], y_p_post[idx], color=colors[i], label=f'Order {order} ({labels[i]})')

plt.title(f'Polynomial order {order} ')
plt.legend()
plt.show()

#2)

n_samples = 500
x_1_500 = np.linspace(-3, 3, n_samples)
y_1_500 = 2 + x_1_500**2 + np.random.normal(0, 2, n_samples) 

x_1p = np.vstack([x_1_500**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1_500 - y_1_500.mean()) / y_1_500.std()

# plt.figure(figsize=(10, 6))
plt.scatter(x_1s[0], y_1s, c='C0', marker='.', alpha=0.3, label='500 data points')

for i, sigma_val in enumerate(beta_sigmas):
    with pm.Model() as model_500:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=sigma_val, shape=order)
        ε = pm.HalfNormal('ε', 5)

        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)

        idata_p = pm.sample(1000, return_inferencedata=True)

    α_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_post = α_post + np.dot(β_post, x_1s)

    plt.plot(x_1s[0][idx], y_post[idx], color=colors[i], label=f'Order {order} ({labels[i]})')

plt.title('500 data points')
plt.legend()
plt.show()

#3)
data = np.loadtxt('date.csv')
x_1 = data[:, 0]
y_1 = data[:, 1]

# plt.figure(figsize=(10, 6))
x_mean = x_1.mean()
x_std = x_1.std()
y_mean = y_1.mean()
y_std = y_1.std()

x_plot = (x_1 - x_mean) / x_std
y_plot = (y_1 - y_mean) / y_std


idatas = {}
orders = [1, 2, 3]
colors = ['C1', 'C2', 'C3']

for i, order in enumerate(orders):
    x_1p = np.vstack([x_1**k for k in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    
    with pm.Model() as model:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
  
        if order == 1:
            μ = α + β[0] * x_1s[0]
        else:
            μ = α + pm.math.dot(β, x_1s)
            
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        
        idata = pm.sample(1000, return_inferencedata=True, progressbar=False)
        pm.compute_log_likelihood(idata)
        idatas[f'order_{order}'] = idata
        

        α_post = idata.posterior['α'].mean(("chain", "draw")).values
        β_post = idata.posterior['β'].mean(("chain", "draw")).values
        
        idx = np.argsort(x_1s[0])
        x_sorted = x_1s[:, idx]
        
        if order == 1:
            y_est = α_post + β_post[0] * x_sorted[0]
        else:
            y_est = α_post + np.dot(β_post, x_sorted)
            
        plt.plot(x_sorted[0], y_est, color=colors[i], linewidth=2, label=f'Order {order}')

plt.title('Polynomial Fits (Standardized)')
plt.xlabel('Standardized x')
plt.ylabel('Standardized y')
plt.legend()
plt.show()

comp_waic = az.compare(idatas, ic="waic")
print("WAIC Comparison:\n", comp_waic)