import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('date_colesterol.csv')
    print("Fisier gasit")
except FileNotFoundError:
    print("Eroare: fisierul cu date nu a fost gasit")
    exit()


t = data['Ore_Exercitii'].values
y = data['Colesterol'].values

t_mean, t_std = t.mean(), t.std()
y_mean, y_std = y.mean(), y.std()

t_scaled = (t - t_mean) / t_std
y_scaled = (y - y_mean) / y_std

print(f"len(data): {len(data)}")
print("-" * 40)

models = {}
traces = {}
idatas = {}

K_values = [3, 4, 5]

for K in K_values:
    
    with pm.Model() as model:

        w = pm.Dirichlet('w', a=np.ones(K))
        
        alpha = pm.Normal('alpha', mu=0, sigma=10, shape=K)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=K)
        gamma = pm.Normal('gamma', mu=0, sigma=10, shape=K)
        
        sigma = pm.HalfNormal('sigma', sigma=10, shape=K)
        mu = alpha + beta * t_scaled[:, None] + gamma * t_scaled[:, None]**2

        obs = pm.NormalMixture('obs', w=w, mu=mu, sigma=sigma, observed=y_scaled)
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, progressbar=True)
        
        #waic and loo pt model
        pm.compute_log_likelihood(trace)
        
        models[K] = model
        traces[K] = trace

#ex1
for K in K_values:
    print(f"\n[ Model K={K} ]")
    summary = az.summary(traces[K], var_names=['w', 'alpha', 'beta', 'gamma', 'sigma'])
    
    post = traces[K].posterior
    w_mean = post['w'].mean(dim=["chain", "draw"]).values

    alpha_s = post['alpha'].mean(dim=["chain", "draw"]).values
    beta_s = post['beta'].mean(dim=["chain", "draw"]).values
    gamma_s = post['gamma'].mean(dim=["chain", "draw"]).values
    sigma_s = post['sigma'].mean(dim=["chain", "draw"]).values


    gamma_orig = gamma_s * (y_std / (t_std ** 2))
    beta_orig = (beta_s * (y_std / t_std)) - (2 * gamma_orig * t_mean)
    alpha_orig = y_mean + (alpha_s * y_std) - (beta_orig * t_mean) - (gamma_orig * t_mean**2)
    sigma_orig = sigma_s * y_std
    
    print(f"{'Subpop':<8} {'Weight':<10} {'Alpha (Int)':<12} {'Beta (Lin)':<12} {'Gamma (Quad)':<12} {'Sigma':<10}")
    print("-" * 70)
    for k in range(K):
        print(f"{k+1:<8} {w_mean[k]:.3f}      {alpha_orig[k]:.2f}        {beta_orig[k]:.2f}        {gamma_orig[k]:.4f}        {sigma_orig[k]:.2f}")


print("\n" + "-"*40)


#ex2
comp_waic = az.compare(traces, ic="waic", scale="deviance")
comp_loo = az.compare(traces, ic="loo", scale="deviance")

print("\nWAIC: ")
print(comp_waic[['rank', 'elpd_waic', 'p_waic', 'weight']])

print("\nLOO: ")
print(comp_loo[['rank', 'elpd_loo', 'p_loo', 'weight']])

best_K = comp_waic.index[0]
print(f"best K = {best_K}")


fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
ax = axes.ravel() 

t_plot = np.linspace(t.min(), t.max(), 200)
t_plot_scaled = (t_plot - t_mean) / t_std

for idx, K in enumerate(K_values):
    current_ax = ax[idx]

    current_ax.scatter(t, y, alpha=0.3, color='gray', s=20, label='Observed Data')
    post = traces[K].posterior
    
    w_mean = post['w'].mean(dim=["chain", "draw"]).values
    alpha_mean = post['alpha'].mean(dim=["chain", "draw"]).values
    beta_mean = post['beta'].mean(dim=["chain", "draw"]).values
    gamma_mean = post['gamma'].mean(dim=["chain", "draw"]).values
    
    y_weighted_sum = np.zeros_like(t_plot)
    
    for k in range(K):

        y_sub_scaled = (alpha_mean[k] + 
                        beta_mean[k] * t_plot_scaled + 
                        gamma_mean[k] * t_plot_scaled**2)
        
        y_sub = y_sub_scaled * y_std + y_mean
        y_weighted_sum += w_mean[k] * y_sub
        
        current_ax.plot(t_plot, y_sub, '--', linewidth=1, alpha=0.6, 
                        label=f'Sub {k+1} (w={w_mean[k]:.2f})')

 
    current_ax.plot(t_plot, y_weighted_sum, 'k-', linewidth=3, label='pop fit')
    
    current_ax.set_title(f'Model pentru K = {K} subpopulatii')
    current_ax.set_xlabel('Ore_Exercitii')
    current_ax.set_ylabel('Colesterol')
    current_ax.legend(fontsize='small')

ax[3].axis('off')
plt.suptitle('Comparatie modele', fontsize=16)
plt.savefig('comparatie_modele.png')
plt.show()