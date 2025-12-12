import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Prices.csv')

#a)
with pm.Model() as model:
    x1_data = pm.Data("x1_data", data['Speed'].values)
    x2_data = pm.Data("x2_data", np.log(data['HardDrive']).values)
    y_data = pm.Data("y_data", data['Price'].values)

    #a priori
    alpha = pm.Normal("alpha", mu=0, sigma=1000)
    beta1 = pm.Normal("beta1", mu=0, sigma=100) 
    beta2 = pm.Normal("beta2", mu=0, sigma=100) 
    sigma = pm.HalfNormal("sigma", sigma=100)

    mu = pm.Deterministic("mu", alpha + beta1 * x1_data + beta2 * x2_data)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

    idata = pm.sample(2000, tune=1000, return_inferencedata=True)
    #print(idata.observed_data)
    

# b si c)
summary = az.summary(idata, var_names=["beta1", "beta2"], hdi_prob=0.95)
print(summary)
#a posteriori
az.plot_posterior(idata, var_names=["beta1", "beta2"], hdi_prob=0.95, ref_val=0)
plt.show()
#pentru hd1=95% avem pentru beta1 [2.352, 7.054] si 0 nu apartine acestui interval, 
# deci beta1 nu poate fi 0, si deci daca e diferit de 0 atunci inseamna ca influenteaza pretul final 
#analog pentru beta2


#d)
x1_new = [33]
x2_new = [np.log(540)]

with model:
    pm.set_data({"x1_data": x1_new, "x2_data": x2_new})
    pp_mu = pm.sample_posterior_predictive(idata, var_names=["mu"], predictions=True)

mu_preds = pp_mu.predictions["mu"].stack(sample=("chain", "draw")).values
hdi_mu = az.hdi(mu_preds, hdi_prob=0.90)
print(f"90% HDI for price (mu): {hdi_mu}")

#e)
with model:
    ppc_e = pm.sample_posterior_predictive(idata, var_names=["y_obs"], predictions=True)

y_preds = ppc_e.predictions["y_obs"].stack(sample=("chain", "draw")).values
hdi_y = az.hdi(y_preds, hdi_prob=0.90)
print(f"90% HDI for price (y): {hdi_y}")


#bonus)
data['premium_binary'] = data['Premium'].apply(lambda x: 1 if x in ['yes', 1] else 0)

with pm.Model() as model_premium:
    x1_data = pm.Data("x1_data", data['Speed'].values)
    x2_data = pm.Data("x2_data", np.log(data['HardDrive']).values)
    y_data = pm.Data("y_data", data['Price'].values)
    x3_data = pm.Data("x3_data", data['premium_binary'].values)

    #a priori
    alpha = pm.Normal("alpha", mu=0, sigma=1000)
    beta1 = pm.Normal("beta1", mu=0, sigma=100)
    beta2 = pm.Normal("beta2", mu=0, sigma=100)
    beta3 = pm.Normal("beta3", mu=0, sigma=100)
    sigma = pm.HalfNormal("sigma", sigma=100)

    mu = alpha + beta1 * x1_data + beta2 * x2_data + beta3 * x3_data
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)
    idata_premium = pm.sample(1000, return_inferencedata=True)

az.plot_posterior(idata_premium, var_names=["beta3"], hdi_prob=0.95, ref_val=0)
plt.title("Posterior for premium (beta3)")
plt.show()