import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#ex1)
try:
    data = pd.read_csv('bike_daily.csv')
    print("Fisier gasit")
except FileNotFoundError:
    print("Eroare: fisierul cu date nu a fost gasit")
    exit()

columns = data.columns

rentals = data['rentals'].values
temp_c = data['temp_c'].values
humidity = data['humidity'].values
wind_kph = data['wind_kph'].values
is_holiday = data['is_holiday'].values
season = data['season'].values

plt.figure(figsize=(10, 6))
plt.scatter(rentals, temp_c, c='C0', marker='.', alpha=0.3, label='rentals-temp_c')
plt.plot(rentals, temp_c, c='C1')
plt.title(f"Puncte {rentals} - {temp_c}")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(rentals, humidity, c='C0', marker='.', alpha=0.3, label='rentals-humidity')
plt.plot(rentals, humidity, c='C1')
plt.title(f'Puncte {rentals} - {humidity} ')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(rentals, wind_kph, c='C0', marker='.', alpha=0.3, label='rentals-wind_kph')
plt.plot(rentals, wind_kph, c='C1')
plt.title(f'Puncte {rentals} - {wind_kph} ')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(rentals, is_holiday, c='C0', marker='.', alpha=0.3, label='rentals-is_holiday')
plt.plot(rentals, is_holiday, c='C1')
plt.title(f'Puncte {rentals} - {is_holiday} ')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(rentals, season, c='C0', marker='.', alpha=0.3, label='rentals-season')
plt.plot(rentals, season, c='C1')
plt.title(f'Puncte {rentals} - {season} ')
plt.legend()
plt.show()


#ex2)
#a)
rentals_scaled = (rentals - rentals.mean()) / rentals.std()
temp_c_scaled = (temp_c - temp_c.mean()) / temp_c.std()
humidity_scaled = (humidity - humidity.mean()) / humidity.std()
wind_kph_scaled = (wind_kph - wind_kph.mean()) / wind_kph.std()

#b)
with pm.Model() as model1:

    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta1 = pm.Normal("beta1", mu=0, sigma=10)
    beta2 = pm.Normal("beta2", mu=0, sigma=10)
    beta3 = pm.Normal("beta3", mu=0, sigma=10)

    mu = alpha + beta1 * temp_c_scaled + beta2 * humidity_scaled + beta3 * wind_kph_scaled
    sigma = pm.HalfNormal("sigma", sigma=10)

    likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=rentals_scaled)
    
    #trace = pm.sample(1000, tune=1000, return_inferencedata=True)

#c)
with pm.Model() as model2:

    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta1 = pm.Normal("beta1", mu=0, sigma=10)
    beta2 = pm.Normal("beta2", mu=0, sigma=10)
    beta3 = pm.Normal("beta3", mu=0, sigma=10)
    beta4 = pm.Normal("beta4", mu=0, sigma=10)

    mu = alpha + beta1 * temp_c_scaled + beta2 * humidity_scaled + beta3 * wind_kph_scaled + beta4 * temp_c_scaled * temp_c_scaled
    sigma = pm.HalfNormal("sigma", sigma=10)

    likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=rentals_scaled)


#ex3)
idatas = {}
with model1:
    idata1 = pm.sample(1000, tune=1000, return_inferencedata=True)
    pm.compute_log_likelihood(idata1)
    idatas[0] = idata1
summary = az.summary(idata1, var_names=["alpha", "beta1", "beta2", "beta3"], hdi_prob=0.95)
print(summary)

with model2:
    idata2 = pm.sample(1000, tune=1000, return_inferencedata=True)
    pm.compute_log_likelihood(idata2)
    idatas[1] = idata2
summary = az.summary(idata2, var_names=["alpha", "beta1", "beta2", "beta3", "beta4"], hdi_prob=0.95)
print(summary)

#ex4)
#a)
comp_waic = az.compare(idatas, ic="waic")
print("WAIC Comparison:\n", comp_waic)


#b)
ppc1 = pm.sample_posterior_predictive(idatas[0], extend_inferencedata=True, progressbar=False)
y_pred1 = ppc1.posterior_predictive["rentals"]        
az.plot_dist(y_pred1, color="orange", label="Predicted Y")


ppc2 = pm.sample_posterior_predictive(idatas[1], extend_inferencedata=True, progressbar=False)
y_pred = ppc2.posterior_predictive["rentals"]        
az.plot_dist(y_pred, color="orange", label="Predicted Y")





    