#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from my_functions import delta_t_naiv, read_in, mean_t50
import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt
import matplotlib as mp

events = read_in("data_data/events6t5.txt")
delta_T, n_values, T_50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values = delta_t_naiv(events, 300)            
# print("minimum n:  ", np.min(n_values))
# print("conter t50 was replaced", counter)
# print("delta T should be 0:  ", np.mean(delta_T))
# fig, axs = plt.subplots(3, 2, figsize=(9, 8)) 

# flattenedT50 = np.ravel(T_50_values)
# flattenedn = np.log10(np.ravel(n_values))
# flattenedTVar = np.ravel(tVar_values)
# flattenedDist = np.ravel(distance_values)
# flattenedZen = np.ravel(zenith_values)
# flattenedSig = np.log10(np.ravel(signal_values))
# flattenedGroup = np.ravel(group_values)
# counts, bins = np.histogram(delta_T, bins=80)

# # Logarithmierte Häufigkeiten berechnen
# log_counts = np.log(counts + 1)  # +1, um log(0) zu vermeiden

# # Balkendiagramm mit logarithmierten Werten plotten
# bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Mittelpunkte der Bins
# axs[0, 0].bar(bin_centers, log_counts, width=np.diff(bins), align='center', alpha=0.7, color='blue', edgecolor='black')
# axs[0, 0].set_title(r"$\Delta T$")
# axs[0, 0].set_xlabel(r"$\Delta T$ in ns")
# axs[0, 0].set_ylabel(r"Count")

# axs[0, 1].hist(flattenedT50, bins=80, color="blue", alpha=0.6)
# axs[0, 1].set_title(r"$T_{50}$")
# axs[0, 1].set_xlabel(r"$T_{50}$ in ns")
# axs[0, 1].set_ylabel(r"Count")

# # Logarithmic x-axis for `n`
# axs[1, 0].hist(flattenedn, bins=80, color="blue", alpha=0.6)
# axs[1, 0].set_title(r"$n$ (Logarithmic X-Scale)")
# axs[1, 0].set_xlabel(r"$n$ (log$_{10}$ scale)")
# axs[1, 0].set_ylabel(r"Count")

# # Logarithmic x-axis for `signal`
# axs[1, 1].hist(flattenedSig, bins=80, color="blue", alpha=0.6)
# axs[1, 1].set_title(r"Signal (Logarithmic X-Scale)")
# axs[1, 1].set_xlabel(r"Signal in VEM (log$_{10}$ scale)")
# axs[1, 1].set_ylabel(r"Count")

# axs[2, 0].hist(flattenedZen, bins=80, color="blue", alpha=0.6)
# axs[2, 0].set_title(r"Zenith")
# axs[2, 0].set_xlabel(r"Zenith in rad")
# axs[2, 0].set_ylabel(r"Count")

# axs[2, 1].hist(flattenedDist, bins=80, color="blue", alpha=0.6)
# axs[2, 1].set_title(r"Distance")
# axs[2, 1].set_xlabel(r"Distance from the core in m")
# axs[2, 1].set_ylabel(r"Count")

# # Adjust layout
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for the main title
# plt.savefig("data_s.pdf", format="pdf", bbox_inches="tight")
# plt.show()
# #print(delta_T)
# print("len values:  ", len(delta_T))
# print("standard deviation of delta_t", np.std(delta_T))






# def V_t0(a, b, c, Theta, signal):
#     return (a * (1 + b * ((1 / np.cos(Theta)) - 1.22)) * np.sqrt(signal)) ** 2 + (c * signal) ** 2

# def V_Delta_T(a, b, c, Theta, signal_i, signal_j):
#     return V_t0(a, b, c, Theta, signal_i) + V_t0(a, b, c, Theta, signal_j)

# def log_likelihood(a, b, c, delta_S, Theta_values, signal_values):
#     log_likelihood_sum = 0
#     for i in range(len(delta_T)):
#         Theta = Theta_values[i]
#         signal_i, signal_j = signal_values[i]

#         V_delta_S_i = V_Delta_T(a, b, c, Theta, signal_i, signal_j)
            
#         term1 = np.log(2 * np.pi * V_delta_S_i)
#         term2 = (delta_S[i]**2) / V_delta_S_i
#         log_likelihood_sum += (term1 + term2)
#         #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
#         #print("term1", term1 , "term2", term2)
#     # print("a", a, "b", b, "d", d, "loglikeli", log_likelihood_sum)        
#     return log_likelihood_sum

# # Initiale Schätzwerte und Grenzen für a, b und d
# initial_params = [0.865, 0.593, 0.023]
# bounds = [(None, None), (None, None), (None, None)]

# minuit = Minuit(
#     lambda a, b, c: log_likelihood(a, b, c, delta_s_values, zenith_values, signal_values),
#     a=initial_params[0],
#     b=initial_params[1],
#     c=initial_params[2]
# )

# # Setze Grenzen für die Parameter
# #minuit.limits = bounds

# # Optional: Fixiere Parameter, wenn nötig (z. B. d bleibt konstant)
# #minuit.fixed["d"] = True

# # Starte die Minimierung
# minuit.migrad()  # Hauptoptimierung

# # Prüfe, ob die Minimierung konvergiert ist
# if minuit.fmin.is_valid:
#     print("Minimierung konvergiert erfolgreich!")
# else:
#     print("Minimierung ist nicht konvergiert.")
#     print(f"EDM (Expected Distance to Minimum): {minuit.fmin.edm}")
#     print(f"Anzahl der Funktionsaufrufe: {minuit.fmin.nfcn}")
#     raise RuntimeError("Minimierung fehlgeschlagen.")

# # Fehleranalyse mit der Hesse-Matrix
# minuit.hesse()  # Berechne Unsicherheiten basierend auf der Hesse-Matrix

# # Optional: Minos-Fehleranalyse für asymmetrische Fehler
# minuit.minos()

# # Extrahiere optimale Parameter und Unsicherheiten
# optimal_params = {name: {"value": val, "error": err} for name, val, err in zip(minuit.parameters, minuit.values, minuit.errors)}
# print("Optimale Parameter:")
# for param, info in optimal_params.items():
#     print(f"{param} = {info['value']} ± {info['error']}")

# # Weitere Details über die Minimierung
# print("\nMinimierungsdetails:")
# print(f"Minimaler Wert der Log-Likelihood: {minuit.fmin.fval}")
# print(f"EDM (Expected Distance to Minimum): {minuit.fmin.edm}")
# print(f"Anzahl der Funktionsaufrufe: {minuit.fmin.nfcn}")

# # Korrelationsmatrix
# print("\nKorrelationsmatrix der Parameter:")
# print(minuit.covariance)

# print("\nZusammenfassung des Fits:")
# print(minuit.fmin)
# print(log_likelihood(0.865, 0.593, 0.023, delta_s_values, zenith_values, signal_values))

def V_t0(a, b, Theta, signal):
    return (a + b * (1 / (np.cos(Theta)))) ** 2 * signal

def V_Delta_T(a, b, Theta, signal_i, signal_j):
    return V_t0(a, b, Theta, signal_i) + V_t0(a, b, Theta, signal_j)

def log_likelihood(a, b, delta_S, Theta_values, signal_values):
    log_likelihood_sum = 0
    for i in range(len(delta_T)):
        Theta = Theta_values[i]
        signal_i, signal_j = signal_values[i]

        V_delta_S_i = V_Delta_T(a, b, Theta, signal_i, signal_j)
            
        term1 = np.log(2 * np.pi * V_delta_S_i)
        term2 = (delta_S[i]**2) / V_delta_S_i
        log_likelihood_sum += (term1 + term2)
        #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
        #print("term1", term1 , "term2", term2)
    # print("a", a, "b", b, "d", d, "loglikeli", log_likelihood_sum)        
    return log_likelihood_sum

# Initiale Schätzwerte und Grenzen für a, b und d
initial_params = [0.865, 0.593]
bounds = [(None, None), (None, None)]

minuit = Minuit(
    lambda a, b: log_likelihood(a, b, delta_s_values, zenith_values, signal_values),
    a=initial_params[0],
    b=initial_params[1]
)

# Setze Grenzen für die Parameter
#minuit.limits = bounds

# Optional: Fixiere Parameter, wenn nötig (z. B. d bleibt konstant)
#minuit.fixed["d"] = True

# Starte die Minimierung
minuit.migrad()  # Hauptoptimierung

# Prüfe, ob die Minimierung konvergiert ist
if minuit.fmin.is_valid:
    print("Minimierung konvergiert erfolgreich!")
else:
    print("Minimierung ist nicht konvergiert.")
    print(f"EDM (Expected Distance to Minimum): {minuit.fmin.edm}")
    print(f"Anzahl der Funktionsaufrufe: {minuit.fmin.nfcn}")
    raise RuntimeError("Minimierung fehlgeschlagen.")

# Fehleranalyse mit der Hesse-Matrix
minuit.hesse()  # Berechne Unsicherheiten basierend auf der Hesse-Matrix

# Optional: Minos-Fehleranalyse für asymmetrische Fehler
minuit.minos()

# Extrahiere optimale Parameter und Unsicherheiten
optimal_params = {name: {"value": val, "error": err} for name, val, err in zip(minuit.parameters, minuit.values, minuit.errors)}
print("Optimale Parameter:")
for param, info in optimal_params.items():
    print(f"{param} = {info['value']} ± {info['error']}")

# Weitere Details über die Minimierung
print("\nMinimierungsdetails:")
print(f"Minimaler Wert der Log-Likelihood: {minuit.fmin.fval}")
print(f"EDM (Expected Distance to Minimum): {minuit.fmin.edm}")
print(f"Anzahl der Funktionsaufrufe: {minuit.fmin.nfcn}")

# Korrelationsmatrix
print("\nKorrelationsmatrix der Parameter:")
print(minuit.covariance)

print("\nZusammenfassung des Fits:")
print(minuit.fmin)

# import matplotlib.colors as mcolors
# def V_t0(a, b, Theta, signal):
#     return (a + b * (1 / (np.cos(Theta)))) ** 2 * signal

# def V_Delta_T(a, b, Theta, signal_i, signal_j):
#     return V_t0(a, b, Theta, signal_i) + V_t0(a, b, Theta, signal_j)

# def log_likelihood(a, b, delta_S, Theta_values, signal_values):
#     log_likelihood_sum = 0
#     for i in range(len(delta_S)):
#         Theta = Theta_values[i]
#         signal_i, signal_j = signal_values[i]

#         V_delta_S_i = V_Delta_T(a, b, Theta, signal_i, signal_j)
            
#         term1 = np.log(2 * np.pi * V_delta_S_i)
#         term2 = (delta_S[i]**2) / V_delta_S_i
#         log_likelihood_sum += (term1 + term2)
        
#     return log_likelihood_sum

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# mp.rc("text", usetex=True)
# mp.rc("font", family="serif")
# pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
# # pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
# mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))

# a_values = np.linspace(0.01, 1, 50)
# b_values = np.linspace(0.01, 1, 50)
# likelihood_grid = np.zeros((len(b_values), len(a_values)))  # Shape must match (b, a)

# # Compute log-likelihood for each (a, b) pair
# for i, a in enumerate(a_values):
#     for j, b in enumerate(b_values):
#         likelihood_grid[j, i] = log_likelihood(a, b, delta_s_values, zenith_values, signal_values)  # Swap indices
#         print(f"a={a}, b={b}, likelihood={likelihood_grid[j, i]}")

# # Set color scale with logarithmic normalization
# min_likelihood = np.min(likelihood_grid)
# max_likelihood = min_likelihood + 1000
# norm = mcolors.LogNorm(vmin=min_likelihood, vmax=max_likelihood)

# # Plot heatmap with correct axis labeling
# plt.figure(figsize=(8, 6))
# c = plt.imshow(likelihood_grid, cmap="inferno_r", norm=norm, origin="lower",
#                extent=[a_values[0], a_values[-1], b_values[0], b_values[-1]])

# # Add colorbar and flip it
# cbar = plt.colorbar(c)
# cbar.set_label(r"$\ell$")

# # Labels and title
# plt.xlabel(r"Parameter $a$")
# plt.ylabel(r"Parameter $b$")
# plt.savefig("likelihoodscanfase22012.pdf", format="pdf", bbox_inches="tight")
# plt.show()