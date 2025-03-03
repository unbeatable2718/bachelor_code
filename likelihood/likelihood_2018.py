#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from iminuit import Minuit
from my_functions import read_in, delta_t, mean_t50

events = read_in("events.txt")
delta_t_values, n_values, t50_values, tVar_values, distance_values, theta_values, signal_values, group_values, counter = delta_t(events)
#print(delta_t_values, n_values, t50_values)

print(len(delta_t_values))
def V_t0(a, b, c, d, e, f, distance, signal):
    x = a * np.log(distance) + b * np.log(signal)
    return np.exp(c / (1 + np.exp(- (x - d) / e)) + f)

# Funktion für V[Delta_T]
def V_Delta_T(a, b, c, d, e, f, distance_i, distance_j, signal_i, signal_j):
    return V_t0(a, b, c, d, e, f, distance_i, signal_i) + V_t0(a, b, c, d, e, f, distance_j, signal_j)

def log_likelihood(a, b, c, d, e, f, delta_T, distance_values, signal_values):
    log_likelihood_sum = 0
    for i in range(len(delta_T)):
        distance_i, distance_j = distance_values[i]
        signal_i, signal_j = signal_values[i]

        # Berechne V[ΔT_i] als Summe von Varianzen
        V_delta_T_i = V_Delta_T(a, b, c, d, e, f, distance_i, distance_j, signal_i, signal_j)
        if V_delta_T_i <= 0:
            return np.inf
            
        else:
            # Prüfe auf numerische Stabilität
            term1 = np.log(2 * np.pi * V_delta_T_i)
            term2 = (delta_T[i]**2) / V_delta_T_i
            if term2 <= 0:
                return np.inf
            log_likelihood_sum += (term1 + term2)
            #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
            #print("term1", term1 , "term2", term2)
    print("a", a, "b", b, "c", c, "d", d, "e", e,"f", f, "loglikeli", log_likelihood_sum)        
    return log_likelihood_sum

# Initiale Schätzwerte und Grenzen für a, b und d
initial_params = [-0.88, 0.31, -2.27, -2.29, 1.94, 4.31]
#bounds = [(1, 500), (None, None), (-50, 50)]

minuit = Minuit(
    lambda a, b, c, d, e, f: log_likelihood(a, b, c, d, e, f, delta_t_values, distance_values, signal_values),
    a=initial_params[0],
    b=initial_params[1],
    c=initial_params[2],
    d=initial_params[3],
    e=initial_params[4],
    f=initial_params[5]
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

# Optionale Qualitätsbewertung
print("\nZusammenfassung des Fits:")
print(minuit.fmin)

# Ergebnisse speichern (optional)
import json
results = {
    "parameters": optimal_params,
    "fval": minuit.fmin.fval,
    "edm": minuit.fmin.edm,
    "is_valid": minuit.fmin.is_valid,
    "correlation_matrix": minuit.covariance.to_dict(),
}
with open("fit_results.json", "w") as f:
    json.dump(results, f, indent=4)








# from scipy.optimize import minimize
# delta_T = delta_t_values  # Array mit den Werten für ΔT_i
# n_values = n_values  # Array mit den Werten für n bei jedem t_i
# T_50_values = t50_values  # Array mit den Werten für T_50 bei jedem t_i
# VT = VT

# def V_t0(a, b, d, T_50, n):
#     #print("a   ", a,"b   ", b, "d    ", d, "T50", T_50, "n    ", n)
#     return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

# # Funktion für V[Delta_T]
# def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j):
#     return V_t0(a, b, d, T_50_i, n_i) + V_t0(a, b, d, T_50_j, n_j)

# # Definiere die Log-Likelihood-Funktion
# def log_likelihood(params, delta_T, T_50_values, n_values):
#     a, b, d = params
#     log_likelihood_sum = 0
#     #print(f'                                                                 {len(delta_T)}')
#     for i in range(len(delta_T)):
#         T_50_i, T_50_j = T_50_values[i]
#         n_i, n_j = n_values[i]
        
#         # Berechne V[ΔT_i] als Summe von Varianzen
#         V_delta_T_i = V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j)
#         if V_delta_T_i < 0:
#             log_likelihood_sum += np.inf
#         else:#print(f"{'meins:':10}", V_t0(a, b, d, T_50_i, n_i), "Tool    ", VT[i][0],"\n meins2   ", V_t0(a, b, d, T_50_j, n_j), "Tool2   ", VT[i][1], "\n\n")
#             # Prüfe auf numerische Stabilität
#             #print(f'V_delta_T_i  {V_delta_T_i}')
#             #print(f'Delta_T  {delta_T[i]}')
#             # Log-Likelihood für das i-te Paar hinzufügen
#             term1 = np.log(2 * np.pi * V_delta_T_i)
#             term2 = (delta_T[i]**2) / V_delta_T_i
#             log_likelihood_sum += term1 + term2
#     #print(log_likelihood_sum)
#     return log_likelihood_sum

# # Initiale Schätzwerte für a, b und d
# initial_params = [100, 3.0, 11]
# bounds = [(1,None), (1,None), (None, None)]

# # Maximierung der Likelihood (Minimierung des negativen Log-Likelihoods)
# result = minimize(lambda params: -log_likelihood(params, delta_T, T_50_values, n_values), initial_params, bounds=bounds)

# # Extrahiere die optimalen Parameter
# optimal_params = result.x

# print(f"Optimale Parameter: {optimal_params}")





# def plot_time_difference(data):
#     avg_distances = []
#     time_difference = []
#     for auger_id, details in data.items():
#         for group_id, stations in details['group_id'].items():
#             distances = []
#             times = []
#             timesN = []
#             if group_id == str(2):
#                 for station in stations:
#                     time = station['timeSecond']
#                     times.append(time)
#                     timeN = station['timeNSecond']
#                     timesN.append(timeN)
#                     distances.append(station['distance'])
                
#                 if times[0] == times[1]:
#                     avg_distance = np.mean(distances)
#                     time_differences = np.abs(timesN[1]-timesN[0])
#                     avg_distances.append(avg_distance)
#                     time_difference.append(time_differences)
#     plt.figure(figsize=(10, 6))
#     plt.scatter(avg_distances, time_difference, alpha=0.7, color='blue')
#     plt.xlabel("Gemittelte Entfernung (m)")
#     plt.ylabel("Zeitliche Differenz (s)")
#     plt.title("Zeitliche Varianz der Stationen in Abhängigkeit der gemittelten Entfernung")
#     plt.grid(True)
#     plt.show()
#     print(avg_distances)
#     print(time_difference)    
# #plot_time_difference(structure)






