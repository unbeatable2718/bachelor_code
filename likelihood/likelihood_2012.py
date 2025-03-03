#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from my_functions import delta_t, read_in, mean_t50, delta_t_naiv
import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt

events = read_in("../data_data/events6t5.txt")
delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values = delta_t(events, 300)
delta_t_valuesN, n_valuesN, t50_valuesN, tVar_valuesN, distance_valuesN, zenith_valuesN, signal_valuesN, group_valuesN, counterN, delta_s_valuesN = delta_t_naiv(events, 300)

#print(delta_t_values, n_values, t50_values)

delta_T = delta_t_values  # Array mit den Werten für ΔT_i
n_values = n_values  # Array mit den Werten für n bei jedem t_i
T_50_values = t50_values  # Array mit den Werten für T_50 bei jedem t_i
def V_t0(a, b, T_50, n):
    return a * pow(2 * T_50 / (n - 1), 2) * n / (n + 2) + b

# # Funktion für V[Delta_T]
# def V_Delta_T(a, b, T_50_i, T_50_j, n_i, n_j):
#     return V_t0(a, b, T_50_i, n_i) + V_t0(a, b, T_50_j, n_j)

# def log_likelihood(a, b, delta_T, T_50_values, n_values):
#     log_likelihood_sum = 0
#     for i in range(len(delta_T)):
#         T_50_i, T_50_j = T_50_values[i]
#         n_i, n_j = n_values[i]

#         # Berechne V[ΔT_i] als Summe von Varianzen
#         V_delta_T_i = V_Delta_T(a, b, T_50_i, T_50_j, n_i, n_j)
#         if V_delta_T_i <= 0:
#             return np.inf
            
#         else:
#             # Prüfe auf numerische Stabilität
#             term1 = np.log(2 * np.pi * V_delta_T_i)
#             term2 = (delta_T[i]**2) / V_delta_T_i
#             if term2 <= 0:
#                 return np.inf
#             log_likelihood_sum += (term1 + term2)
#             #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
#             #print("term1", term1 , "term2", term2)
#     # print("a", a, "b", b, "loglikeli", log_likelihood_sum)        
#     return log_likelihood_sum

# # Initiale Schätzwerte und Grenzen für a, b und d
# initial_params = [1, 1]
# #bounds = [(1, 500), (None, None), (-50, 50)]

# minuit = Minuit(
#     lambda a, b: log_likelihood(a, b, delta_T, T_50_values, n_values),
#     a=initial_params[0],
#     b=initial_params[1]
# )

# # Setze Grenzen für die Parameter
# #minuit.limits = bounds

# # Optional: Fixiere Parameter, wenn nötig (z. B. d bleibt konstant)
# #minuit.fixed["d"] = True
# print("mit Anpassungen a und b")
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

# # Optionale Qualitätsbewertung
# print("\nZusammenfassung des Fits:")
# print(minuit.fmin)






# delta_T = delta_t_valuesN  # Array mit den Werten für ΔT_i
# n_values = n_valuesN  # Array mit den Werten für n bei jedem t_i
# T_50_values = t50_valuesN  # Array mit den Werten für T_50 bei jedem t_i
# # def V_t0(a, b, T_50, n):
# #     return a * pow(2 * T_50 / (n - 1), 2) * n / (n + 2) + b

# # Funktion für V[Delta_T]
# def V_Delta_T(a, b, T_50_i, T_50_j, n_i, n_j):
#     return V_t0(a, b, T_50_i, n_i) + V_t0(a, b, T_50_j, n_j)

# def log_likelihood(a, b, delta_T, T_50_values, n_values):
#     log_likelihood_sum = 0
#     for i in range(len(delta_T)):
#         T_50_i, T_50_j = T_50_values[i]
#         n_i, n_j = n_values[i]

#         # Berechne V[ΔT_i] als Summe von Varianzen
#         V_delta_T_i = V_Delta_T(a, b, T_50_i, T_50_j, n_i, n_j)
#         if V_delta_T_i <= 0:
#             return np.inf
            
#         else:
#             # Prüfe auf numerische Stabilität
#             term1 = np.log(2 * np.pi * V_delta_T_i)
#             term2 = (delta_T[i]**2) / V_delta_T_i
#             if term2 <= 0:
#                 return np.inf
#             log_likelihood_sum += (term1 + term2)
#             #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
#             #print("term1", term1 , "term2", term2)
#     # print("a", a, "b", b, "loglikeli", log_likelihood_sum)        
#     return log_likelihood_sum

# # Initiale Schätzwerte und Grenzen für a, b und d
# initial_params = [1, 1]
# #bounds = [(1, 500), (None, None), (-50, 50)]

# minuit = Minuit(
#     lambda a, b: log_likelihood(a, b, delta_T, T_50_values, n_values),
#     a=initial_params[0],
#     b=initial_params[1]
# )

# # Setze Grenzen für die Parameter
# #minuit.limits = bounds
# print("ohne Anpassungen nur a und b")
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

# # Optionale Qualitätsbewertung
# print("\nZusammenfassung des Fits:")
# print(minuit.fmin)


def V_Delta_T2(T_50_i, T_50_j, n_i, n_j, theta):
    a = 0.64871 + np.cos(theta)*(0.22365 + np.cos(theta)*(-0.49971))
    b = (11.88)**2 + np.cos(theta)*(-(14.45)**2 + np.cos(theta) * (20.3)**2)
    return V_t0(a, b, T_50_i, n_i) + V_t0(a, b, T_50_j, n_j)

def log_likelihood2(delta_T, T_50_values, n_values, theta_values):
    log_likelihood_sum = 0
    for i in range(len(delta_T)):
        T_50_i, T_50_j = T_50_values[i]
        n_i, n_j = n_values[i]
        theta = theta_values[i]

        # Berechne V[ΔT_i] als Summe von Varianzen
        V_delta_T_i = V_Delta_T2(T_50_i, T_50_j, n_i, n_j, theta)
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
    # print("a", a, "b", b, "loglikeli", log_likelihood_sum)        
    return log_likelihood_sum


print("zenithbla mit anpassungen", log_likelihood2(delta_t_values, t50_values, n_values, zenith_values))
print("zenithbla ohne anpassungen", log_likelihood2(delta_t_valuesN, t50_valuesN, n_valuesN, zenith_valuesN))


def V_Delta_T3(T_50_i, T_50_j, n_i, n_j, theta):
    a = 0.64871 + np.cos(theta)*(0.22365 + np.cos(theta)*(-0.49971))
    b = (11.88 * 10 ** (-9))**2 + np.cos(theta)*(-(14.45 * 10 ** (-9))**2 + np.cos(theta)*(20.3 * 10 ** (-9))**2)
    return V_t0(a, b, T_50_i, n_i) + V_t0(a, b, T_50_j, n_j)

def log_likelihood3(delta_T, T_50_values, n_values, theta_values):
    log_likelihood_sum = 0
    for i in range(len(delta_T)):
        T_50_i, T_50_j = T_50_values[i]
        n_i, n_j = n_values[i]
        theta = theta_values[i]

        # Berechne V[ΔT_i] als Summe von Varianzen
        V_delta_T_i = V_Delta_T3(T_50_i, T_50_j, n_i, n_j, theta)
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
    # print("a", a, "b", b, "loglikeli", log_likelihood_sum)        
    return log_likelihood_sum


print("zenithbla mit anpassungen nano", log_likelihood3(delta_t_values, t50_values, n_values, zenith_values))
print("zenithbla ohne anpassungen nano", log_likelihood3(delta_t_valuesN, t50_valuesN, n_valuesN, zenith_valuesN))