#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from my_functions import delta_t_trigger, read_in, mean_t50
import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt
import matplotlib as mp

events = read_in("data_data/events6t5.txt")

# standard diviation of delta t with cut 450: 72.77367336352151 ns

# for i in np.array([1, 2, 3, 4, 5, (450 / 72.77367336352151)]):
for i in np.array([(450 / 72.77367336352151)]):    
    print("cut:  ", i)
    std_diviation_dt = 72.77367336352151
    delta_T, n_values, T_50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values, trigger = delta_t_trigger(events, 300, 'ToT')
    print(len(delta_T))
    def V_t0(a, b, d, T_50, n):
        if n <= 0 or (T_50 + d) <= 0:
            return np.inf
        return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

    def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j):
        return V_t0(a, b, d, T_50_i, n_i) + V_t0(a, b, d, T_50_j, n_j)

    def log_likelihood(a, b, d, delta_T, T_50_values, n_values):
        log_likelihood_sum = 0
        for i in range(len(delta_T)):
            T_50_i, T_50_j = T_50_values[i]
            n_i, n_j = n_values[i]

            # Berechne V[ΔT_i] als Summe von Varianzen
            V_delta_T_i = V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j)
            if V_delta_T_i <= 0:
                return np.inf
                
            else:
                term1 = np.log(2 * np.pi * V_delta_T_i)
                term2 = (delta_T[i]**2) / V_delta_T_i
                if term2 <= 0:
                    return np.inf
                log_likelihood_sum += (term1 + term2)
                #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
                #print("term1", term1 , "term2", term2)
        # print("a", a, "b", b, "d", d, "loglikeli", log_likelihood_sum)        
        return log_likelihood_sum

    # Initiale Schätzwerte und Grenzen für a, b und d
    initial_params = [177, 1.5, 64]
    bounds = [(1, 500), (None, None), (-50, 50)]

    minuit = Minuit(
        lambda a, b, d: log_likelihood(a, b, d, delta_T, T_50_values, n_values),
        a=initial_params[0],
        b=initial_params[1],
        d=initial_params[2]
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








# 2004-2014

# Optimale Parameter:
# a = 177.0060193702065 ± 2.0549507217649237
# b = 0.9002906156277584 ± 0.017107853643154675
# d = 148.3308607049875 ± 3.531628935177749

# Minimierungsdetails:
# Minimaler Wert der Log-Likelihood: 1246435.64707273
# EDM (Expected Distance to Minimum): 2.1370181609185454e-05
# Anzahl der Funktionsaufrufe: 263

# Korrelationsmatrix der Parameter:
# ┌───┬───────────────────────────────┐
# │   │         a         b         d │
# ├───┼───────────────────────────────┤
# │ a │      4.22  -2.70e-3        -0 │
# │ b │  -2.70e-3  0.000293 -57.67e-3 │
# │ d │        -0 -57.67e-3      12.5 │
# └───┴───────────────────────────────┘


# 2014-2024

# Optimale Parameter:
# a = 688.4825339762227 ± 8.404563617479734
# b = 0.40804731543775125 ± 0.0174403486433125
# d = 376.4883366224246 ± 13.24530930615754

# Minimierungsdetails:
# Minimaler Wert der Log-Likelihood: 705704.9528660909
# EDM (Expected Distance to Minimum): 0.00014777650438229712
# Anzahl der Funktionsaufrufe: 305

# Korrelationsmatrix der Parameter:
# ┌───┬──────────────────────────────────┐
# │   │          a          b          d │
# ├───┼──────────────────────────────────┤
# │ a │       70.6  -11.26e-3         -0 │
# │ b │  -11.26e-3   0.000304 -225.46e-3 │
# │ d │         -0 -225.46e-3        175 │
# └───┴──────────────────────────────────┘

# Zusammenfassung des Fits:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                                Migrad                                   │
# ├──────────────────────────────────┬──────────────────────────────────────┤
# │ FCN = 7.057e+05                  │              Nfcn = 305              │
# │ EDM = 0.000148 (Goal: 0.0002)    │           time = 125.5 sec           │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │          Valid Minimum           │   Below EDM threshold (goal x 10)    │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │      No parameters at limit      │           Below call limit           │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │             Hesse ok             │         Covariance accurate          │
# └──────────────────────────────────┴──────────────────────────────────────┘



# 2004-2.2007

# Optimale Parameter:
# a = 192.33364947974655 ± 5.9800481798595415
# b = 0.9662888683503655 ± 0.05016350416961033
# d = 122.04554112806753 ± 8.972612643314521

# Minimierungsdetails:
# Minimaler Wert der Log-Likelihood: 163714.10636718385
# EDM (Expected Distance to Minimum): 1.0446687887409383e-06
# Anzahl der Funktionsaufrufe: 270

# Korrelationsmatrix der Parameter:
# ┌───┬─────────────────────────┐
# │   │       a       b       d │
# ├───┼─────────────────────────┤
# │ a │    35.8 -0.0218      -0 │
# │ b │ -0.0218 0.00252 -0.4289 │
# │ d │      -0 -0.4289    80.5 │
# └───┴─────────────────────────┘

# Zusammenfassung des Fits:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                                Migrad                                   │
# ├──────────────────────────────────┬──────────────────────────────────────┤
# │ FCN = 1.637e+05                  │              Nfcn = 270              │
# │ EDM = 1.04e-06 (Goal: 0.0002)    │           time = 27.3 sec            │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │          Valid Minimum           │   Below EDM threshold (goal x 10)    │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │      No parameters at limit      │           Below call limit           │
# ├──────────────────────────────────┼──────────────────────────────────────┤
# │             Hesse ok             │         Covariance accurate          │
# └──────────────────────────────────┴──────────────────────────────────────┘