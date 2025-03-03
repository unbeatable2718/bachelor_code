#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from pysr import PySRRegressor
from my_functions import read_in, delta_t, mean_t50, delta_t_naiv

# Überprüfen, ob der Dateipfad als Argument übergeben wurde
if len(sys.argv) != 2:
    print("Usage: python script.py <path_to_csv_file>")
    sys.exit(1)

# Der erste Argument ist der Dateipfad
csv_file = sys.argv[1]

# Lesen der CSV-Datei
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: Die Datei {csv_file} wurde nicht gefunden.")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: Die Datei {csv_file} ist leer.")
    sys.exit(1)

def V_t0(a, b, d, T_50, n, distance_i, signal, zenith):
    return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

events = read_in("data_data/events6t5.txt")
delta_T_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values = delta_t_naiv(events, 300)            
a, b, d = 688, 0.41, 376

def compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0_func, distance, signal_values, zenith_values):
    def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j, distance_i, distance_j, signal_i, signal_j, zenith):
        return V_t0_func(a, b, d, T_50_i, n_i, distance_i, signal_i, zenith) + V_t0_func(a, b, d, T_50_j, n_j, distance_j, signal_j, zenith)
    
    normalized_values = []
    for i in range(len(delta_T_values)):
        delta_T_i = delta_T_values[i]
        T_50_i, T_50_j = t50_values[i]
        n_i, n_j = n_values[i]
        distance_i, distance_j = distance[i]
        signal_i, signal_j = signal_values[i]
        zenith = zenith_values[i]
        V_delta_T_i = V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j, distance_i, distance_j, signal_i, signal_j, zenith)
        if V_delta_T_i > 0:
            normalized_value = delta_T_i / np.sqrt(V_delta_T_i)
            normalized_values.append(normalized_value)
    rms = np.std(normalized_values, ddof=1)
    return rms

def compute_likelihood(delta_T_values, n_values, t50_values, a, b, d, V_t0_func, distance_values, signal_values, zenith_values):
    def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j, distance_i, distance_j, signal_i, signal_j, zenith):
        return V_t0_func(a, b, d, T_50_i, n_i, distance_i, signal_i, zenith) + V_t0_func(a, b, d, T_50_j, n_j, distance_j, signal_j, zenith)
    
    log_likelihood_sum = 0
    for i in range(len(delta_T_values)):
        T_50_i, T_50_j = t50_values[i]
        n_i, n_j = n_values[i]
        distance_i, distance_j = distance_values[i]
        signal_i, signal_j = signal_values[i]
        zenith = zenith_values[i]

        # Berechne V[ΔT_i] als Summe von Varianzen
        V_delta_T_i = V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j, distance_i, distance_j, signal_i, signal_j, zenith)
        if np.isnan(V_delta_T_i):
            print(V_delta_T_i)
            
        else:
            #alles.append(V_delta_T_i)
            term1 = np.log(2 * np.pi * V_delta_T_i + 10**(-8))
            term2 = (delta_T_values[i]**2) / (V_delta_T_i + 10**(-8))
            if np.isnan(term1):
                print("t1:", term1)
            if np.isnan(term2):
                print("t2:", term2)    
            log_likelihood_sum += (term1 + term2)
            #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
            #print("term1", term1 , "term2", term2)
    # print("a", a, "b", b, "d", d, "loglikeli", log_likelihood_sum)        
    return log_likelihood_sum

rms_before = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance_values, signal_values, zenith_values)
likelihood_before = compute_likelihood(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance_values, signal_values, zenith_values)

print(f"Old model:   , RMS: {rms_before}, likelihood: {likelihood_before}")



# Durchlaufen der Zeilen und Ausgabe der Komplexität und Formel
for index, row in df.iterrows():
    #alles = np.array([])
    complexity = row["Complexity"]
    formula = row["Equation"]
    
    # Definieren der Funktion V_t0_new, die evaluiert wird
    def V_t0_new(a, b, d, T_50, n, distance, signal, zenith):
        # Ersetze 'x0' durch 'T_50' und 'x1' durch 'n' und evaluiere den String
        return eval(str(formula).replace('x0', 'T_50').replace('x1', 'n').replace('x2', 'distance').replace('x3', 'signal').replace('x4', 'zenith').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('sqrt', 'np.sqrt').replace('^', '**').replace('exp', 'np.exp').replace('log', 'np.log'))

    # Berechne den RMS-Wert mit der evaluierten Formel
    #target_rms = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0_new, distance_values, signal_values, zenith_values)
    likelihood = compute_likelihood(delta_T_values, n_values, t50_values, a, b, d, V_t0_new, distance_values, signal_values, zenith_values)
    print(f"Complexity: {complexity}, likelihood: {likelihood}")
    #print(alles[:100])

# import sys
# import pandas as pd
# import numpy as np
# from pysr import PySRRegressor
# from my_functions import read_in, delta_t, mean_t50, delta_t_naiv_ln

# # Überprüfen, ob der Dateipfad als Argument übergeben wurde
# if len(sys.argv) != 2:
#     print("Usage: python script.py <path_to_csv_file>")
#     sys.exit(1)

# # Der erste Argument ist der Dateipfad
# csv_file = sys.argv[1]

# # Lesen der CSV-Datei
# try:
#     df = pd.read_csv(csv_file)
# except FileNotFoundError:
#     print(f"Error: Die Datei {csv_file} wurde nicht gefunden.")
#     sys.exit(1)
# except pd.errors.EmptyDataError:
#     print(f"Error: Die Datei {csv_file} ist leer.")
#     sys.exit(1)

# def V_t0(a, b, d, T_50, n, distance_i, signal, zenith):
#     return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

# events = read_in("data_data/events6t5.txt")
# delta_T_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values = delta_t_naiv_ln(events, 300)            
# a, b, d = 688, 0.41, 376


# def compute_likelihood(delta_s_values, n_values, t50_values, a, b, d, V_t0_func, distance_values, signal_values, zenith_values):
#     def V_Delta_T(distance_i, distance_j, signal_i, signal_j, zenith):
#         return V_t0_func(distance_i, signal_i, zenith) + V_t0_func(distance_j, signal_j, zenith)
    
#     log_likelihood_sum = 0
#     for i in range(len(delta_s_values)):
#         distance_i, distance_j = distance_values[i]
#         signal_i, signal_j = signal_values[i]
#         zenith = zenith_values[i]

#         # Berechne V[ΔT_i] als Summe von Varianzen
#         V_delta_T_i = V_Delta_T(distance_i, distance_j, signal_i, signal_j, zenith)
#         if np.isnan(V_delta_T_i):
#             print(V_delta_T_i)
            
#         else:
#             #alles.append(V_delta_T_i)
#             term1 = np.log(2 * np.pi * V_delta_T_i + 10**(-8))
#             term2 = (delta_s_values[i]**2) / (V_delta_T_i + 10**(-8))
#             if np.isnan(term1):
#                 print("t1:", term1)
#             if np.isnan(term2):
#                 print("t2:", term2)    
#             log_likelihood_sum += (term1 + term2)
#             #log_likelihood_sum += - np.log(1 / (np.sqrt(2* np.pi * V_delta_T_i)) * np.exp(-((delta_T[i]**2) / (2 * V_delta_T_i))))
#             #print("term1", term1 , "term2", term2)
#     # print("a", a, "b", b, "d", d, "loglikeli", log_likelihood_sum)        
#     return log_likelihood_sum

# # rms_before = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance_values, signal_values, zenith_values)
# # likelihood_before = compute_likelihood(delta_T_values, n_values, t50_values, a, b, d, V_t0, distance_values, signal_values, zenith_values)

# # print(f"Old model:   , RMS: {rms_before}, likelihood: {likelihood_before}")



# # Durchlaufen der Zeilen und Ausgabe der Komplexität und Formel
# for index, row in df.iterrows():
#     #alles = np.array([])
#     complexity = row["Complexity"]
#     formula = row["Equation"]
    
#     # Definieren der Funktion V_t0_new, die evaluiert wird
#     def V_t0_new(distance, signal, zenith):
#         # Ersetze 'x0' durch 'T_50' und 'x1' durch 'n' und evaluiere den String
#         return eval(str(formula).replace('x0', 'distance').replace('x1', 'signal').replace('x2', 'zenith').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('sqrt', 'np.sqrt').replace('^', '**').replace('exp', 'np.exp').replace('log', 'np.log'))

#     # Berechne den RMS-Wert mit der evaluierten Formel
#     #target_rms = compute_rms(delta_T_values, n_values, t50_values, a, b, d, V_t0_new, distance_values, signal_values, zenith_values)
#     likelihood = compute_likelihood(delta_s_values, n_values, t50_values, a, b, d, V_t0_new, distance_values, signal_values, zenith_values)
#     print(f"Complexity: {complexity}, likelihood: {likelihood}")
#     #print(alles[:100])
