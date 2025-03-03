#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from my_functions import delta_t, read_in, mean_t50
import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt

events = read_in("../data_test4/events.txt")

# standard diviation of delta t with cut 450: 72.77367336352151 ns
likelihoods = []
# for i in np.array([1, 2, 3, 4, 5, (450 / 72.77367336352151)]):
for i in np.array([(450 / 72.77367336352151)]):    
    print("cut:  ", i)
    std_diviation_dt = 72.77367336352151
    delta_T, n_values, T_50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter = delta_t(events, (i * std_diviation_dt))            
    print("minimum n:  ", np.min(n_values))
    print("conter t50 was replaced", counter)
    print("delta T should be 0:  ", np.mean(delta_T))
    flattened = np.ravel(T_50_values)
    plt.hist(flattened, bins=250)
    plt.show()
    #print(delta_T)
    print("len values:  ", len(delta_T))
    print("standard deviation of delta_t", np.std(delta_T))

    def V_t0(a, b, d, e, f, g, T_50, n, distance):
        if n <= 0 or (T_50 + d) <= 0:
            return np.inf
        return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2)) * ( e + f * distance + g * distance**2)

    def V_Delta_T(a, b, d, e, f, g, T_50_i, T_50_j, n_i, n_j, distance):
        return V_t0(a, b, d, e, f, g, T_50_i, n_i, distance) + V_t0(a, b, d, e, f, g, T_50_j, n_j, distance)

    def log_likelihood(a, b, d, e, f, g, delta_T, T_50_values, n_values, distance_values):
        log_likelihood_sum = 0
        for i in range(len(delta_T)):
            T_50_i, T_50_j = T_50_values[i]
            n_i, n_j = n_values[i]
            distance = distance_values[i][0]

            # Berechne V[ΔT_i] als Summe von Varianzen
            V_delta_T_i = V_Delta_T(a, b, d, e, f, g, T_50_i, T_50_j, n_i, n_j, distance)
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
        likelihoods.append(log_likelihood_sum)
        print("a", a, "b", b, "d", d, "e", e, "f", f, "g", g, "loglikeli", log_likelihood_sum)        
        return log_likelihood_sum

    # Initiale Schätzwerte und Grenzen für a, b und d
    initial_params = [177, 1.562, 59, 1, 0, 0]
    bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]

    minuit = Minuit(
        lambda a, b, d, e, f, g: log_likelihood(a, b, d, e, f, g, delta_T, T_50_values, n_values, distance_values),
        a=initial_params[0],
        b=initial_params[1],
        d=initial_params[2],
        e=initial_params[3],
        f=initial_params[4],
        g=initial_params[5]
    )

    # Setze Grenzen für die Parameter
    #minuit.limits = bounds

    # Optional: Fixiere Parameter, wenn nötig (z. B. d bleibt konstant)
    minuit.fixed["a"] = True
    minuit.fixed["b"] = True
    minuit.fixed["d"] = True


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
print(min(likelihoods))






