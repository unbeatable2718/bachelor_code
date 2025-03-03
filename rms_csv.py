#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from my_functions import read_in, delta_t, mean_t50, delta_t_naiv
import sys
import pandas as pd
from pysr import PySRRegressor
# plt.rcParams.update({'font.size': 14})

mp.rc("text", usetex=True)
mp.rc("font", family="serif")
pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
# pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))

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

events = read_in("data_data/events6t5.txt")
delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, s_values, group_id_values, counter, delta_s_values = delta_t_naiv(events, 300) 

def rms_by_value_with_errors(delta_T_values, n_values, t50_values, value_values, VT0, distance, num_bins=25):

    def V_Delta_T(T_50_i, T_50_j, n_i, n_j, distance_i, distance_j):
        return VT0(T_50_i, n_i, distance_i) + VT0(T_50_j, n_j, distance_j)
    
    def rms_delta_t(delta_T_values, n_values, t50_values, distance):
        if len(delta_T_values) != len(n_values) or len(delta_T_values) != len(t50_values) or len(delta_T_values) != len(distance):
            raise ValueError("Die Listen delta_T_values, n_values und t50_values müssen gleich lang sein.")
        normalized_values = []
        for i in range(len(delta_T_values)):
            delta_T_i = delta_T_values[i]
            T_50_i, T_50_j = t50_values[i]
            n_i, n_j = n_values[i]
            distance_i, distance_j = distance[i]
            V_delta_T_i = V_Delta_T(T_50_i, T_50_j, n_i, n_j, distance_i, distance_j)
            if V_delta_T_i > 0: 
                normalized_value = delta_T_i / np.sqrt(V_delta_T_i)
                normalized_values.append(normalized_value)
        # rms = np.sqrt(np.mean(np.square(normalized_values)))
        rms = np.std(normalized_values, ddof=1)
        return rms
    
    mean_values = np.array([np.mean(dist_pair) for dist_pair in value_values])
    bins = np.linspace(np.min(mean_values), np.max(mean_values), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
    rms_values = []
    errors = []

    for i in range(num_bins):
        indices = np.where((mean_values >= bins[i]) & (mean_values < bins[i + 1]))[0]
        delta_T_bin = [delta_T_values[idx] for idx in indices]
        t50_bin = [t50_values[idx] for idx in indices]
        n_bin = [n_values[idx] for idx in indices]
        distance_bin = [distance[idx] for idx in indices]
        N = len(delta_T_bin)  

        if N > 0:
            rms = rms_delta_t(delta_T_bin, n_bin, t50_bin, distance_bin)
            rms_values.append(rms)
            errors.append(1 / np.sqrt(2 * N))  
        else:
            rms_values.append(0)  
            errors.append(0) 

    return bin_centers, rms_values, errors

# def V_t0(a, b, d, T_50, n, distance):
#     return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

# def V_t0(a, b, d, T_50, n, distance, signal, zenith):
#     return 177 + 1.562 * ((T_50 + 59) / (n + 1))**2 * (n / (n + 2))

# # Berechne den RMS-Wert mit der evaluierten Formel
# fig, axs = plt.subplots(2, 2, figsize=(15, 10)) 
# bin_centersdist, rms_valuesdist, errorsdist = rms_by_value_with_errors(
#     delta_t_values, n_values, t50_values, distance_values, a, b, d, V_t0, distance_values, s_values, zenith_values
# )
# bin_centersn, rms_valuesn, errorsn = rms_by_value_with_errors(
#     delta_t_values, n_values, t50_values, n_values, a, b, d, V_t0, distance_values, s_values, zenith_values
# )
# bin_centerst, rms_valuest, errorst = rms_by_value_with_errors(
#     delta_t_values, n_values, t50_values, t50_values, a, b, d, V_t0, distance_values, s_values, zenith_values
# )

# axs[0, 0].errorbar(
#     bin_centersdist, rms_valuesdist, yerr=errorsdist, fmt="o", alpha=0.7, linestyle="", color="b", label="Mine"
# )
# axs[0, 0].set_ylim(0, 2)
# axs[0, 0].set_xlabel(r"Distance from the core in m")
# axs[0, 0].set_ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# axs[0, 0].set_title(r"bp")
# axs[0, 0].grid()
# axs[0, 0].legend()
# axs[1, 0].errorbar(
#     bin_centersn, rms_valuesn, yerr=errorsn, fmt="o", alpha=0.7, linestyle="", color="b", label="Mine"
# )
# axs[1, 0].set_ylim(0, 2)
# axs[1, 0].set_xlabel(r"n")
# axs[1, 0].set_ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# axs[1, 0].set_title(r"n")
# axs[1, 0].grid()
# axs[1, 0].legend()
# axs[0, 1].errorbar(
#     bin_centerst, rms_valuest, yerr=errorst, fmt="o", alpha=0.7, linestyle="", color="b", label="Mine"
# )
# axs[0, 1].set_ylim(0, 2)
# axs[0, 1].set_xlabel(r"$t_{50}$")
# axs[0, 1].set_ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# axs[0, 1].set_title(r"$t_{50}$")
# axs[0, 1].grid()
# axs[0, 1].legend()
# axs[1, 1].axis("off") 
# plt.tight_layout()  
# plt.show()


# # Definiere Formeln symbolisch

# def vt1(a, b, d, T_50, n, distance):
#     return ((177 + (T_50 + 65)**2 * (n / (n + 2)) * 1.5 / ((n + 1)**2)) * (0.05 *((2* (distance/200))**2))) 
# def vt2(a, b, d, T_50, n, distance):
#     return ((177 + (T_50 + 65)**2 * (n / (n + 2)) * 1.5 / ((n + 1)**2)) * (0.05 *((2* (distance/100))**2))) 
# def vt3(a, b, d, T_50, n, distance):
#     return ((177 + (T_50 + 65)**2 * (n / (n + 2)) * 1.5 / ((n + 1)**2)) * (0.05 *((2* (distance/100))**2))) 
# def vt4(a, b, d, T_50, n, distance):
#     return ((177 + (T_50 + 65)**2 * (n / (n + 2)) * 1.5 / ((n + 1)**2)) * (0.5 *((1 + (distance/500))**2))) / 2
# def vt5(a, b, d, T_50, n, distance):
#     return ((177 + (T_50 + 65)**2 * (n / (n + 2)) * 1.5 / ((n + 1)**2)) * (0.5 *((1 + (distance/600))**2))) / 2
# def vt6(a, b, d, T_50, n, distance):
#     return ((177 + (T_50 + 65)**2 * (n / (n + 2)) * 1.5 / ((n + 1)**2)) * (0.5 *((1 + (distance/400))**2))) / 2
# def vt7(a, b, d, T_50, n, distance):
#     return ((177 + (T_50 + 65)**2 * (n / (n + 2)) * 1.5 / ((n + 1)**2)) * (0.5 *((1 + (distance/400))**2))) / 2

# eq = np.array([vt3])
# # Durchlaufen der Formeln und Ausgabe
# for formula in eq:

#     # Berechne den RMS-Wert mit der evaluierten Formel
#     bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, distance_values, a, b, d, formula, distance_values)
#     plt.figure(figsize=(15, 10))
#     plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'Mine')
#     # plt.ylim(0, 2)
#     plt.xlabel(r"Distance from the core in m")
#     plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
#     plt.title(formula)
#     plt.grid()
#     plt.legend()
#     plt.show()

bin_centersdist = {}
rms_valuesdist = {}
errorsdist = {}
bin_centersn = {}
rms_valuesn = {}
errorsn = {}
bin_centerst = {}
rms_valuest = {}
errorst = {}
bin_centerszen = {}
rms_valueszen = {}
errorszen = {}

# Durchlaufen der Zeilen und Ausgabe der Komplexität und Formel
for index, row in df.iterrows():

    complexity = row["Complexity"]
    formula = row["Equation"]
    if complexity == 1 or complexity == 8 or complexity == 14 or complexity == 16:  
        # Definieren der Funktion V_t0_new, die evaluiert wird
        def V_t0_new(T_50, n, distance):
            # Ersetze 'x0' durch 'T_50' und 'x1' durch 'n' und evaluiere den String
            return eval(str(formula).replace('x0', 'T_50').replace('x1', 'n').replace('x2', 'distance').replace('x3', 'signal').replace('x4', 'zenith').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('sqrt', 'np.sqrt').replace('^', '**').replace('exp', 'np.exp').replace('log', 'np.log'))
# delta_T_values, n_values, t50_values, value_values, VT0, distance, num_bins=25
        bin_centersdist[complexity], rms_valuesdist[complexity], errorsdist[complexity] = rms_by_value_with_errors(
            delta_t_values, n_values, t50_values, distance_values, V_t0_new, distance_values
        )
        s_log = np.log10(s_values)
        bin_centersn[complexity], rms_valuesn[complexity], errorsn[complexity] = rms_by_value_with_errors(
            delta_t_values, n_values, t50_values, s_log, V_t0_new, distance_values
        )
        bin_centerst[complexity], rms_valuest[complexity], errorst[complexity] = rms_by_value_with_errors(
            delta_t_values, n_values, t50_values, t50_values, V_t0_new, distance_values
        )
        sin2zen = (np.sin(zenith_values))**2
        bin_centerszen[complexity], rms_valueszen[complexity], errorszen[complexity] = rms_by_value_with_errors(
            delta_t_values, n_values, t50_values, sin2zen, V_t0_new, distance_values
        )
fig, axs = plt.subplots(2, 2, figsize=(9, 8)) 

def calculate_n_from_errors(errors):
    return 1 / (2 * np.array(errors) ** 2)

N = calculate_n_from_errors(errorsdist[1])
# Plot 1: Distance
ax1 = axs[0, 0]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centersdist[1], N, width=(np.max(bin_centersdist[1]) - np.min(bin_centersdist[1]))/24, alpha=0.2, color='gray')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax1.errorbar(
    bin_centersdist[1], rms_valuesdist[1], yerr=errorsdist[1], fmt="o", alpha=0.7, linestyle="", color="b", label="1"
)
ax1.errorbar(
    bin_centersdist[8], rms_valuesdist[8], yerr=errorsdist[8], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="8"
)    
ax1.errorbar(
    bin_centersdist[14], rms_valuesdist[14], yerr=errorsdist[14], fmt="o", alpha=0.7, linestyle="", color="#009988", label="14"
)
ax1.errorbar(
    bin_centersdist[16], rms_valuesdist[16], yerr=errorsdist[16], fmt="o", alpha=0.7, linestyle="", color="r", label="16"
)
axs[0, 0].set_ylim(0, None)
axs[0, 0].set_xlabel(r"Distance from the core in m")
axs[0, 0].set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
axs[0, 0].set_title(r"Distance")
axs[0, 0].grid()
axs[0, 0].legend()

N = calculate_n_from_errors(errorsn[1])

ax1 = axs[0, 1]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centersn[1], N, width=(np.max(bin_centersn[1]) - np.min(bin_centersn[1]))/24, alpha=0.2, color='gray')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[0, 1].errorbar(
    bin_centersn[1], rms_valuesn[1], yerr=errorsn[1], fmt="o", alpha=0.7, linestyle="", color="b", label="1"
)
axs[0, 1].errorbar(
    bin_centersn[8], rms_valuesn[8], yerr=errorsn[8], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="8"
)
axs[0, 1].errorbar(
    bin_centersn[14], rms_valuesn[14], yerr=errorsn[14], fmt="o", alpha=0.7, linestyle="", color="#009988", label="14"
)
axs[0, 1].errorbar(
    bin_centersn[16], rms_valuesn[16], yerr=errorsn[16], fmt="o", alpha=0.7, linestyle="", color="r", label="16"
)
axs[0, 1].set_ylim(0, None)
axs[0, 1].set_xlabel(r"Signal in VEM (log$_{10}$ scale)")
axs[0, 1].set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
axs[0, 1].set_title(r"Signal")
axs[0, 1].grid()
axs[0, 1].legend()


N = calculate_n_from_errors(errorst[1])
# Plot 1: Distance
ax1 = axs[1, 0]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centerst[1], N, width=(np.max(bin_centerst[1]) - np.min(bin_centerst[1]))/24, alpha=0.2, color='gray')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[1, 0].errorbar(
    bin_centerst[1], rms_valuest[1], yerr=errorst[1], fmt="o", alpha=0.7, linestyle="", color="b", label="1"
)
axs[1, 0].errorbar(
    bin_centerst[8], rms_valuest[8], yerr=errorst[8], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="8"
)
axs[1, 0].errorbar(
    bin_centerst[14], rms_valuest[14], yerr=errorst[14], fmt="o", alpha=0.7, linestyle="", color="#009988", label="14"
)
axs[1, 0].errorbar(
    bin_centerst[16], rms_valuest[16], yerr=errorst[16], fmt="o", alpha=0.7, linestyle="", color="r", label="16"
)
axs[1, 0].set_ylim(0, None)
axs[1, 0].set_xlabel(r"$T_{50}$ in ns")
axs[1, 0].set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
axs[1, 0].set_title(r"$T_{50}$")
axs[1, 0].grid()
axs[1, 0].legend()




N = calculate_n_from_errors(errorszen[1])
# Plot 1: Distance
ax1 = axs[1, 1]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centerszen[1], N, width=(np.max(bin_centerszen[1]) - np.min(bin_centerszen[1]))/24, alpha=0.2, color='gray')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[1, 1].errorbar(
    bin_centerszen[1], rms_valueszen[1], yerr=errorszen[1], fmt="o", alpha=0.7, linestyle="", color="b", label="1"
)
axs[1, 1].errorbar(
    bin_centerszen[8], rms_valueszen[8], yerr=errorszen[8], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="8"
)
axs[1, 1].errorbar(
    bin_centerszen[14], rms_valueszen[14], yerr=errorszen[14], fmt="o", alpha=0.7, linestyle="", color="#009988", label="14"
)
axs[1, 1].errorbar(
    bin_centerszen[16], rms_valueszen[16], yerr=errorszen[16], fmt="o", alpha=0.7, linestyle="", color="r", label="16"
)
axs[1, 1].set_ylim(0, 2)
axs[1, 1].set_xlabel(r"sin$^2(\theta)$")
axs[1, 1].set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
axs[1, 1].set_title(r"zenith angle")
axs[1, 1].grid()
axs[1, 1].legend()
plt.tight_layout()  
plt.savefig("pysrfase1time.pdf", format="pdf", bbox_inches="tight")
plt.show()

