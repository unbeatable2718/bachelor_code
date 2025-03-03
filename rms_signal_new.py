#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from my_functions import read_in, delta_t, mean_t50, delta_t_naiv_ln
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
delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, s_values, group_id_values, counter, delta_s_values = delta_t_naiv_ln(events, 300) 

a,b,d=177, 1.5, 65
def rms_by_value_with_errors(delta_s_values, distance_values, signal_values, zenith_values, value_values, VT0, num_bins=25):

    def V_Delta_T(distance_i, distance_j, signal_i, signal_j, zenithi):
        return VT0(distance_i, signal_i, zenithi) + VT0(distance_j, signal_j, zenithi)
    
    def rms_delta_t(delta_s_values, distance_values, signal_values, zenith_values):
        if len(delta_s_values) != len(distance_values) or len(delta_s_values) != len(signal_values) or len(delta_s_values) != len(zenith_values):
            raise ValueError("Die Listen delta_T_values, n_values und t50_values müssen gleich lang sein.")
        normalized_values = []
        for i in range(len(delta_s_values)):
            delta_T_i = delta_s_values[i]
            distance_i, distance_j = distance_values[i]
            signal_i, signal_j = signal_values[i]
            zenithi = zenith_values[i]
            V_delta_T_i = V_Delta_T(distance_i, distance_j, signal_i, signal_j, zenithi)
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
        delta_T_bin = [delta_s_values[idx] for idx in indices]
        distance_bin = [distance_values[idx] for idx in indices]
        zenith_bin = [zenith_values[idx] for idx in indices]
        signal_bin = [signal_values[idx] for idx in indices]
        N = len(delta_T_bin)  

        if N > 0:
            rms = rms_delta_t(delta_T_bin, distance_bin, signal_bin, zenith_bin)
            rms_values.append(rms)
            errors.append(1 / np.sqrt(2 * N))  
        else:
            rms_values.append(0)  
            errors.append(0) 

    return bin_centers, rms_values, errors


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
    if complexity == 1 or complexity == 7 or complexity == 12 or complexity == 13:  
        # Definieren der Funktion V_t0_new, die evaluiert wird
        def V_t0_new(distance, signal, zenith):
            # Ersetze 'x0' durch 'T_50' und 'x1' durch 'n' und evaluiere den String
            return eval(str(formula).replace('x0', 'distance').replace('x1', 'signal').replace('x2', 'zenith').replace('x3', 'signal').replace('x4', 'zenith').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('sqrt', 'np.sqrt').replace('^', '**').replace('exp', 'np.exp').replace('log', 'np.log'))
# delta_s_values, distance_values, signal_values, zenith_values, value_values, VT0, num_bins=25
        bin_centersdist[complexity], rms_valuesdist[complexity], errorsdist[complexity] = rms_by_value_with_errors(
            delta_s_values, distance_values, s_values, zenith_values, distance_values, V_t0_new
        )
        bin_centersn[complexity], rms_valuesn[complexity], errorsn[complexity] = rms_by_value_with_errors(
            delta_s_values, distance_values, s_values, zenith_values, s_values, V_t0_new
        )
        bin_centerst[complexity], rms_valuest[complexity], errorst[complexity] = rms_by_value_with_errors(
            delta_s_values, distance_values, s_values, zenith_values, t50_values, V_t0_new
        )
        sin2zen = (np.sin(zenith_values))**2
        bin_centerszen[complexity], rms_valueszen[complexity], errorszen[complexity] = rms_by_value_with_errors(
            delta_s_values, distance_values, s_values, zenith_values, sin2zen, V_t0_new
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
    bin_centersdist[7], rms_valuesdist[7], yerr=errorsdist[7], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="7"
)    
ax1.errorbar(
    bin_centersdist[12], rms_valuesdist[12], yerr=errorsdist[12], fmt="o", alpha=0.7, linestyle="", color="#009988", label="12"
)
ax1.errorbar(
    bin_centersdist[13], rms_valuesdist[13], yerr=errorsdist[13], fmt="o", alpha=0.7, linestyle="", color="r", label="13"
)
axs[0, 0].set_ylim(0, None)
axs[0, 0].set_xlabel(r"Distance from the core in m")
axs[0, 0].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
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
    bin_centersn[7], rms_valuesn[7], yerr=errorsn[7], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="7"
)
axs[0, 1].errorbar(
    bin_centersn[12], rms_valuesn[12], yerr=errorsn[12], fmt="o", alpha=0.7, linestyle="", color="#009988", label="12"
)
axs[0, 1].errorbar(
    bin_centersn[13], rms_valuesn[13], yerr=errorsn[13], fmt="o", alpha=0.7, linestyle="", color="r", label="13"
)
axs[0, 1].set_ylim(0, None)
axs[0, 1].set_xlabel(r"Signal in VEM (ln scale)")
axs[0, 1].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
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
    bin_centerst[7], rms_valuest[7], yerr=errorst[7], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="7"
)
axs[1, 0].errorbar(
    bin_centerst[12], rms_valuest[12], yerr=errorst[12], fmt="o", alpha=0.7, linestyle="", color="#009988", label="12"
)
axs[1, 0].errorbar(
    bin_centerst[13], rms_valuest[13], yerr=errorst[13], fmt="o", alpha=0.7, linestyle="", color="r", label="13"
)
axs[1, 0].set_ylim(0, None)
axs[1, 0].set_xlabel(r"$T_{50}$ in ns")
axs[1, 0].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
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
    bin_centerszen[7], rms_valueszen[7], yerr=errorszen[7], fmt="o", alpha=0.7, linestyle="", color="#CCAA00", label="7"
)
axs[1, 1].errorbar(
    bin_centerszen[12], rms_valueszen[12], yerr=errorszen[12], fmt="o", alpha=0.7, linestyle="", color="#009988", label="12"
)
axs[1, 1].errorbar(
    bin_centerszen[13], rms_valueszen[13], yerr=errorszen[13], fmt="o", alpha=0.7, linestyle="", color="r", label="13"
)
axs[1, 1].set_ylim(0, 2)
axs[1, 1].set_xlabel(r"sin$^2(\theta)$")
axs[1, 1].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
axs[1, 1].set_title(r"zenith angle")
axs[1, 1].grid()
axs[1, 1].legend()
plt.tight_layout()  
plt.savefig("AApysrPhaseIbsignal.pdf", format="pdf", bbox_inches="tight")
plt.show()

