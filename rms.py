#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from my_functions import read_in, delta_t, mean_t50, delta_t_naiv

events = read_in("data_data/events6t5.txt")
delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, s_values, group_id_values, counter, delta_s_values = delta_t_naiv(events, 300) 


print(len(delta_t_values))
def V_t0(a, b, d, T_50, n):
    return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j):
    return V_t0(a, b, d, T_50_i, n_i) + V_t0(a, b, d, T_50_j, n_j)

def rms_delta_t(delta_T_values, n_values, t50_values, a, b, d):
    if len(delta_T_values) != len(n_values) or len(delta_T_values) != len(t50_values):
        raise ValueError("Die Listen delta_T_values, n_values und t50_values müssen gleich lang sein.")
    normalized_values = []
    for i in range(len(delta_T_values)):
        delta_T_i = delta_T_values[i]
        T_50_i, T_50_j = t50_values[i]
        n_i, n_j = n_values[i]
        V_delta_T_i = V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j)
        if V_delta_T_i > 0: 
            normalized_value = delta_T_i / np.sqrt(V_delta_T_i)
            normalized_values.append(normalized_value)
    # rms = np.sqrt(np.mean(np.square(normalized_values)))#
    rms = np.std(normalized_values, ddof=1)
    return rms

def V_t0mine(a, b, d, e, T_50, n):
    return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2)) * (1 + e * T_50)

def V_Delta_Tmine(a, b, d, e, T_50_i, T_50_j, n_i, n_j):
    return V_t0mine(a, b, d, e, T_50_i, n_i) + V_t0mine(a, b, d, e, T_50_j, n_j)

def rms_delta_tmine(delta_T_values, n_values, t50_values, a, b, d, e):
    if len(delta_T_values) != len(n_values) or len(delta_T_values) != len(t50_values):
        raise ValueError("Die Listen delta_T_values, n_values und t50_values müssen gleich lang sein.")
    normalized_values = []
    for i in range(len(delta_T_values)):
        delta_T_i = delta_T_values[i]
        T_50_i, T_50_j = t50_values[i]
        n_i, n_j = n_values[i]
        V_delta_T_i = V_Delta_Tmine(a, b, d, e, T_50_i, T_50_j, n_i, n_j)
        if V_delta_T_i > 0: 
            normalized_value = delta_T_i / np.sqrt(V_delta_T_i)
            normalized_values.append(normalized_value)
    # rms = np.sqrt(np.mean(np.square(normalized_values)))
    rms = np.std(normalized_values, ddof=1)
    return rms

a, b, d = 152, 0.84, 192.79
aPaper, bPaper, dPaper = 134, 2.4, 10
amine, bmine, dmine, emine = 177, 2.29166, 29.61166, -0.000514438
rms_value = rms_delta_t(delta_t_values, n_values, t50_values, a, b, d)
rms_valuePaper = rms_delta_t(delta_t_values, n_values, t50_values, aPaper, bPaper, dPaper)
rms_mine = rms_delta_tmine(delta_t_values, n_values, t50_values, amine, bmine, dmine, emine)
print(f"RMS for mine: {rms_value}\n RMS for Paper  {rms_valuePaper}\n RMS minemine: {rms_mine}")

def rms_by_value_with_errors(delta_T_values, n_values, t50_values, value_values, a, b, d, num_bins=25):
    mean_values = np.array([np.mean(dist_pair) for dist_pair in value_values])
    bins = np.linspace(np.min(mean_values), np.max(mean_values), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
    rms_values = []
    errors = []
    N = []

    for i in range(num_bins):
        indices = np.where((mean_values >= bins[i]) & (mean_values < bins[i + 1]))[0]
        delta_T_bin = [delta_T_values[idx] for idx in indices]
        t50_bin = [t50_values[idx] for idx in indices]
        n_bin = [n_values[idx] for idx in indices]
        n = len(delta_T_bin)  

        if n > 0:
            rms = rms_delta_t(delta_T_bin, n_bin, t50_bin, a, b, d)
            rms_values.append(rms)
            errors.append(1 / np.sqrt(2 * n)) 
            N.append(n) 
        else:
            rms_values.append(0)  
            errors.append(0) 

    return bin_centers, rms_values, errors, N

# def rms_by_value_with_errorsmine(delta_T_values, n_values, t50_values, value_values, a, b, d, e, num_bins=25):
#     mean_values = np.array([np.mean(dist_pair) for dist_pair in value_values])
#     bins = np.linspace(np.min(mean_values), np.max(mean_values), num_bins + 1)
#     bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
#     rms_values = []
#     errors = []
#     N = []

#     for i in range(num_bins):
#         indices = np.where((mean_values >= bins[i]) & (mean_values < bins[i + 1]))[0]
#         delta_T_bin = [delta_T_values[idx] for idx in indices]
#         t50_bin = [t50_values[idx] for idx in indices]
#         n_bin = [n_values[idx] for idx in indices]
#         N = len(delta_T_bin)  

#         if N > 0:
#             rms = rms_delta_tmine(delta_T_bin, n_bin, t50_bin, a, b, d, e)
#             rms_values.append(rms)
#             errors.append(1 / np.sqrt(2 * N))  
#         else:
#             rms_values.append(np.nan)  
#             errors.append(np.nan) 

#     return bin_centers, rms_values, errors, N


# def rms_by_value_with_errors_group_id(delta_T_values, n_values, t50_values, value_values, a, b, d):
#     # Alle eindeutigen diskreten Werte in value_values finden
#     unique_values = np.unique(value_values)
#     print(unique_values)
#     rms_values = []
#     errors = []
#     N = []

#     # Für jeden eindeutigen diskreten Wert berechnen
#     for discrete_value in unique_values:
#         # Indizes, die zu diesem diskreten Wert gehören
#         indices = [idx for idx, value in enumerate(value_values) if value == discrete_value]
        
#         # Werte für die aktuelle Kategorie extrahieren
#         delta_T_bin = [delta_T_values[idx] for idx in indices]
#         t50_bin = [t50_values[idx] for idx in indices]
#         n_bin = [n_values[idx] for idx in indices]
#         N = len(delta_T_bin)  

#         if N > 0:
#             # RMS berechnen
#             rms = rms_delta_t(delta_T_bin, n_bin, t50_bin, a, b, d)
#             rms_values.append(rms)
#             # Fehler berechnen (1 / sqrt(2N))
#             errors.append(1 / np.sqrt(2 * N))
#         else:
#             rms_values.append(np.nan)
#             errors.append(np.nan)

#     return unique_values, rms_values, errors, N


mp.rc("text", usetex=True)
mp.rc("font", family="serif")
pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
# pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))

fig, axs = plt.subplots(2, 2, figsize=(9, 8)) 


bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, distance_values, a, b, d)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, distance_values, aPaper, bPaper, dPaper)
#bin_centersMine, rms_valuesMine, errorsMine = rms_by_value_with_errorsmine(delta_t_values, n_values, t50_values, distance_values, amine, bmine, dmine, emine)

# Plot 1: Distance
ax1 = axs[0, 0]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse

# Balkendiagramm für die Anzahl der Datenpunkte auf der rechten Skala
print(N, "\n", bin_centers)
ax2.bar(bin_centers, N, width=200, alpha=0.2, color='gray')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# Fehlerbalken und rms-Werte auf der linken Skala
ax1.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha=0.7, linestyle='', color='r', label=r'GAP-2007-057')
ax1.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha=0.7, linestyle='', color='b', label=r'New parametrization')
# ax1.errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha=0.7, linestyle='', color='g', label=r'MineMine')

# Einstellungen für die linke Achse
ax1.set_ylim(0, None)
ax1.set_xlabel(r"Distance from the core in m")
ax1.set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
ax1.set_title(r"Distance")
ax1.grid()
ax1.legend()





log_s = np.log10(np.array(s_values))
bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, log_s, a, b, d)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, log_s, aPaper, bPaper, dPaper)

ax1 = axs[0, 1]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.2, color='gray', label='Number of data points (N)')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[0, 1].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2007-057')
axs[0, 1].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'New parametrization')
#axs[0, 1].errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')
axs[0, 1].set_ylim(0, 3)
axs[0, 1].set_xlabel(r"Signal in VEM (log$_{10}$ scale)")
axs[0, 1].set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
axs[0, 1].set_title(r"Signal")
axs[0, 1].grid(which='both', linestyle='--', linewidth=0.5)
axs[0, 1].legend()

bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, t50_values, a, b, d)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, t50_values, aPaper, bPaper, dPaper)
#bin_centersMine, rms_valuesMine, errorsMine = rms_by_value_with_errorsmine(delta_t_values, n_values, t50_values, t50_values, amine, bmine, dmine, emine)

# Plot 3: T50
ax1 = axs[1, 0]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centers, N, (np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.2, color='gray', label='Number of data points (N)')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[1, 0].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2007-057')
axs[1, 0].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'New parametrization')
#axs[1, 0].errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')
axs[1, 0].set_ylim(0, 2)
axs[1, 0].set_xlabel(r"$T_{50}$ in ns")
axs[1, 0].set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
axs[1, 0].set_title(r"$T_{50}$")
axs[1, 0].grid()
axs[1, 0].legend()


sin2_zenith = np.array(np.sin(zenith_values)**2)
bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, sin2_zenith, a, b, d)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_t_values, n_values, t50_values, sin2_zenith, aPaper, bPaper, dPaper)
#bin_centersMine, rms_valuesMine, errorsMine = rms_by_value_with_errorsmine(delta_t_values, n_values, t50_values, zenith_values, amine, bmine, dmine, emine)

# Plot 4: Zenith
ax1 = axs[1, 1]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.2, color='gray', label='Number of data points (N)')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[1, 1].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2007-057')
axs[1, 1].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'New parametrization')
#axs[1, 0].errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')
axs[1, 1].set_ylim(0, 2)
axs[1, 1].set_xlabel(r"$sin^2 \theta$")
axs[1, 1].set_ylabel(r"std of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
axs[1, 1].set_title(r"Zenith angle")
axs[1, 1].grid()
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for the main title
plt.savefig("2007_fase2_time.pdf", format="pdf", bbox_inches="tight")
plt.show()

# # distance

# bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, distance_values, a, b, d)
# bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors(delta_t_values, n_values, t50_values, distance_values, aPaper, bPaper, dPaper)
# bin_centersMine, rms_valuesMine, errorsMine = rms_by_value_with_errorsmine(delta_t_values, n_values, t50_values, distance_values, amine, bmine, dmine, emine)

# plt.figure(figsize=(10, 6))
# print(bin_centersPaper, rms_valuesPaper, errorsPaper)
# plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
# plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'Mine')
# plt.errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')
# plt.ylim(0, 2)
# # plt.xlim(0, 1500)
# plt.xlabel(r"Distance from the core in m")
# plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# plt.title(r"RMS of Normalized $\Delta$ T vs Distance")
# plt.grid()
# plt.legend()
# plt.show()

# # signal

# # Signal plot with logarithmic bins
# bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, s_values, a, b, d)
# bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors(delta_t_values, n_values, t50_values, s_values, aPaper, bPaper, dPaper)
# bin_centersMine, rms_valuesMine, errorsMine = rms_by_value_with_errorsmine(delta_t_values, n_values, t50_values, s_values, amine, bmine, dmine, emine, fmine)

# # Define logarithmic bins
# log_min = np.log10(np.min(s_values))
# log_max = np.log10(np.max(s_values))
# bins = np.logspace(log_min, log_max, 21)
# bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))  # Midpoints in log space

# bin_centersPaper = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))  # Midpoints in log space

# bin_centersMine = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))  # Midpoints in log space

# plt.figure(figsize=(10, 6))
# print(bin_centersPaper, rms_valuesPaper, errorsPaper)
# plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
# plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'Mine')
# plt.errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')

# # Set logarithmic scale for x-axis
# plt.xscale('log')

# plt.ylim(0, 2)
# plt.xlabel(r"Signal in VEM (log$_{10}$ scale)")
# plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# plt.title(r"RMS of Normalized $\Delta$ T vs Signal (log$_{10}$ Scale)")
# plt.grid(which='both', linestyle='--', linewidth=0.5)  # Grid lines for both major and minor ticks
# plt.legend()
# plt.show()


# # t50

# bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, t50_values, a, b, d)
# bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors(delta_t_values, n_values, t50_values, t50_values, aPaper, bPaper, dPaper)
# bin_centersMine, rms_valuesMine, errorsMine = rms_by_value_with_errorsmine(delta_t_values, n_values, t50_values, t50_values, amine, bmine, dmine, emine, fmine)

# plt.figure(figsize=(10, 6))
# print(bin_centersPaper, rms_valuesPaper, errorsPaper)
# plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
# plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'Mine')
# plt.errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')
# plt.ylim(0, 2)
# plt.xlabel(r"$T_{50}$ in ns")
# plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# plt.title(r"RMS of Normalized $\Delta$ T vs $T_50$")
# plt.grid()
# plt.legend()
# plt.show()

# #zenith

# bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, zenith_values, a, b, d)
# bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors(delta_t_values, n_values, t50_values, zenith_values, aPaper, bPaper, dPaper)

# plt.figure(figsize=(10, 6))
# print(bin_centersPaper, rms_valuesPaper, errorsPaper)
# plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
# plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'Mine')
# plt.ylim(0, 2)
# plt.xlabel(r"Zenith in radians")
# plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# plt.title(r"RMS of Normalized $\Delta$ T vs Zenith")
# plt.grid()
# plt.legend()
# plt.show()


# plt.hist(delta_t_values)
# plt.show()


# # groups

# bin_centers, rms_values, errors = rms_by_value_with_errors_group_id(delta_t_values, n_values, t50_values, group_id_values, a, b, d)
# bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors_group_id(delta_t_values, n_values, t50_values, group_id_values, aPaper, bPaper, dPaper)

# plt.figure(figsize=(10, 6))
# plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
# plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'Mine')
# plt.xlabel(r"Groups")
# plt.ylim(0, 2)
# plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
# plt.title(r"Group_ID")
# plt.grid()
# plt.legend()
# plt.show()