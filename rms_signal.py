#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from my_functions import read_in, delta_t, mean_t50, delta_t_naiv

events = read_in("data_data/events6t5.txt")
delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, s_values, group_id_values, counter, delta_s_values = delta_t_naiv(events, 300) 

print(len(delta_t_values))
def V_t0(a, b, Theta, signal):
    return (a + b * (1 / (np.cos(Theta)))) ** 2 * signal

def V_Delta_T(a, b, Theta, signal_i, signal_j):
    return V_t0(a, b, Theta, signal_i) + V_t0(a, b, Theta, signal_j)

def rms_delta_t(delta_s_values, zenith_values, signal_values, a, b):
    normalized_values = []
    for i in range(len(delta_s_values)):
        delta_s_i = delta_s_values[i]
        theta = zenith_values[i]
        s_i, s_j = signal_values[i]
        V_delta_T_i = V_Delta_T(a, b, theta, s_i, s_j)
        if V_delta_T_i > 0: 
            normalized_value = delta_s_i / np.sqrt(V_delta_T_i)
            normalized_values.append(normalized_value)
    # rms = np.sqrt(np.mean(np.square(normalized_values)))#
    rms = np.std(normalized_values, ddof=1)
    return rms

a, b = 0.83, 0.266
aPaper, bPaper = 0.34, 0.46
rms_value = rms_delta_t(delta_s_values, zenith_values, s_values, a, b)
rms_valuePaper = rms_delta_t(delta_s_values, zenith_values, s_values, aPaper, bPaper)
print(f"RMS for mine: {rms_value}\n RMS for Paper  {rms_valuePaper}")

def rms_by_value_with_errors(delta_s_values, zenith_values, s_values, value_values, a, b, num_bins=25):
    mean_values = np.array([np.mean(dist_pair) for dist_pair in value_values])
    bins = np.linspace(np.min(mean_values), np.max(mean_values), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
    rms_values = []
    errors = []
    N = []

    for i in range(num_bins):
        indices = np.where((mean_values >= bins[i]) & (mean_values < bins[i + 1]))[0]
        delta_T_bin = [delta_s_values[idx] for idx in indices]
        zenith_bin = [zenith_values[idx] for idx in indices]
        s_bin = [s_values[idx] for idx in indices]
        n = len(delta_T_bin)  

        if n > 0:
            rms = rms_delta_t(delta_T_bin, zenith_bin, s_bin, a, b)
            rms_values.append(rms)
            errors.append(1 / np.sqrt(2 * n)) 
            N.append(n) 
        else:
            rms_values.append(0)  
            errors.append(0) 

    return bin_centers, rms_values, errors, N

mp.rc("text", usetex=True)
mp.rc("font", family="serif")
pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
# pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))

fig, axs = plt.subplots(2, 2, figsize=(9, 8))


bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, distance_values, a, b)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, distance_values, aPaper, bPaper)

# Plot 1: Distance
ax1 = axs[0, 0]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
axs[0, 0].axhline(1, color='grey', linestyle='--', alpha=0.5)

# Balkendiagramm für die Anzahl der Datenpunkte auf der rechten Skala
ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# Fehlerbalken und rms-Werte auf der linken Skala
ax1.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha=0.7, linestyle='', color='r', label=r'GAP-2012-012')
ax1.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha=0.7, linestyle='', color='b', label=r'New parametrization')
# ax1.errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha=0.7, linestyle='', color='g', label=r'MineMine')

# Einstellungen für die linke Achse
ax1.set_xlabel(r"Distance from the core in m")
ax1.set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
ax1.set_title(r"Distance")
axs[0, 0].set_ylim(bottom=0)
ax1.grid()
ax1.legend()


log_s = np.log10(np.array(s_values))
print(log_s[:7])
print(s_values[:7])
bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, log_s, a, b)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, log_s, aPaper, bPaper)

ax1 = axs[0, 1]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
axs[0, 1].axhline(1, color='grey', linestyle='--', alpha=0.5)
# Balkendiagramm für die Anzahl der Datenpunkte auf der rechten Skala
ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

axs[0, 1].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2012-012')
axs[0, 1].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'New parametrization')
axs[0, 1].set_xlabel(r"Signal in VEM (log$_{10}$ scale)")
axs[0, 1].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
axs[0, 1].set_title(r"Signal")
axs[0, 1].grid()
axs[0, 1].set_ylim(bottom=0)
axs[0, 1].legend()

bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, t50_values, a, b)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, t50_values, aPaper, bPaper)

ax1 = axs[1, 0]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[1, 0].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2012-012')
axs[1, 0].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'New parametrization')
#axs[1, 0].errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')
axs[1, 0].set_xlabel(r"$T_{50}$ in ns")
axs[1, 0].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
axs[1, 0].set_title(r"$T_{50}$")
axs[1, 0].set_ylim(0, None)
axs[1, 0].grid()
axs[1, 0].legend()


sin2_zenith = np.array(np.sin(zenith_values)**2)
bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, sin2_zenith, a, b)
bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, sin2_zenith, aPaper, bPaper)

# Plot 4: Zenith
ax1 = axs[1, 1]  # Linke Achse
ax2 = ax1.twinx()  # Rechte Achse
ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
ax2.set_ylabel("Number of data points (N)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
axs[1, 1].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2012-012')
axs[1, 1].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'New parametrization')
axs[1, 1].set_xlabel(r"sin$^2(\theta)$")
axs[1, 1].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
axs[1, 1].set_title(r"Zenith angle")
axs[1, 1].set_ylim(0, 2)
axs[1, 1].grid()
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for the main title
plt.savefig("AAsignalPhaseIb2012.pdf", format="pdf", bbox_inches="tight")
plt.show()



# def V_t0(a, b, c, Theta, signal):
#     return (a * (1 + b * ((1 / np.cos(Theta)) - 1.22)) * np.sqrt(signal)) ** 2 + (c * signal) ** 2

# def V_Delta_T(a, b, c, Theta, signal_i, signal_j):
#     return V_t0(a, b, c, Theta, signal_i) + V_t0(a, b, c, Theta, signal_j)

# def rms_delta_t(delta_s_values, zenith_values, signal_values, a, b, c):
#     normalized_values = []
#     for i in range(len(delta_s_values)):
#         delta_s_i = delta_s_values[i]
#         theta = zenith_values[i]
#         s_i, s_j = signal_values[i]
#         V_delta_T_i = V_Delta_T(a, b, c, theta, s_i, s_j)
#         if V_delta_T_i > 0: 
#             normalized_value = delta_s_i / np.sqrt(V_delta_T_i)
#             normalized_values.append(normalized_value)
#     # rms = np.sqrt(np.mean(np.square(normalized_values)))#
#     rms = np.std(normalized_values, ddof=1)
#     return rms

# a, b, c = 0.718, 0.579, 0.129
# aPaper, bPaper ,cPaper= 0.865, 0.593, 0.023
# rms_value = rms_delta_t(delta_s_values, zenith_values, s_values, a, b, c)
# rms_valuePaper = rms_delta_t(delta_s_values, zenith_values, s_values, aPaper, bPaper, cPaper)
# print(f"RMS for mine: {rms_value}\n RMS for Paper  {rms_valuePaper}")

# def rms_by_value_with_errors(delta_s_values, zenith_values, s_values, value_values, a, b, c, num_bins=25):
#     mean_values = np.array([np.mean(dist_pair) for dist_pair in value_values])
#     bins = np.linspace(np.min(mean_values), np.max(mean_values), num_bins + 1)
#     bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
#     rms_values = []
#     errors = []
#     N = []

#     for i in range(num_bins):
#         indices = np.where((mean_values >= bins[i]) & (mean_values < bins[i + 1]))[0]
#         delta_T_bin = [delta_s_values[idx] for idx in indices]
#         zenith_bin = [zenith_values[idx] for idx in indices]
#         s_bin = [s_values[idx] for idx in indices]
#         n = len(delta_T_bin)  

#         if n > 0:
#             rms = rms_delta_t(delta_T_bin, zenith_bin, s_bin, a, b, c)
#             rms_values.append(rms)
#             errors.append(1 / np.sqrt(2 * n)) 
#             N.append(n) 
#         else:
#             rms_values.append(0)  
#             errors.append(0) 

#     return bin_centers, rms_values, errors, N

# mp.rc("text", usetex=True)
# mp.rc("font", family="serif")
# pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
# # pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
# mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))
# #plt.rcParams.update({'font.size': 12})
# fig, axs = plt.subplots(2, 2, figsize=(9, 8)) 


# bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, distance_values, a, b, c)
# bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, distance_values, aPaper, bPaper, cPaper)

# # Plot 1: Distance
# ax1 = axs[0, 0]  # Linke Achse
# ax2 = ax1.twinx()  # Rechte Achse
# # axs[0, 0].axhline(1, color='grey', linestyle='--', alpha=0.5)
# # Balkendiagramm für die Anzahl der Datenpunkte auf der rechten Skala
# print(N,"\n", bin_centers)
# ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
# ax2.set_ylabel("Number of data points (N)", color='gray')
# ax2.tick_params(axis='y', labelcolor='gray')

# # Fehlerbalken und rms-Werte auf der linken Skala
# ax1.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha=0.7, linestyle='', color='r', label=r'GAP-2014-035')
# ax1.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha=0.7, linestyle='', color='b', label=r'New parametrization')
# # ax1.errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha=0.7, linestyle='', color='g', label=r'MineMine')
# # Einstellungen für die linke Achse
# ax1.set_xlabel(r"Distance from the core in m")
# ax1.set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
# ax1.set_title(r"Distance")
# axs[0, 0].set_ylim(bottom=0)
# ax1.grid()
# ax1.legend()


# log_s = np.log10(np.array(s_values))
# print(log_s[:7])
# print(s_values[:7])
# bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, log_s, a, b, c)
# bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, log_s, aPaper, bPaper, cPaper)

# ax1 = axs[0, 1]  # Linke Achse
# ax2 = ax1.twinx()  # Rechte Achse
# # axs[0, 1].axhline(1, color='grey', linestyle='--', alpha=0.5)

# # Balkendiagramm für die Anzahl der Datenpunkte auf der rechten Skala
# ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
# ax2.set_ylabel("Number of data points (N)", color='gray')
# ax2.tick_params(axis='y', labelcolor='gray')

# axs[0, 1].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2014-035')
# axs[0, 1].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'New parametrization')
# axs[0, 1].set_xlabel(r"Signal in VEM (log$_{10}$ scale)")
# axs[0, 1].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
# axs[0, 1].set_title(r"Signal")
# axs[0, 1].grid()
# axs[0, 1].set_ylim(bottom=0)
# axs[0, 1].legend()

# bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, t50_values, a, b, c)
# bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, t50_values, aPaper, bPaper, cPaper)

# ax1 = axs[1, 0]  # Linke Achse
# ax2 = ax1.twinx()  # Rechte Achse
# ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
# ax2.set_ylabel("Number of data points (N)", color='gray')
# ax2.tick_params(axis='y', labelcolor='gray')
# axs[1, 0].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2014-035')
# axs[1, 0].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'New parametrization')
# #axs[1, 0].errorbar(bin_centersMine, rms_valuesMine, yerr=errorsMine, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'MineMine')
# axs[1, 0].set_xlabel(r"$T_{50}$ in ns")
# axs[1, 0].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
# axs[1, 0].set_title(r"$T_{50}$")
# axs[1, 0].set_ylim(0, None)
# axs[1, 0].grid()
# axs[1, 0].legend()

# sin2_zenith = np.array(np.sin(zenith_values)**2)
# bin_centers, rms_values, errors, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, sin2_zenith, a, b, c)
# bin_centersPaper, rms_valuesPaper, errorsPaper, N = rms_by_value_with_errors(delta_s_values, zenith_values, s_values, sin2_zenith, aPaper, bPaper, cPaper)

# # Plot 4: Zenith
# ax1 = axs[1, 1]  # Linke Achse
# ax2 = ax1.twinx()  # Rechte Achse
# ax2.bar(bin_centers, N, width=(np.max(bin_centers) - np.min(bin_centers))/24, alpha=0.3, color='gray', label='Number of data points (N)')
# ax2.set_ylabel("Number of data points (N)", color='gray')
# ax2.tick_params(axis='y', labelcolor='gray')
# axs[1, 1].errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'GAP-2014-035')
# axs[1, 1].errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'New parametrization')
# axs[1, 1].set_xlabel(r"sin$^2(\theta)$")
# axs[1, 1].set_ylabel(r"std of $\frac{\Delta S}{\sqrt{V[\Delta S]}}$")
# axs[1, 1].set_title(r"Zenith angle")
# axs[1, 1].set_ylim(0, 2)
# axs[1, 1].grid()
# axs[1, 1].legend()
# plt.tight_layout()  # Reserve space for the main title
# plt.savefig("AAsignalPhaseIa2014.pdf", format="pdf", bbox_inches="tight")
# plt.show()
