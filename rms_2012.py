#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from my_functions import delta_t, read_in
            
events = read_in("data_test4/events_2007.txt")            
delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, s_values, group_id_values, counter = delta_t(events, 450)

print(len(delta_t_values))
def V_t0(a, b, T_50, n):
    return a * pow(2 * T_50 / (n - 1), 2) * n / (n + 2) + b

def V_Delta_T(a, b, T_50_i, T_50_j, n_i, n_j):
    return V_t0(a, b, T_50_i, n_i) + V_t0(a, b, T_50_j, n_j)

def rms_delta_t(delta_T_values, n_values, t50_values, zenith_values, a, b, c, d, e, f):
    if len(delta_T_values) != len(n_values) or len(delta_T_values) != len(t50_values):
        raise ValueError("Die Listen delta_T_values, n_values und t50_values müssen gleich lang sein.")
    normalized_values = []
    for i in range(len(delta_T_values)):
        delta_T_i = delta_T_values[i]
        T_50_i, T_50_j = t50_values[i]
        n_i, n_j = n_values[i]
        cos_theta = np.cos(zenith_values[i])
        a1 = a + cos_theta * (b - c * cos_theta)
        b1 = (d + cos_theta * (e * cos_theta - f))
        V_delta_T_i = V_Delta_T(a1, b1, T_50_i, T_50_j, n_i, n_j)
        if V_delta_T_i > 0: 
            normalized_value = delta_T_i / np.sqrt(V_delta_T_i)
            normalized_values.append(normalized_value)
    rms = np.sqrt(np.mean(np.square(normalized_values)))
    return rms

def V_t02(a, b, T_50, n):
    return a * pow(2 * T_50 / (n - 1), 2) * n / (n + 2) + b

def V_Delta_T2(a, b, T_50_i, T_50_j, n_i, n_j):
    return V_t02(a, b, T_50_i, n_i) + V_t02(a, b, T_50_j, n_j)

def rms_delta_t2(delta_T_values, n_values, t50_values, zenith_values, a, b):
    if len(delta_T_values) != len(n_values) or len(delta_T_values) != len(t50_values):
        raise ValueError("Die Listen delta_T_values, n_values und t50_values müssen gleich lang sein.")
    normalized_values = []
    for i in range(len(delta_T_values)):
        delta_T_i = delta_T_values[i]
        T_50_i, T_50_j = t50_values[i]
        n_i, n_j = n_values[i]
        cos_theta = np.cos(zenith_values[i])
        V_delta_T_i = V_Delta_T(a, b, T_50_i, T_50_j, n_i, n_j)
        if V_delta_T_i > 0: 
            normalized_value = delta_T_i / np.sqrt(V_delta_T_i)
            normalized_values.append(normalized_value)
    rms = np.sqrt(np.mean(np.square(normalized_values)))
    return rms

a1, b1, c1, d1, e1, f1 = -0.048, 0.8245, 0.5451, 319.38, 881.79, 752.654
a2, b2, c2, d2, e2, f2 = 0.64871, 0.22365, 0.49971, 141.24, 412.19, 208.9
a, b = 0.264127, 260.212160
rms_value = rms_delta_t(delta_t_values, n_values, t50_values, zenith_values, a1, b1, c1, d1, e1, f1)
rms_value_old = rms_delta_t(delta_t_values, n_values, t50_values, zenith_values, a2, b2, c2, d2, e2, f2)
rms_value2 = rms_delta_t2(delta_t_values, n_values, t50_values, zenith_values, a, b)
#rms_valuePaper = rms_delta_t(delta_t_values, n_values, t50_values, aPaper, bPaper, dPaper)
print(f"RMS for 2012: {rms_value},\n old: {rms_value_old},\n simple: {rms_value2}")

def rms_by_value_with_errors(delta_T_values, n_values, t50_values, value_values, zenith_values, num_bins, a, b, c, d, e, f):
    mean_values = np.array([np.mean(dist_pair) for dist_pair in value_values])
    bins = np.linspace(np.min(mean_values), 1500, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
    rms_values = []
    errors = []

    for i in range(num_bins):
        indices = np.where((mean_values >= bins[i]) & (mean_values < bins[i + 1]))[0]
        delta_T_bin = [delta_T_values[idx] for idx in indices]
        t50_bin = [t50_values[idx] for idx in indices]
        n_bin = [n_values[idx] for idx in indices]
        N = len(delta_T_bin)  

        if N > 0:
            rms = rms_delta_t(delta_T_bin, n_bin, t50_bin, zenith_values, a, b, c, d, e, f)
            rms_values.append(rms)
            errors.append(1 / np.sqrt(2 * N))  
        else:
            rms_values.append(np.nan)  
            errors.append(np.nan) 

    return bin_centers, rms_values, errors


def rms_by_value_with_errors2(delta_T_values, n_values, t50_values, value_values, zenith_values, num_bins, a, b):
    mean_values = np.array([np.mean(dist_pair) for dist_pair in value_values])
    bins = np.linspace(np.min(mean_values), 1500, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
    rms_values = []
    errors = []

    for i in range(num_bins):
        indices = np.where((mean_values >= bins[i]) & (mean_values < bins[i + 1]))[0]
        delta_T_bin = [delta_T_values[idx] for idx in indices]
        t50_bin = [t50_values[idx] for idx in indices]
        n_bin = [n_values[idx] for idx in indices]
        N = len(delta_T_bin)  

        if N > 0:
            rms = rms_delta_t2(delta_T_bin, n_bin, t50_bin, zenith_values, a, b)
            rms_values.append(rms)
            errors.append(1 / np.sqrt(2 * N))  
        else:
            rms_values.append(np.nan)  
            errors.append(np.nan) 

    return bin_centers, rms_values, errors


mp.rc("text", usetex=True)
mp.rc("font", family="serif")
pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
# pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))



# distance
num_bins = 20
bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, distance_values, zenith_values, num_bins, a1, b1, c1, d1, e1, f1)
bin_centers_old, rms_values_old, errors_old = rms_by_value_with_errors(delta_t_values, n_values, t50_values, distance_values, zenith_values, num_bins, a2, b2, c2, d2, e2, f2)
bin_centers_simple, rms_values_simple, errors_simple = rms_by_value_with_errors2(delta_t_values, n_values, t50_values, distance_values, zenith_values, num_bins, a, b)

plt.figure(figsize=(10, 6))
#print(bin_centersPaper, rms_valuesPaper, errorsPaper)
plt.errorbar(bin_centers_old, rms_values_old, yerr=errors_old, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Old')
plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'Mine')
plt.errorbar(bin_centers_simple, rms_values_simple, yerr=errors_simple, fmt='o', alpha = 0.7, linestyle='', color='g', label=r'Simple')
plt.ylim(0, 2)
plt.xlim(0, 1500)
plt.xlabel(r"Distance from the core in m")
plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
plt.title(r"RMS of Normalized $\Delta$ T vs Distance")
plt.grid()
plt.legend()
plt.show()

# signal

# Signal plot with logarithmic bins
bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, s_values, zenith_values)
#bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors(delta_t_values, n_values, t50_values, s_values, aPaper, bPaper, dPaper)

# Define logarithmic bins
log_min = np.log10(np.min(s_values))
log_max = np.log10(np.max(s_values))
bins = np.logspace(log_min, log_max, 21)
bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))  # Midpoints in log space

#bin_centersPaper = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))  # Midpoints in log space

plt.figure(figsize=(10, 6))
#print(bin_centersPaper, rms_valuesPaper, errorsPaper)
#plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', alpha = 0.7, linestyle='', color='b', label=r'Mine')

# Set logarithmic scale for x-axis
plt.xscale('log')

# plt.ylim(0, 2)
plt.xlabel(r"Signal in VEM (log$_{10}$ scale)")
plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
plt.title(r"RMS of Normalized $\Delta$ T vs Signal (log$_{10}$ Scale)")
plt.grid(which='both', linestyle='--', linewidth=0.5)  # Grid lines for both major and minor ticks
plt.legend()
plt.show()


# t50

bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, t50_values, zenith_values)
#bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors(delta_t_values, n_values, t50_values, t50_values, aPaper, bPaper, dPaper)

plt.figure(figsize=(10, 6))
#print(bin_centersPaper, rms_valuesPaper, errorsPaper)
#plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'Mine')
# plt.ylim(0, 2)
plt.xlabel(r"$T_{50}$ in ns")
plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
plt.title(r"RMS of Normalized $\Delta$ T vs $T_50$")
plt.grid()
plt.legend()
plt.show()

#zenith

bin_centers, rms_values, errors = rms_by_value_with_errors(delta_t_values, n_values, t50_values, zenith_values, zenith_values)
#bin_centersPaper, rms_valuesPaper, errorsPaper = rms_by_value_with_errors(delta_t_values, n_values, t50_values, zenith_values, aPaper, bPaper, dPaper)

plt.figure(figsize=(10, 6))
#print(bin_centersPaper, rms_valuesPaper, errorsPaper)
#plt.errorbar(bin_centersPaper, rms_valuesPaper, yerr=errorsPaper, fmt='o', alpha = 0.7, linestyle='', color='r', label=r'Paper')
plt.errorbar(bin_centers, rms_values, yerr=errors, fmt='o', linestyle='', alpha = 0.7, color='b', label=r'Mine')
# plt.ylim(0, 2)
plt.xlabel(r"Zenith in radians")
plt.ylabel(r"RMS of $\frac{\Delta T}{\sqrt{V[\Delta T]}}$")
plt.title(r"RMS of Normalized $\Delta$ T vs Zenith")
plt.grid()
plt.legend()
plt.show()
