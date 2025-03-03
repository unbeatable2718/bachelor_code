#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

input_file = Path("events.txt")
events = {}

with input_file.open("r") as f:
    current_event = None
    for line in f:
        line = line.strip()  # Remove leading and trailing whitespace
        if line.startswith("Event ID:"):
            # Extract the Event ID and initialize its dictionary
            current_event = line.split("Event ID:")[1].strip()
            events[current_event] = {}
        elif current_event and ":" in line:
            # Extract the key-value pair
            key, value = map(str.strip, line.split(":", 1))
            try:
                # Attempt to convert value to Python literal (e.g., dict, float, int)
                value = eval(value)
            except:
                # If eval fails, keep it as a string
                pass
            events[current_event][key] = value


def delta_t(data):
    delta_t_values = []  
    n_values = []
    t50_values = []
    for auger_id, details in data.items():
        t_0 = []
        t_0Nano = []
        t50 = []
        n = []
        tPFSec = []
        tPFSimple = []
        for group_id, stations in details['group_id'].items():  
            for station in stations:
                t_0i = station["timeSecond"]
                t_0iNano = station["timeNSecond"]
                t50i = station["time50"]
                ni = station["n"]
                tPFSeci = station["coreTimeSec"]
                tPFSimplei = station["timeSimple"]

                t_0.append(t_0i)
                t_0Nano.append(t_0iNano)
                t50.append(t50i)
                n.append(ni)
                tPFSec.append(tPFSeci)
                tPFSimple.append(tPFSimplei)
       
        if len(t_0) == 2:
            delta_t_0 = t_0[1] - t_0[0]
            if delta_t_0 == 0 and tPFSec[1] == tPFSec[0] and t_0[1] == tPFSec[1]:
                delta_T_i = t_0Nano[0] - tPFSimple[0]
                delta_T_j = t_0Nano[1] - tPFSimple[1]
                if np.abs(delta_T_i - delta_T_j)<=200:
                    n_values.append(n)
                    t50_values.append(t50)
                    delta_t_values.append((delta_T_i - delta_T_j))

    return delta_t_values, n_values, t50_values
            
delta_t_values, n_values, t50_values = delta_t(events)

print(len(delta_t_values))
def V_t0(a, b, d, T_50, n):
    return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j):
    return V_t0(a, b, d, T_50_i, n_i) + V_t0(a, b, d, T_50_j, n_j)

def V_delta_t(delta_T_values, n_values, t50_values, a, b, d):
    if len(delta_T_values) != len(n_values) or len(delta_T_values) != len(t50_values):
        raise ValueError("Die Listen delta_T_values, n_values und t50_values müssen gleich lang sein.")
    V_delta_T = []
    for i in range(len(delta_T_values)):
        T_50_i, T_50_j = t50_values[i]
        n_i, n_j = n_values[i]
        V_delta_T.append(V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j))
    return V_delta_T

a, b, d = 193, 0.73, 139
aPaper, bPaper, dPaper = 134, 2.4, 10

V_value = V_delta_t(delta_t_values, n_values, t50_values, a, b, d)
V_valuePaper = V_delta_t(delta_t_values, n_values, t50_values, aPaper, bPaper, dPaper)




mp.rc("text", usetex=True)
mp.rc("font", family="serif")
pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
# pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))
# max_angle = np.max(angles_in_degrees)
# print(f"Maximum angle: {max_angle:.2f} degrees")
# plt.hist(angles_in_degrees, bins=30, color='purple', edgecolor='black', alpha=0.7)
# plt.title(r"Opening Angles")
# plt.xlabel(r"Angle (Degrees)")
# plt.ylabel(r"Frequency")
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.show()    

plt.figure(figsize=(10, 6))
plt.scatter(np.abs(delta_t_values), V_valuePaper, color='r', label=r'Paper', alpha=0.6)
plt.scatter(np.abs(delta_t_values), V_value, color='b', label=r'Mine', alpha=0.6)
plt.xlabel(r"$\Delta$ T")
plt.ylabel(r"V [$\Delta$ T]")
plt.title(r"Comparison of former and new astimation of V[$\Delta$ T]")
plt.grid()
plt.legend()
plt.show()

t50_valuesSolo1 = []
t50_valuesSolo2 = []
for i in range(len(t50_values)):
    t50_valuesSolo1.append(t50_values[i][0])
    t50_valuesSolo2.append(t50_values[i][1])

plt.figure(figsize=(10, 6))
plt.scatter(t50_valuesSolo1, V_valuePaper, color='r', label=r'Paper', alpha=0.6)
plt.scatter(t50_valuesSolo2, V_valuePaper, color='r', alpha=0.6)
plt.scatter(t50_valuesSolo1, V_value, color='b', label=r'Mine', alpha=0.6)
plt.scatter(t50_valuesSolo2, V_value, color='b', alpha=0.6)

plt.xlabel(r"$T_{50}$")
plt.ylabel(r"V [$\Delta$ T]")
plt.title(r"Comparison of former and new astimation of V[$\Delta$ T]")
plt.grid()
plt.legend()
plt.show()

