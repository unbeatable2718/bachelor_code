#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def delta_t(data):
    delta_t_values_old = []  
    delta_t_values_cool = []
    delta_t_values_raw = []
    for auger_id, details in data.items():
        t_0 = []
        t_0Nano = []
        x = []
        tPFSec = []
        tPFNano = []
        tPFSimple = []
        xb = []
        
        aVec = np.array([details['aVecX'], details['aVecY'], details['aVecZ']])
        for group_id, stations in details['group_id'].items():  
            for station in stations:
                t_0i = station["timeSecond"]
                t_0iNano = station["timeNSecond"]
                xi = np.array([station["stationCoorX"], station["stationCoorY"], station["stationCoorZ"]])
                xbi = np.array([station["corePositionX"], station["corePositionY"], station["corePositionZ"]])
                tPFSeci = station["coreTimeSec"]
                tPFNanoi = station["coreTimeNano"]
                tPFSimplei = station["timeSimple"]
                t_0.append(t_0i)
                t_0Nano.append(t_0iNano)
                x.append(xi)
                xb.append(xbi)
                tPFSec.append(tPFSeci)
                tPFSimple.append(tPFSimplei)
                tPFNano.append(tPFNanoi)


        
        if len(t_0) == 2 and len(x) == 2:
            delta_t_0 = t_0[1] - t_0[0]
            if delta_t_0 == 0 and tPFSec[1] == tPFSec[0] and t_0[1] == tPFSec[1]:
                delta_T_i_old = t_0Nano[0] - tPFSimple[0]
                delta_T_j_old = t_0Nano[1] - tPFSimple[1]
                delta_T_cool = t_0Nano[0] - t_0Nano[1] + ((np.dot(aVec, (x[0] - x[1]))) / (299792458 * 10**(-9)))
                delta_t_raw = (t_0Nano[0] - (tPFNano[0] - (np.dot(aVec, (x[0] - xb[0])) / (299792458 * 10**(-9))))) - (t_0Nano[1] - (tPFNano[1] - (np.dot(aVec, (x[1] - xb[1])) / (299792458 * 10**(-9)))))
                delta_t_values_old.append((delta_T_i_old - delta_T_j_old))
                delta_t_values_cool.append(delta_T_cool)
                delta_t_values_raw.append(delta_t_raw)
    return delta_t_values_old, delta_t_values_cool, delta_t_values_raw

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



            
delta_t_values_old, delta_t_values_cool, delta_t_values_raw = delta_t(events)


delta_t_values_old = np.array(delta_t_values_old)
delta_t_values_cool = np.array(delta_t_values_cool)
delta_t_values_raw = np.array(delta_t_values_raw)


difference_old_cool = delta_t_values_old - delta_t_values_cool
difference_old_raw = delta_t_values_old - delta_t_values_raw
difference_cool_raw = delta_t_values_cool - delta_t_values_raw

# Ausgabe der Differenzen
print("Difference (old - cool):", difference_old_cool)
print("Difference (old - raw):", difference_old_raw)
print("Difference (cool - raw):", difference_cool_raw)

# Optional: Zusammenfassung der Statistiken der Differenzen
print("\nStatistics:")
print("Mean difference (old - cool):", np.mean(difference_old_cool))
print("Mean difference (old - raw):", np.mean(difference_old_raw))
print("Mean difference (cool - raw):", np.mean(difference_cool_raw))
print("Max difference (old - cool):", np.max(difference_old_cool))
print("Max difference (old - raw):", np.max(difference_old_raw))
print("Max difference (cool - raw):", np.max(difference_cool_raw))
