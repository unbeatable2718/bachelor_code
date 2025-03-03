#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def calculate_and_plot_delta_t(data):
    # Dictionary, um die Delta-T-Werte nach Station zu speichern
    station_delta_t = defaultdict(list)

    def process_station_group(t_0Nano, tPFSimple, id):
        # Delta-T-Berechnung pro Station
        for i in range(1, len(t_0Nano)):
            delta_T_i = t_0Nano[0] - tPFSimple[0]
            delta_T_j = t_0Nano[i] - tPFSimple[i]
            delta = delta_T_j - delta_T_i
            station_delta_t[id[0]].append(delta_T_i)
            station_delta_t[id[i]].append(delta_T_j)

    # Daten aus den Gruppen verarbeiten
    for auger_id, details in data.items():
        for group_id, stations in details['group_id'].items():
            t_0Nano, tPFSimple, id = [], [], []
            for station in stations:
                t_0Nano.append(station["timeNSecond"])
                tPFSimple.append(station["timeSimple"])
                id.append(station["station_id"])
            
            if len(t_0Nano) >= 2:
                process_station_group(t_0Nano, tPFSimple, id)

    # Durchschnittswerte und Unsicherheiten berechnen
    station_ids = []
    mean_delta_t = []
    std_delta_t = []

    for station_id, deltas in station_delta_t.items():
        station_ids.append(station_id)
        mean_delta_t.append(np.mean(deltas))
        std_delta_t.append(np.std(deltas) / np.sqrt(len(deltas)))


    # Sortiere die Stations-IDs und die zugehörigen Daten
    sorted_data = sorted(zip(station_ids, mean_delta_t, std_delta_t), key=lambda x: x[0])
    station_ids_sorted, mean_delta_t_sorted, std_delta_t_sorted = zip(*sorted_data)

    # Äquidistante Positionen für die Stations-IDs
    x_positions = np.arange(len(station_ids_sorted))

    # Plot erstellen
    plt.figure(figsize=(14, 7))
    plt.errorbar(
        x_positions, mean_delta_t_sorted, yerr=std_delta_t, fmt='o', color='r', label='dT Mean'
    )
    plt.axhline(0, color='gray', linestyle='--', label='Baseline (0)')
    plt.title("Mean dT per Station ID")
    plt.xlabel("Station ID")
    plt.ylabel("Mean dT (ns)")
    plt.xticks(x_positions, station_ids_sorted, rotation=90)  # Sortierte IDs äquidistant
    plt.ylim(-1000, 1000)  # Setze die Y-Achse-Limits
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(station_delta_t[72])



def read_in(file):
    input_file = Path(file)
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
    return events   


def read_in_list(files):
    events = {}

    for file in files:
        input_file = Path(file)

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

    return events

def delta_t(data, cut):
    counter = 0
    delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, delta_s_values = [], [], [], [], [], [], [], [], []
    number_lost_multiplets = 0
    number_lost_times = 0

    def process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated):
        nonlocal number_lost_times
        nonlocal counter
        delta_t_group, n_group, t50_group, tVar_group, distance_group, zenith_group, signal_group, group_group, delta_s_group = [], [], [], [], [], [], [], [], []


        for i in range(1, len(t_0)):
            if tPFSec[0] == tPFSec[i] == t_0[0] == t_0[i] and saturated[0] == False and saturated[i] == False:


# the smaller id is allways the first


                if int(id[0]) < int(id[i]):
                    # if (int(id[0]) == 72 and int(id[i]) == 688) or (int(id[0]) == 75 and int(id[i]) == 698) or (int(id[0]) == 76 and int(id[i]) == 698) or (int(id[0]) == 79 and int(id[i]) == 819) or (int(id[0]) == 81 and int(id[i]) == 819) or (int(id[0]) == 86 and int(id[i]) == 88) or (int(id[0]) == 88 and int(id[i]) == 660) or (int(id[0]) == 93 and int(id[i]) == 94) or (int(id[0]) == 94 and int(id[i]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000 and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_i - delta_T_j)
                        delta_s_group.append(signal[0] - signal[i])
                        # delta_s_group.append(np.log(signal[0]) - np.log(signal[i]))
                        # n_group.append(np.array([n[0], n[i]]))  
                        n_group.append(np.array([max(n[0], 2), max(n[i], 2)]))  
                        meanT50 = mean_t50(np.cos(zenith), distance[0])
                        if meanT50 > t50[0]:
                            counter += 1
                        if meanT50 > t50[i]:
                            counter += 1    
                        if meanT50 > t50[0] and n[0] < 4 and meanT50 < 1000:
                            t501 = meanT50
                        else:
                            t501 = t50[0]
                        if meanT50 > t50[i] and n[i] < 4 and meanT50 < 1000:
                            t502 = meanT50
                        else:
                            t502 = t50[i]            
                        # t50_group.append(np.array([t50[0], t50[i]]))
                        t50_group.append(np.array([t501, t502]))
                        tVar_group.append(np.array([tVar[0], tVar[i]]))
                        distance_group.append(np.array([distance[0], distance[i]]))
                        zenith_group.append(zenith)
                        # signal_group.append(np.array([np.log(signal[0]), np.log(signal[i])]))
                        signal_group.append(np.array([signal[i], signal[0]]))
                        group_group.append(np.array([id[0], id[i]]))

                elif int(id[0]) > int(id[i]):
                    # if (int(id[i]) == 72 and int(id[0]) == 688) or (int(id[i]) == 75 and int(id[0]) == 698) or (int(id[i]) == 76 and int(id[0]) == 698) or (int(id[i]) == 79 and int(id[0]) == 819) or (int(id[i]) == 81 and int(id[0]) == 819) or (int(id[i]) == 86 and int(id[0]) == 88) or (int(id[i]) == 88 and int(id[0]) == 660) or (int(id[i]) == 93 and int(id[0]) == 94) or (int(id[i]) == 94 and int(id[0]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000 and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_j - delta_T_i)
                        delta_s_group.append(signal[i] - signal[0])
                        # delta_s_group.append(np.log(signal[i]) - np.log(signal[0]))
                        # n_group.append(np.array([n[i], n[0]]))
                        n_group.append(np.array([max(n[i], 2), max(n[0], 2)]))     
                        meanT50 = mean_t50(np.cos(zenith), distance[0])
                        if meanT50 > t50[0]:
                            counter += 1
                        if meanT50 > t50[i]:
                            counter += 1    
                        if meanT50 > t50[0] and n[0] < 4 and meanT50 < 1000:
                            t501 = meanT50
                        else:
                            t501 = t50[0]
                        if meanT50 > t50[i] and n[i] < 4 and meanT50 < 1000:
                            t502 = meanT50
                        else:
                            t502 = t50[i]            
                        t50_group.append(np.array([t502, t501]))
                        tVar_group.append(np.array([tVar[i], tVar[0]]))
                        distance_group.append(np.array([distance[i], distance[0]]))
                        zenith_group.append(zenith)
                        signal_group.append(np.array([signal[i], signal[0]]))
                        # signal_group.append(np.array([np.log(signal[i]), np.log(signal[0])]))
                        group_group.append(np.array([id[i], id[0]]))

                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
            else:
                number_lost_times += 1    


        delta_t_values.extend(delta_t_group)
        delta_s_values.extend(delta_s_group)
        n_values.extend(n_group)
        t50_values.extend(t50_group)
        tVar_values.extend(tVar_group)
        distance_values.extend(distance_group)
        zenith_values.extend(zenith_group)
        signal_values.extend(signal_group)
        group_values.extend(group_group)

    for auger_id, details in data.items():
        aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

        for group_id, stations in details['group_id'].items():
            t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated = [], [], [], [], [], [], [], [], [], details["zenith"], [], [], [], []
            group_ids.append(int(group_id)) 
            for station in stations:
                t_0.append(station["timeSecond"])
                t_0Nano.append(station["timeNSecond"])
                x.append(np.array([station["stationCoorX"], station["stationCoorY"], station["stationCoorZ"]]))
                t50.append(station["time50"])
                n.append(station["n"])
                tPFSec.append(station["coreTimeSec"])
                tPFSimple.append(station["timeSimple"])
                tVar.append(station["timeError"])
                distance.append(station["distance"])
                signal.append(station["signal"])
                id.append(station["station_id"])
                saturated.append(station["saturated"])

            num_stations = len(t_0)
            if num_stations >= 2:
                    process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated)
            else:
                number_lost_multiplets += 1
                print(f"Ungültige Multiplet-Größe: {num_stations}")
 
    print(f"lost multiplets: {number_lost_multiplets}")
    print(f"lost times  : {number_lost_times}")
    # plt.hist(delta_t_values, bins=250)
    # plt.show()

    return delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values

def mean_t50(cos_theta, distance):
    return 0.8 * (0.53 * cos_theta - 0.11) * distance


def delta_t_naiv(data, cut):
    counter = 0
    delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, delta_s_values = [], [], [], [], [], [], [], [], []
    number_lost_multiplets = 0
    number_lost_times = 0

    def process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated):
        nonlocal number_lost_times
        nonlocal counter
        delta_t_group, n_group, t50_group, tVar_group, distance_group, zenith_group, signal_group, group_group, delta_s_group = [], [], [], [], [], [], [], [], []


        for i in range(1, len(t_0)):
            if tPFSec[0] == tPFSec[i] == t_0[0] == t_0[i] and saturated[0] == False and saturated[i] == False:


# the smaller id is allways the first


                if int(id[0]) < int(id[i]):
                    # if (int(id[0]) == 72 and int(id[i]) == 688) or (int(id[0]) == 75 and int(id[i]) == 698) or (int(id[0]) == 76 and int(id[i]) == 698) or (int(id[0]) == 79 and int(id[i]) == 819) or (int(id[0]) == 81 and int(id[i]) == 819) or (int(id[0]) == 86 and int(id[i]) == 88) or (int(id[0]) == 88 and int(id[i]) == 660) or (int(id[0]) == 93 and int(id[i]) == 94) or (int(id[0]) == 94 and int(id[i]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000:# and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_i - delta_T_j)
                        delta_s_group.append(signal[0] - signal[i])
                        # delta_s_group.append(np.log(signal[0]) - np.log(signal[i]))
                        n_group.append(np.array([n[0], n[i]]))  
                        # n_group.append(np.array([max(n[0], 2), max(n[i], 2)]))  
                        # meanT50 = mean_t50(np.cos(zenith), distance[0])
                        # if meanT50 > t50[0]:
                        #     counter += 1
                        # if meanT50 > t50[i]:
                        #     counter += 1    
                        # if meanT50 > t50[0] and n[0] < 2 and meanT50 < 1000:
                        #     t501 = meanT50
                        # else:
                        #     t501 = t50[0]
                        # if meanT50 > t50[i] and n[i] < 2 and meanT50 < 1000:
                        #     t502 = meanT50
                        # else:
                        #     t502 = t50[i]            
                        t50_group.append(np.array([t50[0], t50[i]]))
                        tVar_group.append(np.array([tVar[0], tVar[i]]))
                        distance_group.append(np.array([distance[0], distance[i]]))
                        zenith_group.append(zenith)
                        # signal_group.append(np.array([np.log(signal[0]), np.log(signal[i])]))
                        signal_group.append(np.array([signal[i], signal[0]]))
                        group_group.append(np.array([id[0], id[i]]))

                elif int(id[0]) > int(id[i]):
                    # if (int(id[i]) == 72 and int(id[0]) == 688) or (int(id[i]) == 75 and int(id[0]) == 698) or (int(id[i]) == 76 and int(id[0]) == 698) or (int(id[i]) == 79 and int(id[0]) == 819) or (int(id[i]) == 81 and int(id[0]) == 819) or (int(id[i]) == 86 and int(id[0]) == 88) or (int(id[i]) == 88 and int(id[0]) == 660) or (int(id[i]) == 93 and int(id[0]) == 94) or (int(id[i]) == 94 and int(id[0]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000:# and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_j - delta_T_i)
                        delta_s_group.append(signal[i] - signal[0])
                        # delta_s_group.append(np.log(signal[i]) - np.log(signal[0]))
                        n_group.append(np.array([n[i], n[0]]))    
                        # meanT50 = mean_t50(np.cos(zenith), distance[0])
                        # if meanT50 > t50[0]:
                        #     counter += 1
                        # if meanT50 > t50[i]:
                        #     counter += 1    
                        # if meanT50 > t50[0] and n[0] < 2 and meanT50 < 1000:
                        #     t501 = meanT50
                        # else:
                        #     t501 = t50[0]
                        # if meanT50 > t50[i] and n[i] < 2 and meanT50 < 1000:
                        #     t502 = meanT50
                        # else:
                        #     t502 = t50[i]            
                        t50_group.append(np.array([t50[i], t50[0]]))
                        tVar_group.append(np.array([tVar[i], tVar[0]]))
                        distance_group.append(np.array([distance[i], distance[0]]))
                        zenith_group.append(zenith)
                        signal_group.append(np.array([signal[i], signal[0]]))
                        # signal_group.append(np.array([np.log(signal[i]), np.log(signal[0])]))
                        group_group.append(np.array([id[i], id[0]]))

                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
            else:
                number_lost_times += 1    


        delta_t_values.extend(delta_t_group)
        delta_s_values.extend(delta_s_group)
        n_values.extend(n_group)
        t50_values.extend(t50_group)
        tVar_values.extend(tVar_group)
        distance_values.extend(distance_group)
        zenith_values.extend(zenith_group)
        signal_values.extend(signal_group)
        group_values.extend(group_group)

    for auger_id, details in data.items():
        aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

        for group_id, stations in details['group_id'].items():
            t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated = [], [], [], [], [], [], [], [], [], details["zenith"], [], [], [], []
            group_ids.append(int(group_id)) 
            for station in stations:
                t_0.append(station["timeSecond"])
                t_0Nano.append(station["timeNSecond"])
                x.append(np.array([station["stationCoorX"], station["stationCoorY"], station["stationCoorZ"]]))
                t50.append(station["time50"])
                n.append(station["n"])
                tPFSec.append(station["coreTimeSec"])
                tPFSimple.append(station["timeSimple"])
                tVar.append(station["timeError"])
                distance.append(station["distance"])
                signal.append(station["signal"])
                id.append(station["station_id"])
                saturated.append(station["saturated"])

            num_stations = len(t_0)
            if num_stations >= 2:
                    process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated)
            else:
                number_lost_multiplets += 1
                print(f"Ungültige Multiplet-Größe: {num_stations}")
 
    print(f"lost multiplets: {number_lost_multiplets}")
    print(f"lost times  : {number_lost_times}")
    # plt.hist(delta_t_values, bins=250)
    # plt.show()

    return delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values


def delta_t_trigger(data, cut, triggertype):
    counter = 0
    delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, delta_s_values, trigger_values = [], [], [], [], [], [], [], [], [], []
    number_lost_multiplets = 0
    number_lost_times = 0

    def process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, trigger):
        nonlocal number_lost_times
        nonlocal counter
        delta_t_group, n_group, t50_group, tVar_group, distance_group, zenith_group, signal_group, group_group, delta_s_group, trigger_group = [], [], [], [], [], [], [], [], [], []


        for i in range(1, len(t_0)):
            if tPFSec[0] == tPFSec[i] == t_0[0] == t_0[i] and triggertype == trigger[0] == trigger[i]:


# the smaller id is allways the first


                if int(id[0]) < int(id[i]):
                    # if (int(id[0]) == 72 and int(id[i]) == 688) or (int(id[0]) == 75 and int(id[i]) == 698) or (int(id[0]) == 76 and int(id[i]) == 698) or (int(id[0]) == 79 and int(id[i]) == 819) or (int(id[0]) == 81 and int(id[i]) == 819) or (int(id[0]) == 86 and int(id[i]) == 88) or (int(id[0]) == 88 and int(id[i]) == 660) or (int(id[0]) == 93 and int(id[i]) == 94) or (int(id[0]) == 94 and int(id[i]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000 and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_i - delta_T_j)
                        delta_s_group.append(signal[0] - signal[i])
                        # delta_s_group.append(np.log(signal[0]) - np.log(signal[i]))
                        n_group.append(np.array([n[0], n[i]]))  
                        # n_group.append(np.array([max(n[0], 2), max(n[i], 2)]))  
                        # meanT50 = mean_t50(np.cos(zenith), distance[0])
                        # if meanT50 > t50[0]:
                        #     counter += 1
                        # if meanT50 > t50[i]:
                        #     counter += 1    
                        # if meanT50 > t50[0] and n[0] < 2 and meanT50 < 1000:
                        #     t501 = meanT50
                        # else:
                        #     t501 = t50[0]
                        # if meanT50 > t50[i] and n[i] < 2 and meanT50 < 1000:
                        #     t502 = meanT50
                        # else:
                        #     t502 = t50[i]            
                        t50_group.append(np.array([t50[0], t50[i]]))
                        tVar_group.append(np.array([tVar[0], tVar[i]]))
                        distance_group.append(np.array([distance[0], distance[i]]))
                        zenith_group.append(zenith)
                        signal_group.append(np.array([signal[0], signal[i]]))
                        group_group.append(np.array([id[0], id[i]]))
                        trigger_group.append(np.array([trigger[0], trigger[i]]))

                elif int(id[0]) > int(id[i]):
                    # if (int(id[i]) == 72 and int(id[0]) == 688) or (int(id[i]) == 75 and int(id[0]) == 698) or (int(id[i]) == 76 and int(id[0]) == 698) or (int(id[i]) == 79 and int(id[0]) == 819) or (int(id[i]) == 81 and int(id[0]) == 819) or (int(id[i]) == 86 and int(id[0]) == 88) or (int(id[i]) == 88 and int(id[0]) == 660) or (int(id[i]) == 93 and int(id[0]) == 94) or (int(id[i]) == 94 and int(id[0]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000 and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_j - delta_T_i)
                        delta_s_group.append(signal[i] - signal[0])
                        # delta_s_group.append(np.log(signal[i]) - np.log(signal[0]))
                        n_group.append(np.array([n[i], n[0]]))    
                        # meanT50 = mean_t50(np.cos(zenith), distance[0])
                        # if meanT50 > t50[0]:
                        #     counter += 1
                        # if meanT50 > t50[i]:
                        #     counter += 1    
                        # if meanT50 > t50[0] and n[0] < 2 and meanT50 < 1000:
                        #     t501 = meanT50
                        # else:
                        #     t501 = t50[0]
                        # if meanT50 > t50[i] and n[i] < 2 and meanT50 < 1000:
                        #     t502 = meanT50
                        # else:
                        #     t502 = t50[i]            
                        t50_group.append(np.array([t50[i], t50[0]]))
                        tVar_group.append(np.array([tVar[i], tVar[0]]))
                        distance_group.append(np.array([distance[i], distance[0]]))
                        zenith_group.append(zenith)
                        signal_group.append(np.array([signal[i], signal[0]]))
                        group_group.append(np.array([id[i], id[0]]))
                        trigger_group.append(np.array([trigger[i], trigger[0]]))

                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
            else:
                number_lost_times += 1    


        delta_t_values.extend(delta_t_group)
        delta_s_values.extend(delta_s_group)
        n_values.extend(n_group)
        t50_values.extend(t50_group)
        tVar_values.extend(tVar_group)
        distance_values.extend(distance_group)
        zenith_values.extend(zenith_group)
        signal_values.extend(signal_group)
        group_values.extend(group_group)
        trigger_values.extend(trigger_group)

    for auger_id, details in data.items():
        aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

        for group_id, stations in details['group_id'].items():
            t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, trigger = [], [], [], [], [], [], [], [], [], details["zenith"], [], [], [], []
            group_ids.append(int(group_id)) 
            for station in stations:
                t_0.append(station["timeSecond"])
                t_0Nano.append(station["timeNSecond"])
                x.append(np.array([station["stationCoorX"], station["stationCoorY"], station["stationCoorZ"]]))
                t50.append(station["time50"])
                n.append(station["n"])
                tPFSec.append(station["coreTimeSec"])
                tPFSimple.append(station["timeSimple"])
                tVar.append(station["timeError"])
                distance.append(station["distance"])
                signal.append(station["signal"])
                id.append(station["station_id"])
                trigger.append(station["trigger"])


            num_stations = len(t_0)
            if num_stations >= 2:
                    process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, trigger)
            else:
                number_lost_multiplets += 1
                print(f"Ungültige Multiplet-Größe: {num_stations}")
 
    print(f"lost multiplets: {number_lost_multiplets}")
    print(f"lost times  : {number_lost_times}")
    # plt.hist(delta_t_values, bins=250)
    # plt.show()

    return delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values, trigger_values




def delta_t_naiv_ln(data, cut):
    counter = 0
    delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, delta_s_values = [], [], [], [], [], [], [], [], []
    number_lost_multiplets = 0
    number_lost_times = 0

    def process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated):
        nonlocal number_lost_times
        nonlocal counter
        delta_t_group, n_group, t50_group, tVar_group, distance_group, zenith_group, signal_group, group_group, delta_s_group = [], [], [], [], [], [], [], [], []


        for i in range(1, len(t_0)):
            if tPFSec[0] == tPFSec[i] == t_0[0] == t_0[i] and saturated[0] == False and saturated[i] == False:


# the smaller id is allways the first


                if int(id[0]) < int(id[i]):
                    # if (int(id[0]) == 72 and int(id[i]) == 688) or (int(id[0]) == 75 and int(id[i]) == 698) or (int(id[0]) == 76 and int(id[i]) == 698) or (int(id[0]) == 79 and int(id[i]) == 819) or (int(id[0]) == 81 and int(id[i]) == 819) or (int(id[0]) == 86 and int(id[i]) == 88) or (int(id[0]) == 88 and int(id[i]) == 660) or (int(id[0]) == 93 and int(id[i]) == 94) or (int(id[0]) == 94 and int(id[i]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000 and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_i - delta_T_j)
                        # delta_s_group.append(signal[0] - signal[i])
                        delta_s_group.append(np.log(signal[0]) - np.log(signal[i]))
                        n_group.append(np.array([n[0], n[i]]))  
                        # n_group.append(np.array([max(n[0], 2), max(n[i], 2)]))  
                        # meanT50 = mean_t50(np.cos(zenith), distance[0])
                        # if meanT50 > t50[0]:
                        #     counter += 1
                        # if meanT50 > t50[i]:
                        #     counter += 1    
                        # if meanT50 > t50[0] and n[0] < 2 and meanT50 < 1000:
                        #     t501 = meanT50
                        # else:
                        #     t501 = t50[0]
                        # if meanT50 > t50[i] and n[i] < 2 and meanT50 < 1000:
                        #     t502 = meanT50
                        # else:
                        #     t502 = t50[i]            
                        t50_group.append(np.array([t50[0], t50[i]]))
                        tVar_group.append(np.array([tVar[0], tVar[i]]))
                        distance_group.append(np.array([distance[0], distance[i]]))
                        zenith_group.append(zenith)
                        signal_group.append(np.array([np.log(signal[0]), np.log(signal[i])]))
                        # signal_group.append(np.array([signal[i], signal[0]]))
                        group_group.append(np.array([id[0], id[i]]))

                elif int(id[0]) > int(id[i]):
                    # if (int(id[i]) == 72 and int(id[0]) == 688) or (int(id[i]) == 75 and int(id[0]) == 698) or (int(id[i]) == 76 and int(id[0]) == 698) or (int(id[i]) == 79 and int(id[0]) == 819) or (int(id[i]) == 81 and int(id[0]) == 819) or (int(id[i]) == 86 and int(id[0]) == 88) or (int(id[i]) == 88 and int(id[0]) == 660) or (int(id[i]) == 93 and int(id[0]) == 94) or (int(id[i]) == 94 and int(id[0]) == 710):
                    #     a = 1
                    # else:    
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < cut and distance[0] < 5000 and t50[0] < 1000 and t50[i] < 1000 and t_0[i] > 1072352632.0:
                        delta_t_group.append(delta_T_j - delta_T_i)
                        # delta_s_group.append(signal[i] - signal[0])
                        delta_s_group.append(np.log(signal[i]) - np.log(signal[0]))
                        n_group.append(np.array([n[i], n[0]]))    
                        # meanT50 = mean_t50(np.cos(zenith), distance[0])
                        # if meanT50 > t50[0]:
                        #     counter += 1
                        # if meanT50 > t50[i]:
                        #     counter += 1    
                        # if meanT50 > t50[0] and n[0] < 2 and meanT50 < 1000:
                        #     t501 = meanT50
                        # else:
                        #     t501 = t50[0]
                        # if meanT50 > t50[i] and n[i] < 2 and meanT50 < 1000:
                        #     t502 = meanT50
                        # else:
                        #     t502 = t50[i]            
                        t50_group.append(np.array([t50[i], t50[0]]))
                        tVar_group.append(np.array([tVar[i], tVar[0]]))
                        distance_group.append(np.array([distance[i], distance[0]]))
                        zenith_group.append(zenith)
                        # signal_group.append(np.array([signal[i], signal[0]]))
                        signal_group.append(np.array([np.log(signal[i]), np.log(signal[0])]))
                        group_group.append(np.array([id[i], id[0]]))

                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
            else:
                number_lost_times += 1    


        delta_t_values.extend(delta_t_group)
        delta_s_values.extend(delta_s_group)
        n_values.extend(n_group)
        t50_values.extend(t50_group)
        tVar_values.extend(tVar_group)
        distance_values.extend(distance_group)
        zenith_values.extend(zenith_group)
        signal_values.extend(signal_group)
        group_values.extend(group_group)

    for auger_id, details in data.items():
        aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

        for group_id, stations in details['group_id'].items():
            t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated = [], [], [], [], [], [], [], [], [], details["zenith"], [], [], [], []
            group_ids.append(int(group_id)) 
            for station in stations:
                t_0.append(station["timeSecond"])
                t_0Nano.append(station["timeNSecond"])
                x.append(np.array([station["stationCoorX"], station["stationCoorY"], station["stationCoorZ"]]))
                t50.append(station["time50"])
                n.append(station["n"])
                tPFSec.append(station["coreTimeSec"])
                tPFSimple.append(station["timeSimple"])
                tVar.append(station["timeError"])
                distance.append(station["distance"])
                signal.append(station["signal"])
                id.append(station["station_id"])
                saturated.append(station["saturated"])

            num_stations = len(t_0)
            if num_stations >= 2:
                    process_station_group(t_0, t_0Nano, x, t50, n, tPFSec, tPFSimple, tVar, distance, zenith, signal, id, group_ids, saturated)
            else:
                number_lost_multiplets += 1
                print(f"Ungültige Multiplet-Größe: {num_stations}")
 
    print(f"lost multiplets: {number_lost_multiplets}")
    print(f"lost times  : {number_lost_times}")
    # plt.hist(delta_t_values, bins=250)
    # plt.show()

    return delta_t_values, n_values, t50_values, tVar_values, distance_values, zenith_values, signal_values, group_values, counter, delta_s_values


def commissioned(data):

    times = {
        (49, 64): [], # 1
        (139, 186): [], # 2
        (140, 185): [], # 3
        (72, 688): [], # 4
        (73, 695): [], # 5
        (77, 707): [], # 6
        (78, 824): [], # 7
        (80, 669): [], # 8
        (82, 657): [], # 9
        (83, 736): [], # 10
        (87, 663): [], # 11
        (89, 734): [], # 12
        (90, 651): [], # 13
        (91, 643): [], # 14
        (92, 635): [], # 15
        (75, 76): [], # 16
        (75, 698): [], # 16
        (71, 74): [], # 17
        (71, 713): [], # 17
        (79, 81): [], # 18
        (79, 819): [], # 18
        (84, 85): [], # 19
        (84, 664): [], # 19
        (86, 88): [], # 20
        (86, 660): [], # 20
        (93, 94): [], # 21
        (93, 710): [], # 21
        (95, 96): [], # 22
        (95, 918): [], # 22
        (607, 1847): [], # 23
        (56, 59): [], # 24
        (56, 1739): [], # 24
        (60, 1733): [], # 25
        (1764, 20): [], # 26
        (1764, 22): [], # 26
        (1764, 25): [], # 26
        (1764, 39): [], # 26
        (1764, 41): [], # 26
        (20, 22): [], # 26
        (20, 25): [], # 26
        (20, 39): [], # 26
        (20, 41): [], # 26
        (22, 25): [], # 26
        (22, 39): [], # 26
        (22, 41): [], # 26
        (25, 39): [], # 26
        (25, 41): [], # 26
        (39, 41): [], # 26
    }



    def process_station_group(t_0, t_0Nano, tPFSec, tPFSimple, ids):
        for i in range(1, len(t_0)):
            if tPFSec[0] == tPFSec[i] == t_0[0] == t_0[i]:  
                id_pair = tuple(sorted((int(ids[0]), int(ids[i]))))

                if id_pair in times: 
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = abs(delta_T_i - delta_T_j)

                    if delta < 1000:  
                        times[id_pair].append(t_0[0])
                else: 
                    print(ids[0], ids[1])

    for auger_id, details in data.items():
        aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

        for group_id, stations in details['group_id'].items():
            t_0, t_0Nano, tPFSec, tPFSimple, id = [], [], [], [], []
            
            for station in stations:
                t_0.append(station["timeSecond"])
                t_0Nano.append(station["timeNSecond"])
                tPFSec.append(station["coreTimeSec"])
                tPFSimple.append(station["timeSimple"])
                id.append(station["station_id"])


            num_stations = len(t_0)
            if num_stations >= 2:
                process_station_group(t_0, t_0Nano, tPFSec, tPFSimple, id)
                 
    return times




# def problem(data):
#     groups = {
#         (49, 64): [], # 1
#         (139, 186): [], # 2
#         (140, 185): [], # 3
#         (72, 688): [], # 4
#         (73, 695): [], # 5
#         (77, 707): [], # 6
#         (78, 824): [], # 7
#         (80, 669): [], # 8
#         (82, 657): [], # 9
#         (83, 736): [], # 10
#         (87, 663): [], # 11
#         (89, 734): [], # 12
#         (90, 651): [], # 13
#         (91, 643): [], # 14
#         (92, 635): [], # 15
#         (75, 76): [], # 16
#         (75, 698): [], # 16
#         (76, 698): [], # 16
#         (71, 74): [], # 17
#         (71, 713): [], # 17
#         (74, 713): [], # 17
#         (79, 81): [], # 18
#         (79, 819): [], # 18
#         (81, 819): [], # 18
#         (84, 85): [], # 19
#         (84, 664): [], # 19
#         (85, 664): [], # 19
#         (86, 88): [], # 20
#         (86, 660): [], # 20
#         (88, 660): [], # 20
#         (93, 94): [], # 21
#         (93, 710): [], # 21
#         (94, 710): [], # 21
#         (95, 96): [], # 22
#         (95, 918): [], # 22
#         (96, 918): [], # 22
#         (607, 1847): [] # 23
#     }

#     def process_station_group(t_0, t_0Nano, tPFSec, tPFSimple, ids):
#         for i in range(1, len(t_0)):
#             if tPFSec[0] == tPFSec[i] == t_0[0] == t_0[i]:  
#                 id_pair = tuple(sorted((int(ids[0]), int(ids[i]))))

#                 if id_pair in groups: 
#                     delta_T_i = t_0Nano[0] - tPFSimple[0]
#                     delta_T_j = t_0Nano[i] - tPFSimple[i]
#                     delta = abs(delta_T_i - delta_T_j)

#                     if delta < 1000:  
#                         groups[id_pair].append(delta_T_i - delta_T_j if ids[0] < ids[i] else delta_T_j - delta_T_i)

#                 else: 
#                     print(ids[0], ids[1])

#     for auger_id, details in data.items():
#         aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

#         for group_id, stations in details['group_id'].items():
#             t_0, t_0Nano, tPFSec, tPFSimple, id = [], [], [], [], []
            
#             for station in stations:
#                 t_0.append(station["timeSecond"])
#                 t_0Nano.append(station["timeNSecond"])
#                 tPFSec.append(station["coreTimeSec"])
#                 tPFSimple.append(station["timeSimple"])
#                 id.append(station["station_id"])

#             num_stations = len(t_0)
#             if num_stations >= 2:
#                 process_station_group(t_0, t_0Nano, tPFSec, tPFSimple, id)


#     # plt.hist(delta_t_values, bins = 200)
#     # plt.xlim(-1000, 1000)
#     # plt.xlabel(f"$\Delta T$ in ns")
#     # plt.ylabel(f"counts")
#     # plt.title(f'$\Delta T$ for every pair')
#     # plt.show()


#     # unique_values, counts = np.unique(error_group, return_counts=True)

#     # x_positions = np.arange(len(unique_values))

#     # plt.figure(figsize=(10, 6))
#     # plt.bar(x_positions, counts, color='skyblue', edgecolor='black')
#     # plt.xlabel('Stationen (IDs)', fontsize=12)
#     # plt.ylabel('Häufigkeit', fontsize=12)
#     # plt.title('Häufigkeit der Stationen in error_group', fontsize=14)

#     # # Ändere die xticks, damit die IDs der Stationen angezeigt werden
#     # plt.xticks(x_positions, unique_values, rotation=45)

#     # plt.grid(axis='y', linestyle='--', alpha=0.7)
#     # plt.tight_layout()
#     # plt.show()


#     # plt.hist(group_72_688, bins = 200, color='blue', alpha=0.5, label='4')
#     # plt.xlabel(f"$\Delta$ T in ns")
#     # plt.ylabel(f"counts")
#     # plt.title("pair with stations 72 and 688 (group 4)")
#     # plt.show()
#     # plt.hist(group_75_76, bins = 200, color='blue', alpha=0.5, label='16')
#     # plt.title("75_76")
#     # plt.show()
#     # plt.hist(group_75_698, bins = 200, color='blue', alpha=0.5, label='18')
#     # plt.title("75_698")
#     # plt.show()
#     # plt.hist(group_76_698, bins = 200, color='blue', alpha=0.5, label='20')
#     # plt.title("76_698")
#     # plt.show()
#     # plt.hist(group_79_81, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('79_81')
#     # plt.show()
#     # plt.hist(group_79_819, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('79_819')
#     # plt.show()
#     # plt.hist(group_81_819, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('81_819')
#     # plt.show()
#     # plt.hist(group_86_88, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('86_88')
#     # plt.show()
#     # plt.hist(group_86_660, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('86_660')
#     # plt.show()
#     # plt.hist(group_88_660, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('88_660')
#     # plt.show()
#     # plt.hist(group_93_94, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('93_94')
#     # plt.show()
#     # plt.hist(group_93_710, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('93_710')
#     # plt.show()     
#     # plt.hist(group_94_710, bins = 200, color='blue', alpha=0.5, label='21')
#     # plt.title('94_710')
#     # plt.show()                   
#     return groups    