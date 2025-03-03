#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import Counter
from my_functions import read_in, calculate_and_plot_delta_t, commissioned

events = read_in("data_data/events6t5.txt")
# calculate_and_plot_delta_t(events)


print("Problem:\n\n")

times = commissioned(events)

def gps_to_year_month(gps_seconds):
    gps_epoch = datetime(1980, 1, 6)  # GPS-Epoch-Beginn
    utc_time = gps_epoch + timedelta(seconds=gps_seconds)
    return utc_time.year, utc_time.month


def generate_full_year_month_range(start_year, start_month, end_year, end_month):
    months = []
    current_year, current_month = start_year, start_month

    while (current_year, current_month) <= (end_year, end_month):
        months.append(f"{current_year}-{current_month:02d}")
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1

    return months


def plot_combined_heatmap(times, colormap):
    all_time_points = []
    for time_list in times.values():
        for t in time_list:
            all_time_points.append(gps_to_year_month(t))
    
    min_year, min_month = min(all_time_points)
    max_year, max_month = max(all_time_points)
    full_range = generate_full_year_month_range(min_year, min_month, max_year, max_month)

    # Extract station pairs and group IDs
    pair_indices = {}
    group_ids = []  # Store corresponding group IDs

    for idx, (pair, group_id) in enumerate(zip(times.keys(), [
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,24,24,25,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26
    ])):  
        pair_indices[pair] = idx
        group_ids.append(group_id)  # Store group ID for each pair

    heatmap_data = np.full((len(pair_indices), len(full_range)), np.nan)

    for id_pair, time_list in times.items():
        year_month_list = [gps_to_year_month(t) for t in time_list]
        year_month_str = [f"{year}-{month:02d}" for year, month in year_month_list]
        counts = Counter(year_month_str)

        for i, month in enumerate(full_range):
            if month in counts:
                heatmap_data[pair_indices[id_pair], i] = counts[month]

    # Prepare ticks for the X-axis (only show years)
    year_ticks = []
    year_labels = []
    for i, date in enumerate(full_range):
        year, month = map(int, date.split('-'))
        if month == 1:  # Only add a label at the start of each year
            year_ticks.append(i)
            year_labels.append(str(year))

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.set_facecolor('darkgrey')
    plt.imshow(heatmap_data, aspect='auto', cmap=colormap, interpolation='none')

    for row_idx in range(heatmap_data.shape[0]):
        for col_idx in range(heatmap_data.shape[1]):
            if np.isnan(heatmap_data[row_idx, col_idx]):
                plt.plot(col_idx, row_idx, marker='x', color='black', markersize=3, linestyle='None')

    plt.colorbar(label="Frequency")

    # Update y-axis labels to include group ID
    plt.yticks(range(len(pair_indices)), 
               [f"{pair}, {group_id}" for pair, group_id in zip(pair_indices.keys(), group_ids)], 
               fontsize=10)

    plt.xticks(year_ticks, year_labels, rotation=90, fontsize=10)  # Only show years
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Station IDs and Group", fontsize=12)
    plt.tight_layout()
    plt.savefig("pairsovertime.pdf", format="pdf", bbox_inches="tight")
    plt.show()


plot_combined_heatmap(times, 'inferno')



# def gps_to_year_month(gps_seconds):
#     gps_epoch = datetime(1980, 1, 6)  # GPS-Epoch-Beginn
#     utc_time = gps_epoch + timedelta(seconds=gps_seconds)
#     return utc_time.year, utc_time.month

# def generate_full_year_month_range(start_year, start_month, end_year, end_month):
#     months = []
#     current_year, current_month = start_year, start_month

#     while (current_year, current_month) <= (end_year, end_month):
#         months.append(f"{current_year}-{current_month:02d}")
#         if current_month == 12:
#             current_year += 1
#             current_month = 1
#         else:
#             current_month += 1

#     return months

# def plot_histograms(times):
#     for id_pair, time_list in times.items():
#         if time_list: 
#             year_month_list = [gps_to_year_month(t) for t in time_list]
#             year_month_str = [f"{year}-{month:02d}" for year, month in year_month_list]
#             counts = Counter(year_month_str)
#             min_year, min_month = min(year_month_list)
#             max_year, max_month = max(year_month_list)
#             full_range = generate_full_year_month_range(min_year, min_month, max_year, max_month)
#             full_counts = [counts.get(month, 0) for month in full_range]
#             plt.figure(figsize=(12, 6))
#             plt.bar(full_range, full_counts, color='blue', alpha=0.7, edgecolor='black')
#             plt.title(f"Histogram for Pair {id_pair}", fontsize=16)
#             plt.xlabel("Year-Month", fontsize=14)
#             plt.ylabel("Frequency", fontsize=14)
#             plt.xticks(rotation=45, fontsize=10)
#             plt.grid(True, linestyle='--', alpha=0.6)
#             plt.show()  
# plot_histograms(times)



print("\n\n now s over ct: \n\n")
def delta_t(data, group):
    ct_values_4, signal_values_4 = [], []
    def process_station_group(t_0Nano, tPFSimple, signal, id):
        t_group, signal_group = [], []


        for i in range(1, len(t_0Nano)):


# the smaller id is allways the first


                if int(id[0]) < int(id[i]):
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < 5000:
                        if int(id[0]) == group:
                            ct_values_4.append((2.9979 / 10) * (delta_T_i - delta_T_j))
                            signal_values_4.append(np.mean(np.array([signal[0], signal[i]])))     

                elif int(id[0]) > int(id[i]):
                    delta_T_i = t_0Nano[0] - tPFSimple[0]
                    delta_T_j = t_0Nano[i] - tPFSimple[i]
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < 5000:
                        if int(id[i]) == group:
                            ct_values_4.append((2.9979 / 10) * (delta_T_j - delta_T_i))
                            signal_values_4.append(np.mean(np.array([signal[i], signal[0]])))        

    for auger_id, details in data.items():
        aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

        for group_id, stations in details['group_id'].items():
            t_0Nano, tPFSimple, signal, id = [], [], [], []
            
            for station in stations:
                t_0Nano.append(station["timeNSecond"])
                tPFSimple.append(station["timeSimple"])
                signal.append(station["signal"])
                id.append(station["station_id"])

            num_stations = len(t_0Nano)
            if num_stations >= 2:
                if num_stations == 2:
                    process_station_group(t_0Nano, tPFSimple, signal, id)
                else:
                    process_station_group(t_0Nano, tPFSimple, signal, id)
    return ct_values_4, signal_values_4



# Vorsichtig sein!!!!!! Nur schnell geschrieben, die notierung ist verwirrend
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# delta_s ist nur vorrübergehend

def delta_s(data, group):
    ct_values_4, signal_values_4 = [], []
    def process_station_group(t_0Nano, tPFSimple, signal, id):
        t_group, signal_group = [], []


        for i in range(1, len(t_0Nano)):


# the smaller id is allways the first


                if int(id[0]) < int(id[i]):
                    delta_T_i = np.log(signal[0])
                    delta_T_j = np.log(signal[i])
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < 5000:
                        if int(id[0]) == group:
                            ct_values_4.append((delta_T_i - delta_T_j))
                            signal_values_4.append(np.mean(np.array([signal[0], signal[i]])))     

                elif int(id[0]) > int(id[i]):
                    delta_T_i = np.log(signal[0])
                    delta_T_j = np.log(signal[i])
                    delta = np.abs(delta_T_i - delta_T_j)
                    if delta < 5000:
                        if int(id[i]) == group:
                            ct_values_4.append((delta_T_j - delta_T_i))
                            signal_values_4.append(np.mean(np.array([signal[i], signal[0]])))        

    for auger_id, details in data.items():
        aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])

        for group_id, stations in details['group_id'].items():
            t_0Nano, tPFSimple, signal, id = [], [], [], []
            
            for station in stations:
                t_0Nano.append(station["timeNSecond"])
                tPFSimple.append(station["timeSimple"])
                signal.append(station["signal"])
                id.append(station["station_id"])

            num_stations = len(t_0Nano)
            if num_stations >= 2:
                if num_stations == 2:
                    process_station_group(t_0Nano, tPFSimple, signal, id)
                else:
                    process_station_group(t_0Nano, tPFSimple, signal, id)
    return ct_values_4, signal_values_4


# Beispielgruppen
groups = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 139, 140]

# Anzahl der Plots pro Zeile und Gesamtanzahl der Gruppen
n_cols = 7  # Anzahl der Spalten
n_rows = -(-len(groups) // n_cols)  # Berechnet die Zeilenanzahl (aufgerundet)

# Erstelle die Subplots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 15), constrained_layout=True)  # Skaliere entsprechend
axs = axs.flatten()  # Flache Subplot-Achsenliste für einfachen Zugriff

# Schleife über alle Gruppen
for idx, group in enumerate(groups):
    # Beispiel-Daten (ersetze durch deine Funktion)
    ct_values, signal_values = delta_s(events, group)

    ct_values = np.array(ct_values)
    signal_values = np.array(signal_values)

    # Maske für den CT-Wertebereich
    mask = (ct_values >= -100) & (ct_values <= 100)
    ct_values = ct_values[mask]
    signal_values = signal_values[mask]

    # Streudiagramm für die aktuelle Gruppe
    axs[idx].scatter(ct_values, signal_values, color='blue', alpha=0.7, edgecolor='k', s=50)

    # Logarithmische Y-Achse
    axs[idx].set_yscale('log')

    # Achsenbeschriftungen und Titel
    axs[idx].set_xlabel('CT Values', fontsize=10)
    axs[idx].set_ylabel('Signal (log scale)', fontsize=10)
    axs[idx].set_title(f'Group {group}', fontsize=12)

    # Achsenlimits und Gitter
    # axs[idx].set_xlim(-100, 100)
    axs[idx].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Entferne leere Subplots, falls vorhanden
for ax in axs[len(groups):]:
    ax.axis('off')

# Haupttitel für die gesamte Figur
fig.suptitle('Scatter Plots: CT Values vs Signal Values for Groups')
plt.show()



# events_data = {
#     2004: read_in("events_2004.txt"),
#     2005: read_in("events_2005.txt"),
#     2006: read_in("events_2006.txt"),
#     2007: read_in("events_2007.txt"),
#     2008: read_in("events_2008.txt"),
#     2009: read_in("events_2009.txt"),
#     2010: read_in("events_2010.txt"),
#     2011: read_in("events_2011.txt"),
#     2012: read_in("events_2012.txt"),
#     2013: read_in("events_2013.txt")
# }

# group_data_by_year = {year: problem(event) for year, event in events_data.items()}

# all_group_names = set()
# for groups in group_data_by_year.values():
#     all_group_names.update(groups.keys())

# plt.figure(figsize=(12, 8))
# colors = plt.cm.tab20(np.linspace(0, 1, len(all_group_names)))  
# markers = ['o', 's', 'D', '^', 'v', '*', 'p', 'h', 'X'] 

# for idx, group_name in enumerate(sorted(all_group_names)):
#     years = []
#     means = []
#     uncertainties = []
#     alphaa = 0.3
#     colorr = "grey"
#     labell = None
#     fmtt = "o-"

#     for year, groups in group_data_by_year.items():
#         if group_name in groups and groups[group_name]:
#             years.append(year)
#             means.append(np.mean(groups[group_name]) / 25.0)
#             uncertainties.append(np.std(groups[group_name]) / (np.sqrt(len(groups[group_name])) * 25.0))

#     if means and (np.abs(max(means)) > (200 / 25.0) or min(means) < (-200.0 / 25.0)):
#         alphaa = 1
#         colorr = colors[idx]
#         labell = f"Group {group_name}"
#         fmtt = f'{markers[idx % len(markers)]}-'
#     if years: 
#         plt.errorbar(
#             years,
#             means,
#             yerr=uncertainties,
#             fmt=fmtt, 
#             capsize=4,
#             label=labell,
#             color=colorr,
#             alpha=alphaa
#         )

# plt.title("Annual Mean Values with Uncertainty for All Groups (2004-2013)", fontsize=16, fontweight='bold')
# plt.xlabel("Year", fontsize=14)
# plt.ylabel("Mean Value", fontsize=14)
# y_min = -30
# y_max = 30
# plt.yticks(range(y_min, y_max + 1, 5))
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.xticks(sorted(events_data.keys()), fontsize=12) 
# plt.yticks(fontsize=12)
# plt.legend(fontsize=10, title="Groups", loc="upper left", bbox_to_anchor=(1, 1))
# plt.tight_layout()
# plt.show()