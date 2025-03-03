import ROOT
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT.gSystem.Load("/cr/data01/ellwanger/Offline/build/inst-offline-opt/lib/libRecEventKG.so")

def open_adst_files(adst_files):
    adstvec = ROOT.std.vector('string')()
    for adst in adst_files:
        adstvec.push_back(str(adst))
    recEventFile = ROOT.RecEventFile(adstvec, False)
    recEvent = ROOT.RecEvent()
    recEventFile.SetBuffers(recEvent)
    detectorGeometry = ROOT.DetectorGeometry()
    recEventFile.ReadDetectorGeometry(detectorGeometry)
    fileInfo = ROOT.FileInfo()
    recEventFile.ReadFileInfo(fileInfo)
    return recEventFile, recEvent, detectorGeometry, fileInfo

def read_multiplet_grouplist(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    stationgrouplist = {}
    for station in root.findall('.//station'):
        group_id = station.find('groupId').text.strip()
        station_id = station.get('id')
        if group_id != str(0):
            if group_id not in stationgrouplist:
                stationgrouplist[group_id] = {"station_ids": []}
            stationgrouplist[group_id]["station_ids"].append(station_id)
    return stationgrouplist

class Station:
    def __init__(self, station_id, time, time50, stationPosition, distance, residual, signal, estimatedLDF, zenith):
        self.station_id = station_id
        self.time = time                                  
        self.time50 = time50  
        self.stationPosition = stationPosition
        self.distance = distance
        self.residual = residual
        self.signal = signal
        self.estimatedLDF = estimatedLDF            
        self.residualVsCalculated = ((signal[0] - estimatedLDF) / signal[1]) - residual
        self.effectiveNumberOfParticles = signal[0] * (np.cos(zenith[0]) + 0.424 * np.sin(zenith[0]))
        #self.timePlaneFrontSecond = ((coreTime.timeSecond * 299792458) - np.dot(aVectorInCartesian, (stationPosition - corePosition))) / 299792458
        #self.timePlaneFrontNSecond = ((coreTime.timeNSecond * 299792458 * (10**(-9))) - np.dot(aVectorInCartesian, (stationPosition - corePosition))) / (299792458 * 10**(-9))
        #self.timeSimplePFNano = (( - np.dot(aVectorInCartesian, (stationPosition - corePosition))) / 299792458) * 10**9


class StationGroup:
    def __init__(self, group_id):
        self.group_id = group_id
        self.stations = {}

    def add_station(self, station_id, time, time50, stationPosition, distance, residual, signal, estimatedLDF, zenith):
        if station_id not in self.stations:
            self.stations[station_id] = Station(
                station_id, time, time50, stationPosition, distance, residual,
                signal, estimatedLDF, zenith
            )

class Event:
    def __init__(self, auger_id, azimuth=None, zenith=None, energy=None, aVectorInCartesian=None):
        self.auger_id = auger_id
        self.azimuth = azimuth
        self.zenith = zenith
        self.energy = energy
        self.aVectorInCartesian = aVectorInCartesian             
        self.station_groups = {}

    def add_station_to_group(self, group_id, station_id, time, time50, stationPosition, distance, residual, signal, estimatedLDF, zenith):
        if group_id not in self.station_groups:
            self.station_groups[group_id] = StationGroup(group_id)
        self.station_groups[group_id].add_station(
            station_id, time, time50, stationPosition, distance, residual,
            signal, estimatedLDF, zenith
        )

    def has_multiple_stations(self):
        return any(len(group.stations) > 1 for group in self.station_groups.values())

    def get_groups_with_multiple_stations(self):
        return {group_id: group for group_id, group in self.station_groups.items() if len(group.stations) > 1}

class MultipleHitList:
    def __init__(self):
        self.events = {}
        self.pairlist = []

    def get_or_create_event(self, auger_id, azimuth=None, zenith=None, energy=None, aVectorInCartesian=None):
        if auger_id not in self.events:
            self.events[auger_id] = Event(
                auger_id, azimuth, zenith, energy, aVectorInCartesian
            )
        return self.events[auger_id]

    def process_event(
        self, auger_id, group_id, station_id, time, time50,
        stationPosition, distance, residual,
        signal, estimatedLDF, azimuth, zenith,
        aVectorInCartesian, energy
    ):
        event = self.get_or_create_event(
            auger_id, azimuth, zenith, energy, aVectorInCartesian
        )
        event.add_station_to_group(
            group_id, station_id, time, time50, stationPosition, distance,
            residual, signal, estimatedLDF, zenith
        )
        if event.has_multiple_stations() and auger_id not in self.pairlist:
            self.pairlist.append(auger_id)

    def build_structure(self):
        structure = {}
        for auger_id in self.pairlist:
            event = self.events[auger_id]
            structure[auger_id] = {
                'group_id': {},
                'azimuth': event.azimuth[0],
                'azimuthError': event.azimuth[1],
                'zenith': event.zenith[0],
                'zenithError': event.zenith[1],
                'aVecX': event.aVectorInCartesian[0],
                'aVecY': event.aVectorInCartesian[1],
                'aVecZ': event.aVectorInCartesian[2],
                'energy': event.energy[0],
                'energyError': event.energy[1],
            }
            groups_with_multiple = event.get_groups_with_multiple_stations()
            for group_id, group in groups_with_multiple.items():
                structure[auger_id]['group_id'][group_id] = [
                    {
                        "station_id                                                                                 ": station.station_id,
                        "timeSecond": station.time[0],
                        "timeNSecond": station.time[1],
                        "timeError": station.time[2],
                        "time50": station.time50[0],
                        "time50err": station.time50[1],
                        "stationCoorX": station.stationPosition[0],
                        "stationCoorY": station.stationPosition[1],
                        "stationCoorZ": station.stationPosition[2],
                        "distance": station.distance[0],
                        "distanceError": station.distance[1],
                        "signal": station.signal[0],
                        "signalError": station.signal[1],
                        "estimatedLDF": station.estimatedLDF,
                        "residual": station.residual,
                        "residualVsCalculated": station.residualVsCalculated,
                        "n": station.effectiveNumberOfParticles
                    } for station in group.stations.values()
                ]
        return structure

if __name__ == "__main__":
    filepath = 'data.xml'
    base_dir = Path("/cr/augerdata/Reconstruction/Observer/Public/icrc-2025/test3/SD_PhaseI/out/2022/05")
    adst_list = list(base_dir.glob("*.root"))
    adst_list = adst_list
    print("\n")
    recEventFile, recEvent, detectorGeometry, fileInfo = open_adst_files(adst_list)

    ldfType = fileInfo.GetLDFType()
    print(ldfType, "\n")
    grouplist = read_multiplet_grouplist(filepath)

    multiple_hit_list = MultipleHitList()

    while recEventFile.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
        auger_id = recEvent.GetAugerId()
        sdRecShower = recEvent.GetSDEvent().GetSdRecShower()
        azimuth = np.array([sdRecShower.GetAzimuth(), sdRecShower.GetAzimuthError()])
        zenith = np.array([sdRecShower.GetZenith(), sdRecShower.GetZenithError()])
        aVectorInCartesian = np.array([np.sin(zenith[0]) * np.cos(azimuth[0]), np.sin(zenith[0]) * np.sin(azimuth[0]), np.sin(zenith[0])])
        core = sdRecShower.GetCoreSiteCS()
        energy = np.array([sdRecShower.GetEnergy(), sdRecShower.GetEnergyError()])
        #corePosition = np.array([core.X(), core.Y(), core.Z()])
        #coreTime = sdRecShower.GetCoreTimeSecond()
        #coreNTime = sdRecShower.GetCoreTimeNanoSecond()

        for station in recEvent.GetSDEvent().GetStationVector():
            station_id = station.GetId()
            time = np.array([station.GetTimeSecond(), station.GetTimeNSecond(), station.GetTimeVariance()])
            time50 = np.array([station.GetTime50(), station.GetTime50RMS()])
            stationCoor = detectorGeometry.GetStationPosition(station_id)
            stationPosition = np.array([stationCoor.X(), stationCoor.Y(), stationCoor.Z()])
            distance = np.array([station.GetSPDistance(), station.GetSPDistanceError()])
            residual = station.GetLDFResidual()
            signal = np.array([station.GetTotalSignal(), station.GetTotalSignalError()])
            estimatedLDF = sdRecShower.GetLDF().Evaluate(distance[0], ldfType)




            for group_id, group_data in grouplist.items():
                if int(station_id) in map(int, group_data["station_ids"]):
                    multiple_hit_list.process_event(
                        auger_id, group_id, station_id, time, time50,
                        stationPosition, distance, residual, signal, estimatedLDF, 
                        azimuth, zenith, aVectorInCartesian, energy
                    )

    structure = multiple_hit_list.build_structure()
    print(structure)




    def delta_t(data):
        delta_t_values = []  
        n_values = []
        t50_values = []
        c = 299792458 *10**(-9)
        
        for auger_id, details in data.items():
            t_0 = []
            t_0Nano = []
            t_0Error = []
            x = []
            t50 = []
            n = []
            aVector = np.array([details["aVecX"], details["aVecY"], details["aVecZ"]])
            for group_id, stations in details['group_id'].items():  
                i = 0
                for station in stations:
                    if i == 0:
                        t_0i = station["timeSecond"]
                        t_0iNano = station["timeNSecond"]
                        t_0iError = station["timeError"]
                        xi = np.array([station["stationCoorX"], station["stationCoorY"], station["stationCoorZ"]])
                        t50i = station["time50"]
                        ni = station["n"]
                        t_0.append(t_0i)
                        t_0Nano.append(t_0iNano)
                        t_0Error.append(t_0iError)
                        x.append(xi)
                        t50.append(t50i)
                        n.append(ni)

            
            if len(t_0) == 2 and len(x) == 2:

                print("Überprüfen:\n", "t_0Nano[1]  ", t_0Nano[1], "t_0Nano[0]  ", t_0Nano[0], "aVec"   , aVector)
                delta_t_0 = t_0[1] - t_0[0]
                if delta_t_0 == 0:
                    delta_t_0Nano = t_0Nano[1] - t_0Nano[0]
                    delta_t_pf = np.dot((x[1] - x[0]), aVector) / c
                    n_values.append(n[0])
                    n_values.append(n[1])
                    t50_values.append(t50[0])
                    t50_values.append(t50[1])
                    print("Deltas:           ", delta_t_0Nano, delta_t_pf)
                    delta_t_values.append(delta_t_0Nano - delta_t_pf)

        
        return delta_t_values, n_values, t50_values

    delta_t_values, n_values, t50_values = delta_t(structure)
    #print(delta_t_values, n_values, t50_values)
    





    from scipy.optimize import minimize

    m = 0
    z = 0
    # Beispielhafte Daten
    delta_T = delta_t_values  # Array mit den Werten für ΔT_i
    n_values = n_values  # Array mit den Werten für n bei jedem t_i
    T_50_values = t50_values  # Array mit den Werten für T_50 bei jedem t_i

    # Funktion für V[t_0^{(i)}] mit den Parametern a, b, und d
    def V_t0(a, b, d, T_50, n):
        return a + b * ((T_50 + d) / (n + 1))**2 * (n / (n + 2))

    def V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j):
        return V_t0(a, b, d, T_50_i, n_i) + V_t0(a, b, d, T_50_j, n_j)

    def log_likelihood(params, delta_T, T_50_values, n_values):
        global m, z
        a, b, d = params
        log_likelihood_sum = 0

        # Ausgabe der Eingabewerte bei erstem Durchlauf
        if m == 0:
            print("Delta T     ", delta_T)
            print("T_50        ", T_50_values)
            print("n_values    ", n_values)

        for i in range(0, len(T_50_values) - 1, 2):
            T_50_i = T_50_values[i]
            T_50_j = T_50_values[i + 1]
            n_i = n_values[i]
            n_j = n_values[i + 1]

            # Berechne V[ΔT_i] als Summe von Varianzen für Paare (i, i+1)
            V_delta_T_i = V_Delta_T(a, b, d, T_50_i, T_50_j, n_i, n_j)

            # Log-Likelihood Terme
            term1 = -0.5 * np.log(2 * np.pi * V_delta_T_i)
            term2 = -0.5 * delta_T[z]**2 / V_delta_T_i
            log_likelihood_sum += term1 + term2
            z += 1  # Index für delta_T erhöhen

        # Zurücksetzen von z für den nächsten Minimierungsschritt
        m += 1
        z = 0
        return log_likelihood_sum

    # Beispielwerte für die Parameter und Schranken
    initial_params = [134.00, 2.4000, 10.000]
    bounds = [(0, None), (0, None), (None, None)]

    # Maximierung der Likelihood (Minimierung der negativen Log-Likelihood)
    result = minimize(lambda params: -log_likelihood(params, delta_T, T_50_values, n_values), initial_params, bounds=bounds)
    optimal_params = result.x

    print("Optimale Parameter:", optimal_params)










    def plot_time_difference(data):
        avg_distances = []
        time_difference = []
        for auger_id, details in data.items():
            for group_id, stations in details['group_id'].items():
                distances = []
                times = []
                timesN = []
                if group_id == str(2):
                    for station in stations:
                        time = station['timeSecond']
                        times.append(time)
                        timeN = station['timeNSecond']
                        timesN.append(timeN)
                        distances.append(station['distance'])
                    
                    if times[0] == times[1]:
                        avg_distance = np.mean(distances)
                        time_differences = np.abs(timesN[1]-timesN[0])
                        avg_distances.append(avg_distance)
                        time_difference.append(time_differences)
        plt.figure(figsize=(10, 6))
        plt.scatter(avg_distances, time_difference, alpha=0.7, color='blue')
        plt.xlabel("Gemittelte Entfernung (m)")
        plt.ylabel("Zeitliche Differenz (s)")
        plt.title("Zeitliche Varianz der Stationen in Abhängigkeit der gemittelten Entfernung")
        plt.grid(True)
        plt.show()
        print(avg_distances)
        print(time_difference)    
    #plot_time_difference(structure)





    
