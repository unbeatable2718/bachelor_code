#!/usr/bin/env python3
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
    def __init__(self, station_id, time, time50, stationPosition, zenith, aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2):
        self.station_id = station_id
        self.time = time                                  
        self.time50 = time50  
        self.stationPosition = stationPosition            
        self.effectiveNumberOfParticles = signal[0]*(np.cos(zenith[0])+(0.424*np.sin(zenith[0])))
        self.coreTime = coreTime
        self.timePlaneFront = np.array([(((coreTime[0] * 299792458) - np.dot(aVectorInCartesian, (stationPosition - corePosition))) / 299792458), (((coreTime[1] * 299792458 * (10**(-9))) - np.dot(aVectorInCartesian, (stationPosition - corePosition))) / (299792458 * 10**(-9)))])
        self.timeSimplePFNano = coreTime[1] - (( np.dot(aVectorInCartesian, (stationPosition - corePosition)))) / (299792458 * 10**(-9))
        self.corePosition = corePosition
        self.distance = distance
        self.signal = signal
        self.saturated = saturated
        self.trigger = trigger
        self.triggerBit = triggerBit
        self.triggerBit2 = triggerBit2

class StationGroup:
    def __init__(self, group_id):
        self.group_id = group_id
        self.stations = {}

    def add_station(self, station_id, time, time50, stationPosition, zenith, aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2):
        if station_id not in self.stations:
            self.stations[station_id] = Station(
                station_id, time, time50, stationPosition, zenith, aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2
            )

class Event:
    def __init__(self, auger_id, azimuth=None, zenith=None, aVectorInCartesian=None):
        self.auger_id = auger_id
        self.azimuth = azimuth
        self.zenith = zenith
        self.aVectorInCartesian = aVectorInCartesian             
        self.station_groups = {}

    def add_station_to_group(self, group_id, station_id, time, time50, stationPosition, zenith, aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2):
        if group_id not in self.station_groups:
            self.station_groups[group_id] = StationGroup(group_id)
        self.station_groups[group_id].add_station(
            station_id, time, time50, stationPosition, zenith, aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2
        )

    def has_multiple_stations(self):
        return any(len(group.stations) > 1 for group in self.station_groups.values())

    def get_groups_with_multiple_stations(self):
        return {group_id: group for group_id, group in self.station_groups.items() if len(group.stations) > 1}

class MultipleHitList:
    def __init__(self):
        self.events = {}
        self.pairlist = []

    def get_or_create_event(self, auger_id, azimuth=None, zenith=None, aVectorInCartesian=None):
        if auger_id not in self.events:
            self.events[auger_id] = Event(
                auger_id, azimuth, zenith, aVectorInCartesian
            )
        return self.events[auger_id]

    def process_event(
        self, auger_id, group_id, station_id, time, time50,
        stationPosition, azimuth, zenith,
        aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2
    ):
        event = self.get_or_create_event(
            auger_id, azimuth, zenith, aVectorInCartesian
        )
        event.add_station_to_group(
            group_id, station_id, time, time50, stationPosition, zenith, aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2
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
            }
            groups_with_multiple = event.get_groups_with_multiple_stations()
            for group_id, group in groups_with_multiple.items():
                structure[auger_id]['group_id'][group_id] = [
                    {
                        "station_id": station.station_id,
                        "timeSecond": station.time[0],
                        "timeNSecond": station.time[1],
                        "timeError": station.time[2],
                        "time50": station.time50[0],
                        "time50err": station.time50[1],
                        "stationCoorX": station.stationPosition[0],
                        "stationCoorY": station.stationPosition[1],
                        "stationCoorZ": station.stationPosition[2],
                        "coreTimeSec": station.coreTime[0],
                        "coreTimeNano": station.coreTime[1],
                        "timePFSec": station.timePlaneFront[0],
                        "timePFNano": station.timePlaneFront[1],
                        "timeSimple": station.timeSimplePFNano,
                        "n": station.effectiveNumberOfParticles,
                        "signal": station.signal[0],
                        "corePositionX": station.corePosition[0],
                        "corePositionY": station.corePosition[1],
                        "corePositionZ": station.corePosition[2],
                        "distance": station.distance[0],
                        "saturated": station.saturated,
                        "trigger": station.trigger,
                        "triggerBit": ''.join(['1' if bit else '0' for bit in station.triggerBit]),
                        "triggerBit2":''.join(['1' if bit else '0' for bit in station.triggerBit2])
                    } for station in group.stations.values()
                ]
        return structure

if __name__ == "__main__":
    filepath = 'data.xml'
    base_dir = Path("/cr/augerdata/Reconstruction/Observer/Public/icrc-2025/test4/SD_PhaseI/out/")
    years = [2004]

    adst_list = []
    for year in years:
        year_dir = base_dir / str(year)
        adst_list.extend(year_dir.rglob("*.root"))
    adst_list = ["/cr/augerdata/Reconstruction/Observer/Public/icrc-2025/test4/SD_PhaseI/out/SD_PhaseI_adst_wc_mini.root"]
    print("\n")
    recEventFile, recEvent, detectorGeometry, fileInfo = open_adst_files(adst_list)

    ldfType = fileInfo.GetLDFType()
    print(ldfType, "\n")
    grouplist = read_multiplet_grouplist(filepath)

    multiple_hit_list = MultipleHitList()

    while recEventFile.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
        sdRecShower = recEvent.GetSDEvent().GetSdRecShower()
        zenith = np.array([sdRecShower.GetZenith(), sdRecShower.GetZenithError()])
        t5 = recEvent.GetSDEvent().Is6T5()
        if zenith[0] <= np.deg2rad(60) and t5 == 1:
            auger_id = recEvent.GetEventId()
            azimuth = np.array([sdRecShower.GetAzimuth(), sdRecShower.GetAzimuthError()])
            #aVectorInCartesian = np.array([np.sin(zenith[0]) * np.cos(azimuth[0]), np.sin(zenith[0]) * np.sin(azimuth[0]), np.cos(zenith[0])])
            aVectorInCartesian = np.array([sdRecShower.GetPlaneFrontAxis().X(), sdRecShower.GetPlaneFrontAxis().Y(), sdRecShower.GetPlaneFrontAxis().Z()]) 
            core = sdRecShower.GetCoreSiteCS()
            corePosition = np.array([core.X(), core.Y(), core.Z()])
            coreTime = np.array([sdRecShower.GetCoreTimeSecond(), sdRecShower.GetCoreTimeNanoSecond()])

            for station in recEvent.GetSDEvent().GetStationVector():
                saturated = station.IsLowGainSaturated()
                trigger = station.GetStationTriggerName()
                triggerBit = np.array([station.IsMoPS(), station.IsT1Threshold(), station.IsT2Threshold(), station.IsThreshold(), station.IsToT(), station.IsTOT(), station.IsToTd(), station.IsTOTd()])
                triggerBit2 = np.array([station.IsTrigger(0), station.IsTrigger(1), station.IsTrigger(2), station.IsTrigger(5), station.IsTrigger(6)])
                station_id = station.GetId()
                time = np.array([station.GetTimeSecond(), station.GetTimeNSecond(), station.GetTimeVariance()])
                time50 = np.array([station.GetTime50(), station.GetTime50RMS()])
                stationCoor = detectorGeometry.GetStationPosition(station_id)
                stationPosition = np.array([stationCoor.X(), stationCoor.Y(), stationCoor.Z()])
                signal = np.array([station.GetTotalSignal(), station.GetTotalSignalError()])
                distance = np.array([station.GetSPDistance(), station.GetSPDistanceError()])
                for group_id, group_data in grouplist.items():
                    if int(station_id) in map(int, group_data["station_ids"]):
                        multiple_hit_list.process_event(
                            auger_id, group_id, station_id, time, time50,
                            stationPosition, 
                            azimuth, zenith, aVectorInCartesian, corePosition, coreTime, signal, distance, saturated, trigger, triggerBit, triggerBit2
                        )

    structure = multiple_hit_list.build_structure()
    print(len(structure))
    # File path where the events will be written
    output_file = Path("/cr/users/engel-j/data_data/events6t5.txt")

    # Write the events to a file
    with output_file.open("w") as f:
        for event_id, event_data in structure.items():
            f.write(f"Event ID: {event_id}\n")
            for key, value in event_data.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")  # Add a newline after each event
