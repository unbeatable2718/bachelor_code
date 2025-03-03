import ROOT
import numpy as np
from pathlib import Path
from glob import glob
import pandas as pd
import bz2
from copy import deepcopy
from Rejection_Status import Rejection_Status as rs
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import json
from datetime import datetime
import matplotlib as mp

ROOT.gSystem.Load("/cr/data01/ellwanger/Offline/build/inst-offline-opt/lib/libRecEventKG.so")


def open_ADST_files(adst_files):
    adstvec = ROOT.std.vector('string')()
    for adst in adst_files:
        adstvec.push_back(str(adst))

    rf = ROOT.RecEventFile(adstvec, False)
    event = ROOT.RecEvent()
    rf.SetBuffers(event)

    det = ROOT.DetectorGeometry()
    rf.ReadDetectorGeometry(det)

    return rf, event, det

def read_multiplet_stationlist(filepath):
    tree = ET.parse('data.xml')
    root = tree.getroot()
    stationlist = []
    for station in root.findall('.//station'):
        group_id = station.find('groupId').text.strip()
        station_id = station.get('id')
        if group_id != str(0):
            stationlist.append(int(station_id))
    return stationlist         

def read_multiplet_grouplist(filepath):
    tree = ET.parse('data.xml')
    root = tree.getroot()
    stationgrouplist = {}
    for station in root.findall('.//station'):
        group_id = station.find('groupId').text.strip()
        station_id = station.get('id')
        if group_id != str(0):
            if group_id not in stationgrouplist:
                stationgrouplist[group_id] = {"station_ids":[]}
            stationgrouplist[group_id]["station_ids"].append(station_id)

    #print(stationgrouplist['26']['station_ids'])   
    return stationgrouplist


if __name__ == "__main__":
    #stationlist = read_multiplet_stationlist('data.xml')
    grouplist = read_multiplet_grouplist('data.xml')
    #print(grouplist)

    base_dir = Path("/cr/augerdata/Reconstruction/Observer/Public/icrc-2025/test3/SD_PhaseI/out/2022/05")
    adst_list = list(base_dir.glob("*.root"))
    print(len(adst_list))

    rf, event, det = open_ADST_files(adst_list)
    eventlist = {}
    multiplethitlist = {}
    pairlist = []


    while rf.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
        auger_id = event.GetAugerId()
        if auger_id not in eventlist:
            eventlist[auger_id] = set()
        
        for station in event.GetSDEvent().GetStationVector():
            station_id = station.GetId()
            
            for group_id, group_data in grouplist.items():
                if int(station_id) in map(int, group_data["station_ids"]):
                    # Initialize multiplethitlist for auger_id and group_id if not present
                    if auger_id not in multiplethitlist:
                        multiplethitlist[auger_id] = {"group_id": {}}
                    
                    if group_id not in multiplethitlist[auger_id]["group_id"]:
                        multiplethitlist[auger_id]["group_id"][group_id] = []
                    
                    # Append the station_id under the correct group_id, avoiding duplicates
                    if station_id not in multiplethitlist[auger_id]["group_id"][group_id]:
                        multiplethitlist[auger_id]["group_id"][group_id].append(station_id)
                    
                    # Add auger_id to pairlist if this is the second time it appears for this group
                    if auger_id not in pairlist:
                        if len(multiplethitlist[auger_id]["group_id"][group_id]) > 1:
                            pairlist.append(auger_id)

    print(multiplethitlist)            
    print(pairlist)
    structure = {}

    # Loop through each auger_id in pairlist to build the desired structure
    for auger_id in pairlist:
        # Initialize structure for the auger_id if not already present
        if auger_id not in structure:
            structure[auger_id] = {'group_id': {}}
        
        # Access group and station lists for the current auger_id in multiplethitlist
        group_data = multiplethitlist[auger_id]["group_id"]
        
        # Loop through each group_id and corresponding station list
        for group_id, station_list in group_data.items():
            # Initialize an empty list for each group_id if it doesn't already exist
            if group_id not in structure[auger_id]['group_id']:
                structure[auger_id]['group_id'][group_id] = []
            
            # Add each station_id to the corresponding group_id in the structure
            structure[auger_id]['group_id'][group_id].extend(station_list)

    print(structure)
    print("done ...")
