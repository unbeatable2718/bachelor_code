#!/usr/bin/env python3
import ROOT
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib as mp
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

def angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_theta = dot_product / (norm_vec1 * norm_vec2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle


if __name__ == "__main__":
    max = 0
    filepath = 'data.xml'
    base_dir = Path("/cr/augerdata/Reconstruction/Observer/Public/icrc-2025/test3/SD_PhaseI/out/2022/")
    adst_list = list(base_dir.rglob("*.root"))
    adst_list = adst_list
    print("\n")
    recEventFile, recEvent, detectorGeometry, fileInfo = open_adst_files(adst_list)

    ldfType = fileInfo.GetLDFType()
    print(ldfType, "\n")
    angles_in_degrees = []
    while recEventFile.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
        auger_id = recEvent.GetAugerId()
        sdRecShower = recEvent.GetSDEvent().GetSdRecShower()
        azimuth = np.array([sdRecShower.GetAzimuth(), sdRecShower.GetAzimuthError()])
        zenith = np.array([sdRecShower.GetZenith(), sdRecShower.GetZenithError()])
        aVectorInCartesianSelf = np.array([np.sin(zenith[0]) * np.cos(azimuth[0]), np.sin(zenith[0]) * np.sin(azimuth[0]), np.cos(zenith[0])])
        aVectorInCartesianADST = np.array([sdRecShower.GetPlaneFrontAxis().X(), sdRecShower.GetPlaneFrontAxis().Y(), sdRecShower.GetPlaneFrontAxis().Z()]) 
        core = sdRecShower.GetCoreSiteCS()
        corePosition = np.array([core.X(), core.Y(), core.Z()])
        coreTime = np.array([sdRecShower.GetCoreTimeSecond(), sdRecShower.GetCoreTimeNanoSecond()])

        for station in recEvent.GetSDEvent().GetStationVector():
            station_id = station.GetId()
            time = np.array([station.GetTimeSecond(), station.GetTimeNSecond(), station.GetTimeVariance()])
            time50 = np.array([station.GetTime50(), station.GetTime50RMS()])
            stationCoor = detectorGeometry.GetStationPosition(station_id)
            stationPosition = np.array([stationCoor.X(), stationCoor.Y(), stationCoor.Z()])
            signal = np.array([station.GetTotalSignal(), station.GetTotalSignalError()])
            n = signal[0]/((zenith[0]*(0.69*zenith[0] - 0.55) + 1.2)/1.2)
            #Vmeins = V_t0(time50[0], n)
            distance = np.array([station.GetSPDistance(), station.GetSPDistanceError()])
        angle = angle_between_vectors(aVectorInCartesianSelf, aVectorInCartesianADST)
        angles_in_degrees.append(np.degrees(angle))
        if angle > max:
            max = angle

    mp.rc("text", usetex=True)
    mp.rc("font", family="serif")
    pck = ["amsmath", "amssymb", "newpxtext", "newpxmath"]  # Palatino-like fonts
    # pck = ["amsmath", "amssymb", "mathptmx"]  # Times-like fonts (optional alternative)
    mp.rc("text.latex", preamble="".join([f"\\usepackage{{{p}}}" for p in pck]))
    max_angle = np.max(angles_in_degrees)
    print(f"Maximum angle: {max_angle:.2f} degrees")
    plt.hist(angles_in_degrees, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title(r"Opening Angles")
    plt.xlabel(r"Angle (Degrees)")
    plt.ylabel(r"Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()    