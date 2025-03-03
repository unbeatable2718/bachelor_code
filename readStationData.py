import ROOT
import numpy as np
from pathlib import Path
from glob import glob
import pandas as pd
import bz2
from copy import deepcopy
from Rejection_Status import Rejection_Status as rs

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


if __name__ == "__main__":
    base_dir = Path("/cr/augerdata/Reconstruction/Observer/Public/icrc-2025/test3/SD_PhaseI/out/2022/06/")
    adst_list = list(base_dir.glob("*.root"))
    print(len(adst_list))
    adst_list = adst_list[:1]

    rf, event, det = open_ADST_files(adst_list)

    while rf.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
        print(rf.GetActiveFileName())
        for station in event.GetSDEvent().GetBadStationVector():
            print(rs.resolve(station.GetReason()))

    print("done ...")
