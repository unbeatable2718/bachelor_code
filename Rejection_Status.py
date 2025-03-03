# Author: Philip Meder (2024)

class Rejection_Status:
    """Keep this up to date with ADST/RecEvent/src/StationStatus.h."""

    eNoRejection      = 0
    eLightning        = (1 << 0)  # treat as non extistent
    eBadCompress      = (1 << 1)  # treat as non extistent
    eOutOfTime        = (1 << 2)  # treat as silent
    eOffGrid          = (1 << 3)  # treat as non extistent
    eDenseArray       = (1 << 4)  # treat as non extistent
    eRandomRejection  = (1 << 5)  # treat as silent
    eEngineeringArray = (1 << 6)  # treat as non extistent
    eMCInnerRadiusCut = (1 << 7)  # treat as non extistent
    eNoRecData        = (1 << 8)  # treat as non extistent
    eLonely           = (1 << 9)  # treat as silent
    eNoTrigger        = (1 << 10)  # treat as silent
    eErrorCode        = (1 << 11)  # treat as non extistent
    eNoCalibData      = (1 << 12)  # treat as non extistent
    eNoGPSData        = (1 << 13)  # treat as non extistent
    eBadCalib         = (1 << 14)  # treat as non extistent
    eRegularMC        = (1 << 15)  # treat as non extistent
    eTOTdRejected     = (1 << 16)  # use them
    eMoPSRejected     = (1 << 17)  # use them
    eNotAliveT2       = (1 << 18)  # treat as non extistent
    eNotAliveT120     = (1 << 19)  # treat as non extistent
    eBadSilent        = (1 << 20)  # treat as non extistent
    eAllPMTsBad       = (1 << 21)  # treat as non extistent
    eElectronicsType  = (1 << 22)  # treat as non extistent (UUB)

    @classmethod
    def create(cls, status_list):
        """Create status number out of luist of rejection reasons."""
        status = cls.eNoRejection
        all_stat = vars(cls)
        for stat in status_list:
            status = status | all_stat[stat]
        return status

    @classmethod
    def resolve(cls, status_number):
        """Create list of rejection reasons out of status number."""
        # 'Fast' resolve for 0 -> no rejection, maybe delete
        if status_number == 0:
            return []

        all_stat = vars(cls)
        status_list = []
        for key, item in all_stat.items():
            if key[0] != "e":
                continue
            if status_number & item:
                status_list.append(key)
        # Security routine: If everything is up to date, then this is useless
        #                   If status_number != 0 but it is unkown -> catch
        if status_number != 0 and not status_list:
            status_list = [f"RejectedByUnknownStatus{status_number}"]
        return status_list


if __name__ == "__main__":
    # Example Create:
    # status_list = ["eOffGrid", "eRandomRejection", "eMCInnerRadiusCut"]
    # status_id = Rejection_Status.create(status_list)
    # print(status_id)

    # Example Resolve: You have a station with a certain rejection status id.
    rejection_status_id = 2024

    # Get a list of human readable rejection stati
    rejection_status_list = Rejection_Status.resolve(rejection_status_id)
    print(rejection_status_list)

    # Let's say you want to ignore the MCInnerRadiusCut status
    try:
        rejection_status_list.remove("eMCInnerRadiusCut")
    except ValueError:
        # Status is not in rejection list
        pass

    # Check if station is rejected
    is_rejected = bool(rejection_status_list)
    print(f"Station rejected: {is_rejected}")
