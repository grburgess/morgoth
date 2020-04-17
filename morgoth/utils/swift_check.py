import os
import csv
from datetime import datetime
import numpy as np

swift_data_dir = os.environ.get("SWIFT_VOEVENT_DATA_DIR")

def check_swift(gbm_trigger_time):
    """
    Check if there was a swift trigger +/- 100 seconds around GBM trigger
    :param gbm_trigger_time: Trigger time of GBM; datetime object
    :return: swift ra, dec and id if there was a swift trigger, if not None is returned
    """
    # Get swift trigger times, id and position
    if swift_data_dir!=None:
        file_path = os.path.join(swift_data_dir, "swift_triggers.csv")
        if os.path.exists(file_path):
            with open(file_path, "r") as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                swift_trigger_times = np.array([])
                swift_trigger_ids = np.array([])
                swift_pos = np.array([])
                for row in reader:
                    swift_trigger_times = np.append(swift_trigger_times,datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%f"))
                    swift_trigger_ids= np.append(swift_trigger_ids,int(row[1]))
                    swift_pos = np.append(swift_pos,np.array([float(row[2]), float(row[3])]))

            swift_pos = np.reshape(swift_pos,(len(swift_trigger_times), 2))

            helper = np.vectorize(lambda x: x.total_seconds())

            time_diff = helper(swift_trigger_times-gbm_trigger_time)

            mask = np.abs(time_diff)<100

            if np.sum(mask)==0:
                return None
            else:
                return {"ra":float(swift_pos[mask][0,0]),
                        "dec":float(swift_pos[mask][0,1]),
                        "trigger":int(swift_trigger_ids[mask][0])}
            
    # File with swift bursts not found. Assume there was no swift burst.
    return None
