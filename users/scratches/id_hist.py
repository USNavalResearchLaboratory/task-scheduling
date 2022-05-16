from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from data_importer import ADSBFAData, TRACK_IDENTITIES


data_dir = Path("data_importer/example_data")
# filename = data_dir / 'adsb/Naval_Research_Lab_Sample.csv'
# filename = data_dir / 'adsb/temp/FlightAware/Naval_Research_Lab_2021-01-01_2021-02-01_Random_5000_Flights.csv'
filename = data_dir / 'adsb/temp/FlightAware/Naval_Research_Lab_2021-02-01_2021-02-25_Random_5000_Flights.csv'

obj_data = ADSBFAData.from_file(filename, num_rows=100000, do_cleanup=True)
data = obj_data.data

data_grouped = data.groupby('track_number')

track_numbers, tracks = zip(*data_grouped)

diffs = []
for trk in tracks:
    t = trk['time'].to_numpy().astype('datetime64[s]')
    d = np.diff(t).astype(float)
    diffs.extend(d.tolist())

# plt.hist(diffs, bins=100)
plt.hist(diffs, bins=np.arange(100))
