'''
Created on Jan 23, 2025

@author: Admin
'''

import os
import json
import pandas as pd
from datetime import datetime, date

'''
returns a dataframe having timeseries of earth's magnetic field
source: https://geomag.usgs.gov/plots/
downloaded_fileformat: json
'''
def get_timeseries_magnetic_data():
    folder = r'C:\Users\Admin\eclipse-workspace\magnavis\src\data\temporal_magnetic_field'
    
    files = ['downloadx.json', 'downloady.json', 'downloadz.json']
    
    df_final = pd.DataFrame()
    df_list = []
    
    for fil in files:
        full_fname = os.path.join(folder, fil)
        with open(full_fname, 'r') as ff:
            data = json.load(ff)
            orientation = data['metadata']['intermagnet']['reported_orientation']
            print(data['metadata']['intermagnet']['reported_orientation']) #data['reported_orientation'], data['times'])
            times_str = data['times']
            # print(len(times_str))
            times = [datetime.fromisoformat(t[:-1] + '+00:00') for t in times_str]
            # print(times[0])
            mag_field = data['values'][0]['values']
            # print(mag_field[0])
            df_ = pd.DataFrame({
                    f'time_{orientation}': times,
                    f'mag_{orientation}_nT': mag_field
                })
            df_final = pd.concat([df_final, df_], axis=1)
    return df_final



if __name__ == '__main__':
    df = get_timeseries_magnetic_data()
    print(df) # datetime, float (field in nT) (NaN supported)
    df.astype(str).to_excel('magnetic_time_ser.xlsx', index=False)