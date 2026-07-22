import numpy as np
import pandas as pd 
import json 

def create_input_json(coordinates_file, mach, angle, name, num_points):
    df = pd.read_csv(coordinates_file, header=None)
    target_x = np.array([0, 0.25, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])/100
    cutoff = np.argwhere(np.sign(np.diff(df.values[:,0]))>=0.0).flatten()[0]+1
    print(cutoff)
    xc_u, xc_l = df.values[:cutoff,0], df.values[cutoff:,0]
    zc_u, zc_l = df.values[:cutoff,1], df.values[cutoff:,1]
    coordinates = np.hstack((target_x[:,None], np.interp(target_x, np.flip(xc_u), np.flip(zc_u))[:,None], np.interp(target_x, xc_l, zc_l)[:,None])).tolist()
    output = {
        "coordinates": coordinates,
        "mach": mach,
        "angle": angle,
        "name": name,
        "resolution": int(num_points)
    }
    
    # Convert and write JSON object to file
    with open("sample.json", "w") as outfile: 
        json.dump(output, outfile)


create_input_json('rae2822_coord2.csv', 0.676, 1.9, 'rae2822', 701)








