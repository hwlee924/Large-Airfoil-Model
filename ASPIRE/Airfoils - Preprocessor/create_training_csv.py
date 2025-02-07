#%% Import libraries
import pandas as pd 
import numpy as np
import os 
import glob
import re
import matplotlib.pyplot as plt
import json 
import sys
import coord_to_kulfan
#%% Set directory 
os.chdir('F:/Cloned Respositories/GP-Aero/ASPIRE/Airfoils - Preprocessor') 
#%% Inputs
# Database related
aspire_dir = '../Airfoils' # ASPIRE directory 
aspire_airfoils = os.listdir(aspire_dir) # get all airfoil folders 

# Visualize individual outputs?
validate_entry = True 
#%% Functions 
class data_log():
    def __init__(self):
        self.added_file = []
        self.omitted_file = []
        self.omitted_reason = []
        
    def add_omitted_file(self, file_name, reason_str):
        self.omitted_file.append(file_name)
        self.omitted_reason.append(reason_str)
        
    def add_added_file(self, file_name):
        self.added_file.append(file_name)
        
    def summarize(self):
        return None
        
def read_case_information(file_name, logger): 
    # Extract case information 
    AoA  = re.search(r"\_A(.*?)\_M", file_name)
    Minf = re.search(r"\_M(.*?)\_Re", file_name)
    Re   = re.search(r"\_Re(.*?)\_", file_name) 
    success_flag = True
    
    # AoA validity check 
    if AoA:
        if AoA.group(1)[0] == 'm': # negative 
            AoA = float(AoA.group(1)[1:])*-1
        else: # postive 
            AoA = float(AoA.group(1))
    else: 
        logger.add_omitted_file(file_name, 'Missing angle of attack info')
        print(file_name + ' omitted due to missing angle of attack.')
        success_flag = False
        
    # Minf validity check 
    if Minf:
        Minf = float(Minf.group(1))
    else: 
        logger.add_omitted_file(file_name, 'Missing Mach number info')
        print(file_name + ' omitted due to missing mach number.')
        success_flag = False
        
    # Re validity check 
    if Re:
        Re = float(Re.group(1))
    else: 
        logger.add_omitted_file(file_name, 'Missing Reynolds number info')
        print(file_name + ' omitted due to missing Reynolds number.')
        success_flag = False
        
    if success_flag:
        return success_flag, [AoA, Minf, Re] 
    else: 
        return success_flag, None
    
def read_airfoil_geometry(coord_file_name, logger):
    target_xc_u = np.array([0, 0.0025, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 
                0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 
                0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0]) # reference location, upper surface
    target_xc_l = np.array([0, 0.0025, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 
                0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 
                0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0]) # reference location, lower surface
    coords = pd.read_csv(coord_file_name, header=None).values # reading csv file 
    # print(coords)
    switch_idx = np.argwhere(np.sign(np.diff(coords[:,0])) == 0.0).flatten()[0] + 1  # idx separating upper and lower surfaces
    xc_u, xc_l = np.flip(coords[:switch_idx, 0]), coords[switch_idx:, 0]
    zc_u, zc_l = np.flip(coords[:switch_idx, 1]), coords[switch_idx:, 1]
    # Interpolate coordinates
    target_zc_u = np.interp(target_xc_u, xc_u, zc_u)
    target_zc_l = np.interp(target_xc_l, xc_l, zc_l)
    
    if validate_entry: 
        plt.figure(figsize=(10,2))
        plt.plot(xc_u, zc_u, 'r-')
        plt.plot(xc_l, zc_l, 'b-')
        plt.plot(target_xc_u, target_zc_u, 'ro')
        plt.plot(target_xc_l, target_zc_l, 'bs')
        plt.xlabel('x/c')
        plt.ylabel('z/c')
        plt.title(coord_file_name)
        plt.show()
    return [target_zc_u, target_zc_l]

def read_tag_information(tag_file_name, logger):
    with open(tag_file_name, 'r') as f:
        tag_data = json.load(f)
        
        # Extract airfoil information 
        af_name = tag_data['airfoil']['name'] # AF name 
        if tag_data['airfoil']['camber'] == 'Y': # Camber
            af_camber = 'Cambered'
        else: 
            af_camber = 'Symmetric'
        if tag_data['airfoil']['supercritical'] == 'Y': # Supercriticality
            af_supercrit = 'Supercritical'
        else: 
            af_supercrit = '-'
        af_usage = tag_data['airfoil']['application'] # AF application
        
        # Exctract source information 
        source_name = tag_data['source']['name'] # source document name
        source_type = tag_data['source']['type'] # plot or graph
        
        # Extract noise information
        noise_cp = tag_data['uncertainty']['cp']
        noise_xc = tag_data['uncertainty']['x']
        noise_alpha = tag_data['uncertainty']['alpha']
        noise_mach = tag_data['uncertainty']['mach']
        return [af_name, af_camber, af_supercrit, af_usage], [noise_cp, noise_xc, noise_alpha, noise_mach], [source_name, source_type]

def read_cp(file_name, logger):
    cp_data = pd.read_csv(file_name, na_values=['--', ""])
    cp_data = cp_data.values  
    
    # convert xc to xhat and yhat 
    xc, cp = cp_data[:, 0], cp_data[:, 1]

    switch_idx = np.argwhere(np.sign(np.diff(xc)) != -1).flatten()[0] + 1  
    
    xc_u, xc_l = xc[:switch_idx].astype(float), xc[switch_idx:].astype(float)
    cp_u, cp_l = cp[:switch_idx].astype(float), cp[switch_idx:].astype(float)
    # print(cp_u.dtype, cp_l.dtype)
    # drop NaN (dont drop before since we need the nan for the correct cutoff) 

    xc_u, xc_l = xc_u[~np.isnan(cp_u)], xc_l[~np.isnan(cp_l)]
    cp_u, cp_l = cp_u[~np.isnan(cp_u)], cp_l[~np.isnan(cp_l)]
    
    # convert to xhat yhat
    xhat_u, xhat_l = xc_u*2 - 1, xc_l*2 - 1
    yhat_u, yhat_l = np.sin(np.arccos(xhat_u)), -np.sin(np.arccos(xhat_l))
    xhat, yhat = np.hstack((xhat_u, xhat_l)), np.hstack((yhat_u, yhat_l))

    cp_all = np.hstack((cp_u, cp_l))
    assert not any(np.isnan(yhat)), 'y_hat contains NaN values'
    return [xc_u, xc_l], [cp_u, cp_l, cp_all], [xhat, yhat]
    
def characterize_noise(noise_tag, cp_profile):
    # Extract between markers function
    def extract_substring(s, start_marker, end_marker):
        start = s.find(start_marker) + len(start_marker)
        end = s.find(end_marker, start)
        if start_marker in s and end_marker in s:
            return s[start:end]
        return None

    def extract_before_marker(s, marker):
        end = s.find(marker)
        if end != -1:
            return s[:end]
        return s

    # For Cp 
    if noise_tag[0] == '':  # None
        noise_cp = 0.01
        noise_cp = np.tile(noise_cp, cp_profile.shape[0])
    elif len(noise_tag[0]) > 3 and noise_tag[0][-3:] == 'max': # Percentage of max case 
        relative_percentage = float(extract_substring(noise_tag[0], '<', f'% of max'))
        noise_cp = relative_percentage/100 * np.max(np.abs(cp_profile))
        noise_cp = np.tile(noise_cp, cp_profile.shape[0]) 
    elif len(noise_tag[0]) > 4 and noise_tag[0][-4:] == '|Cp|': 
        relative_term = float(extract_substring(noise_tag[0], '+ ', f'|Cp|'))
        constant_term = float(extract_before_marker(noise_tag[0], ' + '))
        noise_cp = constant_term + relative_term*np.abs(cp_profile) 
    elif noise_tag[0][-1] == '%': # Percentage of dynamic pressure
        target_percentage = float(noise_tag[0][:-1])
        noise_cp = target_percentage/100 #* np.abs(cp_profile)
        noise_cp = np.tile(noise_cp, cp_profile.shape[0])
    else: # Absolute case 
        noise_cp = float(noise_tag[0])
        noise_cp = np.tile(noise_cp, cp_profile.shape[0])
    # For alpha (future implementation)
    
    # For xc (future implementation)
    
    # For mach (future implementation)
    return noise_cp
    
def read_aspire(folder_dir, validate_entry):
    global case_counter
    df_ = []
    #aspire_dir + '/' + i
    airfoil_cp_data = glob.glob(folder_dir + '/*.csv') # list of all csv (pressure data)
    if any(file.endswith('coordinates.csv') for file in airfoil_cp_data): # if coordinates file exists
        matching_idx = next(
            (idx for idx, file in enumerate(airfoil_cp_data) if file.endswith('coordinates.csv')), 
            None
        ) # find index of coordinates file 
        airfoil_coordinates = airfoil_cp_data[matching_idx] # coordinates file 
        airfoil_cp_data.pop(matching_idx) # updated cp data file  
        zc = read_airfoil_geometry(airfoil_coordinates, logger)
        zc_u, zc_l = zc[0], zc[1]
    else: 
        logger.add_omitted_file(folder_dir, 'No coordinates file')
        print(folder_dir + ' omitted due to missing airfoil coordinates.')
        
        return
    
    # Get tag file     
    airfoil_tag = glob.glob(folder_dir + '/*.json') # tag file 
    # omit airfoil if no tag file 
    if not airfoil_tag:
        logger.add_omitted_file(folder_dir, 'No tag file')
        print(folder_dir + ' omitted due to missing tag file.')
        return 
    else: 
        tag_af, tag_noise, tag_source = read_tag_information(airfoil_tag[0], logger)
    
    # Load all cp csv files 
    for j  in airfoil_cp_data: 
        if validate_entry:
            print(j)
        # Extract case information 
        extract_success, extracted_info = read_case_information(j, logger)
        if extract_success: 
            AoA, Minf, Re = extracted_info[0], extracted_info[1], extracted_info[2]
        else: 
            continue # skip if invalid 
        
        # Read CSV file and assemble entry 
        xcs, cps, hats = read_cp(j, logger)
        
        # Use noise tag and Cp profile to characterize noise
        noise_array = characterize_noise(tag_noise, cps[2])  
        
        # Assemble 
        target_xc_u = np.flip(np.array([0, 0.0025, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 
                0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 
                0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0])) # reference location, upper surface
        target_xc_l = np.array([0, 0.0025, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 
                    0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 
                    0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0]) 
        
        tile_shape = (hats[0].shape[0], 1)
        entry_row = [
                        # Numericals
                        np.tile(target_xc_u, tile_shape), # xc_u
                        np.tile(target_xc_l, tile_shape), # xc_l
                        np.tile(zc_u, tile_shape), # zc_u
                        np.tile(zc_l, tile_shape), # zc_l
                        hats[0][:,None], # xhat 
                        hats[1][:,None], # yhat
                        np.tile(Minf, tile_shape), # Mach 
                        np.tile(Re  , tile_shape), # Reynolds
                        np.tile(AoA , tile_shape), # alpha 
                        cps[2][:, None], # Cp 
                        noise_array[:, None], # noise at given Cp
                        np.tile(case_counter, tile_shape), # Case number
                        
                        # Categoricals 
                        np.tile(tag_af[0], tile_shape), # airfoil name 
                        np.tile(tag_af[1], tile_shape), # airfoil camber
                        np.tile(tag_af[2], tile_shape), # airfoil supercriticality
                        np.tile(tag_af[3], tile_shape), # airfoil application
                    ]
        
        # need Noise and familiy 
        # 'M', 'noise', 'af', 'symmetry', 'supercritical', 'usage', 'family', 'case', 'Cp'
        headers = [f'xcu_{i}' for i in range(1, 28+1)] + \
                    [f'xcl_{i}' for i in range(1, 28+1)] + \
                    [f'zcu_{i}' for i in range(1, 28+1)] + \
                    [f'zcl_{i}' for i in range(1, 28+1)] + \
                    ['xhat', 'yhat'] + \
                    ['M', 'Re', 'alpha'] + \
                    ['Cp', 'noise'] + \
                    ['case'] + \
                    ['af', 'symmetry', 'supercritical', 'usage']
                    
                    
        # set dtypes 
        dtypes = {col: 'float64' for col in headers[:-4]}  # Set float64 for numerical columns
        dtypes.update({col: 'object' for col in headers[-4:]})  # Set object for categorical columns
        entry_df_row = pd.DataFrame(np.hstack(entry_row), columns=headers).astype(dtypes)
        
        if validate_entry: 
            plt.figure(figsize=(10, 6))
            plt.plot(xcs[0], cps[0], 'o', label= 'Upper surface')
            plt.plot(xcs[1], cps[1], 's', label= 'Lower surface')
            plt.gca().invert_yaxis()
            plt.xlabel('x/c')
            plt.ylabel('$C_p$')
            plt.title(j)
            plt.legend()
            plt.show()
            
            user_confirmation = input('Confirm data validity. Enter X to quit') 
            if user_confirmation == 'X':
                raise ValueError('User terminated operation')
        case_counter += 1 
        df_.append(entry_df_row)
    return pd.concat(df_)

def assemble_data_from_aspire(savefile_name, range=(0, len(aspire_airfoils)), save_entries=True, validate_entry=False):
    global case_counter 
    aspire_entries_ = []
    for i in aspire_airfoils[range[0]:range[1]]:  
    # check sub folders  
    
        subfolder_list = [f for f in os.listdir(aspire_dir + '/' + i) if os.path.isdir(os.path.join(aspire_dir + '/' + i, f))]
        if not subfolder_list:
            # Get Cp data and coordinates
            entry_ = read_aspire(aspire_dir + '/' + i, validate_entry=validate_entry)
            
        else: 
            for k in subfolder_list:
                entry_ = read_aspire(aspire_dir + '/' + i + '/' + k, validate_entry=validate_entry)
        
        if entry_ is not None:
            aspire_entries_.append(entry_)

    aspire_entries = pd.concat(aspire_entries_)
    if save_entries:
        aspire_entries.to_csv(savefile_name, index=False)

    return aspire_entries

def cst_transform():
    return 1 
#%% 

#%% Process data 
if __name__ == "__main__":
    logger = data_log()
    global case_counter   
    case_counter = 0
    validate_entry = False
    assembled_entries = assemble_data_from_aspire('save_file.csv', save_entries=False, validate_entry=validate_entry)
