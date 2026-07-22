import os 
import torch

"""
print function that adds a tag to the message i.e. [TAG] MESSAGE
message (str): message string to print out
tag     (str): tag string to be prepended to the message
"""
def print_with_tag(message:str, tag:str=None):
    return print(f"[{tag}] {message}" if tag else message)

"""
Sets up the environment for CPU or GPU usage.
use_gpu (bool): True if GPU should be used, False if CPU should be used. 
gpu_id  (int) : Provide an int value to specify which GPU to use. Ignored when using CPU.
"""    
def initialize_devices(use_gpu:bool=False, gpu_id:int=None):
    print_tag = 'DEVICE'
    if use_gpu == True: # USE GPU
        if gpu_id is not None: # user has specfied a GPU number
            # Make sure the gpu_id is an integer 
            assert isinstance(gpu_id, int), f"Invalid GPU ID: {gpu_id}. GPU ID must be an integer."
        else:
            gpu_id = 0  # default GPU id if not specified
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # select the specified GPU 
        output_device = torch.device(f'cuda:{gpu_id}') # output device 
        print_with_tag(f"Planning to run on GPU {gpu_id}", print_tag)
         
    elif use_gpu == False: # DO NOT USE GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "" # make sure no GPUs are available in the environment
        output_device = torch.device('cpu')
        print_with_tag('Using CPU.', print_tag)
        
    else: # INVALID INPUT, USE CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "" # make sure no GPUs are available in the environment
        output_device = torch.device('cpu')
        print_with_tag('Invalid use_gpu parameter. Defaulting to CPU.', print_tag)
    return output_device

def initialize_plot_settings(line_width:int=3, tick_size:int=20, label_size:int=24, legend_size:int=20):
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    
    # Set default line width
    mpl.rcParams['lines.linewidth'] = line_width

    
    # Set default label font size
    mpl.rcParams['axes.labelsize'] = label_size
    
    # Set default tick label sizes
    mpl.rcParams['xtick.labelsize'] = tick_size
    mpl.rcParams['ytick.labelsize'] = tick_size
    mpl.rcParams['mathtext.fontset'] = "stix"
    # Legend fontsize
    mpl.rcParams['legend.fontsize'] = legend_size
    # font_path = '/scratch/hlee981/Fonts/Times New Roman.ttf' 
    # fm.fontManager.addfont(font_path)
    # prop = fm.FontProperties(fname=font_path)
    # font_name = prop.get_name()
    # plt.rcParams['font.family'] = font_name

def mvn_to_device(mvn: torch.distributions.MultivariateNormal, device):
        """
        Takes a GPU-pushed multivariate normal distribution
        and makes a new multivariate normal distribution in CPU
        """
        return torch.distributions.MultivariateNormal(
            mvn.mean.to(device),
            mvn.covariance_matrix.to(device)
        )
    
def n_to_device(normal_distrib:torch.distributions.normal.Normal, device):
    """
    Takes a GPU-pushed univariate normal distribution
    and makes a new univariate normal distribution in CPU
    """
    return torch.distributions.normal.Normal(
        normal_distrib.mean.to(device),
        normal_distrib.scale.to(device)
    )
    