""" This code contains only the code to train and test the model. For the code that includes data analysis, refer to the jupyter notebook
"""
#%% Import Libs
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt 
import os 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
#%% Set up user input 
useGPU = True # Use GPU?
random_test = False # Leave this for now 

# For Model Training
train_new = True # Training new, False if you want to load existing vars
training_iterations = 100 # Total training iterations
swa_startpt = 100 # When to start saving weights for Stochastic weight averaging 
interval_checkpt = 50 # Interval for internal MAE tracking 
swa_load_num = 2 # How many weights for SWA
#%% Set up GPU usage 
if useGPU == True: # 6 is for personal use 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    output_device = torch.device('cuda:0')
    n_devices = torch.cuda.device_count()
    output_device = torch.device('cuda:0')
    print('Planning to run on {} GPUs'.format(n_devices))
elif useGPU == False: 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_device = torch.device('cpu')
    print('Using CPU')
else: 
    raise ValueError("Incorrect value of useGPU variable")

#%% Functions for general utility
# Function to save checkpoint
def save_checkpoint(model, optimizer, history, iters, fileName):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'iterations': iters
                },
            fileName)
    
def gen_save_string(prefix, lr, decay, nn_dims, losses, iters, override_date=None):
    from datetime import date
    if override_date is not None:
        save_str_date = override_date
    else:
        today = date.today()
        save_str_date = today.strftime("%Y%m%d") + '_'
    save_str_prefix = prefix + '_' 
    save_str_optimizer = 'lr' + str(lr) + '_' + 'decay' + str(decay) + '_'
    save_str_dims = 'dims' + str(nn_dims[0]) + '-' + str(nn_dims[1]) + '-' + str(nn_dims[2]) + '-' + str(nn_dims[3]) + '_'
    save_str_checkpt = 'checkpoint' + str(iters) + '_'
    save_str_loss = 'loss' + str(losses[0]) + '_'
    save_str_trainmetric = 'trainMetric' + str(losses[1]) + '_'
    save_str_testmetric = 'testLoss' + str(losses[2])
    return save_str_date + save_str_checkpt + save_str_prefix + save_str_optimizer + save_str_dims + save_str_loss + save_str_trainmetric + save_str_testmetric 

# Functions useful for predictions
def validate_predictions(x_data, y_data, af_df, test_case, num_cases, weights_list = None, save_figs = False):
    input_var = []
    output_var = []
    
    af_u = af_df.unique()
    subtest_af = [af_u[test_case]]
    subtest_x = x_data[af_df.isin(subtest_af).values]
    subtest_y = y_data[af_df.isin(subtest_af).values]

    # Evaluate
    if weights_list is None:
        weights_list = []
        for n in range(0, 5):
            weights_list.append(gen_save_string(save_tag, lr, decay, nn_dims, (history_values['mll'][iters_checkpt-1-n], 
                                                                    history_values['train_err'][iters_checkpt-1-n], 
                                                                    history_values['test_err'][iters_checkpt-1-n]), 
                                    iters_checkpt-1-n))
    preds_subtest = aggregate_posterior(subtest_x.to(output_device), weights_list)
    print('Evaulating posterior predictive distribution for airfoil ' + subtest_af[0] + '...')
    print('Subtest MAE: {}'.format(calc_err(preds_subtest.mean.cpu(), subtest_y.cpu())))

    targetAF = af_u[test_case]
        
    temp_ind_af = np.where(af_df.values == targetAF)[0]
    temp_arr = x_data[temp_ind_af]

    unique_AM_pair = np.unique(temp_arr[:,-4:-2], axis=0)
    if num_cases > unique_AM_pair.shape[0]:
        num_cases = unique_AM_pair.shape[0]
        
    for j in np.arange(0, num_cases):
        plt.figure()

        # Generate predictor and make predictions
        ## Predictor
        test_airfoil = gen_test_data(af_u[test_case], unique_AM_pair[j][0], unique_AM_pair[j][1])
        input_var.append(test_airfoil)
        temp_ind = temp_ind_af[torch.logical_and(temp_arr[:, -4] == unique_AM_pair[j,0], temp_arr[:, -3] == unique_AM_pair[j,1])]
        
        ## Evaulate Posterior Predictive 
        sample_airfoil_pred = aggregate_posterior(test_airfoil.to(output_device), weights_list)
        output_var.append(sample_airfoil_pred)

        # Plot result
        ## posterior
        (f, ax), (xc_u, xc_l), (mu_u, mu_l), (std_u, std_l) = plot_posterior(test_airfoil[:,-2], sample_airfoil_pred, scale = y_scale, mean = y_mean)
        ## experimental validation
        temp_plot_x = ((x_data[temp_ind, -2]+1)/2).cpu().detach().numpy()
        temp_plot_y = (y_data[temp_ind].cpu()+y_mean)/y_scale
        temp_plot_noise = noise[temp_ind]
        ax.errorbar(temp_plot_x, temp_plot_y, yerr=2*np.sqrt(temp_plot_noise)/y_scale, fmt='k.', capsize=2, label='Expt., Flemming 1984') #  $C_p \pm 2\sigma$
        ax.invert_yaxis()
        plt.xlabel('x/c')
        plt.ylabel('$C_p$')
        # plt.legend()
        plt.title(targetAF +'\n' + r'$\alpha$ = ' + str(np.round(np.rad2deg(unique_AM_pair[j,0]),2)) + r'$^\circ,$' + r' $M_\infty = $' + str(unique_AM_pair[j,1]))
        if save_figs == True:
            plt.savefig(targetAF+'_'+'a'+str(np.round(np.rad2deg(unique_AM_pair[j,0]),2))+'_'+str(unique_AM_pair[j,1])+'.png', bbox_inches='tight')
        plt.show() 
    return input_var, output_var

def gen_test_data(airfoil, alpha, mach, num_points = 601, manual_point = None, out_x=False):
    
    global af, data_raw, meanX
    if manual_point is not None:
        num_points = manual_point.shape[0]

    if isinstance(airfoil, str):
        # if string, read airfoil info from training data
        target_af_idx = np.argwhere(af.isin([airfoil]).values)[0]
        target_af_geom = torch.Tensor(data_raw[zu_str+zl_str].values[target_af_idx, :])*y_scale - meanX[:28*2]
        target_af_geom = target_af_geom.tile((num_points,1))
        
    else:
        # manual loading, implement later
        target_xloc_u = torch.Tensor(airfoil[0]).reshape((1,-1))
        target_xloc_l = torch.Tensor(airfoil[1]).reshape((1,-1))
        target_af_geom = torch.hstack((target_xloc_u, target_xloc_l))*y_scale - meanX[:28*2]
        target_af_geom = target_af_geom.tile((num_points,1))
    # Final output 
    if manual_point is None:
        # Define others
        xc_u = torch.linspace(-1, 1, num_points//2+1).reshape((num_points//2+1,1)) 
        xc_l = torch.linspace((1.0 - (xc_u[1].item()-xc_u[0].item())), -1, num_points//2).reshape((num_points//2,1))
        yc_u = torch.sin(torch.arccos(xc_u))
        yc_l = -torch.sin(torch.arccos(xc_l))
        a = torch.ones((num_points, 1))*alpha # angle of attack
        m = torch.ones((num_points, 1))*mach # mach
    
        test_array = torch.hstack((target_af_geom, a, m, torch.vstack((xc_u, xc_l)), torch.vstack((yc_u, yc_l)))).to(output_device)
    else: 
        a = torch.ones((num_points, 1))*alpha # angle of attack
        m = torch.ones((num_points, 1))*mach # mach
        test_array = torch.hstack((target_af_geom, a, m, manual_point[:, 0].reshape((-1,1)), manual_point[:, 1].reshape((-1,1)))).to(output_device)
    
    if out_x == False:
        return test_array 
    elif out_x == True:
        return test_array, (xc_u, xc_l)

def plot_posterior(xc, posterior_predictive, scale = 1, mean = 0, cutoff = None, color='r'):
    f, ax = plt.subplots(1,1)
    
    # obtain cutoff 
    if cutoff is None:
        cutoff = xc.shape[0]//2
    
    xc_u, xc_l = (xc[:cutoff].cpu()+1)/2, (xc[cutoff:].cpu()+1)/2#xc[cutoff:].cpu()
    
    post_mean, post_std = posterior_predictive.mean.cpu(), torch.sqrt(torch.diag(posterior_predictive.covariance_matrix.cpu()))
    mu_u, mu_l = (post_mean[:cutoff] + mean)/scale, (post_mean[cutoff:] + mean)/scale
    std_u, std_l = post_std[:cutoff]/scale, post_std[cutoff:]/scale
    
    # Plot posterior mean
    ax.plot(xc_u, mu_u, color=color, linestyle='-', label = 'DKL GP model', linewidth=2.0)
    ax.plot(xc_l, mu_l, color=color, linestyle='--')
    ax.fill_between(xc_u, mu_u - 2*std_u, mu_u + 2*std_u, color='lightgray', label = 'Predicted $2\sigma$', linewidth=2.0)
    ax.fill_between(xc_l, mu_l - 2*std_l, mu_l + 2*std_l, color='lightgray', linewidth=2.0)
    return (f, ax), (xc_u, xc_l), (mu_u, mu_l), (std_u, std_l)

def aggregate_posterior(test_data, weights_list, get_cl=False):
    n_ = len(weights_list)
    mean = torch.zeros(test_data.shape[0])
    covar = torch.zeros(test_data.shape[0], test_data.shape[0]) 
    for n in range(0, n_):
        load_checkpoint('./weights/' + weights_list[n], model, optimizer)
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            # print(test_data.get_device())
            preds = model(test_data)
            mean += preds.mean.cpu()
            covar += preds.covariance_matrix.cpu()
            
            if get_cl:
                1 # Implementing later for analytical
    
    mean = mean/n_
    covar = covar/n_
    agg_posterior_predictive_dist = gpytorch.distributions.MultivariateNormal(mean, covar + torch.eye(mean.shape[0])*1e-4)
    
    if get_cl:
        return 1
    elif get_cl == False:
        return agg_posterior_predictive_dist 
#%% Import Data & Preprocess 
# Useful variables 
y_scale = 10

# Read data 
"""
Make sure to unpack the rar file
"""
data_raw = pd.read_csv('./data/ASPIRE_subsample_data.csv', low_memory=False) # Read csv 

# Pre-process data 
def detect_num(s):
    return any(i.isdigit() for i in s)

## Identify end of coordinates  
for num_idx in range(data_raw.columns.shape[0]):
    out_ = detect_num(data_raw.columns[num_idx])
    if out_ == False:
        num_xc_str = data_raw.columns[num_idx-1]
        num_xc = int(num_xc_str.split('_')[-1])
        break
## Extract column var names for airfoil geometry (z)
zu_str = []
zl_str = []
for i in range(1, num_xc+1):
    zu_str.append('z_u_' + str(i))
    zl_str.append('z_l_' + str(i))
## Define remaining variables to be extracted
rem_str = ['alpha', 'M', 'xc', 'yc'] # angle of attach, Mach, conformal x, conformal y
input_idx = zu_str + zl_str + rem_str

# Convert input data to tensor and other general conversions 
input_df = data_raw[input_idx]
X = torch.Tensor(input_df.values)
X[:, -4] = torch.deg2rad(X[:, -4]) # Convert AoA to radians
X[:, :-4] *= y_scale # Scale up
meanX = torch.mean(X, axis=0) # Subtracting by mean, not used for this code 
meanX[:] = 0 
X -= meanX

noise = (torch.Tensor(data_raw['noise'])*y_scale)**2 # Define noise values from experiments

# Extract supplementary data
af = data_raw['af']
cat = data_raw[['symmetry', 'supercritical']]
af_unique = np.unique(af)

# Train - test split per airfoil 
if random_test: 
    # Randomly select test set 
    train_afu, test_afu = train_test_split(af_unique, test_size = .1, random_state = 1) # fix this later
elif random_test == False:
    # Manual override of the test set 
    test_afu = ['SC 1095','Supercritical airfoil 9a', 'NACA 63-415'] # 'RISO-A1-21'
## Remove manually chosen af from af list
train_afu = np.delete(af_unique, np.argwhere(af_unique==test_afu[0]))
for i in range(1, len(test_afu)):
    train_afu = np.delete(train_afu, np.argwhere(train_afu==test_afu[i])) 
## Identify corresponding indices
train_idx = af.isin(train_afu).values
test_idx = af.isin(test_afu).values
## Only use subset?
def get_subset(train_indices, test_indices, subset_bounds):
    M_bounds, A_bounds = subset_bounds[0], subset_bounds[1]
    if subset_bounds[0] is not None:
        train_idx = np.logical_and(input_df['M'] <= M_bounds[1], train_indices)
        train_idx = np.logical_and(input_df['M'] >= M_bounds[0], train_idx)
        test_idx =  np.logical_and(input_df['M'] <= M_bounds[1], test_indices)
        test_idx =  np.logical_and(input_df['M'] >= M_bounds[0], test_idx)
    else:
        train_idx = train_indices
        test_idx = train_indices
    if subset_bounds[1] is not None:
        train_idx = np.logical_and(input_df['alpha'] <= A_bounds[1], train_idx)
        train_idx = np.logical_and(input_df['alpha'] >= A_bounds[0], train_idx)
        test_idx =  np.logical_and(input_df['alpha'] <= A_bounds[1], test_idx)
        test_idx =  np.logical_and(input_df['alpha'] >= A_bounds[0], test_idx)
    return train_idx, test_idx
train_idx, test_idx = get_subset(train_idx, test_idx, [(0.0, 0.73), (-4.0, 12.0)])

# Pre-process training targets 
y = torch.Tensor(data_raw['Cp'].values) * y_scale
y_mean = torch.mean(y)
y -= y_mean

# Train - test split 
train_x, test_x, train_y, test_y, train_af, test_af = X[train_idx], X[test_idx], y[train_idx], y[test_idx], data_raw['af'][train_idx], data_raw['af'][test_idx]
train_noise, test_noise, train_cat, test_cat = noise[train_idx], noise[test_idx], cat[train_idx], cat[test_idx]
## push to output device 
train_x, train_y  = train_x.to(output_device), train_y.to(output_device)
#%% Define Model
data_dim = train_x.size(-1)
nn_dims = [1000, 1000, 500, 50, 10] 

from gpytorch.priors import NormalPrior

## Main DKL Model
class DKL_GP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(DKL_GP, self).__init__(train_x, train_y, likelihood)
            
            # Mean Module
            self.mean_module = gpytorch.means.ConstantMean()
            
            # Covariance module 
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=5/2, ard_num_dims=nn_dims[-1]),
                )

            # NN Feature Extractor Module
            self.feature_extractor = LargeFeatureExtractor()
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0) # scale the feature extractor outputs 

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x) # .to(output_device)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice" .to(output_device)

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
## Feature Extractor 
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, nn_dims[0])) # __
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('dropout1', torch.nn.Dropout(0.2))
        self.add_module('linear2', torch.nn.Linear(nn_dims[0], nn_dims[1]))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('dropout2', torch.nn.Dropout(0.2))
        self.add_module('linear3', torch.nn.Linear(nn_dims[1], nn_dims[2]))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('dropout3', torch.nn.Dropout(0.2))
        self.add_module('linear4', torch.nn.Linear(nn_dims[2], nn_dims[3]))
        if len(nn_dims) > 4:
            self.add_module('relu4', torch.nn.ReLU())
            self.add_module('dropout4', torch.nn.Dropout(0.2))
            self.add_module('linear5', torch.nn.Linear(nn_dims[3], nn_dims[4]))
        
# Define model & likelihood  
noise_prior = NormalPrior(0.0, 1.0)
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_noise, learn_additional_noise=True, noise_prior=noise_prior) 
likelihood.second_noise = 0.3
model = DKL_GP(train_x, train_y, likelihood)

## Push to output device 
model = model.to(output_device)
model.feature_extractor = model.feature_extractor.to(output_device)
likelihood = likelihood.to(output_device)

# Define Optimizer  
lr = 1e-3
decay = 1e-4 
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=lr) #  , weight_decay=decay
from torch.optim.lr_scheduler import StepLR, ConstantLR
scheduler = StepLR(optimizer, step_size=1000, gamma = 0.5)

# Set up run or load checkpoint 
load_checkpoint_bool = False # Set this to true if you want to load the best weights
def load_checkpoint(file_path, model, optimizer):
    checkpt = torch.load(file_path, map_location=output_device) 
    model.load_state_dict(checkpt['model_state_dict'])
    optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    history_values = checkpt['history']
    return history_values

def calc_err(true_val, pred_val, err_type = 'MAE'):
        if err_type == 'MAE':
            err = torch.mean(torch.abs(true_val - pred_val))
        if err_type == 'MSE':
            # not implemented yet 
            err = 1
        return err

#%% Train Model 
# Due to the very large covariance matrix, approx 60 GB GPU is required to train the model fully.
import tqdm.notebook as tn 
torch.cuda.empty_cache()

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Set up hyperparameters history
if 'history_values' not in locals():
    history_values = {
        'test_err': np.array([]),
        'train_err': np.array([]),
        'mll': np.array([]),
        'mll_all': np.array([]),
        'n': np.array([])
    }
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 
save_tag = 'output'

mem = []
mem_model = []
save_pt_mll = []
save_pt_test = []
save_pt_train = []
save_pt_iter = []

best_loss = float('inf') # loss
best_model_state_dict = None
iters = 0
iters_checkpt = 0

def train_and_save_checkpoints(begin_cycle): 
    global best_loss, best_model_state_dict, best_optim_state_dict, iters, iters_checkpt, mem, scheduler
    sub_iter = 0
    trpz = []
    iterator = tn.tqdm(range(training_iterations))

    for i in iterator:
        model.train()
        likelihood.train()
        
        # Zero backprop gradients
        optimizer.zero_grad()
     
        # Get output from model
        output = model(train_x)
        
        # Calculate loss 
        loss = -mll(output, train_y)  
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        history_values['mll_all'] = np.append(history_values['mll_all'], loss.cpu().detach().numpy())
        
        # Save checkpoint 
        if (i+1)%interval_checkpt == 0:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
                # Calculate MAE 
                validation_results = model(test_x.to(output_device)) # temp_model(temp_test)#
                train_results = model(train_x) #  temp_model(temp_train)#
                history_values['mll'] = np.append(history_values['mll'], loss.cpu().detach().numpy())
                history_values['test_err'] = np.append(history_values['test_err'], calc_err(validation_results.mean.cpu(), test_y)) # Force cpu 
                history_values['train_err'] = np.append(history_values['train_err'], calc_err(train_results.mean.cpu(), train_y.cpu())) 
                history_values['n'] = np.append(history_values['n'], iters)
                
                # Print Losses and MAE 
                print('Loss: {}'.format(history_values['mll'][iters_checkpt]) + 
                    ' / Train MAE: {}'.format(history_values['train_err'][iters_checkpt]) + 
                    ' / Test MAE: {}'.format(history_values['test_err'][iters_checkpt]) + 
                    ' / Noise: {}'.format(np.sqrt(likelihood.second_noise_covar.noise.item())))
                mem.append(history_values['test_err'][iters_checkpt])
                mem_model.append([model, optimizer, history_values, iters])
                trpz.append(torch.abs(torch.trapz(validation_results.mean.cpu()) - torch.trapz(test_y)))
                # Continuously update minimum 
                
                min_idx = np.argmin(mem)
                min_idx_sub = len(mem)-1-np.argmin(mem)
                fileDir = './' + gen_save_string(save_tag, lr, 0.0, nn_dims, (history_values['mll'][iters_checkpt-min_idx_sub], history_values['train_err'][iters_checkpt-min_idx_sub], 
                                                                                            history_values['test_err'][iters_checkpt-min_idx_sub]), iters_checkpt-min_idx_sub)
                
                if sub_iter >= 199:
                    save_pt_mll.append(history_values['mll'][iters_checkpt-min_idx_sub])
                    save_pt_train.append(history_values['train_err'][iters_checkpt-min_idx_sub])
                    save_pt_test.append(history_values['test_err'][iters_checkpt-min_idx_sub])
                    save_pt_iter.append(iters_checkpt-min_idx_sub)
                    save_checkpoint(model=mem_model[min_idx][0], optimizer=mem_model[min_idx][1], history=mem_model[min_idx][2], fileName=fileDir, iters=mem_model[min_idx][3])
                    optimizer.param_groups[0]['lr'] = 1e-3
                    scheduler = StepLR(optimizer, step_size=500, gamma=0.999)
                    mem, trpz = [], []
                    sub_iter = 0 
                    print('New cycle')

                iters_checkpt += 1 
        optimizer.step()
        scheduler.step()
        if i >= begin_cycle:
            sub_iter += 1 
        iters += 1 

if train_new:
    # Run Training 
    train_and_save_checkpoints(swa_startpt)

    # Plot Results 
    f, ax = plt.subplots(1,2, figsize=(12, 4))
    # Loss vs iterations
    ax[0].semilogy(history_values['mll_all'])
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Marginal Log Likelihood')
    ax[0].title.set_text('Loss vs. Iterations')

    # Validation error vs iterations
    ax[1].semilogy(np.arange(interval_checkpt, iters+1, interval_checkpt), np.array(history_values['test_err'][:]),'--', label='Test Data')
    ax[1].legend()

    # Labels
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('MAE')
    ax[1].title.set_text('MAE vs. Iterations')
    plt.show()

#%% See predictions for Cp 
model.eval()
likelihood.eval()
weights_ = []

# If train new 
if train_new:
    weight_sweep_inds = np.lexsort((save_pt_train, save_pt_test, save_pt_mll))
    for i in range(0, swa_load_num):
        load_str = gen_save_string(save_tag, lr, 0.0, nn_dims, (save_pt_mll[weight_sweep_inds[i]], 
                                                                        save_pt_train[weight_sweep_inds[i]], 
                                                                        save_pt_test[weight_sweep_inds[i]]), 
                                    save_pt_iter[weight_sweep_inds[i]], override_date=None)
        weights_.append(load_str)
else:
    weights_ = ['weights_1', 
                'weights_2',
                'weights_3',] 
    
num_figs = 3
# 0: NACA 63-415, 1: Supercritical airfoil 9a, 2: SC1095
_, _ = validate_predictions(test_x, test_y, test_af, 0, num_figs, weights_list=weights_, save_figs=False)
#%% Code for Calculating Cn, Ca, Cm
# Use the following if you want to calculate coefficients
import scipy
upper_xc_af = np.flip(np.array(data_raw.values[0,:28], dtype = 'float'))
lower_xc_af = np.array(data_raw.values[0, 28:28*2], dtype = 'float')

def brute_cn(samples, samples_af):
    cn = []
    u_bound = samples_af.shape[0]//2+1
    l_bound = samples_af.shape[0]//2
    samples = samples.detach().numpy()
    upper_xc = (samples_af[:u_bound,-2].cpu().detach().numpy()+1)/2
    lower_xc = np.flip(samples_af[l_bound:,-2].cpu().detach().numpy()+1)/2
    yu = np.interp(x = upper_xc, xp=upper_xc_af, fp=np.flip(samples_af[0,:28].cpu().detach().numpy()/y_scale))
    yl = np.interp(x = lower_xc, xp=lower_xc_af         , fp=samples_af[0, 28:28*2].cpu().detach().numpy()/y_scale)

    dyudx = np.gradient(yu, upper_xc)
    dyldx = np.gradient(yl, lower_xc)
    
    for i in np.arange(0, samples.shape[0]):
        cn.append(-np.trapz(samples[i, :u_bound]/y_scale, x=upper_xc) + np.trapz(np.flip(samples[i, l_bound:])/y_scale, x=lower_xc) ) # -np.trapz(samples[i, :u_bound]/y_scale, x=upper_xc) + np.trapz(np.flip(samples[i, l_bound:])/y_scale, x=lower_xc) 
    return np.mean(cn), np.std(cn)

def brute_cm(samples, samples_af, loc=0.25):
    cl = []
    samples = samples.detach().numpy()
    u_bound = samples_af.shape[0]//2+1
    l_bound = samples_af.shape[0]//2
    upper_xc = (samples_af[:u_bound,-2].cpu().detach().numpy()+1)/2
    lower_xc = np.flip(samples_af[l_bound:,-2].cpu().detach().numpy()+1)/2

    yucs = scipy.interpolate.CubicSpline(upper_xc_af, np.flip(samples_af[0,:28].cpu().detach().numpy()/y_scale))#np.interp(x = upper_xc, xp=np.flip(upper_xc_af), fp=np.flip(samples_af[0,:28].cpu().detach().numpy()/y_scale))
    yu = yucs(upper_xc)
    ylcs = scipy.interpolate.CubicSpline(lower_xc_af, samples_af[0, 28:28*2].cpu().detach().numpy()/y_scale)
    yl = ylcs(lower_xc)

    dyudx = np.gradient(yu, upper_xc, edge_order=1)
    dyldx = np.gradient(yl, lower_xc, edge_order=1)

    for i in np.arange(0, samples.shape[0]):
        term1 = -np.trapz(samples[i, :u_bound]/y_scale*(loc - upper_xc), x=upper_xc) + np.trapz(np.flip(samples[i, l_bound:])/y_scale*(loc - lower_xc), x=lower_xc)
        term2 = np.trapz(samples[i, :u_bound]/y_scale * dyudx * yu, x=upper_xc) - np.trapz(samples[i, l_bound:]/y_scale * dyldx * yl, x=lower_xc)
        cl.append(term1 + term2)
    return np.mean(cl), np.std(cl)

def brute_ca(samples, samples_af, manual_coords = None):
    ca = []
    samples = samples.detach().numpy()
    u_bound = samples_af.shape[0]//2+1
    l_bound = samples_af.shape[0]//2
    upper_xc = (samples_af[:u_bound,-2].cpu().detach().numpy()+1)/2
    lower_xc = np.flip(samples_af[l_bound:,-2].cpu().detach().numpy()+1)/2
    yucs = scipy.interpolate.CubicSpline(upper_xc_af, np.flip(samples_af[0,:28].cpu().detach().numpy()/y_scale))#np.interp(x = upper_xc, xp=np.flip(upper_xc_af), fp=np.flip(samples_af[0,:28].cpu().detach().numpy()/y_scale))
    yu = yucs(upper_xc)
    ylcs = scipy.interpolate.CubicSpline(lower_xc_af, samples_af[0, 28:28*2].cpu().detach().numpy()/y_scale)
    yl = ylcs(lower_xc)#np.interp(x = lower_xc, xp=lower_xc_af         , fp=samples_af[0, 28:28*2].cpu().detach().numpy()/y_scale)
    if manual_coords is None:
        dyudx = np.gradient(yu, upper_xc, edge_order=2)
        dyldx = np.gradient(yl, lower_xc, edge_order=2)
    else:
        dyudx = np.interp(upper_xc, manual_coords[0], np.gradient(manual_coords[1], manual_coords[0], edge_order=1))# - np.tan(ang)
        dyldx = np.interp(lower_xc, manual_coords[2], np.gradient(manual_coords[3], manual_coords[2], edge_order=1))# - np.tan(ang)
    for i in np.arange(0, samples.shape[0]):
        ca.append(np.trapz(samples[i, :u_bound]/y_scale*dyudx, x=upper_xc) - np.trapz(np.flip(samples[i, l_bound:]/y_scale)*dyldx, x=lower_xc)) # np.trapz(((samples[i, :u_bound-1]+samples[i, 1:u_bound])/2)*dyudx, x=upper_xc) 
    return np.mean(ca), np.std(ca)
