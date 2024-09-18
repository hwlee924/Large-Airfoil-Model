# import spaces
#%%
import json
import gpytorch
import numpy as np
import warnings
import torch 
# import gradio as gr 
import time 
import json 
#%%
""" Converts website inputs to the model input """
def convert_inputs(data, num_points = 121, out_x=False, use_gpu=False):
    mach_ = data['mach'] # Mach 
    alph_ = data['angle'] # Angle of attack 
    xcyc_ = torch.tensor(data['coordinates']) # Coordinates
    scaler = 10 # Scaled in the model

    assert xcyc_[:,1].shape[0] == 28
    assert xcyc_[:,2].shape[0] == 28

    dyudx = torch.tensor(np.gradient(xcyc_[:,1].detach().numpy(), xcyc_[:,0].detach().numpy()))
    dyldx = torch.tensor(np.gradient(xcyc_[:,2].detach().numpy(), xcyc_[:,0].detach().numpy()))
    if use_gpu:
        dyudx = dyudx.cuda()
        dyldx = dyldx.cuda()
    af_geom = torch.hstack((torch.flip(xcyc_[:,1].flatten(), [0]).reshape((1,-1)), xcyc_[:,2].flatten().reshape((1,-1)))) * scaler

    # Generate points in conformal domain
    xc_u = -2*(torch.sin(torch.linspace(0,1,num_points//2+1)*np.pi/2)-0.5).reshape((num_points//2+1,1))  #torch.linspace(-1, 1, num_points//2+1).reshape((num_points//2+1,1)) 
    xc_l = torch.flip(xc_u[:-1] ,(0,))
    yc_u = torch.sin(torch.arccos(xc_u))
    yc_l = -torch.sin(torch.arccos(xc_l))
    
    alph = torch.ones((num_points, 1))*np.deg2rad(alph_) # angle of attack
    mach = torch.ones((num_points, 1))*mach_ # mach
    
    # Combine 
    new_data = torch.hstack((torch.tile(af_geom, (num_points, 1)), alph, mach, torch.vstack((xc_u, xc_l)), torch.vstack((yc_u, yc_l))))
    if use_gpu:
        new_data = new_data.cuda()
    return new_data, np.vstack(((xc_u+1)/2, (xc_l+1)/2)).flatten(), (dyudx, dyldx)

""" Define DKL model as a gpytorch model"""
class DKL_GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        nn_dims = [1000, 1000, 500, 50, 10] 
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
        projected_x = self.feature_extractor(x)  
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice" 
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def project_input(self, x):
        projected_x = self.feature_extractor(x)  
        projected_x = self.scale_to_bounds(projected_x)
        return projected_x 
""" deep neural network """ 
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        data_dim = 60 
        nn_dims = [1000, 1000, 500, 50, 10] 
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

""" Load model and weights """
def load_model(use_gpu=False):
    # Load likelihood
    noise_prior = gpytorch.priors.NormalPrior(0.0, 1.0)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_noise*100, learn_additional_noise=True, noise_prior=noise_prior) # Placeholder likelihood to initialize model
    
    # Initialize model 
    model = DKL_GP(train_x, train_y, likelihood)
    
    # Set to eval mode
    model.to(torch.float32)
    if use_gpu: 
        model.cuda()
    model.eval()
    likelihood.eval()
    return model, likelihood 

""" Load pre-saved pickles (K(x,x) + \sigma I)^-1"""
def load_global_model_params():
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore") # warning suppression is required for website
        train_x, train_y, train_noise = torch.load('./model/train.pt', map_location='cpu')
        L = torch.load('./model/lam_L.pt', map_location='cpu')
    return L, train_x, train_y, train_noise

""" Do operations with model """
def airfoil_model(data, use_gpu=False):
    # Load model
    model, likelihood = load_model(use_gpu=use_gpu)

    # Calculate posterior
    (mu, cov), (cn, cn_std), (cm, cm_std), (ca, ca_std) = calc_posterior(model, data, use_gpu=use_gpu)

    mu = mu.cpu().detach().numpy().astype(np.float32)
    std = torch.sqrt(torch.diag(cov)).cpu().detach().numpy().astype(np.float32)
    std[np.isnan(std)] = 1e-4

    # Generate output files 
    # af_info = ['airfoil,'+af_name, 'alpha,'+str(np.round(alpha,2)), 'mach,'+str(np.round(mach,2)), '', ',value, 2 sigma', 'c_n,'+str(np.round(cn,3))+','+str(np.round(2*cn_std,3)), 'c_a,'+str(np.round(ca,3))+','+str(np.round(2*ca_std,3)), 'c_m,'+str(np.round(cm,3))+','+str(np.round(2*cm_std,3)), '', 'C_p Distribution']
    # filename = "output.csv"
    # with open(filename, 'w') as f:
    #     for line in af_info:
    #         f.write(line + "\n")
    # with open(filename, 'a') as f:
    #     np.savetxt(f, np.hstack((xc.reshape((-1,1)), mu.reshape((-1,1)), std.reshape((-1,1)))), delimiter=',', header='x/c, Cp, 2 sigma', comments='')

    results = {
        'xc': xc.astype(np.float32).tolist(),
        'cp': mu.tolist(),
        'lo': (mu - 2*std).tolist(),
        'hi': (mu + 2*std).tolist(),
        'cn': cn.astype(np.float32).tolist(),
        'cn_2std': (2*cn_std).astype(np.float32).tolist(),
        'ca': ca.astype(np.float32).tolist(),
        'ca_2std': (2*ca_std).astype(np.float32).tolist(),
        'cm': cm.astype(np.float32).tolist(),
        'cm_2std': (2*cm_std).astype(np.float32).tolist(),
    }
    return results, model, likelihood

""" Calculate posterior mean and covariance GP """
def calc_posterior(model, test_data, use_gpu=False):
    start_time = time.time()
    with torch.no_grad():
        scaler = 10
        Kxs = gpytorch.lazy.ZeroLazyTensor((train_y.shape[0], test_data.shape[0]))
        Kss = gpytorch.lazy.ZeroLazyTensor((test_data.shape[0], test_data.shape[0]))
        
        for i in [1, 2, 3]:
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore") # warning suppression is required
                checkpt = torch.load('./model/weights_'+str(i), map_location='cpu'); 
            model.load_state_dict(checkpt['model_state_dict'])
            
            proj_train_ = model.project_input(train_x)
            proj_test_  = model.project_input(test_data)
            
            Kxs += model.covar_module(proj_train_, proj_test_)
            Kss += model.covar_module(proj_test_)
        
        Kxs /= 3.0
        Kss /= 3.0
        y_mean = -2.7447 # This may need to be a saved number later
    
        if use_gpu:
            Kxs = Kxs.to_dense()
            A = torch.linalg.solve_triangular(L.cuda(), Kxs.cuda(), upper=False)
            Kss = Kss.to_dense()
            v = torch.linalg.solve_triangular(L.cuda(), train_y.reshape((-1,1)), upper=False)
            
            mu = A.T @ v
            cov = Kss.cuda() - A.T @ A
        else: 
            Kxs = Kxs.to_dense()
            A = torch.linalg.solve_triangular(L, Kxs, upper=False)
            Kss = Kss.to_dense()
            v = torch.linalg.solve_triangular(L, train_y.reshape((-1,1)), upper=False)

            mu = A.T @ v
            cov = Kss - A.T @ A
        
        out_Cp = (mu.flatten() + y_mean)/scaler
        out_cov = cov/(scaler**2)
         
        
        #
        num_points = xc.shape[0]
        upper_xc_, lower_xc_ = xc[:num_points//2+1].flatten(), xc[num_points//2+1:].flatten()
        yu = torch.tensor(np.interp(upper_xc_, np.array(coords)[:,0], np.array(coords)[:,1]))
        yl = torch.tensor(np.interp(lower_xc_, np.array(coords)[:,0], np.array(coords)[:,2]))
        dyudx = torch.tensor(np.interp(upper_xc_, np.array(coords)[:,0], np.gradient(np.array(coords)[:,1], np.array(coords)[:,0])))
        dyldx = torch.tensor(np.interp(lower_xc_, np.array(coords)[:,0], np.gradient(np.array(coords)[:,2], np.array(coords)[:,0])))
        upper_xc = torch.tensor(upper_xc_)
        lower_xc = torch.tensor(lower_xc_)
        
        
        jitter = torch.eye(out_cov.shape[0])*1e-4
        if use_gpu:
            yu, yl, dyudx, dyldx = yu.cuda(), yl.cuda(), dyudx.cuda(), dyldx.cuda()
            upper_xc, lower_xc = upper_xc.cuda(), lower_xc.cuda()
            jitter = jitter.cuda()
            
        # Get distribution  
        distr = torch.distributions.multivariate_normal.MultivariateNormal(loc=out_Cp.flatten(), covariance_matrix=out_cov+jitter)
        num_samples = 100
        samples = distr.sample((num_samples,))

        cn, cm, ca = [], [], []
        for i in np.arange(0, num_samples):    
            Cp_upper, Cp_lower = samples[i, :num_points//2+1].flatten(), samples[i, num_points//2+1:].flatten()
            # Calculate Cn
            cn.append((torch.trapz(y=Cp_upper, x=upper_xc) + torch.trapz(y=Cp_lower, x=lower_xc)).cpu().detach().numpy())
            
            # Calculate Cm
            cm_term1 = -(torch.trapz(y=Cp_upper*(upper_xc - .25), x=upper_xc) + torch.trapz(y=Cp_lower*(lower_xc - .25), x=lower_xc)).cpu().detach().numpy()
            cm_term2 = (-torch.trapz(y=Cp_upper*dyudx*yu, x=upper_xc) - torch.trapz(y=Cp_lower*dyldx*yl, x=lower_xc)).cpu().detach().numpy()
            cm.append(cm_term1 + cm_term2)
            
            # Calculate Ca 
            ca.append((torch.trapz(y=torch.flip(Cp_upper*dyudx, (0,)), x=torch.flip(upper_xc, (0,))) - torch.trapz(Cp_lower*dyldx, x=lower_xc)).cpu().detach().numpy())
        
        cn_mu, cn_std = np.mean(cn), np.std(cn)
        cm_mu, cm_std = np.mean(cm), np.std(cm)
        ca_mu, ca_std = np.mean(ca), np.std(ca)
        end_time = time.time()
        print(end_time - start_time)
    return (out_Cp, out_cov), (cn_mu, cn_std), (cm_mu, cm_std), (ca_mu, ca_std)

""" The main predict function """
def predict(inputs):
    use_gpu = False
    global L, train_x, train_y, train_noise  # Declare global variables
    # push globals to gpu if available
    if use_gpu==True:
        L, train_x, train_y, train_noise = L.cuda(), train_x.cuda(), train_y.cuda(), train_noise.cuda()
    # input_data = json.loads(inputs) # Load json from website input 
    with open(inputs) as f:
        input_data = json.load(f)
        print('Done loading json')
    global xc, af_name, alpha, mach, grads, coords 
    af_name, alpha, mach, coords, output_reso = input_data['name'], input_data['angle'], input_data['mach'], input_data['coordinates'], input_data['resolution'] # Extract information
    converted_input, xc, grads = convert_inputs(input_data, num_points=output_reso, use_gpu=use_gpu) # Convert to model input 
    results, _, _  = airfoil_model(converted_input, use_gpu=use_gpu) # Calculate results
    
    a = np.hstack((np.array(results['xc'])[:,None], np.array(results['cp'])[:,None]))
    np.savetxt("output.csv", a, delimiter=",")
    return json.dumps(results) # output json

global L, train_x, train_y, train_noise
L, train_x, train_y, train_noise = load_global_model_params()
print('Done loading files ')

# %%
predict('sample.json')

# %%
