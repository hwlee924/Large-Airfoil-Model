#%% 
import gpytorch
import numpy as np
import warnings
import torch  
import os  
import pandas as pd 
torch.set_default_dtype(torch.float32)
#%%
""" Defines the DKL model as a gpytorch model class"""
class lam_adapt(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """ 
        train_x = training data - inputs | torch.tensor
        train_y = training data - targets | torch.tensor 
        likelihood = likelihood model of this GP | gpytorch.likelihood class 
        """
        nn_dims = [1000, 1000, 500, 50, 10] # fully connected neural network dimension for each layer 
        # 10 output latent variables 

        # keep track of weights files 
        self.weights_dir = './model/'

        # run checks here 
        assert train_x.shape[1] == 60, 'Incorrect train_x dimension' # check dimension of train_x 
        assert train_y.shape[0] == train_x.shape[0], 'Dimensions of train_y does not equal to that of train_x' # make sure train_y agrees with train_x 
        assert isinstance(train_x, torch.Tensor), 'Input must be a Tensor'
        assert isinstance(train_y, torch.Tensor), 'Input must be a Tensor'
        assert os.path.exists(self.weights_dir), 'model directory does not exist'
        # initialize 
        super(lam_adapt, self).__init__(train_x, train_y, likelihood)

        # mean module
        self.mean_module = gpytorch.means.ConstantMean() # constant mean 
        
        # covariance module 
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.MaternKernel(nu=5/2, ard_num_dims=nn_dims[-1]),
            ) # ARD Matern 5/2 kernel
        
        # NN Feature Extractor module
        self.feature_extractor = fcnn_layer(train_x, nn_dims)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0) # scale the feature extractor outputs 

        self.airfoil_coordinates = None 
        self.scale = 10 

    """ default forward method in gpytorch.model class, not used due to memory costraints"""
    def forward(self, x):
        """
        x: test data | torch.Tensor 
        ---
        returns a MultivariateNormal distribution based on the mean and covariance 
        """
        # Neural network layer 
        projected_x = self.feature_extractor(x)  # feed model input thru NN 
        projected_x = self.scale_to_bounds(projected_x)  # Make the output NN values "nice" to work with 
        # Gaussian process layer 
        mean_x = self.mean_module(projected_x) # mean from latent vars
        covar_x = self.covar_module(projected_x) # covariance from latent vars
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    """ helper function: feeds input data through feature extractor """
    def project_input(self, x):
        """
        x: torch.Tensor object to feed through feature extractor
        ---
        returns the values after pushing the input through the neural network + scaling
        """
        projected_x = self.feature_extractor(x)  # feed model input thru NN 
        projected_x = self.scale_to_bounds(projected_x) 
        return projected_x 
    
    def load_weights(self, weight_num):
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore") # warning suppression is required  
            checkpt = torch.load(self.weights_dir + 'weights_' + str(weight_num), map_location='cpu');    
        self.load_state_dict(checkpt['model_state_dict'])
    
    def predict(self, test_data, get_coeff=False, coeff_samples = 10000):
        """
        test_data: the data to perform prediction on | input_data (automated tensor generation) class or torch.Tensor (manual)
        get_coeff (optional): calculate aerodynamic forces and moment coefficients from Cp via MonteCarlo | Boolean
        coeff_samples (optional): number of samples to use for Monte Carlo coefficient calculation
        (Not used currently) num_weights (optional): number of weights used in the posterior averaging, improves prediction robustness, higher computational cost | int up to 3
        ---
        model_output: a dict containing output information 
            cp_distribution: model posterior predictive distribution | gpytorch.distributions.MultivariateNormal
            xc: x/c location for cp_distribution, mostly for plotting | torch.Tensor
            'cX_mean': coefficient, mean
            'cX_stdev': coefficient, standard deviation
        """
        # generate input tensor to the model 
        if isinstance(test_data, input_data):
            test_x = test_data.assemble_tensor().to(torch.float32)
        elif isinstance(test_data, torch.Tensor):
            test_x = test_data.to(torch.float32)
        else:
            raise ValueError('Invalid input data')
        
        # check 
        assert isinstance(get_coeff, bool), 'get_coeff must be a bool' 
        # push to gpu if used 
        if self.use_gpu:
            test_x = test_x.cuda()
        # begin prediction
        with torch.no_grad():
            Kxs = gpytorch.lazy.ZeroLazyTensor((train_y.shape[0], test_x.shape[0]))
            Kss = gpytorch.lazy.ZeroLazyTensor((test_x.shape[0], test_x.shape[0]))
            # for loop thru each weights and calculate posterior
            num_weights = 3
            for i in range(1, num_weights+1): 
                # load in weights 
                self.load_weights(i) 
                # calculate Kxs Kss
                proj_train_ = self.project_input(train_x)
                proj_test_  = self.project_input(test_x)
                Kxs += self.covar_module(proj_train_, proj_test_)
                Kss += self.covar_module(proj_test_, proj_test_)
            # do weight averaging
            Kxs /= num_weights
            Kss /= num_weights
            y_mean = -2.7447

            if self.use_gpu:
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
            
            posterior_cp = (mu.flatten() + y_mean)/self.scale
            posterior_cov = cov/(self.scale**2)
            posterior_dist = gpytorch.distributions.MultivariateNormal(posterior_cp, posterior_cov + torch.eye(posterior_cov.shape[0])*1e-4)
            xc_loc = (test_x[:, -2] + 1)/2
            if get_coeff==False:
                model_output = { 
                    'cp_distribution': posterior_dist,
                    'xc': xc_loc,
                }
                return model_output
            else:
                posterior_samples = posterior_dist.sample(torch.Size([coeff_samples]))

                def get_coeff_samples(coefficient_str, test_data, cp_samples):
                    """ get coefficients from samples via MC """
                    xc_u, xc_l = xc_loc[:test_data.num_pts[0]], xc_loc[test_data.num_pts[0]:]
                    samples_u, samples_l = cp_samples[:, :test_data.num_pts[0]], cp_samples[:, test_data.num_pts[0]:]
                    dzdx_u, dzdx_l = torch.from_numpy(np.gradient(test_data.splines[0](xc_u), xc_u)), torch.from_numpy(np.gradient(test_data.splines[1](xc_l), xc_l))
                    z_u, z_l = torch.from_numpy(test_data.splines[0](xc_u)), torch.from_numpy(test_data.splines[1](xc_l))
                    
                    # match coefficient_str: # for python 3.10 upwards
                    #     case 'ca':
                    #         # upper = torch.trapz(y=samples_u * dzdx_u, x=xc_u, dim=1)
                    #         # lower = torch.trapz(y=samples_l * dzdx_l, x=xc_l, dim=1)

                    #         dzdx_u = torch.diff(z_u) / torch.diff(xc_u)
                    #         dzdx_l = torch.diff(z_l) / torch.diff(xc_l)
                    #         upper = torch.sum(samples_u[:, 1:] * dzdx_u * torch.diff(xc_u), dim=1)
                    #         lower = torch.sum(samples_l[:, :-1] * dzdx_l * torch.diff(xc_l), dim=1)
                    #         coefficient_samples = upper - lower
                    #     case 'cn':
                    #         upper = torch.trapz(y=samples_u, x=xc_u, dim=1)
                    #         lower = torch.trapz(y=samples_l, x=xc_l, dim=1)
                    #         coefficient_samples = lower - upper 
                    #     case 'cm': 
                    #         dzdx_u, dzdx_l = torch.from_numpy(np.gradient(test_data.splines[0](xc_u), xc_u)), torch.from_numpy(np.gradient(test_data.splines[1](xc_l), xc_l))
                    #         term1u = torch.trapz(samples_u*(xc_u - 0.25), x=xc_u)
                    #         term1l = torch.trapz(samples_l*(xc_l - 0.25), x=xc_l)  
                    #         term1 = term1u - term1l
                            
                    #         term2u = torch.trapz((samples_u * dzdx_u) * z_u, x=xc_u)
                    #         term2l = torch.trapz((-samples_l * dzdx_l) * z_l, x=xc_l)
                    #         term2 = term2u + term2l
                    #         coefficient_samples = term1 + term2 
                            
                    if coefficient_str == 'ca':
                        dzdx_u = torch.diff(z_u) / torch.diff(xc_u)
                        dzdx_l = torch.diff(z_l) / torch.diff(xc_l)
                        upper = torch.sum(samples_u[:, 1:] * dzdx_u * torch.diff(xc_u), dim=1)
                        lower = torch.sum(samples_l[:, :-1] * dzdx_l * torch.diff(xc_l), dim=1)
                        coefficient_samples = upper - lower
                    elif coefficient_str == 'cn':
                        upper = torch.trapz(y=samples_u, x=xc_u, dim=1)
                        lower = torch.trapz(y=samples_l, x=xc_l, dim=1)
                        coefficient_samples = lower - upper 
                    elif coefficient_str == 'cm': 
                        dzdx_u, dzdx_l = torch.from_numpy(np.gradient(test_data.splines[0](xc_u), xc_u)), torch.from_numpy(np.gradient(test_data.splines[1](xc_l), xc_l))
                        term1u = torch.trapz(samples_u*(xc_u - 0.25), x=xc_u)
                        term1l = torch.trapz(samples_l*(xc_l - 0.25), x=xc_l)  
                        term1 = term1u - term1l
                        
                        term2u = torch.trapz((samples_u * dzdx_u) * z_u, x=xc_u)
                        term2l = torch.trapz((-samples_l * dzdx_l) * z_l, x=xc_l)
                        term2 = term2u + term2l
                        coefficient_samples = term1 + term2 
                    return coefficient_samples
                ca_samples = get_coeff_samples('ca', test_data, posterior_samples)
                cn_samples = get_coeff_samples('cn', test_data, posterior_samples)
                cm_samples = get_coeff_samples('cm', test_data, posterior_samples)

                cl_samples = cn_samples*np.cos(np.deg2rad(test_data.alph)) - ca_samples*np.sin(np.deg2rad(test_data.alph))
                cd_samples = cn_samples*np.sin(np.deg2rad(test_data.alph)) + ca_samples*np.cos(np.deg2rad(test_data.alph))

                model_output = { 
                    'cp_distribution': posterior_dist,
                    'xc': xc_loc,
                    'cl_mean': torch.mean(cl_samples).item(),
                    'cl_stdev': torch.std(cl_samples).item(),
                    'cd_mean': torch.mean(cd_samples).item(),
                    'cd_stdev': torch.std(cd_samples).item(), 
                    'cm_mean': torch.mean(cm_samples).item(),
                    'cm_stdev': torch.std(cm_samples).item(), 
                }
                return model_output
            
""" Class to control input tensor for the model """
"""
airfoil_input: what airfoil is this? Provide directory to csv file or define "NACA XXXX" | str
alpha: angle of attack [degrees] | float
mach: freestream mach number | float
num_auto_points: number of uniformly distributed points that is auto-generated | float
manual_points: user-provided manual point distribution | list of 2 torch.Tensors
use_gpu = Determine whether to use GPU or not | boolean
"""
from scipy.interpolate import CubicSpline
class input_data():
    def __init__(self, airfoil_input, alpha, mach, num_auto_points=120, manual_points=None, use_gpu=False): 
        # xc locations where airfoil geometry is defined
        self.xc_u = torch.tensor([0, 0.0025, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075, 0.1, 
                                  0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                                    0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0]) # upper surface
        self.xc_l = torch.flip(self.xc_u, [0, ]) # lower surface
        self.mach = mach # mach number
        self.alph = alpha  # angle of attack (degrees)
        self.num_pts = [num_auto_points, num_auto_points] # default resolution
        self.zc_u = None # initialize airfoil coordinates
        self.zc_l = None 
        self.scale = 10
        self.manual_pts = manual_points
        self.use_gpu = use_gpu
        if manual_points is not None: 
            assert isinstance(manual_points[0], torch.Tensor), 'Manually defined points must be a Tensor'
            assert isinstance(manual_points[1], torch.Tensor), 'Manually defined points must be a Tensor'
            assert all(torch.diff(manual_points[0])>0), 'Manually defined points must be in ascending order'
            assert all(torch.diff(manual_points[1])>0), 'Manually defined points must be in ascending order'
        # get airfoil
        # read from csv 
        if os.path.exists(airfoil_input):
            # loaded CSV file 
            loaded_af = pd.read_csv(airfoil_input).values
            loaded_xc, loaded_zc = loaded_af[:,0], loaded_af[:,1] 
            # interpolate to target location
            surf_bound = int(np.argwhere(np.diff(np.sign(np.diff(loaded_xc))) != 0.)[0][0]+2)
            loaded_xcu, loaded_xcl = np.flip(loaded_xc[:surf_bound]), loaded_xc[surf_bound:]
            loaded_zcu, loaded_zcl = np.flip(loaded_zc[:surf_bound]), loaded_zc[surf_bound:]
            self.zc_u = torch.tensor(np.interp(self.xc_u, loaded_xcu, loaded_zcu))
            self.zc_l = torch.tensor(np.interp(self.xc_l, loaded_xcl, loaded_zcl))

            # get numerical gradient 
            cs_u, cs_l = CubicSpline(loaded_xcu, loaded_zcu), CubicSpline(loaded_xcl, loaded_zcl)
            self.splines = [cs_u, cs_l]
            ref_xc = torch.linspace(0., 1., 501)
            temp_zcu, temp_zcl = cs_u(ref_xc), cs_l(ref_xc)
            self.dzdx_u = torch.from_numpy(np.interp(self.xc_u, ref_xc, np.gradient(temp_zcu, ref_xc)))
            self.dzdx_l = torch.from_numpy(np.interp(self.xc_l, ref_xc, np.gradient(temp_zcl, ref_xc)))
        # auto generate NACA
        elif airfoil_input[:5] == 'NACA ':  # 4/5 digit only 
            digits = airfoil_input[5:]
            if len(digits)==4:
                self.get_naca4digit(airfoil_input)
            elif len(digits)==5:  
                self.get_naca5digit(airfoil_input)
            else:
                raise ValueError('Invalid NACA 4-digit or 5-digit code.') 
        else: 
            raise ValueError('Invalid input airfoil. Please use a csv file or a NACA 4-digit airfoil.')

    def assemble_tensor(self):
        assert self.zc_u is not None, 'Airfoil geometry not defined.'

        if self.manual_pts is None: 
            # generate linear spacing x/c range if not provided by user
            test_xcu = torch.linspace(0., 1., self.num_pts[0])
            test_xcl = torch.linspace(0., 1., self.num_pts[1])
            test_xcu = torch.flip(-2*(torch.sin(torch.linspace(0,1, self.num_pts[0])*np.pi/2)-0.5)  , [0, ])
            test_xcl = torch.flip(-2*(torch.sin(torch.linspace(0,1, self.num_pts[0])*np.pi/2)-0.5)  , [0, ])
            # convert to conformal coordinates
            xhat_u, xhat_l = test_xcu, test_xcl #2*test_xcu-1, 2*test_xcl-1
            yhat_u, yhat_l = torch.sin(torch.arccos(xhat_u)), -torch.sin(torch.arccos(xhat_l))
            xhat = torch.hstack((xhat_u, xhat_l))
            yhat = torch.hstack((yhat_u, yhat_l))
        else: 
            test_xcu = self.manual_pts[0]
            test_xcl = self.manual_pts[1]
            # convert to conformal coordinates
            xhat_u, xhat_l = 2*test_xcu-1, 2*test_xcl-1
            yhat_u, yhat_l = torch.sin(torch.arccos(xhat_u)), -torch.sin(torch.arccos(xhat_l))
            xhat = torch.hstack((xhat_u, xhat_l))
            yhat = torch.hstack((yhat_u, yhat_l))
        # put tensor together
        cols_af_u = torch.tile(torch.flip(self.zc_u, [0, ]), (xhat.shape[0], 1)) # airfoil zc upper 
        cols_af_l = torch.tile(torch.flip(self.zc_l, [0, ]), (xhat.shape[0], 1)) # airfoil zc lower
        cols_alph = torch.ones((xhat.shape[0], 1))*torch.deg2rad(torch.as_tensor(self.alph)) # angle of attack
        cols_mach = torch.ones((xhat.shape[0], 1))*self.mach # mach number
        out_tensor = torch.hstack((cols_af_u*self.scale, cols_af_l*self.scale, cols_alph, cols_mach, xhat[:,None], yhat[:,None]))
        return out_tensor
    
    def get_naca4digit(self, airfoil_str):
        """ Calculates NACA 4 digit airfoil coordinates for model input
        airfoil_str: 'NACA XXXX' string
        """
        digits = airfoil_str[5:]
        m = float(digits[0])/100 # maxmimum camber
        p = float(digits[1])/10 # location of maximum camber
        t = float(digits[2:])/100 # thickness 
        
        if m == 0 and p == 0:  # Symmetric NACA 4-digit airfoil coordinates
            self.zc_u = 5*t*(0.2969*torch.sqrt(self.xc_u) - 0.1260*self.xc_u - 0.3516*self.xc_u**2 + 0.2843*self.xc_u**3 - 0.1015*self.xc_u**4)
            self.zc_l = -5*t*(0.2969*torch.sqrt(self.xc_l) - 0.1260*self.xc_l - 0.3516*self.xc_l**2 + 0.2843*self.xc_l**3 - 0.1015*self.xc_l**4)

            ref_xc = torch.linspace(0, 1, 501)
            new_xcu, new_xcl = ref_xc, ref_xc
            new_zcu = 5*t*(0.2969*torch.sqrt(ref_xc) - 0.1260*ref_xc - 0.3516*ref_xc**2 + 0.2843*ref_xc**3 - 0.1015*ref_xc**4)
            new_zcl = -5*t*(0.2969*torch.sqrt(ref_xc) - 0.1260*ref_xc - 0.3516*ref_xc**2 + 0.2843*ref_xc**3 - 0.1015*ref_xc**4)
        else: # Cambered airfoil
            ref_xc = torch.linspace(0, 1, 501) # reference xc 
            # thickness 
            yt = 5*t*(0.2969*torch.sqrt(ref_xc) - 0.1260*ref_xc - 0.3516*ref_xc**2 + 0.2843*ref_xc**3 - 0.1015*ref_xc**4)
            # Calculate mean camber line
            yc, theta = torch.ones_like(ref_xc), torch.ones_like(ref_xc)
            yc[ref_xc <= p] = (m/(p**2)) * (2*p*ref_xc[ref_xc <= p] - ref_xc[ref_xc <= p]**2) # camber line
            yc[(ref_xc <= 1) & (ref_xc > p)] = (m/((1-p)**2)) * ((1-2*p) + 2*p*ref_xc[(ref_xc <= 1) & (ref_xc > p)] - ref_xc[(ref_xc <= 1) & (ref_xc > p)]**2)
            # Get normal angles (theta)
            theta[ref_xc <= p] = torch.arctan((2*m / (p**2)) * (p-ref_xc[ref_xc <= p]))
            theta[(ref_xc <= 1) & (ref_xc > p)] = torch.arctan((2*m / ((1-p)**2)) * (p-ref_xc[(ref_xc <= 1) & (ref_xc > p)]))
            # calculate over full range 
            new_xcu, new_xcl = ref_xc - yt*torch.sin(theta), ref_xc + yt*torch.sin(theta)
            new_zcu, new_zcl = yc + yt*torch.cos(theta), yc - yt*torch.cos(theta)
            # interpolate to target locations
            self.zc_u = torch.from_numpy(np.interp(self.xc_u.numpy(), new_xcu.numpy(), new_zcu.numpy()))
            self.zc_l = torch.from_numpy(np.interp(self.xc_l.numpy(), new_xcl.numpy(), new_zcl.numpy()))

        # get gradient
        self.dzdx_u = torch.from_numpy(np.interp(self.xc_u.numpy(), new_xcu, np.gradient(new_zcu, new_xcu)))
        self.dzdx_l = torch.from_numpy(np.interp(self.xc_l.numpy(), new_xcl, np.gradient(new_zcl, new_xcl)))
        cs_u, cs_l = CubicSpline(new_xcu, new_zcu), CubicSpline(new_xcl, new_zcl)
        self.splines = [cs_u, cs_l]

    def get_naca5digit(self, airfoil_str):
        """ Calculates NACA 4 digit airfoil coordinates for model input
        airfoil_str: 'NACA XXXXX' string
        """
        digits = airfoil_str[5:]
        t = float(digits[-2:])/100 # thickness 
        camberline_profile = digits[:3]
        reflex = digits[2]  
        ref_xc = torch.linspace(0, 1, 501) # reference xc 
        # get camberline coefficient values 
        # Non reflexed
        if camberline_profile=='210':
            p = 0.05
            r = 0.0580
            k1 = 361.40
        elif camberline_profile=='220':
            p = 0.1
            r = 0.126
            k1 = 51.640
        elif camberline_profile=='230':
            p = 0.15
            r = 0.2025
            k1 = 15.957
        elif camberline_profile=='240':
            p = 0.20
            r = 0.290
            k1 = 6.643
        elif camberline_profile=='250':
            p = 0.25
            r = 0.391
            k1 = 3.230
        elif camberline_profile=='221':
            p = 0.1
            r = 0.130
            k1 = 51.990
            k2k1 = 0.000764
        elif camberline_profile=='231':
            p = 0.15
            r = 0.217
            k1 = 15.793
            k2k1 = 0.00677
        elif camberline_profile=='241':
            p = 0.20
            r = 0.318
            k1 = 6.520
            k2k1 = 0.0303
        elif camberline_profile=='251':
            p = 0.25
            r = 0.441
            k1 = 3.191
            k2k1 = 0.1355
        else:
            raise ValueError('Please enter a valid NACA 5-digit airfoil.') 
        
        yt = 5*t*(0.2969*torch.sqrt(ref_xc) - 0.1260*ref_xc - 0.3516*ref_xc**2 + 0.2843*ref_xc**3 - 0.1015*ref_xc**4)
        yc, theta = torch.ones_like(ref_xc), torch.ones_like(ref_xc)
        if reflex == '0': 
            yc[ref_xc <= r] = k1/6*(ref_xc[ref_xc <= r]**3 - 3*r*ref_xc[ref_xc <= r]**2 + r**2*(3-r)*ref_xc[ref_xc <= r])
            yc[(ref_xc > r) & (ref_xc <= 1)] = k1/6 * r**3 *(1-ref_xc[(ref_xc > r) & (ref_xc <= 1)])
            theta[ref_xc <= r] = torch.arctan(k1/6 * (3*ref_xc[ref_xc <= r]**2 - 6*r*ref_xc[ref_xc <= r] + r**2*(3-r)))
            theta[(ref_xc > r) & (ref_xc <= 1)] = torch.arctan(torch.tensor(-k1/6 * r**3)) 
        else: 
            yc[ref_xc <= r] = k1/6 * ((ref_xc[ref_xc <= r] - r)**3 - k2k1*ref_xc[ref_xc <= r]*(1-r)**3 - r**3*ref_xc[ref_xc <= r] + r**3)
            yc[(ref_xc > r) & (ref_xc <= 1)] = k1/6* (k2k1*(ref_xc[(ref_xc > r) & (ref_xc <= 1)]-r)**3 - k2k1*(1-r)**3*ref_xc[(ref_xc > r) & (ref_xc <= 1)] - r**3*ref_xc[(ref_xc > r) & (ref_xc <= 1)] + r**3)
            theta[ref_xc <= r] = torch.arctan(k1/6 * (3*(ref_xc[ref_xc <= r] - r)**2 - k2k1*(1-r)**3 - r**3) )
            theta[(ref_xc > r) & (ref_xc <= 1)] = torch.arctan(k1/6 * (3*k2k1*(ref_xc[(ref_xc > r) & (ref_xc <= 1)]-r)**2 - k2k1*(1-r)**3 - r**3)) 
        
        new_xcu, new_xcl = ref_xc - yt*torch.sin(theta), ref_xc + yt*torch.sin(theta)
        new_zcu, new_zcl = yc + yt*torch.cos(theta), yc - yt*torch.cos(theta)
        # interpolate to target locations
        self.zc_u = torch.from_numpy(np.interp(self.xc_u.numpy(), new_xcu.numpy(), new_zcu.numpy()))
        self.zc_l = torch.from_numpy(np.interp(self.xc_l.numpy(), new_xcl.numpy(), new_zcl.numpy()))
        
        self.dzdx_u = torch.from_numpy(np.interp(self.xc_u.numpy(), new_xcu, np.gradient(new_zcu, new_xcu)))
        self.dzdx_l = torch.from_numpy(np.interp(self.xc_l.numpy(), new_xcl, np.gradient(new_zcl, new_xcl)))

        cs_u, cs_l = CubicSpline(new_xcu, new_zcu), CubicSpline(new_xcl, new_zcl)
        self.splines = [cs_u, cs_l]

""" Defines the fully connected neural network for latent variable mapping """ 
# inherits methods from pytorch nn class
class fcnn_layer(torch.nn.Sequential):
    def __init__(self, train_x, nn_dims):
        data_dim = train_x.shape[1]
        super(fcnn_layer, self).__init__()
        # Layer 1 
        self.add_module('linear1', torch.nn.Linear(data_dim, nn_dims[0])) 
        self.add_module('relu1', torch.nn.ReLU()) # ReLu activation 
        self.add_module('dropout1', torch.nn.Dropout(0.2)) # dropout to reduce overfitting 
        # Layer 2 
        self.add_module('linear2', torch.nn.Linear(nn_dims[0], nn_dims[1]))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('dropout2', torch.nn.Dropout(0.2))
        # Layer 3 
        self.add_module('linear3', torch.nn.Linear(nn_dims[1], nn_dims[2]))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('dropout3', torch.nn.Dropout(0.2))
        # Layer 4 
        self.add_module('linear4', torch.nn.Linear(nn_dims[2], nn_dims[3]))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('dropout4', torch.nn.Dropout(0.2))
        # Final linear layer
        self.add_module('linear5', torch.nn.Linear(nn_dims[3], nn_dims[4]))

""" Unpack the model from the model folder 
use_gpu: determine whether to use GPU | Boolean
----
returns model and likelihood which are used to make predictions
"""
def unpack_model(use_gpu=False):
    print('Loading in model...')
    assert os.path.isdir('./model/'), 'Model directory does not exist'
    # reads in train_x, train_y, noise_model, and Lxx (cholesky Kxx)
    global L, train_x, train_y, train_noise
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore") # warning suppression is required for website
        # Load in training data
        train_x, train_y, train_noise = torch.load('./model/train.pt', map_location='cpu')
        # Load in pickled main covariance matrix Kxx 
        if 'L' not in globals():
            if os.path.exists('./model/lam_L.pt'):
                print('    lam_L.pt file already exists. Loading file...')
                L = torch.load('./model/lam_L.pt', map_location='cpu')
            else:
                # if not included, message to download the file  
                file_url = 'https://drive.google.com/uc?export=download&id=1uN1zAYjMSJgsjuYRmBlAVaKJ9rTXA42Q' 
                save_path = './model/lam_L.pt'
                print('    Missing lam_L.pt file.\nThis is a large file (6.1 GB) that will need to be downloaded from ' + file_url)
                print('    Would you like to automatically download this file? y/n')
                while True:
                    download_response = input()
                    if download_response == 'y':
                        # Helper method handling downloading large files 
                        import gdown
                        file_id = '1uN1zAYjMSJgsjuYRmBlAVaKJ9rTXA42Q'
                        gdown.download(f"https://drive.google.com/uc?id={file_id}", save_path, quiet=False)
                        print('    Download finished')
                        break
                    elif download_response== 'n':
                        raise ValueError('Missing lam_L.pt file')
                    else:
                        download_response = input()
        else:
            print('    lam_L.pt file already loaded...')
        print('    Loading complete!')
        scaler = 10 # model scaling 

    # Load likelihood
    noise_prior = gpytorch.priors.NormalPrior(0.0, 1.0)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_noise*scaler**2, learn_additional_noise=True, noise_prior=noise_prior) # Placeholder likelihood to initialize model
    
    # Initialize model 
    model = lam_adapt(train_x, train_y, likelihood)
    # model.load_weights(1) # intialize to weights 1 
    # Set to eval mode
    model.to(torch.float32) # float32 for memory reduction
    if use_gpu: # push model to cuda if using GPU
        model.cuda()
    model.eval()
    likelihood.eval()
    model.use_gpu = use_gpu
    return model, likelihood 