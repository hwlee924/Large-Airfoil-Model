"""
includes functions relating to airfoil geometry 
"""

import torch 
import pickle
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np 
from cst_modeling.section import cst_foil_fit, cst_foil

class lam_pca_transformer():
    def __init__(self, target_device):
        # Set up things required to do PCA ops 
        pca_path = Path(__file__).resolve().parent / 'model' / 'airfoil_pca.pkl'
        with pca_path.open('rb') as file:
            pca_data = pickle.load(file)
            
            # Mean
            self.pca_mean = pca_data['pca_mean'].to(target_device) 
            self.pca_mean.requires_grad = False 
            
            # Number of components 
            self.pca_num_components = pca_data['pca_num_component'] 
            
            # Components
            self.pca_components = pca_data['pca_component'].to(target_device).detach() 
            self.pca_components.requires_grad = False
            
            # Normalizer (only to be used with old one)
            self.normalizer = torch.tensor([4.4, 1.4, 1.0, 0.54, 0.46, 0.3, 0.3, 0.3, 0.14, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]).to(target_device)  
            self.pca_mean.requires_grad = False

        # device
        self.output_device = target_device 
        
    # Convert ASPIRE-style airfoil geom description [n x 56]
    # to LAM-input style tensor [n x 15] 
    # this consists of 15 principal components
    def forward(self, input_tensor:torch.Tensor, from_physical_coords:bool=False):
        if input_tensor.device != self.output_device:
            1
        if from_physical_coords:
            transform_x = torch.matmul((input_tensor.to(self.output_device)*10 - torch.tensor([-1.0183e-03,  6.6037e-02,  1.2920e-01,  1.5010e-01,  1.8197e-01,
                                                                        2.1010e-01,  2.3404e-01,  3.2614e-01,  3.9228e-01,  4.4484e-01,
                                                                        5.2446e-01,  5.7991e-01,  6.1825e-01,  6.4223e-01,  6.5326e-01,
                                                                        6.5214e-01,  6.3889e-01,  6.1551e-01,  5.8339e-01,  5.4348e-01,
                                                                        4.9676e-01,  4.4340e-01,  3.8353e-01,  3.1746e-01,  2.4448e-01,
                                                                        1.6637e-01,  8.1677e-02, -6.5470e-03, -1.0183e-03, -5.0847e-02,
                                                                        -1.0268e-01, -1.1823e-01, -1.4116e-01, -1.6067e-01, -1.7664e-01,
                                                                        -2.3604e-01, -2.7721e-01, -3.0889e-01, -3.5454e-01, -3.8519e-01,
                                                                        -4.0371e-01, -4.1090e-01, -4.0732e-01, -3.9429e-01, -3.7214e-01,
                                                                        -3.4229e-01, -3.0554e-01, -2.6262e-01, -2.1380e-01, -1.6124e-01,
                                                                        -1.1162e-01, -6.8811e-02, -3.6338e-02, -1.5701e-02, -1.1301e-02,
                                                                        -2.9101e-02,], requires_grad=False).to(self.output_device)) - self.pca_mean, self.pca_components.T)
        else:     
            transform_x = torch.matmul(input_tensor - self.pca_mean, self.pca_components.T)
        transform_x /= self.normalizer # <- delete this later
        return transform_x
    
    # Convert LAM-input style tensor [n x 15]
    # to ASPIRE-style airfoil geom description [n x 56] 
    # get_physical_coords=True returns the actual airfoil coordinates in terms of x/c and z/c
    def inverse(self, input_pca, get_physical_coords:bool=False):
        # if the input is just torch.Tensor samples 
        if isinstance(input_pca, torch.Tensor):
            i_transform_x = self.__inverse_tensor(input_pca_tensor=input_pca,
                                                  get_physical_coords=get_physical_coords)
        
        elif isinstance(input_pca, np.ndarray):
            i_transform_x = self.__inverse_tensor(input_pca_tensor=torch.from_numpy(input_pca),
                                                  get_physical_coords=get_physical_coords)
        # input is a multivariate normal distribution
        elif isinstance(input_pca, torch.distributions.MultivariateNormal):
            i_transform_x = self.__inverse_mvn(input_pca_distrib=input_pca,
                                               get_physical_coords=get_physical_coords)
        else: 
            raise ValueError("Invalid format for principal components!")  
        return i_transform_x
    
    def __inverse_tensor(self, input_pca_tensor:torch.Tensor, get_physical_coords:bool=False):
        if input_pca_tensor.device != self.output_device:
            input_pca_tensor = input_pca_tensor.to(self.output_device)
        i_transform_x = torch.matmul(input_pca_tensor * self.normalizer, self.pca_components) + self.pca_mean
        if get_physical_coords:
            i_transform_x += torch.tensor([-1.0183e-03,  6.6037e-02,  1.2920e-01,  1.5010e-01,  1.8197e-01,
                            2.1010e-01,  2.3404e-01,  3.2614e-01,  3.9228e-01,  4.4484e-01,
                            5.2446e-01,  5.7991e-01,  6.1825e-01,  6.4223e-01,  6.5326e-01,
                            6.5214e-01,  6.3889e-01,  6.1551e-01,  5.8339e-01,  5.4348e-01,
                            4.9676e-01,  4.4340e-01,  3.8353e-01,  3.1746e-01,  2.4448e-01,
                            1.6637e-01,  8.1677e-02, -6.5470e-03, -1.0183e-03, -5.0847e-02,
                            -1.0268e-01, -1.1823e-01, -1.4116e-01, -1.6067e-01, -1.7664e-01,
                            -2.3604e-01, -2.7721e-01, -3.0889e-01, -3.5454e-01, -3.8519e-01,
                            -4.0371e-01, -4.1090e-01, -4.0732e-01, -3.9429e-01, -3.7214e-01,
                            -3.4229e-01, -3.0554e-01, -2.6262e-01, -2.1380e-01, -1.6124e-01,
                            -1.1162e-01, -6.8811e-02, -3.6338e-02, -1.5701e-02, -1.1301e-02,
                            -2.9101e-02,], requires_grad=False).to(self.output_device)
            i_transform_x /= 10
        return i_transform_x
    
    def __inverse_mvn(self, input_pca_distrib:torch.distributions.MultivariateNormal, get_physical_coords:bool=False):
        """
        Invert PCA but in distribution nform 
        """
        # Push distribution to right device 
        if input_pca_distrib.mean.device != self.output_device:
            input_pca_distrib = torch.distributions.MultivariateNormal(
                input_pca_distrib.mean.to(self.output_device),
                input_pca_distrib.covariance_matrix.to(self.output_device),
                )
            
        # Conversion matrices
        norm_diag = torch.diag(self.normalizer)
        conversion_matrix = (norm_diag @ self.pca_components).to(self.output_device).T

        # Convert mean vector
        loc = self.inverse(input_pca_distrib.mean, get_physical_coords=get_physical_coords).to(self.output_device)
        
        # Convert covariance matrix 
        cov = (conversion_matrix @ input_pca_distrib.covariance_matrix @ conversion_matrix.T) / 100
        cov = cov + 1e-12 * torch.eye(cov.shape[0], device=self.output_device)
        
        # Create multivariate normal object 
        i_transform_distrib = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov) 
        return i_transform_distrib
    
    def assemble_lam_tensor_from_pcs(self, guess:torch.Tensor, target_alpha:float, target_mach:float, pts_per_side:int=200, normalize:bool=False):  
        if normalize:
            normalized_guess = guess * self.normalizer
        else :
            normalized_guess = guess
        test_xcu = torch.flip(-2*(torch.sin(torch.linspace(0,1, pts_per_side)*torch.pi/2)-0.5)  , [0, ]).to(self.output_device) 
        test_xcl = torch.flip(-2*(torch.sin(torch.linspace(0,1, pts_per_side)*torch.pi/2)-0.5)  , [0, ]).to(self.output_device) 
        
        # convert to conformal coordinates
        xhat_u, xhat_l = test_xcu, test_xcl
        yhat_u, yhat_l = torch.sin(torch.arccos(xhat_u)), -torch.sin(torch.arccos(xhat_l))
        xhat = torch.hstack((xhat_u, xhat_l))
        yhat = torch.hstack((yhat_u, yhat_l))
        
        # put tensor together
        cols_alph = torch.ones((xhat.shape[0], 1)).to(self.output_device) * torch.deg2rad(torch.as_tensor(target_alpha)) # angle of attack
        cols_mach = torch.ones((xhat.shape[0], 1)).to(self.output_device) * target_mach # mach number
        cols_pca = normalized_guess.repeat(xhat.shape[0], 1).to(self.output_device) #guess.repeat(xhat.shape[0], 1).to(self.output_device)
        out_tensor = torch.hstack((cols_pca, cols_alph, cols_mach, xhat[:,None], yhat[:,None]))
        return out_tensor.to(self.output_device)
    
    
"""
Airfoil
"""
def smooth_airfoil_from_nodes(design_result:dict, show_result:bool=False, num_samples:int=100):
    # If torch.tensor = deterministic airfoil parametrization
    if isinstance(design_result['params'], torch.Tensor):
        smooth_xc, smooth_zcu, smooth_zcl = __determinstic_airfoil_smoothing(design_result, show_result)
        return smooth_xc, smooth_zcu, smooth_zcl
    elif isinstance(design_result['params'], np.ndarray): 
        # convert to tensor if in numpy 
        design_result['params'] = torch.from_numpy(design_result['params'])
        smooth_xc, smooth_zcu, smooth_zcl = __determinstic_airfoil_smoothing(design_result, show_result)
        return smooth_xc, smooth_zcu, smooth_zcl
    # If torch.distribution = probabilistic airfoil parametrization
    elif isinstance(design_result['params'], torch.distributions.distribution.Distribution):
        smooth_xc, smooth_zcu_mean, smooth_zcl_mean, smooth_zcu_stdev, smooth_zcl_stdev = __probabilistic_airfoil_smoothing(design_result, show_result, num_samples)
        return smooth_xc, smooth_zcu_mean, smooth_zcl_mean, smooth_zcu_stdev, smooth_zcl_stdev
    else: 
        raise ValueError("Invalid design parameters.")
    
    

def __determinstic_airfoil_smoothing(design_result:dict, show_result:bool=False):
    te_t = design_result['zc_u'][-1] - design_result['zc_l'][-1] #0.008 # Trailing edge thickness - pre-defined
    te_offset = (design_result['zc_u'][-1] + design_result['zc_l'][-1])*0.5

    cst_fit_u, cst_fit_l = cst_foil_fit(xu=design_result['xc'], xl=design_result['xc'], 
                                yu=design_result['zc_u']-te_offset*design_result['xc']+te_t*design_result['xc'], 
                                yl=design_result['zc_l']+te_offset*design_result['xc']+te_t*design_result['xc'], 
                                n_cst=7) 
    
    smooth_xc, smooth_zcu, smooth_zcl, _, _ = cst_foil(200, cst_fit_u, cst_fit_l, tail=te_t, te_offset=te_offset)
    
    # Plot the result 
    if show_result:
        f, ax = plt.subplots(1,1,figsize=(8,6))
        ax.plot(design_result['xc'], design_result['zc_u'], 'ko', label='LAM nodes')
        ax.plot(design_result['xc'], design_result['zc_l'], 'ko')
        ax.plot(smooth_xc, smooth_zcu, 'tab:blue', label='Smoothed')
        ax.plot(smooth_xc, smooth_zcl, 'tab:blue')
        ax.legend()
        ax.set_xlabel('Chordwise location, $x/c$')
        ax.set_ylabel('Thickness, $z/c$')
        ax.set_ylim([np.min(design_result['zc_l'])-0.1, np.max(design_result['zc_u'])+0.1])
        plt.show
    return smooth_xc, smooth_zcu, smooth_zcl

def __probabilistic_airfoil_smoothing(design_result:dict, show_result:bool=False, num_samples:int=100, num_cst:int=7):
    # Initialize transformer 
    pca_transformer = lam_pca_transformer(target_device='cpu')
    
    # Generate samples 
    param_samples = design_result['params'].rsample(torch.Size([num_samples])).detach() 
    
    # Enumerate through all samples 
    posterior_zcu = []
    posterior_zcl = []
    for i in range(num_samples): 
        # Convert the principal components into physical representation of LAM 
        posterior_zc_ = pca_transformer.inverse(param_samples[i], get_physical_coords=True) 
        posterior_zcu_ = posterior_zc_[:28].cpu().detach().numpy() # upper 
        posterior_zcl_ = posterior_zc_[28:].cpu().detach().numpy() # lower 
        te_t = posterior_zcu_[-1] - posterior_zcl_[-1] # Trailing edge thickness - manually capculated 
        te_offset = (posterior_zcu_[-1] + posterior_zcl_[-1])*0.5 # how much the trailing edge offset from z/c = 0.0

        # Do CST fit to obtain a smooth version of the airfoil 
        cst_fit_u, cst_fit_l = cst_foil_fit(xu=design_result['xc'], xl=design_result['xc'], 
                                    yu=posterior_zcu_-te_offset*design_result['xc']+te_t*design_result['xc'], 
                                    yl=posterior_zcl_+te_offset*design_result['xc']+te_t*design_result['xc'], 
                                    n_cst=num_cst) 
        smooth_xc, smooth_zcu, smooth_zcl, _, _ = cst_foil(200, cst_fit_u, cst_fit_l, tail=te_t, te_offset=te_offset)
        
        # Aggregate all samples
        posterior_zcu.append(smooth_zcu)
        posterior_zcl.append(smooth_zcl)
    posterior_zcu = np.vstack(posterior_zcu)
    posterior_zcl = np.vstack(posterior_zcl) 
    
    # Obtain mean and standard deviation of the aggregate 
    zcu_mean = np.mean(posterior_zcu, axis=0)
    zcl_mean = np.mean(posterior_zcl, axis=0)
    zcu_stdev = np.std(posterior_zcu, axis=0)
    zcl_stdev = np.std(posterior_zcl, axis=0)
    
    if show_result: 
        f, ax = plt.subplots(1,1,figsize=(8,6))
        ax.plot(design_result['xc'], design_result['zc_u'], 'ko', label='LAM nodes')
        ax.plot(design_result['xc'], design_result['zc_l'], 'ko')
        ax.plot(smooth_xc, zcu_mean, 'tab:blue', label='Smoothed, mean')
        ax.plot(smooth_xc, zcl_mean, 'tab:blue')
        ax.fill_between(smooth_xc, 
                        zcu_mean+2*zcu_stdev, 
                        zcu_mean-2*zcu_stdev,
                        alpha=0.3,
                        color='tab:blue', 
                        label='Smoothed, 2$\sigma$')
        ax.fill_between(smooth_xc, 
                        zcl_mean+2*zcl_stdev, 
                        zcl_mean-2*zcl_stdev,
                        alpha=0.3,
                        color='tab:blue', 
                        ) 
        ax.legend()
        ax.set_xlabel('Chordwise location, $x/c$')
        ax.set_ylabel('Thickness, $z/c$')
        ax.set_ylim([np.min(design_result['zc_l'])-0.1, np.max(design_result['zc_u'])+0.1])
        plt.show
    return smooth_xc, zcu_mean, zcl_mean, zcu_stdev, zcl_stdev