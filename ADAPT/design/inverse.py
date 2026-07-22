from dataclasses import dataclass
import numpy as np
import torch 
import gpytorch
from scipy.optimize import minimize
from ADAPT.geometry import lam_pca_transformer
from ADAPT.math import quadratic_interp
import tqdm.notebook as tn
import time 
import pyro
import pyro.distributions as dist 
from pyro.infer import SVI, Trace_ELBO  
from abc import ABC, abstractmethod         

@dataclass
class airfoil_design_input:
    """ 
    Container class for all relevant information required in the inverse design process.
    This class serves as the aerodynamic design "input file" 
    """
    
    # Operating conditions 
    alpha:list # Angle of attack [deg]
    mach:list # Freestream mach number 
    # Since this model utilizes the LAM, the underlying assumption is that Re ~ O(6)
    
    # Airfoil geometry 
    initial_design:torch.Tensor=None # Initial guess in the principal component (PC) space 
    design_variance:torch.Tensor=None # Variance of the design parameters in the PC space
    pods_sample:torch.Tensor=None # A sample from the posterior design space (pods)
    # The posterior design space is identified via stochastic variational inference (SVI).
    # It represents the 95% CI in which the objective if minimized (approximated)
    # It is approximated by a multivariate normal distribution
    pods_distrib:torch.distributions.distribution.Distribution=None # PODS distribution object stored during inference
    # The prior design space is the original design space (much wider than PODS)
    prds_distrib:torch.distributions.distribution.Distribution=None
    lam_xc:torch.Tensor = torch.tensor([0, 0.0025, 0.0075, 0.01, 0.015, 0.02, 
                                        0.025, 0.05, 0.075, 0.1,  0.15, 0.2, 
                                        0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                                        0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 
                                        0.95, 1.0]) # the x/c coordinates used in LAM's af geometry definition (1 x 28)
    
    # Design results 
    design_pca_distrib:torch.distributions.distribution.Distribution=None # distributional definition of the final design in the PC space
    design_airfoil_distrib:torch.distributions.distribution.Distribution=None # distributional definition of the final design in the physical space
    # the physical coordinates are represented by y/c values at a predefined set of x/c locations (stored as lam_xc).
    # (1 x 56) where the first 28 are for the upper surface, the latter 28 are for the lower surface
    # The following aerodynamic coefficients' are characterized as a Gaussian mixture distribution unless the source code is modified 
    design_cp_distrib:torch.distributions.distribution.Distribution=None # distributional definition of the final design's Cp curve
    # (1 x 400) where the first 200 are for the upper surface, the latter 200 are for the lower surface
    design_cl_distrib:torch.distributions.distribution.Distribution=None # distributional definition of the final design's cl
    design_cd_distrib:torch.distributions.distribution.Distribution=None # distributional definition of the final design's cm at quarter chord
    design_cm_distrib:torch.distributions.distribution.Distribution=None # distributional definition of the final design's cd (specifically pressure drag)
    
    # Active space settings 
    num_active_dims:int='auto' # number of active dimensions in the principal design space (15D)
    if num_active_dims == 'auto':
        num_inactive_dims:int=None
    else: 
        num_inactive_dims:int=15 - num_active_dims # number of inactive dimensions - determines the tolerance level
    num_gradient_samples:int=1000 # number of samples used to approximate the gradient covariance matrix 
    
    # Aerodynamic properties of the mean airfoil
    # Intermediate results for SVI is stored here;
    # This is because the inverse design objectives are computed wrt to the aerodynamic properties of the
    # MEAN airfoil of the posterior design space rather than taking into account the covariance
    mean_cp:torch.distributions.distribution.Distribution=None # distributional definition of the mean intermediate design's Cp distribution
    mean_cl:torch.distributions.distribution.Distribution=None # distributional definition of the mean intermediate design's cl
    mean_cm:torch.distributions.distribution.Distribution=None # distributional definition of the mean intermediate design's cm at quarter chord
    mean_cd:torch.distributions.distribution.Distribution=None # distributional definition of the mean intermediate design's cd (specifically pressure drag)
    
    # Target aerodynamic properties 
    # At least one of the target coefficients must be provided to perform inverse design
    target_design:torch.Tensor=None # target design's parametrization in the PC space; not used for design, only for plotting purposes
    target_cp:list=None # distributional definition of the target design's Cp distribution
    # torch.distributions.distribution.Distribution
    target_cl:torch.distributions.distribution.Distribution=None # distributional definition of the target design's cl
    target_cm:torch.distributions.distribution.Distribution=None # distributional definition of the target design's cm at quarter chord
    target_cd:torch.distributions.distribution.Distribution=None # distributional definition of the target design's cd (specifically pressure drag)
    
    # Geometric constraints 
    constraint_max_t:float = None # maximum thickness constraint (as a fraction of chord)

    # Transformer 
    # Any transformer that can convert the evaluation_model input to physical space coordinates
    # In our case, this is the conversion from PC to y/c
    pca_transformer:lam_pca_transformer=None
    
    # Convergence parameters  
    # Convergence is determined via:
    # calculating the moving average of the first _tracking_num_ principal component values 
    # over _mavg_window_ number of iterations. If the difference in the PC values fall below _tolerance_
    # the results are deemed to be convertged
    tracking_num:int=6
    mavg_window:int=10
    tolerance:float=5e-4
    
    def __post_init__(self):
        """Post-initialization processing for the dataclass."""
        self.__check_multipoint_target()
    
    
    """
    These are not used yet 
    """
    def parameters_to_physical(self):
        if self.pca_transformer is None: 
            return self.pods_distrib
        else: 
            return self.__parameters_to_physical()
        return 
    
    def __parameters_to_physical(self):
        out = self.pca_transformer.inverse(self.pods_distrib.mean, get_physical_coords=True)  
        return out
    
    def __parameters_to_physical_distrib(self):
        return 

    def __check_multipoint_target(self):
        """ 
        If multiple targets are provided, combine them into a single MVN 
        """

        # def combine_targets(target_list:list):
        #     mu_ = []
        #     cov_ = [] 
        #     for target in target_list:
        #         mu_.append(target.mean)
        #         cov_.append(target.covariance_matrix) 
        #     mu = torch.concat([m for m in mu_], dim=0)
        #     cov = torch.block_diag(*[c for c in cov_])
        #     return gpytorch.distributions.MultivariateNormal(mu, cov)
        def combine_targets(target_list: list):
            mus = []
            covs = []

            for target in target_list:
                # Case 1: GPyTorch MultivariateNormal
                if isinstance(target, gpytorch.distributions.MultivariateNormal):
                    mu = target.mean
                    cov = target.covariance_matrix

                # Case 2: Torch Normal
                elif isinstance(target, torch.distributions.Normal):
                    mu = target.mean.reshape(-1)
                    var = target.variance.reshape(-1)
                    cov = torch.diag(var)

                else:
                    raise TypeError(
                        f"Unsupported distribution type: {type(target)}"
                    )

                mus.append(mu)
                covs.append(cov)

            mu = torch.cat(mus, dim=0)
            cov = torch.block_diag(*covs)
            return gpytorch.distributions.MultivariateNormal(mu, cov)

        
        if isinstance(self.target_cp, list):
            self.target_cp = combine_targets(self.target_cp)  
        if isinstance(self.target_cl, list):
            self.target_cl = combine_targets(self.target_cl)
        if isinstance(self.target_cm, list):
            self.target_cm = combine_targets(self.target_cm)
        if isinstance(self.target_cd, list):
            self.target_cd = combine_targets(self.target_cd)


class evaluation_model(ABC):
    """
    Generic wrapper class for machine learning models used in the probabilistic 
    airfoil inverse design framework.
    """
    def __init__(self, ml_model):
        self.model = ml_model  
        
    @abstractmethod
    def predict(self, candidate_design: "airfoil_design_input"):
        """ 
        Predict the pressure coefficient (Cp) for the given candidate design.
        Must be implemented by subclasses.
        """
        pass

    def __call__(self, *args, **kwargs):
        """Allows the model to be called directly, forwarding to `predict`."""
        return self.predict(*args, **kwargs) 
    
class probabilistic_designer:
    def __init__(self, 
                 evaluation_model:evaluation_model, # Should be the LAM by default   
                 user_input:airfoil_design_input,  
                 optimizer = None,   
                 max_iters:int=3000, 
                 output_device='cpu',
                 verbose=True): 
        """
        opt_alg: choice of optimization algorithm, currently "Nelder-Mead" or "Adam" [str]
        initial_guess: initial guess of CST coefficients, [torch.Tensor([cst,u_1, ... , cst,u_6, cst,l_1, ... , cst,l_6])]
        design_AoA: design angle of attack in degrees [float]
        design_mach: design freestream mach number [float]
        target_cp: target pressure coefficient distribution [MultivariateNormal]
        target_cl: target lift coefficient [normal distribution]
        target_cm: target LE moment coefficient [normal distribution]
        target_cd: target pressure drag coefficient [normal distribution]
        target_cst: NOT used for optimization, used to calculate residuals 
        max_iters: maximum number of iterations [int]
        """
        self.max_iters = max_iters  
        self.model = evaluation_model  
        self.optimizer = optimizer
        self.airfoil = user_input  
        self.output_device = output_device 
        self.verbose = verbose 
        
        # Keeps track of the inverse design/optimization history throughout iterations
        # and metadata
        self.history = {
            # Overall 
            'iters': [], # iteration counter
            'loss': [], # loss 
            'total_iter': 0, # total number of iterations
            'run_time': None, # algorithm run-time 
            
            # Aero 
            'pods_distrib': [], # design parameter @ each iteration
            'cp': [], # Cp distribution @ each iteration
            'cl': [], # cl distribution @ each iteration
            'cm': [], # cm distribution @ each iteration
            'cd': [], # cd distribution @ each iteration
            
            # For convergence check 
            'delta_distrib': []
        }
 
    def run_design(self, *args, **kwargs):
        """
        Run the design process using stochastic variational inference (SVI).

        This method ensures that:
        1. At least one design objective is provided (e.g., Cp, Cl, Cm, Cd).
        2. An initial design is specified.

        After checks, it initializes the design state and runs the SVI procedure.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the `svi` method.
        **kwargs : dict
            Keyword arguments passed to the `svi` method.

        Returns
        -------
        result : airfoil_design_input
            Final optimized design parameters or SVI output.
        history : dict
            Design iteration history and metadata from the SVI process.
        """ 
        
        # ---- Pre-checks before running SVI ----
        # Ensure that at least one design objective is specified
        objective_count = 0
        if self.airfoil.target_cp is not None: 
            objective_count += 1
        if self.airfoil.target_cl is not None: 
            objective_count += 1
        if self.airfoil.target_cm is not None: 
            objective_count += 1
        if self.airfoil.target_cd is not None: 
            objective_count += 1
        assert objective_count >= 1, (
        "You must provide at least one design objective "
        "(target_cp, target_cl, target_cm, or target_cd)."
        )
        
        # Ensure initial airfoil design parameters are provided
        assert self.airfoil.initial_design is not None, (
            "You must provide the initial airfoil design parameters."
        )
        
        # ---- Initialization ----
        # Set the sample from the posterior design space 
        # to the initial design before beginning inverse desing
        self.airfoil.pods_sample = self.airfoil.initial_design

        # ---- Run optimization via SVI ----
        result, history = self.svi(*args, **kwargs)

        # ---- Determine the tolerance range of the airfoil ----
        start_time = time.time()
        tolerance_pca_distrib, tolerance_geom_distrib = self.determine_airfoil_tolerance()
        self.airfoil.design_pca_distrib = tolerance_pca_distrib
        self.airfoil.design_airfoil_distrib = tolerance_geom_distrib
        end_time = time.time()
        print('Tol covar runtime', end_time - start_time)
        # Given this new geometry, approximate the full Cp distribution via Gaussian mixture model
        print(f"Approximating design airfoil's aerodynamic characteristics using Gaussian Mixtures..." if self.verbose else '')
        self.airfoil.design_cp_distrib, self.airfoil.design_cl_distrib, self.airfoil.design_cm_distrib, self.airfoil.design_cd_distrib,= self.create_aero_gaussian_mixture(pca_distrib=tolerance_pca_distrib)
        print(f'Finished!' if self.verbose else '')
        return result, history
    
    
    def svi(self, num_particles:int=1, 
            prior_design_space:dist=None,
            prior_loss_scale:float=0.1,
            variational_init_scale:float=0.1, # 0.05
            betas:list=[1, 100, 1000, 100]):
        """
        Runs inverse airfoil design probabilistically using stochastic variational inference,
        which enables a multivariate normal description of the airfoil geometry (princpal components ).
        The optimization loop minimizes
        1) the Jensen-shannon divergence b/n the candidate Cp distribution and the target Cp distribution
        2) the deviation of the CST coefficients from the initialization
        """
        pyro.clear_param_store() 
                
        # Main model where the sampling and loss calculation happends 
        def pyro_model():
            # Define the prior design space (in the principal component space) from which airfoil candidates are sampled
            if prior_design_space is not None:
                vars_prior = prior_design_space.to_event(1)
            else:  
                # Default airgoil geometry prior 
                initial_af = self.airfoil.initial_design.clone().detach()
                vars_scale = torch.ones((15, ), device=self.model.output_device) * 0.05 if self.airfoil.design_variance is None else self.airfoil.design_variance
                vars_prior = dist.Normal(loc=initial_af, scale=vars_scale).to_event(1)
                
            # prior_loss_scale affects how much the deviation from the prior distribution contributes to loss
            # reduce the value to enable more flexibility, increase to constrain to the initial geometry
            with pyro.poutine.scale(scale=prior_loss_scale): 
                vars_sample = pyro.sample("vars_sample", vars_prior)#.to(self.model.output_device) 

            # From candidate geometry predict the corresponding aerodynamic quantities 
            import time 
            # init_time_pred = time.time()
            candidate_cp = self.model(self.airfoil, sample_tensor=vars_sample) 
            # end_time_pred = time.time()
            # print(f"Prediction time: {end_time_pred - init_time_pred}")
            if any(x is not None for x in (self.airfoil.target_cl, self.airfoil.target_cm, self.airfoil.target_cd)):
                candidate_cl, candidate_cm, candidate_cd = self.get_coefficients(candidate_cp, vars_sample)   
            else: 
                candidate_cl, candidate_cm, candidate_cd = None, None, None 
                
            
            # Computes Jensen-Shannon divergence as negative log likelihood (hence the -1 factor)
            # init_time_loss = time.time()
            loss_cp, loss_cl, loss_cm, loss_cd = self.__get_loss(current_coeffs=[candidate_cp, candidate_cl, candidate_cm, candidate_cd], 
                                                    target_coeffs=[self.airfoil.target_cp, self.airfoil.target_cl, self.airfoil.target_cm, self.airfoil.target_cd],
                                                    get_components=True)
            # end_time_loss = time.time()
            # print(f"Loss calculation time: {end_time_loss - init_time_loss}")
            # loss_cp = -self.airfoil.target_cp.log_prob(candidate_cp.mean) <- fully Bayesian
            
            # pressure distribution 
            if self.airfoil.target_cp is not None:  
                # Need to minimize KL, ELBO is maximized, so negative required   
                pyro.factor("loss_cp", -1 * betas[0] * loss_cp)  
            if self.airfoil.target_cl is not None:   
                pyro.factor("loss_cl", -1 * betas[1] * loss_cl)  
            if self.airfoil.target_cm is not None:  
                pyro.factor("loss_cm", -1 * betas[2] * loss_cm)  
            if self.airfoil.target_cd is not None:  
                pyro.factor("loss_cd", -1 * betas[3] * loss_cd)  
            
            # Get constraint losses 
            if self.airfoil.constraint_max_t is not None:
                sample_coords = self.airfoil.pca_transformer.inverse(vars_sample, get_physical_coords=True) 
                penalty_thickness = (torch.max(sample_coords[:28] - sample_coords[28:]) - self.airfoil.constraint_max_t).pow(2).sqrt() 
                pyro.factor("penalty_maxt", -1000 * penalty_thickness)  

        # Here we assume that the principal components can be approximated as a multivaraite-normal distribution 
        # init_scale will determine the error bar on the airfoil geom, 
        guide = pyro.infer.autoguide.AutoMultivariateNormal(pyro_model, init_scale=variational_init_scale) 

        # Set up gradient-descent optimizer 
        if self.optimizer is None: 
            # These are the default settings
            initial_lr = 0.01 # initial learning rate 
            self.optimizer = pyro.optim.PyroLRScheduler(
                torch.optim.lr_scheduler.StepLR,  # Scheduler type
                optim_args={
                    "optimizer": torch.optim.Adam,         # Optimizer type
                    "optim_args": {"lr": initial_lr},       # Optimizer parameters
                    "step_size": 300,                      # Decay every 300 steps
                    "gamma": 0.5                            # Multiply LR by 0.5
                }
            )
        
        # Define SVI instance 
        svi = SVI(pyro_model, 
                  guide, 
                  self.optimizer, 
                  loss=Trace_ELBO(num_particles=num_particles)
                  )  
        
        # Measuring runtime 
        start_time = time.time()
        
        # Main SVI loop 
        print("Narrowing down the design space via SVI..." if self.verbose else "")
        with tn.tqdm(range(0, self.max_iters), desc="Iterations") as prog_bar:
            for _ in prog_bar:
                # init_time_iter = time.time()
                loss = svi.step()  
                # end_time_iter = time.time()
                # print(f"Iteration time: {end_time_iter - init_time_iter}")
                # Update progress bar 
                prog_bar.set_postfix(loss=loss)
                
                # Update history & perform convergence check here
                with torch.no_grad():  
                    # Get multivariate-normal approximations of the current design parameters
                    posterior_design_space = guide.get_posterior()
                    # Detach 
                    posterior_design_space = torch.distributions.multivariate_normal.MultivariateNormal(
                        posterior_design_space.mean.detach(),
                        posterior_design_space.covariance_matrix.detach()
                    )

                    # Update current airfoil and generate predictions
                    self.airfoil.pods_distrib = posterior_design_space
                    self.airfoil.pods_sample = posterior_design_space.mean
                    self.airfoil.mean_cp = self.model(self.airfoil, sample_tensor=posterior_design_space.mean)
                    self.airfoil.mean_cl, self.airfoil.mean_cm, self.airfoil.mean_cd = self.get_coefficients(self.airfoil.mean_cp, 
                                                                         self.airfoil.pods_sample)    
                    # Update history - aerodynamics
                    self.history['cp'].append(self.__mvn_to_cpu(self.airfoil.mean_cp))
                    if len(self.airfoil.alpha)==1: 
                        self.history['cl'].append(self.__mvn_to_cpu(self.airfoil.mean_cl))
                        self.history['cm'].append(self.__mvn_to_cpu(self.airfoil.mean_cm))
                        self.history['cd'].append(self.__mvn_to_cpu(self.airfoil.mean_cd)) 
                    else: 
                        self.history['cl'].append(self.__mvn_to_cpu(self.airfoil.mean_cl))
                        self.history['cm'].append(self.__mvn_to_cpu(self.airfoil.mean_cm))
                        self.history['cd'].append(self.__mvn_to_cpu(self.airfoil.mean_cd)) 
                    # Update history - training related
                    self.history['iters'].append(self.history['total_iter']) # iteration
                    self.history['loss'].append(loss) # ELBO of loss 
                    
                    # Update history - parameters 
                    self.history['pods_distrib'].append(self.__mvn_to_cpu(posterior_design_space)) 
                    self.history['delta_distrib'].append(posterior_design_space.mean.cpu()) # only take the mean
                    if self.history['total_iter'] >= self.airfoil.mavg_window+1:
                        # Calculate the percentage change in the moving average
                        delta_params = np.vstack(self.history['delta_distrib'])
                        prev_moving_avg = np.mean(delta_params[-self.airfoil.mavg_window-1:-2, :self.airfoil.tracking_num], axis=0) 
                        current_moving_avg = np.mean(delta_params[-self.airfoil.mavg_window:-1, :self.airfoil.tracking_num], axis=0) 
                        diff_moving_avg = (current_moving_avg - prev_moving_avg) / prev_moving_avg 
                        
                        if np.all(np.abs(diff_moving_avg) <= self.airfoil.tolerance) and np.all(np.abs(diff_moving_avg) > 0.0): # zero is false here
                            print("    Convergence achieved." if self.verbose else "")
                            end_time = time.time()
                            self.history['runtime'] = end_time - start_time
                            self.history['total_iter'] += 1
                            break
                        
                self.history['total_iter'] += 1  # counter  
        end_time = time.time()
        print('SVI runtime', end_time - start_time)
        # Update current airfoil one last time into CPU
        self.airfoil.pods_distrib = self.__mvn_to_cpu(self.airfoil.pods_distrib)
        self.airfoil.mean_cp = self.__mvn_to_cpu(self.airfoil.mean_cp)
        if len(self.airfoil.alpha)==1: 
            self.history['cl'].append(self.__mvn_to_cpu(self.airfoil.mean_cl))
            self.history['cm'].append(self.__mvn_to_cpu(self.airfoil.mean_cm))
            self.history['cd'].append(self.__mvn_to_cpu(self.airfoil.mean_cd)) 
        else: 
            self.history['cl'].append(self.__mvn_to_cpu(self.airfoil.mean_cl))
            self.history['cm'].append(self.__mvn_to_cpu(self.airfoil.mean_cm))
            self.history['cd'].append(self.__mvn_to_cpu(self.airfoil.mean_cd)) 
        return self.airfoil, self.history
    
    """
    For calculating aerodynamic coefficients and losses 
    """
    def __get_loss(self, current_coeffs:list, target_coeffs:list, get_components:bool=False):
        # coeffs should be a list in the order of cp, cl, cm, cd 
        # for current and target respectively
        
        # Jensen-Shannon divergence can be thought of as the "symmetrized" version of the KL divergence 
        # a list in the order of cp, cl, cm, cd 
        losses = [] 
        for coeff_idx, target_coeff in enumerate(target_coeffs): 
            if target_coeff is not None: 
                losses.append(self.__KLsymm(current_coeffs[coeff_idx], target_coeff))
            else: 
                losses.append(None) 
        
        if get_components:
            loss_cp, loss_cl, loss_cm, loss_cd = losses 
            return loss_cp, loss_cl, loss_cm, loss_cd
        else: 
            loss_sum = 0.0
            for loss_component in losses:
                if loss_component is not None: 
                    loss_sum += loss_component
            return loss_sum
        
    def __KLsymm(self, current, target):
        """
        Compute the symmetrized KL divergence between two distributions.
            KL_symm(p || q) ≈ 0.5 * ( KL(p || q) + KL(q || p) )

        Parameters
        ----------
        current : torch.distributions.Distribution
            The current distribution (e.g., model prediction).
        target : torch.distributions.Distribution
            The target distribution (e.g., design objective).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the Jensen-Shannon divergence.
        """
        
        # Compute the symmetric KL divergence (forward + reverse)
        # loss = 0.5 * (
        #     torch.distributions.kl.kl_divergence(target, current) +
        #     torch.distributions.kl.kl_divergence(current, target)
        # ) 
        jitter = 1e-4
        if isinstance(target, gpytorch.distributions.MultivariateNormal) or isinstance(target, torch.distributions.MultivariateNormal):
            target_w_jitter = torch.distributions.MultivariateNormal(
                loc=target.mean,
                covariance_matrix=target.covariance_matrix + torch.eye(target.covariance_matrix.shape[0]).to(target.covariance_matrix.device) * jitter
            )
            current_w_jitter = torch.distributions.MultivariateNormal(
                loc=current.mean,
                covariance_matrix=current.covariance_matrix + torch.eye(current.covariance_matrix.shape[0]).to(target.covariance_matrix.device) * jitter
            )

        else: 
            target_w_jitter = torch.distributions.Normal(
                loc=target.mean,
                scale=target.scale + jitter
            )
            current_w_jitter = torch.distributions.Normal(
                loc=current.mean,
                scale=current.scale + jitter
            )
        loss = 0.5 * (
            torch.distributions.kl.kl_divergence(target_w_jitter, current_w_jitter) +
            torch.distributions.kl.kl_divergence(current_w_jitter, target_w_jitter)
        )
        
        
        return loss
     
    def determine_airfoil_tolerance(self, ):
        """ Comment 
        """
        print(f'Determining airfoil tolerances from the provided design space...' if self.verbose else '') 
        H = self.__approximate_gradient_cov(self.airfoil.num_gradient_samples) 
        _, _, W, V = self.__partition_active_space(H, self.airfoil.num_active_dims)
        tolerance_pca_distrib, tolerance_geom_distrib = self.__determine_inactive_space(
            self.airfoil.pods_distrib.mean, W, V
            )
        return tolerance_pca_distrib, tolerance_geom_distrib
    
    def __approximate_gradient_cov(self, num_samples:int): 
        """
        Approximates the gradient covariance matrix H (Eq. 10 in Ref [2].) 
        by averaging the outer product of gradients over samples drawn from a 
        control zone distribution and summing across multiple weighted objectives.

        Args:
            design_space (torch.distributions.Distribution): 
                Distribution from which candidate design points are sampled. 
                Should typically be the posterior design space obtained SVI 
            num_samples (int): 
                Number of samples (candidate designs) to use for estimating H.

        Returns:
            torch.Tensor: 
                Approximate gradient covariance matrix H of shape (15, 15).
        """ 
        print(f'    Approximating C from {num_samples} samples...' if self.verbose else '')
        
        # ---- Generate samples from the original design space ---- 
        candidate_designs = self.airfoil.pods_distrib.sample(torch.Size([num_samples])) 
        
        # ---- Determine design objectives ---- 
        # The objective list is always set up in the following order: [cp, cl, cm, cd]
        # this is used when iterating thru different objectives provided in the airfoil_inverse_designer
        target_arr = np.array([self.airfoil.target_cp,  
                               self.airfoil.target_cl,
                               self.airfoil.target_cd,
                               self.airfoil.target_cm])
        loss_weights_arr = np.array([1, 100, 1000, 100])
        target_idx = np.argwhere(target_arr!=None).flatten()
        target_arr = target_arr[target_idx]
        target_str = np.array(['C_p', 'c_l', 'c_m', 'c_d'])[target_idx]
        
        # ---- Define diagonal weight matrix (Ref [2], Eq.7) ----
        omega = loss_weights_arr[target_idx] 
        
        # Initialize H 
        H = torch.zeros((15, 15)) 
        for i, (omega_i, target_metric, loss_weight) in enumerate(zip(omega, target_arr, loss_weights_arr)):
            print(f'        Approximating C_i with respect to {target_str[i]}...' if self.verbose else '')
            
            # Create C, a covariance matrix that is the averaged outer product of the gradient (Ref [2], Eq.9)
            # Zero the gradient covariance matrix 
            C = torch.zeros((15, 15))
            
            # Iterate through a number of candidate designs to approximate the gradient 
            for _, candidate in enumerate(candidate_designs):
                # Create a leaf tensor with requires_grad
                candidate_design = candidate.clone().detach().requires_grad_(True) #  ((candidate + design_space_mean)*design_space_scale)
                
                # Forward pass through the LAM to obtain the QoI
                candidate_cp = self.model(self.airfoil, sample_tensor=candidate_design.to(self.model.output_device), get_grad=True)
                candidate_cl, candidate_cm, candidate_cd = self.get_coefficients(candidate_cp, candidate_design, num_samples=5000)  
                candidate_metrics = [candidate_cp, candidate_cl, candidate_cm, candidate_cd]
                    
                # Compute loss
                loss = self.__KLsymm(target_metric, candidate_metrics[target_idx[i]]) * loss_weight 
                
                # Backward pass to calculate the gradient wrt to candidate_design
                loss.backward()
                
                # Access gradient w.r.t. this candidate  
                C += torch.outer(candidate_design.grad.detach().cpu(), candidate_design.grad.detach().cpu()) 
            C /= num_samples
            
            # Construct H, the weighted sum of the gradient covariance matrices corresponding to each matrices
            H += omega_i * C # (Ref [2], Eq. 10) 
        return H
    
    def __partition_active_space(self, H:torch.Tensor, num_active_dims:int): 
        """
        Partition the gradient covariance matrix H into active and inactive subspaces
        using eigenvalue decomposition  

        Args:
            H (torch.Tensor): 
                Gradient covariance matrix of shape (d, d); typically 15 x 15.
            num_active_dims (int): 
                Number of active dimensions to retain (defines the active subspace rank).

        Returns:
            L1 (torch.Tensor): 
                Eigenvalues corresponding to the active subspace (largest num_active_dims).
            L2 (torch.Tensor): 
                Eigenvalues corresponding to the inactive subspace (remaining eigenvalues).
            W (torch.Tensor): 
                Eigenvectors spanning the active subspace (columns corresponding to L1).
            V (torch.Tensor): 
                Eigenvectors spanning the inactive subspace (columns corresponding to L2).
        """
        print(f'    Partitioning C into active and inactive space...' if self.verbose else '') 
        
        # Eigenvalue decomposition of H 
        eigval, eigvec = torch.linalg.eig(H) # eigenvalues, eigenvectors
        if num_active_dims=='auto':
            num_active_dims = np.argmax(np.abs(np.diff(eigval.real))) + 1
            self.airfoil.num_active_dims = num_active_dims
            self.airfoil.num_inactive_dims = 15 - num_active_dims
            print(f'    A total of {num_active_dims} active dimensions found...' if self.verbose else '') 
        # Partition the eigendecomposition into active and inactive dimensions 
        # depends on the user input 
        L1, L2 = eigval[:num_active_dims].real, eigval[num_active_dims:].real
        W, V = eigvec[:, :num_active_dims].real, eigvec[:, num_active_dims:].real
        # import matplotlib.pyplot as plt 
        # plt.rcParams['figure.subplot.left']   = 0.15
        # plt.rcParams['figure.subplot.right']  = 0.85
        # plt.rcParams['figure.subplot.top']    = 0.95
        # plt.rcParams['figure.subplot.bottom'] = 0.15
        # f, ax = plt.subplots(1,1, figsize=(8,6))
        # plt.semilogy(np.arange(1, eigval.real.shape[0]+1), np.flip(np.sort(eigval.real.numpy())),'o')
        # plt.ylabel('Eigenvalues')
        # plt.xlabel('Design variables')
        # print('Condition num: ', np.max(eigval.real.numpy())/np.min(eigval.real.numpy()))
        return L1, L2, W, V 
    
    def __determine_inactive_space(self, x0:torch.Tensor, W:torch.Tensor, V:torch.Tensor):
        """
        Sample points in the inactive subspace.

        Args:
            x0 (torch.Tensor): 
                feasible point in original space (n_dim,)
            W (torch.Tensor):
                active subspace eigenvectors (n_dim, n_dim - n_inactive) 
            V (torch.Tensor):
                inactive subspace eigenvectors (n_dim, n_inactive) 
        Returns:
            samples: array of shape (n_samples, n_dim) FIX THIS 
        """
        print(f'    Characterizing the inactive space...' if self.verbose else '')
        
        # Calculate covariances to obtain Schur's complement 
        Sig = self.airfoil.pods_distrib.covariance_matrix
        Sig_22 = V.T @ Sig @ V
        Sig_11 = W.T @ Sig @ W
        Sig_12 = W.T @ Sig @ V
        L = torch.linalg.cholesky(Sig_11)   
        X = torch.linalg.solve(L, Sig_12)             
        
        # Schur's complement 
        S = Sig_22 - X.T @ X
        S = 0.5 * (S + S.T) # Symmetrize
        
        # Analytical distribution in the principal component space 
        x_distrib_pca = torch.distributions.MultivariateNormal(
            loc = x0, 
            covariance_matrix = V @ S @ V.T + 1e-6*torch.eye(V.shape[0])
        )
        
        # Analytical distribution in the physical space  
        x_distrib_phys = self.airfoil.pca_transformer.inverse(x_distrib_pca, get_physical_coords=True,)
        x_distrib_phys = torch.distributions.MultivariateNormal(loc=x_distrib_phys.mean.cpu(), 
                                                                covariance_matrix=x_distrib_phys.covariance_matrix.cpu())
        return x_distrib_pca, x_distrib_phys
    
    def create_aero_gaussian_mixture(self,
                                     pca_distrib:torch.distributions.MultivariateNormal, 
                                     num_distrib:int=20):
        # Generate samples 
        airfoil_samples = pca_distrib.sample((num_distrib, ))
        
        # Obtain aerodynamic coeffficients' MVNs from the LAM predictions for each sample  
        # cp_distrib is MVN, the rest is just univariate N 
        cp_distribs, cl_distribs, cm_distribs, cd_distribs = [], [], [], []
        
        for candidate in airfoil_samples:
            candidate_cp = self.model(self.airfoil, sample_tensor=candidate.to(self.model.output_device))
            candidate_cl, candidate_cm, candidate_cd = self.get_coefficients(candidate_cp, candidate, num_samples=5000)  # <- redo this  

            cp_distribs.append(candidate_cp)
            cl_distribs.append(candidate_cl)
            cm_distribs.append(candidate_cm)
            cd_distribs.append(candidate_cd)
        
        # Create Gaussian mixture model from the LAM outputs
        gmm_weights = torch.distributions.Categorical(torch.ones(torch.Size([num_distrib])))
        cp_gaussian_mixture = torch.distributions.MixtureSameFamily(gmm_weights, 
                                                                    torch.distributions.MultivariateNormal(loc=torch.stack([i.mean.cpu().detach() for i in cp_distribs]),
                                                                                    covariance_matrix=torch.stack([i.covariance_matrix.cpu().detach() for i in cp_distribs])))
        print('Dlete below ')
        # if len(self.airfoil.alpha)==1: 
        #     cl_gaussian_mixture = torch.distributions.MixtureSameFamily(gmm_weights, 
        #                                                             torch.distributions.Normal(loc=torch.stack([i.loc.cpu().detach() for i in cl_distribs]),
        #                                                                                     scale=torch.stack([i.scale.cpu().detach() for i in cl_distribs])))
        #     cm_gaussian_mixture = torch.distributions.MixtureSameFamily(gmm_weights, 
        #                                                                 torch.distributions.Normal(loc=torch.stack([i.loc.cpu().detach() for i in cm_distribs]), 
        #                                                                                         scale=torch.stack([i.scale.cpu().detach() for i in cm_distribs])))
        #     cd_gaussian_mixture = torch.distributions.MixtureSameFamily(gmm_weights, 
        #                                                                 torch.distributions.Normal(loc=torch.stack([i.loc.cpu().detach() for i in cd_distribs]),
        #                                                                                         scale=torch.stack([i.scale.cpu().detach() for i in cd_distribs])))
        # else: 
        cl_gaussian_mixture = torch.distributions.MixtureSameFamily(gmm_weights, 
                                                                torch.distributions.MultivariateNormal(loc=torch.stack([i.mean.cpu().detach() for i in cl_distribs]),
                                                                covariance_matrix=torch.stack([i.covariance_matrix.cpu().detach() for i in cl_distribs])))
        cm_gaussian_mixture = torch.distributions.MixtureSameFamily(gmm_weights, 
                                                                torch.distributions.MultivariateNormal(loc=torch.stack([i.mean.cpu().detach() for i in cm_distribs]),
                                                                covariance_matrix=torch.stack([i.covariance_matrix.cpu().detach() for i in cm_distribs])))
        cd_gaussian_mixture = torch.distributions.MixtureSameFamily(gmm_weights, 
                                                                torch.distributions.MultivariateNormal(loc=torch.stack([i.mean.cpu().detach() for i in cd_distribs]),
                                                                covariance_matrix=torch.stack([i.covariance_matrix.cpu().detach() for i in cd_distribs])))
        # Sample if needed, otherwise return distribution 
        return cp_gaussian_mixture, cl_gaussian_mixture, cm_gaussian_mixture, cd_gaussian_mixture
        
    """
    Utilities
    """
    def __mvn_to_cpu(self, mvn: gpytorch.distributions.MultivariateNormal):
        """
        Takes a GPU-pushed multivariate normal distribution
        and makes a new multivariate normal distribution in CPU
        """
        return gpytorch.distributions.MultivariateNormal(
            mvn.mean.cpu(),
            mvn.covariance_matrix.cpu()
        )
    
    def __n_to_cpu(self, normal_distrib:torch.distributions.normal.Normal):
        """
        Takes a GPU-pushed univariate normal distribution
        and makes a new univariate normal distribution in CPU
        """
        return torch.distributions.normal.Normal(
            normal_distrib.mean.cpu(),
            normal_distrib.scale.cpu()
        )
        
    def save_result(self, file_path: str):
        """
        Save the airfoil design result and optimization history to a file.

        The results are stored as a dictionary containing:
            - 'design': the current airfoil object (with design parameters and targets)
            - 'history': the optimization history from the SVI/design process

        Parameters
        ----------
        file_path : str
            Path to the file where results will be saved.
            The file will be saved in .pkl format.
        """
        
        # Package design and history into a dictionary
        save_dict = { 
            'design': self.airfoil,
            'history': self.history
        }

        # Save dictionary to the given path using torch's serialization
        # with open(file_path, "wb") as file:
        torch.save(save_dict, file_path)
    
    """
    may wanna move this to lam_adapt
    """
    def get_coefficients(self, pressure_coeff, parameters, num_samples=5000): 
        params_to_coords =  self.airfoil.pca_transformer.inverse(parameters, get_physical_coords=True) # normalizer?
        ycu_lam = params_to_coords[:28].to(self.model.output_device)
        ycl_lam = params_to_coords[28:].to(self.model.output_device)
        ul_idx = 200 # hard-coded for now 

        # --- high resolution x/c values corresponding to default lam preds --- 
        x = (torch.flip(-2*(torch.sin(torch.linspace(0, 1, 200)*np.pi/2)-0.5)  , [0, ]).reshape((1, 1, 200)) + 1) / 2
        x = x.to(self.model.output_device)
        x = x.flatten()

        # --- interpolate lam y/c values into high resolution x/c in a differentiable manner --- 
        yu = quadratic_interp(x.to(self.model.output_device), 
                                        self.airfoil.lam_xc.to(self.model.output_device), 
                                        ycu_lam).flatten().to(self.model.output_device)
        yl = quadratic_interp(x.to(self.model.output_device), 
                                        self.airfoil.lam_xc.to(self.model.output_device), 
                                        ycl_lam).flatten().to(self.model.output_device)

        # --- compute panel vectors --- 
        dxu = torch.gradient(x)[0]  
        dxl = torch.gradient(x)[0] 

        dyu = torch.gradient(yu)[0] 
        dyl = torch.gradient(yl)[0] 

        dsu = torch.sqrt(dxu**2 + dyu**2)
        dsl = torch.sqrt(dxl**2 + dyl**2)
        

        num_targets = len(self.airfoil.alpha)
        cl_mean_list, cm_mean_list, cd_mean_list = [], [], []
        cl_std_list, cm_std_list, cd_std_list = [], [], []
        for i in range(num_targets):
            # --- sample pressure distribution mvn --- 
            cp = pressure_coeff.rsample(torch.Size([num_samples])).to(self.model.output_device)
            cpu_samples = cp[:, 2*ul_idx*i : 2*ul_idx*i + ul_idx] # upper surface  0:200, 400:600, ...
            cpl_samples = cp[:, 2*ul_idx*i + ul_idx : 2*ul_idx*i + 2*ul_idx] # lower surface 200:400, 600:800, ... 

            # --- calculate normals ---
            nxu = -dyu / dsu
            nyu = dxu / dsu
            nxl = dyl / dsl
            nyl = -dxl / dsl 

            # --- y-component panel forces ---  
            dFy_u = -cpu_samples * nyu * dsu
            dFy_l = -cpl_samples * nyl * dsl
            Fy_u, Fy_l = dFy_u.sum(dim=1), dFy_l.sum(dim=1)

            # --- x-component panel forces ---
            dFx_u = -cpu_samples * nxu * dsu
            dFx_l = -cpl_samples * nxl * dsl
            Fx_u, Fx_l = dFx_u.sum(dim=1), dFx_l.sum(dim=1)

            # --- compute lift and drag from panel forces ---
            aoa_rad = torch.deg2rad(torch.tensor(self.airfoil.alpha[i], device=self.model.output_device)) 
            cos_aoa, sin_aoa = torch.cos(aoa_rad), torch.sin(aoa_rad)
            ca = Fx_u + Fx_l
            cn = Fy_u + Fy_l
            cl = cn * cos_aoa - ca * sin_aoa
            cd = cn * sin_aoa + ca * cos_aoa 

            # --- compute qc moment using panel forces ---
            rx = x - 0.25
            ryu = yu 
            ryl = yl 
            dmu = rx.unsqueeze(0) * dFy_u - ryu.unsqueeze(0) * dFx_u
            dml = rx.unsqueeze(0) * dFy_l - ryl.unsqueeze(0) * dFx_l
            cm = -1 * (dmu.sum(dim=1) + dml.sum(dim=1))

            cl_mean_list.append(cl.mean())
            cl_std_list.append(cl.std(unbiased=False))
            cm_mean_list.append(cm.mean())
            cm_std_list.append(cm.std(unbiased=False))
            cd_mean_list.append(cd.mean())
            cd_std_list.append(cd.std(unbiased=False))

        def make_distrib_from_samples(samples):
                return torch.distributions.Normal(samples.mean(), samples.std(unbiased=False))
    
        def make_distrib_from_mean(mean_value, std_value):
                return torch.distributions.Normal(mean_value, std_value)
        
        def make_mvn_distrib_from_mean(mean_list, std_list): 
            mu = torch.concat([m.reshape(-1) for m in mean_list])
            cov = torch.block_diag(*[c.pow(2) for c in std_list]) # make sure this is variance 
            return gpytorch.distributions.MultivariateNormal(mu, cov)
    
        # if num_targets == 1: 
        #     return make_distrib_from_mean(cl_mean_list[0], cl_std_list[0]), make_distrib_from_mean(cm_mean_list[0], cm_std_list[0]), make_distrib_from_mean(cd_mean_list[0], cd_std_list[0])
        # else: 
        return make_mvn_distrib_from_mean(cl_mean_list, cl_std_list), make_mvn_distrib_from_mean(cm_mean_list, cm_std_list), make_mvn_distrib_from_mean(cd_mean_list, cd_std_list)
        

    # def get_coefficients(self, pressure_coeff, parameters, num_samples=5000):  
    #     params_to_coords =  self.airfoil.pca_transformer.inverse(parameters, get_physical_coords=True)
    #     zcu = params_to_coords[:28].to(self.model.output_device)
    #     zcl = params_to_coords[28:].to(self.model.output_device)

    #     # this is the default xc created from create_tensor fx converted to physical coordinates,
    #     # reshaped into interpolable dims 
    #     xc_highres = (torch.flip(-2*(torch.sin(torch.linspace(0, 1, 200)*np.pi/2)-0.5)  , [0, ]).reshape((1, 1, 200)) + 1) / 2
    #     xc_highres = xc_highres.to(self.model.output_device)
    #     xc_highres_flatten = xc_highres.flatten()
    #     # do high resolution interpolation of the coordinates based on 
        
    #     # Interpolation (diff'able)
    #     zcu_highres = quadratic_interp(xc_highres.to(self.model.output_device), 
    #                                    self.airfoil.lam_xc.to(self.model.output_device), 
    #                                    zcu).flatten() 
    #     zcl_highres = quadratic_interp(xc_highres.to(self.model.output_device), 
    #                                    self.airfoil.lam_xc.to(self.model.output_device), 
    #                                    zcl).flatten() 
        
    #     # Central differenc 
    #     dx = torch.gradient(xc_highres.flatten())[0]
    #     dzcu = torch.gradient(zcu_highres)[0]
    #     dzcl = torch.gradient(zcl_highres)[0]
        
    #     # Calculate tangent vectors  
    #     tangents_u = torch.column_stack((dx, dzcu))
    #     tangents_l = torch.column_stack((dx, dzcl))
        
    #     # Normalize tangents 
    #     norm_tangents_u = tangents_u / torch.linalg.norm(tangents_u, axis=1)[:, None]
    #     norm_tangents_l = tangents_l / torch.linalg.norm(tangents_l, axis=1)[:, None]
        
    #     # Calculate normals 
    #     normals_u = torch.column_stack((-norm_tangents_u[:,1], norm_tangents_u[:,0]))
    #     normals_l = torch.column_stack((norm_tangents_l[:,1], -norm_tangents_l[:,0]))

    #     # Get pressures samples 
    #     cp_samples = pressure_coeff.rsample(torch.Size([num_samples])).to(self.model.output_device)
    #     cpu_samples = cp_samples[:, :200]
    #     cpl_samples = cp_samples[:, 200:]
        
    #     # Dot product 
    #     weighted_dx = dx[None, :, None]  
    #     upper_dot_samples = (cpu_samples[:, :, None] * normals_u[None, :, :] * weighted_dx).sum(dim=1)  # [n,2]
    #     lower_dot_samples = (cpl_samples[:, :, None] * normals_l[None, :, :] * weighted_dx).sum(dim=1)  # [n,2]

    #     # Compute forces
    #     cn_samples = -(upper_dot_samples[:, 1] + lower_dot_samples[:, 1]) # [n]
    #     ca_samples = -(upper_dot_samples[:, 0] - lower_dot_samples[:, 0]) # [n]

    #     # Lift / Drag
    #     aoa_rad = torch.deg2rad(torch.tensor(self.airfoil.alpha, device=self.model.output_device))
    #     cos_aoa, sin_aoa = torch.cos(aoa_rad), torch.sin(aoa_rad)

    #     cl_samples = cn_samples * cos_aoa + ca_samples * sin_aoa
    #     cd_samples = cn_samples * sin_aoa + ca_samples * cos_aoa 
        
    #     # Calculate moment 
    #     arm = (xc_highres_flatten - 0.25)[None, :]  # [1,n]

    #     upper_cm_samples = torch.trapezoid(cpu_samples * arm, x=xc_highres_flatten, dim=1)
    #     lower_cm_samples = torch.trapezoid(cpl_samples * arm, x=xc_highres_flatten, dim=1)
    #     dzdx_u = (dzcu / dx)[None, :]
    #     dzdx_l = (dzcl / dx)[None, :]

    #     a = torch.trapezoid(cpu_samples * dzdx_u * zcu_highres[None, :], x=xc_highres_flatten, dim=1)
    #     b = torch.trapezoid(cpl_samples * dzdx_l * zcl_highres[None, :], x=xc_highres_flatten, dim=1)

    #     cm_samples = upper_cm_samples - lower_cm_samples + a - b
        
    #     def make_distrib(samples):
    #         return torch.distributions.Normal(samples.mean(), samples.std(unbiased=False))

    #     return make_distrib(cl_samples), make_distrib(cm_samples), make_distrib(cd_samples)

""" 

"""
def helper_airfoil_plotting(airfoil:airfoil_design_input, manual_distrib=None, return_dict=False):
    """
    Converts the airfoil geometry distribution into a plottable elements (numpy)
    """
    
    # By default, we are plotting the final design (w tolerances), unless manual distribution is provided
    if manual_distrib is None: 
        plotting_distrib = airfoil.design_airfoil_distrib
    elif isinstance(manual_distrib, torch.Tensor): # if not a distribution, just output the mean 
        if manual_distrib.shape[0] == 15: # this is PCA -> automatically convert to physical 
            plotting_distrib = airfoil.pca_transformer.inverse(
                manual_distrib.to(airfoil.pca_transformer.output_device),
                get_physical_coords=True
            ).cpu().detach().numpy() 
            
        # Make sure that it is in airfoil format
        assert plotting_distrib.shape[0] == 56, "Invalid plotting distribution size"
        
        xc = airfoil.lam_xc.numpy()
        zcu_mean = plotting_distrib[:28]
        zcl_mean = plotting_distrib[28:]
        if return_dict: 
            out_dict = (
                    {'xc':xc, 'upper':zcu_mean, 'lower':zcl_mean}, 
                    {'upper_bound':None, 'lower_bound':None}
                    )
            return out_dict
        else: 
            return xc, zcu_mean, zcl_mean
    
    elif isinstance(manual_distrib, torch.distributions.distribution.Distribution):
        plotting_distrib = manual_distrib
        
    else: 
        raise ValueError("Invalid manual distribution provided.")
        
    xc = airfoil.lam_xc.numpy()
    zcu_mean = plotting_distrib.mean[:28].numpy()
    zcl_mean = plotting_distrib.mean[28:].numpy()
    zcu_std = np.sqrt(
                    np.diag(
                        plotting_distrib.covariance_matrix.numpy()
                        )
                    )[:28]
    zcl_std = np.sqrt(
                    np.diag(
                        plotting_distrib.covariance_matrix.numpy()
                        )
                    )[28:]
    if return_dict: 
        out_dict = (
                    {'xc':xc, 'upper':zcu_mean, 'lower':zcl_mean}, 
                    {'upper_bound':2*zcu_std, 'lower_bound':2*zcl_std}
                    )
        return out_dict
    else: 
        return xc, zcu_mean, zcl_mean, 2*zcu_std, 2*zcl_std


def helper_cp_plotting(distrib:torch.distributions.distribution.Distribution, 
                                num_samples:int=10000,
                                return_dict=False):
    num_pts = 200
    num_cps = distrib.mean.shape[0] // (num_pts*2)

    xc = torch.flip(-2*(torch.sin(torch.linspace(0,1, 200)*torch.pi/2)-0.5)  , [0, ]).numpy()
    xc += 1 
    xc /= 2 
    mean_, std_ = sample_from_distrib(distrib, num_samples) 

    

    if return_dict: 
        if num_cps == 1:     
            out_dict = (
                        {'xc':xc, 'upper':mean_[:200], 'lower':mean_[200:]}, 
                        {'upper_bound':2*std_[:200], 'lower_bound':2*std_[200:]}
                        )
        else:  
            out_dict = []
            for i in range(num_cps):
                out_dict.append(
                    (
                        {'xc':xc, 'upper':mean_[i*num_pts*2 : i*num_pts*2 + num_pts], 'lower':mean_[i*num_pts*2 + num_pts : i*num_pts*2 + num_pts*2]}, 
                        {'upper_bound':2*std_[i*num_pts*2 : i*num_pts*2 + num_pts], 'lower_bound':2*std_[i*num_pts*2 + num_pts : i*num_pts*2 + num_pts*2]}
                    )
                ) 
        return out_dict
    else: 
        if num_cps == 1:     
            upper_mean = mean_[:200]
            lower_mean = mean_[200:]
            upper_std = std_[:200]
            lower_std = std_[200:]
        else: 
            upper_mean = [mean_[i*num_pts*2 : i*num_pts*2 + num_pts] for i in range(num_cps)]
            lower_mean = [mean_[i*num_pts*2 + num_pts : i*num_pts*2 + num_pts*2] for i in range(num_cps)]
            upper_std = [std_[i*num_pts*2 : i*num_pts*2 + num_pts] for i in range(num_cps)]
            lower_std = [std_[i*num_pts*2 + num_pts : i*num_pts*2 + num_pts*2] for i in range(num_cps)]
        return xc, upper_mean, lower_mean, 2*upper_std, 2*lower_std 

def sample_from_distrib(
    distrib: torch.distributions.distribution.Distribution, 
    num_samples: int = 10000
    ):
    """
    Estimate the mean and standard deviation from a given PyTorch distribution.

    This function handles two cases:
    1. If the distribution is a Multivariate Normal, it uses the analytical
       mean and covariance matrix to compute mean and standard deviation.
    2. Otherwise, it falls back to brute-force Monte Carlo sampling.

    Parameters
    ----------
    distrib : torch.distributions.Distribution
        The PyTorch probability distribution to analyze.
    num_samples : int, optional (default=10000)
        Number of samples to draw if brute-force sampling is required.

    Returns
    -------
    mean_ : numpy.ndarray
        Estimated mean vector of the distribution.
    std_ : numpy.ndarray
        Estimated standard deviation (per dimension).
    """

    if isinstance(distrib, torch.distributions.MultivariateNormal):
        # Case 1: Multivariate Normal → use exact mean & covariance
        mean_ = distrib.mean.cpu().detach().numpy()
        cov = distrib.covariance_matrix.cpu().detach().numpy()
        std_ = np.sqrt(np.diag(cov))  # extract diagonal elements (variances → std dev)
    else:
        # Case 2: Other distributions → sample to estimate statistics
        samples = distrib.sample(torch.Size([num_samples])).cpu().detach().numpy()
        mean_ = np.mean(samples, axis=0)
        std_ = np.std(samples, axis=0)

    return mean_, std_

class deterministic_designer:
    def __init__(self, 
                 evaluation_model:evaluation_model, # Should be the LAM by default   
                 user_input:airfoil_design_input,  
                 optimizer = None,   
                 max_iters:int=3000, 
                 output_device='cpu',
                 verbose=True): 
        """
        opt_alg: choice of optimization algorithm, currently "Nelder-Mead" or "Adam" [str]
        initial_guess: initial guess of CST coefficients, [torch.Tensor([cst,u_1, ... , cst,u_6, cst,l_1, ... , cst,l_6])]
        design_AoA: design angle of attack in degrees [float]
        design_mach: design freestream mach number [float]
        target_cp: target pressure coefficient distribution [MultivariateNormal]
        target_cl: target lift coefficient [normal distribution]
        target_cm: target LE moment coefficient [normal distribution]
        target_cd: target pressure drag coefficient [normal distribution]
        target_cst: NOT used for optimization, used to calculate residuals 
        max_iters: maximum number of iterations [int]
        """
        self.max_iters = max_iters  
        self.model = evaluation_model  
        self.optimizer = optimizer
        self.airfoil = user_input  
        self.output_device = output_device 
        self.verbose = verbose  
        # Keeps track of the inverse design/optimization history throughout iterations
        # and metadata
        self.history = {
            # Overall 
            'iters': [], # iteration counter
            'loss': [], # loss 
            'total_iter': 0, # total number of iterations
            'run_time': None, # algorithm run-time 
            
            # Aero 
            'pods_distrib': [], # design parameter @ each iteration
            'cp': [], # Cp distribution @ each iteration
            'cl': [], # cl distribution @ each iteration
            'cm': [], # cm distribution @ each iteration
            'cd': [], # cd distribution @ each iteration
            
            # For convergence check 
            'delta_distrib': []
        }
 
    def run_design(self, runtype='Nelder-Mead', *args, **kwargs):
        
        # ---- Pre-checks before running SVI ----
        # Ensure that at least one design objective is specified
        objective_count = 0
        if self.airfoil.target_cp is not None: 
            objective_count += 1
        if self.airfoil.target_cl is not None: 
            objective_count += 1
        if self.airfoil.target_cm is not None: 
            objective_count += 1
        if self.airfoil.target_cd is not None: 
            objective_count += 1
        assert objective_count >= 1, (
        "You must provide at least one design objective "
        "(target_cp, target_cl, target_cm, or target_cd)."
        )
        
        # Ensure initial airfoil design parameters are provided
        assert self.airfoil.initial_design is not None, (
            "You must provide the initial airfoil design parameters."
        )
        
        # ---- Initialization ----
        # Set the sample from the posterior design space 
        # to the initial design before beginning inverse desing
        self.airfoil.pods_sample = self.airfoil.initial_design

        # ---- Run optimization via SVI ----
        if runtype == 'Nelder-Mead':
            _ = self.__neldermead(*args, **kwargs)
        elif runtype == 'Adam':
            _ = self.__adam(*args, **kwargs)
        else: 
            return 
        return self.airfoil, self.history
    
    def __evaluate_lam(self, parameters):
            """ Take in CST and feed thru LAM """
            # From candidate geometry predict the corresponding aerodynamic quantities 
            candidate_cp = self.model(self.airfoil, sample_tensor=parameters)
            return candidate_cp 

    def __evaluate(self, parameters):
        """ optimization main body"""
        candidate_cp = self.__evaluate_lam(parameters)
        candidate_cl, candidate_cm, candidate_cd = None, None, None 
                
            
        # Computes Jensen-Shannon divergence as negative log likelihood (hence the -1 factor)
        loss_cp, loss_cl, loss_cm, loss_cd = self.__get_loss(current_coeffs=[candidate_cp, candidate_cl, candidate_cm, candidate_cd], 
                                                target_coeffs=[self.airfoil.target_cp, self.airfoil.target_cl, self.airfoil.target_cm, self.airfoil.target_cd],
                                                get_components=True)
        if isinstance(parameters, np.ndarray):
            self.airfoil.pods_sample = torch.from_numpy(parameters).cpu()
        else: 
            self.airfoil.pods_sample = parameters.cpu()
        self.airfoil.design_cp_distrib = candidate_cp
        # Calculate loss 
        loss = loss_cp.item() # Get scalar since gradient is NOT used here .item()
        return loss

    def __neldermead(self):
        from scipy.optimize import minimize
        result = minimize(self.__evaluate, self.airfoil.initial_design.cpu().detach().numpy(), method='Nelder-Mead')
        return result
    
    def __adam(self):
        parameters = self.airfoil.initial_design.clone().detach().requires_grad_(True)
        # Start at x = 5

        # Define the optimizer (Adam)
        optimizer = torch.optim.Adam([parameters], lr=0.01)

        # Training loop
        for step in range(2000):
            optimizer.zero_grad()  # Clear gradients

            # Dummy loss: minimize (x - target)^2
            loss = self.__evaluate(parameters)

            # Backpropagate
            loss.backward()

            # Update x
            optimizer.step()
        return 
    
    # def __callback(x):
    #     """ keeps track of optimization history (airfoil shapes, loss, etc)"""
    #     cp_ = __evaluate(x)
        
    #     # Update current airfoil
    #     self.current['cst'] = x
    #     self.current['cp'] = cp_
        
    #     # Calculate error  
    #     loss = self.get_loss().item() 
        
    #     # Print loss 
    #     print(f'Iteration {self.history["total_iter"]} loss: {loss}')
        
    #     # Add to optimization history 
    #     self.history['iters'].append(self.history['total_iter']) # iteration
    #     self.history['loss'].append(loss) # KL-divergence
    #     self.history['airfoil_cst'].append(x) # CST values 
    #     self.history['airfoil_cp'].append(self.mvn_to_cpu(cp_)) # Cp distribution, gpytorch.distributions.MultivariateNormal object cp_
    #     # Get Cp residual
    #     # Get airfoil residual
    #     self.history['total_iter'] += 1  # counter 
    
    
    """
    For calculating aerodynamic coefficients and losses 
    """
    def __get_loss(self, current_coeffs:list, target_coeffs:list, get_components:bool=False):
        # coeffs should be a list in the order of cp, cl, cm, cd 
        # for current and target respectively
        
        # Jensen-Shannon divergence can be thought of as the "symmetrized" version of the KL divergence 
        # a list in the order of cp, cl, cm, cd 
        losses = [] 
        for coeff_idx, target_coeff in enumerate(target_coeffs):
            if target_coeff is not None:
                losses.append(self.__KLsymm(current_coeffs[coeff_idx], target_coeff))
            else: 
                losses.append(None) 
        
        if get_components:
            loss_cp, loss_cl, loss_cm, loss_cd = losses 
            return loss_cp, loss_cl, loss_cm, loss_cd
        else: 
            loss_sum = 0.0
            for loss_component in losses:
                if loss_component is not None: 
                    loss_sum += loss_component
            return loss_sum
        
    def __KLsymm(self, current, target):
        """
        Compute the Symmetrized KL divergence between two distributions.

            KL_symm(p || q) ≈ 0.5 * ( KL(p || q) + KL(q || p) )

        Parameters
        ----------
        current : torch.distributions.Distribution
            The current distribution (e.g., model prediction).
        target : torch.distributions.Distribution
            The target distribution (e.g., design objective).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the Jensen-Shannon divergence.
        """
        
        # Compute the symmetric KL divergence (forward + reverse)
        loss = 0.5 * (
            torch.distributions.kl.kl_divergence(target, current) +
            torch.distributions.kl.kl_divergence(current, target)
        )
        
        return loss