import torch
import gpytorch
import numpy as np


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, num_tasks, num_inputs):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, 16, num_inputs)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # ------------------------------------------------------
        # Note: maybe use Matern kernel instead of RBF kernel for more flexibility in modeling the data
        # ------------------------------------------------------



        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # type: ignore 
    
def load_model(model_path):
    #checkpoint = torch.load('learning/models/gp_model.pth')
    checkpoint = torch.load(model_path)

    # Recreate the model with the saved hyperparameters
    model = MultitaskGPModel(
        num_latents=checkpoint['num_latents'],
        num_tasks=checkpoint['num_tasks'],
        num_inputs=checkpoint['num_inputs']
    )

    # Load the saved state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=checkpoint['num_tasks'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    return model, likelihood

def predict(model, likelihood, x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(x))


class CorrectedDLOModel:
    def __init__(self, model, likelihood):
        self.model = model
        self.likelihood = likelihood
        self.model.eval()
        self.likelihood.eval()

    def predict(self, sl_pos, ft_tcp, torque_tcp):
        """
        sl_pos    : np.ndarray (N_nodes, 3) — Simulink node positions
        ft_tcp    : np.ndarray (3,)         — force at TCP from Simulink
        torque_tcp: np.ndarray (3,)         — torque at TCP from Simulink

        Returns corrected node positions as np.ndarray (N_nodes, 3)
        """
        # Build the same input format your GP was trained on
        x_input = np.concatenate([ft_tcp, torque_tcp])           # (6,)
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)  # (1, 6)

        # Get GP residual prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(x_tensor))
            residual = preds.mean.numpy()   # (1, N_nodes * 3)

        residual = residual.reshape(sl_pos.shape)  # (N_nodes, 3)

        # Corrected position = Simulink + residual
        return sl_pos + residual

    def predict_with_uncertainty(self, sl_pos, ft_tcp, torque_tcp):
        """Same but also returns 2-sigma confidence bounds."""
        x_input = np.concatenate([ft_tcp, torque_tcp])
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(x_tensor))
            residual = preds.mean.numpy().reshape(sl_pos.shape)
            lower, upper = preds.confidence_region()
            lower = lower.numpy().reshape(sl_pos.shape)
            upper = upper.numpy().reshape(sl_pos.shape)

        corrected = sl_pos + residual
        return corrected, sl_pos + lower, sl_pos + upper
