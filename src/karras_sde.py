import numpy as np
import torch
import torch.nn as nn
from functools import partial
import math

class KarrasSDE(nn.Module):
    def __init__(
            self,
            x_dim: int,
            cond_dim: int,
            model,
            device: str,
            sigma_data=0.5,
            sigma_min=0.002,
            sigma_max=80,
            sigma_sample_density_type: str = 'lognormal',
    ) -> None:
        super().__init__()

        self.device = device
        # use the score wrapper
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.model = model # this is diffmodel
        self.mapping_noise = nn.Linear(1, self.cond_dim).to(self.device) # was 2

    def get_diffusion_scalings(self, sigma):
        """
        Computes the scaling factors for diffusion training at a given time step sigma.

        Args:
        - self: the object instance of the model
        - sigma (float or torch.Tensor): the time step at which to compute the scaling factors

        , where self.sigma_data: the data noise level of the diffusion process, set during initialization of the model

        Returns:
        - c_skip (torch.Tensor): the scaling factor for skipping the diffusion model for the given time step sigma
        - c_out (torch.Tensor): the scaling factor for the output of the diffusion model for the given time step sigma
        - c_in (torch.Tensor): the scaling factor for the input of the diffusion model for the given time step sigma

        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def diffusion_train_step(self, x, cond, noise=None, mask=None, t_chosen=None, weights = None):
        """
        Computes the training loss and performs a single update step for the score-based model.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, dim)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)

        Returns:
        - loss.item() (float): the scalar value of the training loss for this batch

        """

        self.model.train()
        x = x.to(self.device)
        cond = cond.to(self.device)

        if t_chosen is None:
            t_chosen = self.make_sample_density()(shape=(len(x),), device=self.device).reshape(-1,1)

        if mask is not None:
            mask = mask.to(self.device)


        #loss = self.diffusion_loss(x, cond, t_chosen, noise)
        loss = self.diffusion_loss(x, cond, t_chosen, noise, mask, weights)
        return loss



    def diffusion_loss(self, x, cond, t, noise, mask = None, weights = None):
        """
        Computes the diffusion training loss for the given model, input, condition, and time.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, channels, height, width)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)
        - t (torch.Tensor): the time step tensor of shape (batch_size,)

        Returns:
        - loss (torch.Tensor): the diffusion training loss tensor of shape ()

        The diffusion training loss is computed based on the following equation from Karras et al. 2022:
        loss = (model_output - target)^2.mean()
        where,
        - noise: a tensor of the same shape as x, containing randomly sampled noise
        - x_1: a tensor of the same shape as x, obtained by adding the noise tensor to x
        - c_skip, c_out, c_in: scaling tensors obtained from the diffusion scalings for the given time step
        - t: a tensor of the same shape as t, obtained by taking the natural logarithm of t and dividing it by 4
        - model_output: the output tensor of the model for the input x_1, condition cond, and time t
        - target: the target tensor for the given input x, scaling tensors c_skip, c_out, c_in, and time t
        """

        # apply mask here in the beginning, change xt from 2dim to 1dim
        if mask is not None:
            mask = mask.squeeze(1)[:, 1:3]
            x = x.squeeze(1)
            x = x * mask  # element-wise masking, [batch_size, 2]
            x = x.sum(dim=1)
            x = x.unsqueeze(1)

        if noise is None:
            noise = torch.randn_like(x, device=x.device)

        xt = x + noise * append_dims(t, x.ndim)
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4 # noise cond


        # added the following, mapping 2 column y to cond_dim columns
        noisy_target = self.mapping_noise(xt)
        noisy_target = noisy_target.squeeze(1)
        cond = cond.squeeze(1)


        model_output = self.model(noisy_target* c_in + cond, cond, t)
        #model_output = self.model(xt * c_in, cond, t) # this is where the MLP is called


        # denoised_x = c_out * model_output + c_skip * x_1
        target = (x - c_skip * xt) / c_out

        # Compute squared error
        loss = (model_output - target).pow(2)

        # Apply weights if available
        if weights is not None:
            weights = weights.unsqueeze(1)
            loss = loss * weights

        loss = loss.mean()


        return loss

    def round_sigma(self, sigma):
        return torch.tensor(sigma)

    def denoise(self, xt, cond, t):
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        # also add the mapping
        noisy_target = self.mapping_noise(xt) #xt (512,2)
        model_output = self.model(noisy_target * c_in, cond, t)
        #model_output = self.model(xt * c_in, cond, t)

        denoised_x = c_out * model_output + c_skip * xt
        return denoised_x

    def make_sample_density(self):
        """
        Returns a function that generates random timesteps based on the chosen sample density.

        Args:
        - self: the object instance of the model

        Returns:
        - sample_density_fn (callable): a function that generates random timesteps

        The method returns a callable function that generates random timesteps based on the chosen sample density.
        The available sample densities are:
        - 'lognormal': generates random timesteps from a log-normal distribution with mean and standard deviation set
                    during initialization of the model also used in Karras et al. (2022)
        - 'loglogistic': generates random timesteps from a log-logistic distribution with location parameter set to the
                        natural logarithm of the sigma_data parameter and scale and range parameters set during initialization
                        of the model
        - 'loguniform': generates random timesteps from a log-uniform distribution with range parameters set during
                        initialization of the model
        - 'uniform': generates random timesteps from a uniform distribution with range parameters set during initialization
                    of the model
        - 'v-diffusion': generates random timesteps using the Variational Diffusion sampler with range parameters set during
                        initialization of the model
        - 'discrete': generates random timesteps from the noise schedule using the exponential density
        - 'split-lognormal': generates random timesteps from a split log-normal distribution with mean and standard deviation
                            set during initialization of the model
        """
        sd_config = []

        if self.sigma_sample_density_type == 'lognormal':
            # loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            # scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            loc = - 1.2
            scale = 1.2
            return partial(rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(rand_discrete, values=sigmas)
        else:
            raise ValueError('Unknown sample density type')

    def edm_sampler(self, cond, randn_like=torch.randn_like,
            num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        latents = torch.randn((cond.size(0), self.x_dim)).to(cond.device)

        # Time step discretization.
        step_indices = torch.arange(num_steps, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.denoise(x_hat, cond, t_hat * torch.ones(cond.size(0),1).to(self.device))
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x_next, cond, t_next * torch.ones(cond.size(0),1).to(self.device))
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample(self, cond, num_steps=18):
        generate = self.edm_sampler(cond, num_steps=num_steps)

        return generate



def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def append_zero(action):
    return torch.cat([action, action.new_zeros([1])])


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from a lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an uniform distribution."""
    return torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value


def rand_discrete(shape, values, device='cpu', dtype=torch.float32):
    probs = [1 / len(values)] * len(values)  # set equal probability for all values
    return torch.tensor(np.random.choice(values, size=shape, p=probs), device=device, dtype=dtype)


def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data

