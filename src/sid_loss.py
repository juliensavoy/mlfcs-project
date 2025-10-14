import torch
import torch.nn.functional as F
"""Loss functions used in the paper
"Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""


# ----------------------------------------------------------------------------
class SID_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, beta_d=19.9, beta_min=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min

    def generator_loss(self, true_score, fake_score, x, cond=None, alpha=1.2, tmax=800, mask = None, weights = None):
# tmax changed from 800 to 100
        sigma_min = 0.002
        sigma_max = 80 #80
        rho = 7.0
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        rnd_t = torch.rand([x.shape[0], 1], device=x.device) * tmax / 1000
        sigma = (max_inv_rho + (1 - rnd_t) * (min_inv_rho - max_inv_rho)) ** rho

        n = torch.randn_like(x) * sigma



        x_real = true_score.edm.denoise(x + n, cond, sigma) # x 1dim, y0 or y1
        x_fake = fake_score.denoise(x + n, cond, sigma)

        with torch.no_grad():
            weight_factor = abs(x - x_real).to(torch.float32).mean(dim=[1], keepdim=True).clip(min=0.00001)


        loss = (x_real - x_fake) * ((x_real - x) - alpha * (x_real - x_fake)) / weight_factor

        # weights from propnet
        if weights is not None:
            weights = weights.unsqueeze(1)
            loss = loss* weights  # Apply importance weighting

        if mask is not None:
            mask = mask.squeeze(1)[:, 1:3]
            num_observed = mask.sum(0)
            loss = (loss * mask).sum(0) / torch.where(num_observed > 0, num_observed,
                                                      torch.tensor(1.0, device=loss.device)) #this is 2-dim, loss for y0 and loss for y1

            batch_size = mask.shape[0]
            loss = loss * batch_size
            loss = loss.sum()
        else:
            loss = loss.sum()  # Standard mean loss if no mask


        return loss

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0):
        # from https://github.com/crowsonkb/k-diffusion
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def dmd_loss(self, true_score, fake_score, x, cond=None, alpha=1.2, tmax=800):
        sigma_min = 0.002
        sigma_max = 80
        rho = 7.0

        karras_sigmas = self.get_sigmas_karras(tmax, sigma_min, sigma_max, rho).to(x.device)

        min_step_percent = 0.02
        max_step_percent = 0.98
        num_train_timesteps = 800
        min_step = int(min_step_percent * num_train_timesteps)
        max_step = int(max_step_percent * num_train_timesteps)

        with torch.no_grad():
            timesteps = torch.randint(
                min_step,
                min(max_step + 1, num_train_timesteps),
                [x.shape[0], 1],
                device=x.device,
                dtype=torch.long
            )

            noise = torch.randn_like(x)
            #timestep_sigma = karras_sigmas[timesteps]
            timestep_sigma = karras_sigmas[timesteps.view(-1)]  #
            n = torch.randn_like(x) * timestep_sigma.reshape(-1, 1) * noise
            x_real = true_score.edm.denoise(x + n, cond, timestep_sigma.reshape(-1, 1))
            x_fake = fake_score.denoise(x + n, cond, timestep_sigma.reshape(-1, 1))

            p_real = (x - x_real)
            p_fake = (x - x_fake)

            # weight_factor = torch.abs(p_real).mean(dim=[1], keepdim=True)
            grad = (p_real - p_fake)  # / weight_factor

            grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(x, (x - grad).detach(), reduction="mean")

        return loss
