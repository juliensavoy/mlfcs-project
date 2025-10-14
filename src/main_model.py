import numpy as np
import torch
import torch.nn as nn
from src.diff_model import diff_CSDI
from src.karras_sde import KarrasSDE
import yaml


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        # keep the __init__ the same
        super().__init__()
        self.device = device
        self.target_dim = config["train"]["batch_size"]
        

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim

        self.cond_dim = config["diffusion"]["cond_dim"]
        self.mapping_noise = nn.Linear(2, self.cond_dim)

        if self.is_unconditional == False:
            self.emb_total_dim += 1

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2 # here it is 1, may need to change?
        self.diffmodel = diff_CSDI(config_diff, input_dim) # is the diffusion architecture

        self.num_steps = config_diff["num_steps"]


        # Initialize KarrasSDE for EDM-style noise scheduling
        self.edm = KarrasSDE(
            x_dim=1,  # Number of target variables (y0, y1), should be 1, y0 or y1
            cond_dim=self.cond_dim,
            model=self.diffmodel, # the network architecture of diffpo
            device=self.device
        )



    #
    def calc_loss_edm(self, observed_data, cond_mask, gt_mask, is_train, set_t=-1, propnet=None):
        B, K, L = observed_data.shape

        # Extract the conditioning data
        a = observed_data[:, :, 0].unsqueeze(2)  # Treatment indicator
        x = observed_data[:, :, 5:]  # Covariates
        cond_obs = torch.cat([a, x], dim=2)
        t_batch = observed_data[:, :, 0].squeeze()

        # Extract the target variables (y0, y1)
        y_data = observed_data[:, :, 1:3]

        # Apply EDM loss computation using KarrasSDE
        target_mask = gt_mask - cond_mask

        #     x_batch = observed_data[:, :, 5:] # all the covariates
        if len(x.shape) > 2:
            x = x.squeeze()
        if len(x.shape) < 2:
            x = x.unsqueeze(1)

        if propnet is not None:
            pi_hat = propnet.forward(x.float())
            weights = (t_batch / pi_hat[:, 1]) + ((1 - t_batch) / pi_hat[:, 0])
            weights = weights.reshape(-1, 1, 1)
            # Clip weights for stable training
            weights = torch.clamp(weights, min=0.1, max=5)
        else:
            weights = None
        loss = self.edm.diffusion_train_step(y_data, cond_obs, mask=target_mask, weights=weights)



        return loss

    def impute_edm(self, observed_data, cond_mask, n_samples): # used in evaluation
        B, K, L = observed_data.shape
        #imputed_samples = torch.zeros(B, n_samples, 2).to(self.device)

        # Use only covariates for conditioning
        x = observed_data[:, :, 5:]  # shape: [B, K, L-5]

        # Create constant a=0 and a=1 tensors
        a0 = torch.zeros(B, K, 1).to(self.device)
        a1 = torch.ones(B, K, 1).to(self.device)

        # Concatenate treatment and covariates to form conditioning inputs
        cond_obs_0 = torch.cat([a0, x], dim=2)
        cond_obs_1 = torch.cat([a1, x], dim=2)

        # === Instead of loop, repeat BATCH for n_samples ===
        cond_obs_0_repeated = cond_obs_0.repeat_interleave(n_samples, dim=0)  # shape: [B * n_samples, K, L-4]
        cond_obs_1_repeated = cond_obs_1.repeat_interleave(n_samples, dim=0)

        # Sample all at once
        y0_samples = self.edm.edm_sampler(
            cond=cond_obs_0_repeated,
            num_steps=50,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            S_churn=0,
            S_noise=0
        ).detach()

        y1_samples = self.edm.edm_sampler(
            cond=cond_obs_1_repeated,
            num_steps=50,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            S_churn=0,
            S_noise=0
        ).detach()

        # Reshape outputs
        y0_samples = y0_samples.view(B, n_samples)
        y1_samples = y1_samples.view(B, n_samples)

        # Stack into final shape [B, n_samples, 2]
        imputed_samples = torch.stack([y0_samples, y1_samples], dim=-1)

        return imputed_samples

    def forward(self, batch, is_train=1, propnet = None):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
    
        # cond_mask: specify which elements are included in the conditional input to the model
        if is_train == 0: # in evaluation mode
            cond_mask = gt_mask.clone()
        else: # in training mode
            cond_mask = gt_mask.clone()
            
            cond_mask[:, :, 1] = 0
            cond_mask[:, :, 2] = 0

        loss_func = self.calc_loss_edm(observed_data, cond_mask, gt_mask, is_train, propnet=propnet)

        return loss_func

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            cond_mask[:,:,0] = 0
            target_mask = observed_mask - cond_mask
            #side_info = self.get_side_info(observed_tp, cond_mask) # this is just cond_mask

            samples = self.impute_edm(observed_data, cond_mask, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class DiffPO(CSDI_base):
    def __init__(self, config, device, target_dim=1):
        super(DiffPO, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"][:, np.newaxis, :]
        observed_data = observed_data.to(self.device).float()

        observed_mask = batch["observed_mask"][:, np.newaxis, :]
        observed_mask = observed_mask.to(self.device).float()

        observed_tp = batch["timepoints"].to(self.device).float()

        gt_mask = batch["gt_mask"][:, np.newaxis, :]

        gt_mask = gt_mask.to(self.device).float()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )