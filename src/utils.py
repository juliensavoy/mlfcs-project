import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import wandb
import copy
import torch.optim as optim

import torch.nn.functional as F

from scipy.stats import wasserstein_distance
from src.sid_loss import SID_EDMLoss
from src.karras_sde import KarrasSDE
from src.main_model import DiffPO


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def torch_wasserstein_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    return torch.mean(torch.abs(x_sorted - y_sorted))


############ Pretrain, train and evaluate

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=500,
    foldername="",
    propnet = None
):
    torch.manual_seed(0)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p0 = int(0.25 * config["epochs"])
    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    p3 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    wandb.log({"propnet_input": "None" if propnet is None else "Provided"})
    history = {'train_loss':[], 'val_rmse':[]}
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model.forward(batch = train_batch, is_train=1, propnet = propnet)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        wandb.log({"train loss":avg_loss / batch_no})

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            print(f'Start validation at Epoch: {epoch_no}')

            model.eval()
            val_nsample = 50


            pehe_val = AverageMeter()
            y0_val = AverageMeter()
            y1_val = AverageMeter()


            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        output = model.evaluate(valid_batch, val_nsample)
                        samples, observed_data, target_mask, observed_mask, observed_tp = output
                        samples_median = torch.median(samples, dim=1).values
                        
                        obs_data = observed_data.squeeze(1)
                        true_ite = obs_data[:, 3] - obs_data[:, 4]

                        pred_y0, pred_y1 = samples_median[:, 0], samples_median[:, 1]

                        diff_y0 = ((pred_y0 - obs_data[:, 3]) ** 2).mean() # use mu0, mu1
                        diff_y1 = ((pred_y1 - obs_data[:, 4]) ** 2).mean()
                        diff_ite = ((true_ite - (pred_y0 - pred_y1)) ** 2).mean()

                        y0_val.update(diff_y0.item(), obs_data.size(0))
                        y1_val.update(diff_y1.item(), obs_data.size(0))
                        pehe_val.update(diff_ite.item(), obs_data.size(0))



                # Compute final averaged validation metrics
                y0, y1 = torch.sqrt(torch.tensor(y0_val.avg)), torch.sqrt(torch.tensor(y1_val.avg))
                pehe = torch.sqrt(torch.tensor(pehe_val.avg))


                print(f"RMSE Y0 = {y0:.5g}, RMSE Y1 = {y1:.5g}, PEHE VAL = {pehe:.5g}, Wasserstein Y0 = {wass_y0:.5g}, Wasserstein Y1 = {wass_y1:.5g}")

        # Final in-sample evaluation after training
    print("Starting final in-sample evaluation on training set...")
    y0_rmse, y1_rmse, pehe, wass_y0, wass_y1 = evaluate(model, train_loader, nsample=50)
    wandb.log({
        "in_sample/y0 RMSE": y0_rmse,
        "in_sample/y1 RMSE": y1_rmse,
        "in_sample/PEHE": pehe,
    })








def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    torch.manual_seed(0)

    with torch.no_grad():
        model.eval()

        pehe_test = AverageMeter()
        y0_test = AverageMeter()
        y1_test = AverageMeter()
        wass_y0_test = AverageMeter()
        wass_y1_test = AverageMeter()

        y0_samples_list = []
        y1_samples_list = []
        y0_true_list = []
        y1_true_list = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, observed_data, target_mask, observed_mask, observed_tp = output

                samples_median = torch.median(samples, dim=1).values

                obs_data = observed_data.squeeze(1)
                true_ite = obs_data[:, 3] - obs_data[:, 4]

                pred_y0, pred_y1 = samples_median[:, 0], samples_median[:, 1]
                pred_y0_samples = samples[:, :, 0]  # [B, nsample]
                pred_y1_samples = samples[:, :, 1]
                pred_ite_samples = pred_y0_samples - pred_y1_samples
                pred_ite = torch.median(pred_ite_samples, dim=1).values

                diff_y0 = ((pred_y0 - obs_data[:, 3]) ** 2).mean()
                diff_y1 = ((pred_y1 - obs_data[:, 4]) ** 2).mean()
                diff_ite = ((true_ite - pred_ite) ** 2).mean()
                wass_y0 = torch_wasserstein_distance(obs_data[:, 3], pred_y0)
                wass_y1 = torch_wasserstein_distance(obs_data[:, 4], pred_y1)

                y0_test.update(diff_y0.item(), obs_data.size(0))
                y1_test.update(diff_y1.item(), obs_data.size(0))
                pehe_test.update(diff_ite.item(), obs_data.size(0))
                wass_y0_test.update(wass_y0.item(), obs_data.size(0))
                wass_y1_test.update(wass_y1.item(), obs_data.size(0))

                # For uncertainty estimation
                y0_samples_list.append(pred_y0_samples)
                y1_samples_list.append(pred_y1_samples)
                y0_true_list.append(obs_data[:, 3])
                y1_true_list.append(obs_data[:, 4])

        print('====================================')
        print('Finish test')

        # Compute RMSE and Wasserstein
        y0 = torch.sqrt(torch.tensor(y0_test.avg))
        y1 = torch.sqrt(torch.tensor(y1_test.avg))
        pehe = torch.sqrt(torch.tensor(pehe_test.avg))
        wass_y0 = torch.tensor(wass_y0_test.avg)
        wass_y1 = torch.tensor(wass_y1_test.avg)

        # Concatenate prediction samples and ground truth
        pred_samples_y0 = torch.cat(y0_samples_list, dim=0)  # [N, nsample]
        pred_samples_y1 = torch.cat(y1_samples_list, dim=0)
        truth_y0 = torch.cat(y0_true_list, dim=0)
        truth_y1 = torch.cat(y1_true_list, dim=0)

        # Uncertainty estimation for 90% and 95%
        prob_0_90, width_0_90 = compute_interval(pred_samples_y0, truth_y0, confidence_level=0.90)
        prob_0_95, width_0_95 = compute_interval(pred_samples_y0, truth_y0, confidence_level=0.95)
        prob_1_90, width_1_90 = compute_interval(pred_samples_y1, truth_y1, confidence_level=0.90)
        prob_1_95, width_1_95 = compute_interval(pred_samples_y1, truth_y1, confidence_level=0.95)

        print(f"Uncertainty (90%):  Y0 coverage = {prob_0_90:.3f}, median width = {width_0_90:.3f} | "
              f"Y1 coverage = {prob_1_90:.3f}, median width = {width_1_90:.3f}")
        print(f"Uncertainty (95%):  Y0 coverage = {prob_0_95:.3f}, median width = {width_0_95:.3f} | "
              f"Y1 coverage = {prob_1_95:.3f}, median width = {width_1_95:.3f}")

        wandb.log({
            "y0 TEST RMSE": y0.item(),
            "y1 TEST RMSE": y1.item(),
            "PEHE TEST": pehe.item(),
        })

        return y0.item(), y1.item(), pehe.item(), wass_y0.item(), wass_y1.item()


def check_interval(confidence_level, y_pred, y_true):
    lower = (1 - confidence_level) / 2
    upper = 1 - lower
    lower_quantile = torch.quantile(y_pred, lower)
    upper_quantile = torch.quantile(y_pred, upper)
    in_interval = torch.logical_and(y_true >= lower_quantile, y_true <= upper_quantile)
    return lower_quantile, upper_quantile, in_interval

def compute_interval(po_samples, y_true, confidence_level):
    """
    Computes the empirical coverage probability and median width of 95% prediction intervals.

    Args:
        po_samples (torch.Tensor): Tensor of shape [N, num_samples] — each row is a sample distribution for one instance.
        y_true (torch.Tensor): Tensor of shape [N] — true values.

    Returns:
        prob (float): Proportion of true values that fall within the 95% prediction interval.
        median_width (float): Median width of the prediction intervals.
    """
    counter = 0
    width_list = []

    for i in range(po_samples.shape[0]):
        lower_quantile, upper_quantile, in_interval = check_interval(confidence_level=confidence_level, y_pred=po_samples[i, :],
                                                                     y_true=y_true[i])

        if in_interval.item():  # convert from tensor to bool
            counter += 1

        width = upper_quantile - lower_quantile
        width_list.append(width.unsqueeze(0))

    prob = counter / po_samples.shape[0]
    all_width = torch.cat(width_list, dim=0)
    median_width = torch.median(all_width).item()

    return prob, median_width

def emp_dist(po_samples, true_samples):
    dist_list = []
    for i in range(po_samples.shape[0]):
        # Out-sample empirical Wasserstein distance
        out_wd = wasserstein_distance(po_samples[i, :], true_samples[i, :])
        dist_list.append(out_wd)
    total_dist = np.stack(dist_list, axis=0)
    avg_dist = np.mean(total_dist)
    return total_dist, avg_dist


####################################
############ SiD, train and evaluate
####################################

def train_sid(
    model_DiffPO,
    config,
    pretrain_path,
    num_epochs,
    train_loader,
    valid_loader,
    device,
    valid_epoch_interval=500,
    alpha=1.2,
    propnet=None
):
    print("Loading pre-trained DiffPO model for SID training...")

    # Load trained DiffPO model
    true_diffusion = model_DiffPO(config, device).to(device)
    true_diffusion.load_state_dict(torch.load(pretrain_path))
    true_diffusion.diffmodel.requires_grad_(False)

    sample_batch = next(iter(train_loader))
    observed_data, observed_mask, observed_tp, gt_mask, _, _ = true_diffusion.process_data(sample_batch)

    batch_x = observed_data[:, :, 5:].to(device)  # Covariates (x)
    batch_z = observed_data[:, :, 0].unsqueeze(2).to(device)  # Treatment (z)
    # batch_y = observed_data[:, :, 1:3].to(device)  # Outcome (y)

    # Concatenate condition: (x, z)
    batch_cond = torch.cat([batch_x, batch_z], dim=-1)

    # Create a fake model for SID training
    fake_model = copy.deepcopy(true_diffusion.diffmodel)

    # change to DiffPO
    fake_diffusion = KarrasSDE(x_dim=1, cond_dim=batch_cond.shape[-1], device=device, model=fake_model)
    fake_diffusion.model.requires_grad_(True)

    generator = copy.deepcopy(true_diffusion.diffmodel)  # Copy AFTER moving
    generator.requires_grad_(True)

    optimizer_fake = optim.Adam(fake_model.parameters(), lr=1e-4, betas=(0.0, 0.999), weight_decay=1e-6)
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999), weight_decay=1e-6)

    loss_fn = SID_EDMLoss()
    init_sigma = 2.5 #2.5
    torch.manual_seed(0)

    for epoch in range(num_epochs):

        #train_iter = iter(train_loader)
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer_fake.zero_grad()

                # === Process main batch ===
                observed_data, observed_mask, observed_tp, gt_mask, _, _ = true_diffusion.process_data(train_batch)
                batch_x = observed_data[:, :, 5:].to(device)
                batch_z = observed_data[:, :, 0].unsqueeze(2).to(device)
                batch_y = observed_data[:, :, 1:3].to(device)

                bernoulli_z_batch = torch.bernoulli(0.01 * torch.ones(batch_z.shape, device=batch_z.device))

                batch_x_shuffled = batch_x[torch.randperm(batch_x.shape[0])]
                batch_z_shuffled = batch_z[torch.randperm(batch_z.shape[0])]

                ######
                batch_cond = torch.cat([batch_x_shuffled, bernoulli_z_batch], dim=-1) #iwdd batch_cond
                # getting the data to match between fake and pretrain
                batch_cond_fake = torch.cat([batch_x, batch_z], dim=-1)#iwdd batch_cond_fake

                # sid batch_cond & batch_cond_fake
                #batch_cond = torch.cat([batch_x, batch_z], dim=-1)


                # true_diffusion is DiffPO
                # fake_diffusion is KarrasSDE

                ###### train generator, here fake is in eval, generator is in train
                # generator should use data x indep of z
                fake_diffusion.model.eval().requires_grad_(False)
                generator.train().requires_grad_(True)  # generator is diffmodel
                noise = torch.randn(batch_x.shape[0], batch_cond.shape[-1], device=batch_y.device) * init_sigma  ###  note that instead of 2, input a 178-dim noise
                sigma = torch.ones(batch_y.shape[0], 1, device=batch_y.device) * init_sigma
                fake_y = generator(noise, batch_cond, sigma)  # forward(self, x, cond_info, diffusion_step):
                loss = loss_fn.generator_loss(true_diffusion, fake_diffusion, fake_y, batch_cond, alpha=alpha, mask = None, weights =None) #weights.detach() 3 was batch_cond

                # ablation: DMD
                #loss = loss_fn.dmd_loss(true_diffusion, fake_diffusion, fake_y, cond = batch_cond, alpha=alpha) #weights.detach() 3 was batch_cond

                optimizer_g.zero_grad()
                loss.backward()

                optimizer_g.step()

                ### train fake diffusion, generator is in eval
                fake_diffusion.train().requires_grad_(True)
                generator.eval().requires_grad_(False)

                fake_y = fake_y.detach()
                fake_loss = fake_diffusion.diffusion_train_step(fake_y, batch_cond_fake, mask = None, weights = None) # weights.detach()
                optimizer_fake.zero_grad()
                fake_loss.backward()
                optimizer_fake.step()

                it.set_postfix(ordered_dict={"G loss": loss.item(), "epoch": epoch}, refresh=False)


        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, SID Loss: {loss.item() / len(train_loader):.4f}")
            wandb.log({"SID Train Loss": loss.item() / len(train_loader)})

        # Validation Step
        if valid_loader is not None and (epoch + 1) % valid_epoch_interval == 0:
            print("Starting validation...")
            eval_metrics = evaluate_sid(true_diffusion, generator, valid_loader, nsample=50, init_sigma=init_sigma, device=device)

    print("SID training complete.")
    print("Starting final in-sample evaluation on training set (SID)...")
    eval_metrics_in_sample = evaluate_sid(true_diffusion, generator, train_loader, nsample=50, init_sigma=init_sigma,
                                          device=device, prefix="in_sample/")

    wandb.log(eval_metrics_in_sample)

    return generator


def evaluate_sid(true_diffusion, generator, test_loader, nsample=100, init_sigma=5, device="cuda", prefix=""):
    """
    Evaluate the SID on the test set.
    Computes RMSE, PEHE, Wasserstein distance, and uncertainty (90% and 95%) for y0 and y1.

    Args:
        prefix (str): optional prefix to add to logged metric names (e.g., "out_sample/").
    """
    generator.eval()

    pehe_meter = AverageMeter()
    y0_rmse_meter, y1_rmse_meter = AverageMeter(), AverageMeter()

    y0_samples_list, y1_samples_list = [], []
    y0_true_list, y1_true_list = [], []

    with torch.no_grad():
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for test_batch in it:
                observed_data, observed_mask, observed_tp, gt_mask, _, _ = true_diffusion.process_data(test_batch)

                # Extract covariates and treatment
                batch_x = observed_data[:, :, 5:].to(device)
                z0 = torch.zeros_like(observed_data[:, :, 0:1]).to(device)
                z1 = torch.ones_like(observed_data[:, :, 0:1]).to(device)

                ground_truth_y0 = observed_data[:, :, 3].to(device)
                ground_truth_y1 = observed_data[:, :, 4].to(device)
                true_ite = ground_truth_y0 - ground_truth_y1

                cond_0 = torch.cat([batch_x, z0], dim=-1).squeeze(1)
                cond_1 = torch.cat([batch_x, z1], dim=-1).squeeze(1)

                batch_size = cond_0.shape[0]
                sigma = torch.ones(batch_size * nsample, 1, device=device) * init_sigma

                def sample_outputs(cond):
                    cond_repeat = cond.repeat_interleave(nsample, dim=0)
                    noise = torch.randn_like(cond_repeat) * init_sigma
                    preds = generator(noise, cond_repeat, sigma).view(batch_size, nsample, -1)
                    return preds

                preds_all_0 = sample_outputs(cond_0)
                preds_all_1 = sample_outputs(cond_1)

                pred_y0 = torch.median(preds_all_0, dim=1).values
                pred_y1 = torch.median(preds_all_1, dim=1).values
                pred_ite = pred_y0 - pred_y1

                diff_y0 = ((pred_y0 - ground_truth_y0.squeeze(1)) ** 2).mean()
                diff_y1 = ((pred_y1 - ground_truth_y1.squeeze(1)) ** 2).mean()
                diff_ite = ((true_ite.squeeze(1) - pred_ite) ** 2).mean()


                y0_rmse_meter.update(diff_y0.item(), batch_size)
                y1_rmse_meter.update(diff_y1.item(), batch_size)
                pehe_meter.update(diff_ite.item(), batch_size)


                y0_samples_list.append(preds_all_0)
                y1_samples_list.append(preds_all_1)
                y0_true_list.append(ground_truth_y0.squeeze(1))
                y1_true_list.append(ground_truth_y1.squeeze(1))



    y0_rmse = torch.sqrt(torch.tensor(y0_rmse_meter.avg))
    y1_rmse = torch.sqrt(torch.tensor(y1_rmse_meter.avg))
    pehe = torch.sqrt(torch.tensor(pehe_meter.avg))


    eval_metrics = {
        "IWDD RMSE_y0": y0_rmse.item(),
        "IWDD RMSE_y1": y1_rmse.item(),
        "IWDD PEHE": pehe.item(),
    }

    # Add prefix if given
    if prefix:
        eval_metrics = {f"{prefix}{k}": v for k, v in eval_metrics.items()}

    print("SID Evaluation Metrics:", eval_metrics)
    wandb.log(eval_metrics)

    return eval_metrics