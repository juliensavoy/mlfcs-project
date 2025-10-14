import argparse
import torch
import datetime
import json
import yaml
import os

from src.main_model import DiffPO
from src.utils import train, evaluate, train_sid, evaluate_sid
from dataset_acic import get_dataloader

from PropensityNet import load_data

import wandb
import copy


# start a new wandb_ihdp run to track this script
wandb.init(
    # set the wandb_ihdp project where this run will be logged
    project="DiffPO",
    notes="DiffPO"
)

wandb.define_metric("stage")

# Ensure a new WandB run is started if needed
def ensure_wandb():
    if wandb.run is None:  # Check if WandB is inactive
        wandb.init(project="DiffPO", notes="DiffPO")


torch.manual_seed(0)

parser = argparse.ArgumentParser(description="DiffPO")
parser.add_argument("--config", type=str, default="acic2018.yaml")
parser.add_argument("--current_id", type=str, default="")


parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2) # what is the test missing ratio?
parser.add_argument("--nfold", type=int, default=1, help="for 5-fold test") # what is the nfold here?
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--pretrain", type=int, default=1, help="Train pretrain model if necessary")
parser.add_argument("--num_epochs", type=int, default=2000, help="Number of SID training epochs")
parser.add_argument("--valid_epoch_interval", type=int, default=200, help="Validation interval")
parser.add_argument("--train_sid", type=int, default=1, help="Whether to train SID")
parser.add_argument("--alpha", type=float, default=0.5, help="alpha parameter in SiD")


args = parser.parse_args()
print(args)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print('Dataset is:')
print(config["dataset"]["data_name"])

print(json.dumps(config, indent=4))

# Create folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/acic_fold" + str(args.nfold) + "_" + current_time + "/"
# print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


current_id = args.current_id
print('Start exe_acic on current_id', current_id)

# Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    dataset_name = config["dataset"]["data_name"],
    current_id = current_id
)

# #=======================First train and fix propnet======================
# # Train a propensitynet on this dataset
#
# propnet = load_data(dataset_name = config["dataset"]["data_name"], current_id=current_id)
# # frozen the trained_propnet
# print('Finish training propnet and fix the parameters')
# propnet.eval()
# # ========================================================================
# propnet = propnet.to(args.device)
model = DiffPO(config, args.device).to(args.device)

if args.pretrain:
    wandb.log({"stage": "pretrain"})
    # save training setting
    wandb.config = {"epochs": config["train"]["epochs"], "num_steps": config["diffusion"]["num_steps"], "lr": config["train"]["lr"]}

    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        valid_epoch_interval=config["train"]["valid_epoch_interval"],
        foldername=foldername,
        propnet = None
    )
    print('----------------Finish pretraining------------')

    print("---------------Start testing pretraining---------------")

    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
    # save test model
    directory = "./save_model/" + args.current_id
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), "./save_model/" + args.current_id + "/model_weights.pth") # modify!!


else:
    print("Using existing pretrained model.")
    model.load_state_dict(torch.load("./save_model/"+ args.current_id + "/model_weights.pth"))
    model.to(args.device)
    model.eval()  # Set to evaluation mode

    print("---------------Start testing---------------")
    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

pretrain_model_path = f"./save_model/{args.current_id}/model_weights.pth" # modify!!



if args.train_sid:
    ensure_wandb()
    wandb.log({"stage": "sid"})

    print("----------------Start SID Training---------------")
    generator = train_sid(
        model_DiffPO=DiffPO,
        config=config,
        num_epochs=args.num_epochs,
        pretrain_path=pretrain_model_path,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_epoch_interval=args.valid_epoch_interval,
        device=args.device,
        alpha=args.alpha,
        propnet = None
    )
    print("----------------Finish SID training------------")
    print("----------------Start testing SID---------------")

    eval_metrics = evaluate_sid(model, generator, test_loader, nsample=args.nsample, device=args.device, prefix="out_of_sample/")

    model_save_path = os.path.join(f"./save_model/{args.current_id}/sid_model_weights.pth")

    torch.save(generator.state_dict(), model_save_path)


else:
    ensure_wandb()
    true_diffusion = DiffPO(config, args.device).to(args.device)
    true_diffusion.load_state_dict(torch.load(pretrain_model_path))
    true_diffusion.diffmodel.requires_grad_(False)
    generator = copy.deepcopy(true_diffusion.diffmodel)
    generator.load_state_dict(torch.load(f"./save_model/{args.current_id}/sid_model_weights.pth"))
    generator.to(args.device)
    generator.eval()

    print("---------------Start testing SID---------------")
    evaluate_sid(true_diffusion, generator, train_loader, nsample=args.nsample, device=args.device)

wandb.finish()