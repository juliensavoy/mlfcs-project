CUDA_VISIBLE_DEVICES=1 python exe_acic.py --config acic2018.yaml --current_id "9333a461d3944d089ef60cdf3b88fd40" --pretrain 1 --train_sid 1


# to run multiple datasets together
# Define GPUs and number of jobs per GPU
gpu_list=(1 2 3 4 6 7)  # Adjust based on available GPUs
num_gpus=${#gpu_list[@]}  # Number of GPUs
num_jobs_per_gpu=1  # Jobs per GPU

# Define dataset IDs
dataset_ids=(
    "110f6dc8583c456ea0dd242d5d598497"
    "3ebc51612e034ff99e8632a228dae430"
    "5a147c7e542a4ea5b22da127b654666b"
    "5ad181455e954bcba44743e1f2d7824e"
    "74420a1794304013bb7a5a8f61994d71"
    "8ff38d337ec842dab1b8c01076e24816"
    "9333a461d3944d089ef60cdf3b88fd40"
    "ac6e494cbc254dc599be26a2a17f229c"
    "ae51149d38ce42609e00bf5701e4fe88"
    "d1546da12d8e4daf8fe6771e2187954d"
    "d4ae3280e4e24ca395533e429726fafc"
    "e36aca1030264e638452ea4053cbb42c"
)

# Define alpha values
alphas=(0.5)

# Initialize counter
i=0

# Loop through alpha and dataset combinations
for alpha in "${alphas[@]}"; do
    for dataset_id in "${dataset_ids[@]}"; do
        gpu=${gpu_list[$((i % num_gpus))]}  # Assign GPU in round-robin manner
        echo "Processing dataset: $dataset_id with alpha: $alpha on GPU: $gpu"

        # Launch the job in the background
        CUDA_VISIBLE_DEVICES=$gpu python exe_acic.py --config acic2018.yaml --current_id "$dataset_id" --pretrain 1 --train_sid 1 --num_epochs 2000 --alpha "$alpha" &

        ((i++))  # Increment counter

        # Wait after launching `num_gpus * num_jobs_per_gpu` processes
        if (( i % (num_gpus * num_jobs_per_gpu) == 0 )); then
            wait  # Wait for all background jobs before launching more
        fi
    done
done

wait  # Ensure all jobs finish
echo "All jobs completed!"

