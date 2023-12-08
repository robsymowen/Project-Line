#!/bin/bash -l
#SBATCH --job-name=lines
#SBATCH --export=ALL

# ssh into cluster (must be on vpn): ssh username@login.rc.fas.harvard.edu
# cd to this directory cd $PROJECT_DIR/Project-Line

# testing: ./train.sh alexnet in1k_rgb --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv --test_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv --stop_early_epoch 2 --use_submitit 1 --total_batch_size 512

# train rgb, val rgb, test anime_style: 
# ./train.sh alexnet in1k_rgb --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv --test_dataset imagenet1k-line-ffcv/imagenet1k_anime_val_jpg_q100_s256_lmax512_crop.ffcv

# train anime_style, val anime_style, test rgb: 
# /train.sh alexnet in1k_anime_style --train_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_train_jpg_q100_s256_lmax512_crop-2b7bdbda.ffcv --val_dataset imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop-872c1585.ffcv --test_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv

# resume training: ./train.sh alexnet in1k_rgb --train_dataset imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv --val_dataset imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv --test_dataset imagenet1k-line-ffcv/imagenet1k_anime_val_jpg_q100_s256_lmax512_crop.ffcv --uuid 20231114_115005

# run without submitit: ./train.sh alexnet --stop_early_epoch 2 --use_submitit 0 --log_subfolder debug

# check job status with squeue: squeue -u username or squeue -a -p kempner
# you can ssh into nodes listed for your job: ssh holygpu8a19305
# then monitor gpu usage: watch -n .1 nvidia-smi

# Exit the script if any command fails
set -e

# set for debugging
# set -x

# Check if SHARED_DATA_DIR is set
if [ -z "$SHARED_DATA_DIR" ]; then
    echo "ERROR: SHARED_DATA_DIR is not set. Update your ~/.bashrc file."
    exit 1
fi

# Find the full path to the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Name of the github repo
PROJECT_NAME="$(basename "$SCRIPT_DIR")"

# fixed values
BUCKET_NAME='visionlab-members'
LOG_ROOT=$SCRIPT_DIR/logs
USERNAME=$USER
IN_MEMORY=1
NUM_WORKERS=16
NUM_GPUS=4
CONDA_ENV=workshop

# Default values for named arguments
PARTITION=kempner
ACCOUNT=kempner_alvarez_lab
IGNORE_ACCOUNT_WARN=0
USE_SUBMITIT=1
LOG_SUBFOLDER='models'
RECIPE="standard"
NUM_NODES=1
TOTAL_BATCH_SIZE=2048
EPOCHS=100
OPTIMIZER="adamw"
LR=0.0001
TRAIN_DATASET=$SHARED_DATA_DIR/imagenet1k-ffcv/imagenet1k_train_jpg_q100_s256_lmax512_crop.ffcv
VAL_DATASET=$SHARED_DATA_DIR/imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv
TEST_DATASET=$SHARED_DATA_DIR/imagenet1k-ffcv/imagenet1k_val_jpg_q100_s256_lmax512_crop.ffcv
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
STOP_EARLY_EPOCH=0
IMAGE_STATS_RGB=imagenet_rgb_avg
IMAGE_STATS_LINE=imagenet_line_stdonly

# Required positional arguments
MODEL_ARCH="$1"
DATASET="$2"
shift 2

# Process named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --partition) PARTITION="$2"; shift ;;
        --account) ACCOUNT="$2"; shift ;;
        --use_submitit) USE_SUBMITIT="$2"; shift ;;
        --log_subfolder) LOG_SUBFOLDER="$2"; shift ;;
        --recipe) RECIPE="$2"; shift ;;
        --total_batch_size) TOTAL_BATCH_SIZE="$2"; shift ;;
        --num_nodes) NUM_NODES="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --stop_early_epoch) STOP_EARLY_EPOCH="$2"; shift ;;
        --optimizer) OPTIMIZER="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --train_dataset) TRAIN_DATASET="$SHARED_DATA_DIR/$2"; shift ;;
        --val_dataset) VAL_DATASET="$SHARED_DATA_DIR/$2"; shift ;;  
        --test_dataset) TEST_DATASET="$SHARED_DATA_DIR/$2"; shift ;;
        --image_stats_rgb) IMAGE_STATS_RGB="$2"; shift ;;
        --image_stats_line) IMAGE_STATS_LINE="$2"; shift ;;
        --test_dataset) TEST_DATASET="$SHARED_DATA_DIR/$2"; shift ;;
        --ignore_account_warning_at_risk_of_burning_lab_fairshare_unnecessarily) IGNORE_ACCOUNT_WARN="$2"; shift ;;
        --uuid) UUID="$2"; shift ;;  # capture provided UUID
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# compute total world size (total num gpus)
WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS))

# compute per-gpu batch_size
BATCH_SIZE=$(($TOTAL_BATCH_SIZE / $WORLD_SIZE))

# If UUID hasn't been provided, generate a new one
# [ -z "$UUID" ] && UUID="${TIMESTAMP}_$(uuidgen)"
[ -z "$UUID" ] && UUID="${TIMESTAMP}"

# Check if the required arguments have been provided
if [ -z "$MODEL_ARCH" ] || [ -z "$DATASET" ]; then
    echo "Usage: $0 <model_arch> <dataset_name> [--partition PARTITION] [--account ACCOUNT] [--use_submitit USE_SUBMITIT] [--recipe RECIPE] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--stop_early_epoch STOP_EARLY_EPOCH] [--optimizer OPTIMIZER] [--lr LR] [--train_dataset TRAIN_DATASET] [--val_dataset VAL_DATASET] [--test_dataset TEST_DATASET] [--uuid UUID]"
    exit 1
fi

# Check if WORLD_SIZE divides evenly into TOTAL_BATCH_SIZE
if [ $(($TOTAL_BATCH_SIZE % $WORLD_SIZE)) -ne 0 ]; then
    echo "Error: WORLD_SIZE does not divide evenly into TOTAL_BATCH_SIZE"

    nearest_total_batch_size=$(($TOTAL_BATCH_SIZE/$WORLD_SIZE * $WORLD_SIZE))

    echo "You could set --total_batch_size=$nearest_total_batch_size, or any other multiple of $WORLD_SIZE"
else
    echo "TOTAL_NUM_GPUS (aka WORLD_SIZE)=$WORLD_SIZE divides evenly into TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    echo "Per-GPU batch_size is set to $BATCH_SIZE"
fi

# Function to check file existence and read permissions
check_file() {
    FILE=$1
    if [ -f "$FILE" ]; then
        if [ -r "$FILE" ]; then
          echo "File '$FILE' exists and is readable."
        else
          echo "Error: File '$FILE' is not readable."
          exit 1
        fi
    else
        echo "Error: File '$FILE' does not exist."
        exit 1
    fi
}

# Function to check if the user has access to the S3 bucket
check_s3_access() {
    echo "Verifying access to: s3://$BUCKET_NAME/$USERNAME/"
    aws s3 ls "s3://$BUCKET_NAME/$USERNAME/" --profile=wasabi
    if [ $? -eq 0 ]; then
        echo "Access to s3://$BUCKET_NAME/$USERNAME/ confirmed."
    else
        echo "Error: Cannot access s3://$BUCKET_NAME/$USERNAME/. Check your AWS credentials and permissions."
        exit 1
    fi
}

# Function to check partition and account consistency
check_partition_account() {
    if [[ $PARTITION != *"kempner"* && $ACCOUNT == *"kempner"* ]]; then
        echo "Warning: PARTITION does not contain 'kempner' but ACCOUNT does. It's recommended to set ACCOUNT to use your 'regular' account for non-kempner partitions, e.g., 'alvarez_lab' instead of 'kempner_alvarez_lab'"
        
        if [ $IGNORE_ACCOUNT_WARN -ne 1 ]; then
            echo "If you want to ignore this recommendation and proceed, rerun the script with '--ignore_account_warning_at_risk_of_burning_lab_fairshare_unnecessarily 1'"
            exit 1
        else
            echo "Warning ignored. Proceeding with the submission..."
        fi
    fi
}

# generate local and remote storage directories from arguments
LOCAL_LOG_SUBFOLDER=$LOG_SUBFOLDER/$DATASET/$MODEL_ARCH/$UUID
BUCKET_SUBFOLDER=$USERNAME/Projects/$PROJECT_NAME/$LOCAL_LOG_SUBFOLDER

echo "LOCAL_LOG_SUBFOLDER: $LOCAL_LOG_SUBFOLDER"
echo "BUCKET_SUBFOLDER: $BUCKET_SUBFOLDER"

# load anaconda
# module load Mambaforge/23.3.1-fasrc01
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, loading Mambaforge module..."
    module load Mambaforge/23.3.1-fasrc01
fi

source activate $CONDA_ENV
echo "PATH after activation: $PATH"

# Check if the dataset files exist
check_file "$TRAIN_DATASET"
check_file "$VAL_DATASET"
check_file "$TEST_DATASET"

# Verify S3 access
check_s3_access

# Verify choice of partition + account
check_partition_account

# Echo the python version
echo "Using Python from: $(which python)"

python3 train.py \
    --data.train_dataset $TRAIN_DATASET \
    --data.val_dataset $VAL_DATASET \
    --data.test_dataset $TEST_DATASET \
    --data.num_workers $NUM_WORKERS \
    --data.in_memory $IN_MEMORY \
    --data.image_stats_rgb $IMAGE_STATS_RGB \
    --data.image_stats_line $IMAGE_STATS_LINE \
    --logging.folder $LOG_ROOT/$LOCAL_LOG_SUBFOLDER \
    --logging.bucket_name $BUCKET_NAME \
    --logging.bucket_subfolder $BUCKET_SUBFOLDER \
    --training.distributed 1 \
    --dist.partition $PARTITION \
    --dist.account $ACCOUNT \
    --dist.world_size $WORLD_SIZE \
    --dist.ngpus $NUM_GPUS \
    --dist.nodes $NUM_NODES \
    --dist.use_submitit $USE_SUBMITIT \
    --model.arch "$MODEL_ARCH" \
    --training.batch_size "$BATCH_SIZE" \
    --training.epochs "$EPOCHS" \
    --training.stop_early_epoch "$STOP_EARLY_EPOCH" \
    --training.optimizer "$OPTIMIZER" \
    --training.base_lr "$LR" 0