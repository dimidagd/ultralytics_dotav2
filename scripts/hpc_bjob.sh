#!/bin/sh
### General options
### –- specify queue --
#BSUB -q "gpu_id"
### -- set the job Name --
#BSUB -J EXPERIMENT
### -- ask for number of cores (default: 1) --
#BSUB -n NCORES
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=NGPU:mode=exclusive_process:aff=yes"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB] SPANSETTING"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u dimidagd@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ./hpc_logs/gpu_%J.out
#BSUB -e ./hpc_logs/gpu_%J.err
#BSUB -R "selectgpumemory"
# -- end of LSF options --

echo "Preparing to run hpc script"
# Function to be executed when the Python script exits
function cleanup {
    echo "Python script terminated. Exiting..."
    # kill $nvidia_smi_pid
    ../scripts/cleanup.sh TEMPORARYDIR
    exit 1
}

# Trap the EXIT signal and call the cleanup function
trap cleanup EXIT SIGINT SIGTERM
DDP=false
distributed_cmd=""
NNODES=1
num_gpus=NGPU
batch_size=32
distributed_cmd="device=0"
if [ "$DDP" = true ] ; then
    echo "DDP true"!
    # Get the list of nodes-addresses
    echo "Getting list"
    if [ -e "$LSB_DJOB_HOSTFILE" ] && [ -f "$LSB_DJOB_HOSTFILE" ]; then
        List=$(cat "$LSB_DJOB_HOSTFILE" | uniq)
    else
        echo "The file '$LSB_DJOB_HOSTFILE' does not exist or is not a regular file."
    fi
    echo "List of nodes: $List"
    # Set the port for the rendezvous protocol
    PORT=$((RANDOM % 101 + 29400))
    echo "Random rendev port $PORT"
    if [ "$num_gpus" = 1 ] ; then
        distributed_cmd="device=0"
    fi
    if [ "$num_gpus" = 2 ] ; then
        distributed_cmd="device=0,1"
    fi
    if [ "$num_gpus" = 3 ] ; then
        distributed_cmd="device=0,1,2"
    fi
    # If NNODES > 2 then multinode=true
    if [ "$NNODES" -gt 1 ] ; then
        distributed_cmd="$distributed_cmd ddp_multinode=True ddp_hostname=$HOSTNAME ddp_multinode_port=$PORT ddp_multinode_nnodes=$NNODES"
    fi
    # Calculate 64* num_gpus and cast to int and divisible by the ngpus variable
    echo "numgpus $num_gpus"
    batch_size=$((batch_size * num_gpus))
    echo "Batch size $batch_size"
fi

echo "Hostname $HOSTNAME: and cuda devices: $CUDA_VISIBLE_DEVICES"

pretrained=False
multi_scale=False
inputsz=640
dataset=DOTAv2.0-patches-ship
basemodel=yolov8x-obb
date_time=$(date '+%Y%m%d_%H%M%S')
task=obb
# Calculate new learning rate based on batch size and 64
 # default lr for adam in ultralytics yolov8 is 0.01

echo "Loading stash from TEMPORARYDIR"
./load_stash.sh TEMPORARYDIR && \
cd TEMPORARYDIR && \
echo "Current working directory $PWD" && \
echo "Sourcing .bashrc-yolo" && \
source $HOME/.bashrc-yolo && echo "Running train python script"
optimizer=Adam
# base lr is 0.01 if optimizer is SDG, 0.001 if Adam, else 0.01
if [ "$optimizer" = "SGD" ] ; then
    lr=0.01
elif [ "$optimizer" = "Adam" ] ; then
    lr=0.001
else
    lr=0.01
fi

lr=$(echo "scale=5; $lr * sqrt($batch_size / 64)" | bc)

name=$task-$basemodel-$dataset-pre-trained-$pretrained-multi_scale-$multi_scale-$optimizer-$date_time
# Save the base command in a variable
base_command="LOGLEVEL=INFO yolo $task train optimizer=$optimizer data=$dataset.yaml exist_ok=True lr0=$lr model=$basemodel.yaml imgsz=$inputsz pretrained=$pretrained multi_scale=$multi_scale epochs=100 save_period=5 name=$name workers=8 batch=$batch_size $distributed_cmd 2>&1 | tee  ./scripts/hpc_logs/EXPERIMENT.log"
export HF_DATASETS_CACHE="/work1/dimda/cache/huggingface" \
WANDB_CACHE_DIR="/work1/dimda/cache/wandb" \
WANDB_DATA_DIR="/work1/dimda/cache/wandb" \

base_command="cd examples && python3 train_classifier.py"
if [[ "$DDP" = true ]]; then
  blaunch -z "$List" -- $base_command
else
  eval "$base_command"
fi
