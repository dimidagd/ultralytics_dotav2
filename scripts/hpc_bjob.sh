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
#BSUB -u pgrigor22@gmail.com
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


num=NGPU
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
    # if num=2, then distributed_cmd="device=0,1"
    if [ "$num" = 2 ] ; then
        distributed_cmd="device=0,1"
    fi
    if [ "$num" = 3 ] ; then
        distributed_cmd="device=0,1,2"
    fi
fi


batch_size=64	
# Check if numgpus is equal to anything else than NGPU
if [ "$DDP" = true ] ; then
    echo "numgpus $num"
    # Calculate 128/numgpus and cast to int and divisible by the ngpus variable
    batch_size=$((64 * num))    
fi

# nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 10 &
# nvidia_smi_pid=$!
echo "Hostname $HOSTNAME: and cuda devices: $CUDA_VISIBLE_DEVICES"

echo "Loading stash from TEMPORARYDIR"
./load_stash.sh TEMPORARYDIR && \
cd TEMPORARYDIR && \
echo "Current working directory $PWD" && \
cd ./examples && echo "Sourcing bashrc" && \
source $HOME/.bashrc-yolo && echo "Running train python script" && \
LOGLEVEL=INFO yolo obb train data=DOTAv2.0-patches.yaml exist_ok=True model=yolov8n-obb.yaml pretrained=yolov8n-obb.pt epochs=100 save_period=1 project=runs name=train-obb imgsz=640 batch=$batch_size $distributed_cmd \
2>&1 | tee  ../scripts/hpc_logs/EXPERIMENT.log
../scripts/cleanup.sh TEMPORARYDIR

# Monitor job with $ bnvtop JOBID