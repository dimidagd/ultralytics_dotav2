#!/bin/bash
source var/ship.asciiart
source ~/.bashrc
set -e

# Define default values
GPU=""
NGPU=1
NHOSTS=1
EXPERIMENT=""
use_bsub=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpuqueue)
      GPU="$2"
      shift 2
      ;;
    --ngpu)
      NGPU="$2"
      shift 2
      ;;
    --nhosts)
      NHOSTS="$2"
      shift 2
      ;;
    --expname)
      EXPERIMENT="$2"
      shift 2
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
      ;;
  esac
done

# Check if GPU is empty
if [ -z "$GPU" ]; then
  echo "GPU argument not provided, exiting"
  exit 1
fi
# Set Distributed if NGPU >2
if [[ $NGPU -gt 1 ]]; then
    DDP=true
else
    DDP=false
fi

echo "Run on queue: $GPU"
echo "Distributed gpu: $DDP"
echo "Num gpus: $NGPU"
echo "Num hosts: $NHOSTS"
if [ "$GPU" = "gpua100" ]; then
    GPUMEM="select[gpu40gb]"
    use_bsub=true
elif [ "$GPU" = "gpuv100" ]; then
    GPUMEM="select[gpu32gb]"
    use_bsub=true
elif [ "$GPU" = "gpua100+" ]; then
    GPU=gpua100
    GPUMEM="select[gpu80gb]"
    use_bsub=true
elif [ "$GPU" = "bash" ]; then
    GPU=gpua100
    GPUMEM="select[gpu80gb]"
    use_bsub=false
else
    echo "Invalid argument. Usage: $0 [gpua100|gpua100+|gpuv100|bash]"
    exit 1
fi

source utils.sh
set -a
source .env
set +a
mkdir -p ../notebooks/runs/logs


if [ "$use_bsub" = true ]; then
([[ -z $(git status -s) ]] && \
print_ok "No uncommited changes" ) \
|| (print_failed "Changes detected below, please commit first" && git status -s && exit 1)
fi
NCORES=$(($NGPU * 7 * $NHOSTS))
NCORESPERNODE=$(($NGPU * 7))
if [ "$NHOSTS" -gt 1 ]; then
  HOSTSET="span[ptile=$NCORESPERNODE]"
  STANDALONE=""
else
  HOSTSET="span[hosts=1]"
  STANDALONE="--standalone"
fi
EXPERIMENT=${job:-$(date '+%Y%m%d_%H%M%S')}

if [ "$use_bsub" = true ]; then
    (bjobs -w | grep " $EXPERIMENT ") && echo "Job $EXPERIMENT already exists" && exit 1
    export gitcommit=$(git rev-parse HEAD) && \
source ./stash_src.sh && \
 (sed  "s:EXPERIMENT:$EXPERIMENT:g" hpc_bjob.sh | \
  sed  "s:num=NGPU:num=$NGPU:g" | \
  sed  "s:gpu_id:$GPU:g" | \
  sed  "s:selectgpumemory:$GPUMEM:g"| \
  sed  "s:TEMPORARYDIR:$temp_dir:g" | \
  sed  "s:GITCOMMIT:$gitcommit:g" | \
  sed "s:BSUB -n NCORES:BSUB -n $NCORES:g" | \
  sed 's/\+\+loader\.num_workers=NWORKERS //' | \
  sed "s/DDP=false/DDP=$DDP/g" | \
  sed "s/-m torch.distributed.run/-m torch.distributed.run $STANDALONE /g" | \
  sed "s/--nnodes=NNODES --nproc-per-node=NPROCPERNODE/--nnodes=$NHOSTS --nproc-per-node=$NGPU/g" | \
  sed "s:SPANSETTING:$HOSTSET:g" | \
  bsub)
else
read -p "Enter the batch size: " BATCHSIZE
   export gitcommit=$(git rev-parse HEAD) && \
source ./stash_src.sh && \
 (sed  "s:EXPERIMENT:$EXPERIMENT:g" hpc_bjob.sh | \
  sed  "s:num=NGPU:num=$NGPU:g" | \
  sed  "s:gpu_id:$GPU:g" | \
  sed  "s:selectgpumemory:$GPUMEM:g"| \
  sed  "s:TEMPORARYDIR:$temp_dir:g" | \
  sed  "s:GITCOMMIT:$gitcommit:g" | \
  sed "s:BSUB -n NCORES:BSUB -n $NCORES:g" | \
  sed "s/DDP=false/DDP=$DDP/g" | \
  sed "s/--nnodes=NNODES --nproc-per-node=NPROCPERNODE/--nnodes=$NHOSTS --nproc-per-node=$NGPU/g" | \
  sed 's/\+\+loader\.num_workers=NWORKERS/\+\+loader\.num_workers=BATCHSIZE \+\+loader\.batch_size=BATCHSIZE/g' | \
  sed  "s:BATCHSIZE:$BATCHSIZE:g" | \
  sed '/blaunch -z   "$List"/d' | \
  bash)
fi
