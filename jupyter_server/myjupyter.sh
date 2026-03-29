#!/bin/bash
#
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --gpus=h200-141:1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=jupyter-%j.out
#SBATCH --error=jupyter-%j.err

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"


# 1. Activate conda environment
source /home/e/e0958668/miniforge3/etc/profile.d/conda.sh
source /home/e/e0958668/miniforge3/etc/profile.d/mamba.sh
conda activate qwen_embedding

# 2. Define a log file to capture Jupyter's stdout/stderr (where the token URL appears)
JUPYTER_TEMP_LOG="${SLURM_SUBMIT_DIR}/jupyter-${SLURM_JOB_ID}.log"

# 3. Launch Jupyter in the background, redirecting output to the log file
jupyter lab --no-browser --ip=0.0.0.0 --port=5001 &> "${JUPYTER_TEMP_LOG}" &
JUPYTER_PID=$!
echo "Jupyter PID: $JUPYTER_PID � logging to: $JUPYTER_TEMP_LOG"

# 4. Poll the log until the token URL appears (timeout after 60s)
echo "Waiting for Jupyter Lab to initialise..."
timeout 60 bash -c \
    "until grep -qE 'http.*token=[a-f0-9]+' '${JUPYTER_TEMP_LOG}'; do sleep 2; done"

# 5. Extract and display the URL after Jupyter has started
JUPYTER_URL=$(grep -oE "http://[^ ]+token=[a-f0-9]+" "${JUPYTER_TEMP_LOG}" | head -n 1)

echo "========================================================"
echo "Jupyter Lab is ready at $(date)"
echo ""
echo "  1. Open an SSH tunnel from your local machine:"
echo "     ssh -L 5001:$(hostname):5001 <your_login_node_address>"
echo ""
echo "  2. Then open this URL in your browser:"
echo "     ${JUPYTER_URL}"
echo "========================================================"

# 6. Wait for Jupyter to exit (keeps the SLURM job alive for the session)
wait $JUPYTER_PID
echo "Jupyter Lab process ended at $(date)."

