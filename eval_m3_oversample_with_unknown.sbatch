#!/usr/bin/env bash
#SBATCH --job-name=eval_m3_unk
#SBATCH --partition=NA100q
#SBATCH --nodelist=node01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/eval_m3_oversample_with_unknown/slurm_%j.log
#SBATCH --error=experiments/eval_m3_oversample_with_unknown/slurm_%j.log

set -euo pipefail
cd /export/home2/wa0009xi/ots-lid

source /export/home2/wa0009xi/miniconda3/etc/profile.d/conda.sh
conda activate ots-lid-torch210

export CUDA_VISIBLE_DEVICES=3

mkdir -p experiments/eval_m3_oversample_with_unknown

echo "===================================================="
echo "job_id: $SLURM_JOB_ID  node: $SLURMD_NODENAME"
echo "===================================================="

"$CONDA_PREFIX/bin/python" scripts/eval_ambernet_lid.py \
    --ckpt checkpoints/ambernet_lid_19class_oversample_20epoch.nemo \
    --eval_manifest manifests/lid_eval_en_cv_mix_3s_with_unknown.json \
    --work_dir experiments/eval_m3_oversample_with_unknown \
    --labels "de,en,es,fr,hi,id,ja,km,ko,ms,pt,ru,th,tl,tr,unknown,vi,yue,zh" \
    --batch_size 256 \
    --num_workers 8 \
    --accelerator gpu \
    --precision 16-mixed \
    --per_label \
    --top_confusions 30

echo "[$(date)] eval done"
