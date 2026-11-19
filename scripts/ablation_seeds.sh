#!/bin/bash
# Multi-seed ablation of the auxiliary tasks (WHAM features, Human3.6M).
#   baseline      : no aux tasks
#   joint_mr0.10  : joint masking only (ratio 0.1)
#   frame_mr0.50  : frame masking only (ratio 0.5)
#
# Each config runs on its own GPU pair, over all seeds, and the 3 pairs run in parallel.
# Re-running skips finished runs and resumes unfinished ones from their last checkpoint.
# Run it inside tmux (it takes many hours). Progress: tail -f logs/ablation_multiseed_*/progress.log

# Data paths
DATA_3D=/media/data/ferreiral1/data_3d_h36m.npz
DATA_2D=/media/data/ferreiral1/data_2d_h36m_gt.npz
FEATURES=/media/data/ferreiral1/WHAM/feats_h36m.npy

export PYTHONPATH=. NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
source /home/group-cvg/cvg-students/drlo/miniconda/etc/profile.d/conda.sh
conda activate /home/ferreiral1/.conda/envs/thesis

SEEDS=(1126 42 2024 7 123)
CONFIGS=(baseline joint_mr0.10 frame_mr0.50)
GPU_PAIRS=("2,3" "4,5" "6,7")             # one pair per config, same order
TAG=bs256_lr3e-4_2gpu
LOG_DIR=logs/ablation_multiseed_$TAG
mkdir -p "$LOG_DIR" checkpoints

COMMON=(
    --wham --d_model 128 --feat_dim 563 --n_heads 4 --num_layers 4 --dim_ff 256
    --dataset_path "$DATA_3D"
    --dataset_2d_path "$DATA_2D"
    --features_path "$FEATURES"
    --subjects_train 'S1,S6,S7,S8,S9' --subjects_val 'S11' --subjects_test 'S5'
    --batch_size 128 --lr 3e-4             # 128 per GPU x 2 GPUs = 256
    --downsample 2 --stride 2 --workers 4 --epochs 50
)

# Extra arguments for each config
AUX="--aux --aux_frame_mode iid --aux_frame_mask_context"
config_args() {
    case $1 in
        baseline)     echo "--lambda_aux_joint 0.0 --lambda_aux_frame 0.0 --aux_joint_ratio 0.30 --aux_frame_ratio 0.30" ;;
        joint_mr0.10) echo "$AUX --lambda_aux_joint 1.0 --lambda_aux_frame 0.0 --aux_joint_ratio 0.10 --aux_frame_ratio 0.30" ;;
        frame_mr0.50) echo "$AUX --lambda_aux_joint 0.0 --lambda_aux_frame 1.0 --aux_joint_ratio 0.30 --aux_frame_ratio 0.50" ;;
    esac
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/progress.log"; }

# Train (up to 3 attempts, resuming from the last checkpoint) and then test one run
run() {
    local NAME=$1 SEED=$2 GPUS=$3 PORT=$4
    local RUN=wham_aux_${TAG}_${NAME}_s${SEED}
    local CDIR=checkpoints/$RUN
    local TEST_LOG=$LOG_DIR/${RUN}_test.log

    if grep -qs "Future horizons (mm)" "$TEST_LOG"; then log "skip $RUN (already tested)"; return; fi
    mkdir -p "$CDIR"
    local ARGS=("${COMMON[@]}" $(config_args "$NAME") --seed "$SEED"
                --checkpoint_dir "$CDIR" --checkpoint_save_name "$RUN.pth")

    for attempt in 1 2 3; do
        local LAST=$(ls -t "$CDIR"/*.pth 2>/dev/null | head -1)
        log "[GPU $GPUS] train $RUN (attempt $attempt) ${LAST:+resuming from $LAST}"
        CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=2 --master_port="$PORT" src/train_motionst.py \
            "${ARGS[@]}" --mode train --wandb_run_name "$RUN" ${LAST:+--resume "$LAST"} \
            >> "$LOG_DIR/${RUN}_train.log" 2>&1 && break
        sleep 60
    done

    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=2 --master_port="$PORT" src/train_motionst.py \
        "${ARGS[@]}" --mode test --wandb_mode disabled > "$TEST_LOG" 2>&1

    if grep -qs "Future horizons (mm)" "$TEST_LOG"; then log "[GPU $GPUS] done $RUN"
    else log "[GPU $GPUS] ERROR $RUN failed, see $LOG_DIR/${RUN}_*.log"; fi
}

# Ctrl-C stops everything
trap 'echo "stopping all runs"; kill 0; exit 1' INT TERM

echo "Started: $(date)"
for i in "${!CONFIGS[@]}"; do
    ( for SEED in "${SEEDS[@]}"; do run "${CONFIGS[$i]}" "$SEED" "${GPU_PAIRS[$i]}" $((29600 + i)); done ) &
    sleep 20                               # stagger start-up
done
wait

# Summary: mean ± std over seeds, and the difference to the baseline on the same seed
python - "$LOG_DIR" "$TAG" "${CONFIGS[*]}" "${SEEDS[@]}" <<'EOF'
import sys, re, statistics as st

log_dir, tag, configs, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4:]

def numbers(line):
    return [float(x) for x in re.findall(r"\d+\.\d+", line.split(": ", 1)[1])]

def mean_std(xs):
    return f"{st.mean(xs):.2f}±{st.stdev(xs) if len(xs) > 1 else 0:.2f}"

# results[config][seed] = (past MPJPE, [future MPJPE per horizon])
results = {c: {} for c in configs}
for c in configs:
    for s in seeds:
        try:
            lines = open(f"{log_dir}/wham_aux_{tag}_{c}_s{s}_test.log").read().splitlines()
        except FileNotFoundError:
            continue
        past = [numbers(l)[0] for l in lines if "Past reconstruction MPJPE" in l]
        fut = [numbers(l) for l in lines if "Future horizons (mm)" in l]
        if past and fut:
            results[c][s] = (past[-1], fut[-1])

print(f"\n===== SUMMARY ({tag}, seeds: {' '.join(seeds)}) =====")
print(f"{'config':<13} | n | {'past':<11} | {'future avg':<11} | future per horizon")
for c, runs in results.items():
    if not runs:
        print(f"{c:<13} | no results yet")
        continue
    past = [p for p, _ in runs.values()]
    avg = [st.mean(f) for _, f in runs.values()]
    horizons = [mean_std(h) for h in zip(*[f for _, f in runs.values()])]
    print(f"{c:<13} | {len(runs)} | {mean_std(past):<11} | {mean_std(avg):<11} | {' '.join(horizons)}")

print("\nDifference vs baseline, same seed (negative = better):")
base = results["baseline"]
for c in configs[1:]:
    shared = [s for s in results[c] if s in base]
    if not shared:
        continue
    diffs = [st.mean(results[c][s][1]) - st.mean(base[s][1]) for s in shared]
    better = sum(d < 0 for d in diffs)
    print(f"{c:<13} | future avg {mean_std(diffs)} mm | better on {better}/{len(diffs)} seeds | "
          + " ".join(f"{d:+.2f}" for d in diffs))
EOF

echo "Finished: $(date)"