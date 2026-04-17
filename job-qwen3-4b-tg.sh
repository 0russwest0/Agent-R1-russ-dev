#!/bin/bash
#SBATCH -J tg_qwen3_4b
#SBATCH -p A100
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -n 80
#SBATCH --gres=gpu:A100:5
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -xeuo pipefail

cd /home/zrliu14/workspace/Agent-R1
mkdir -p logs
mkdir -p logs/rollout_smoke
mkdir -p logs/validation_smoke
RUN_TMPDIR="/dev/shm/a$$"
mkdir -p "$RUN_TMPDIR/ray"
ml Anaconda3/2025.06
ml cuda/13.0
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate verl

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export RAY_ADDRESS=local

export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com

export TEACHER_MODEL_NAME=/home/zrliu14/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export TEACHER_API_KEY=empty

mkdir -p "/data2/group_刘淇/$USER/checkpoints"
ln -sfn "/data2/group_刘淇/$USER/checkpoints" "$HOME/checkpoints_data2"

TEACHER_LOG="logs/teacher_${SLURM_JOB_ID}.log"
TEACHER_STARTUP_TIMEOUT_SECS=300

cleanup() {
  if [[ -n "${TEACHER_PID:-}" ]] && kill -0 "$TEACHER_PID" 2>/dev/null; then
    kill "$TEACHER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

CUDA_VISIBLE_DEVICES=4 vllm serve "$TEACHER_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --served-model-name "$TEACHER_MODEL_NAME" \
  > "$TEACHER_LOG" 2>&1 &
TEACHER_PID=$!

teacher_wait_start=$SECONDS
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
  if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
    echo "Teacher process exited before becoming ready. See $TEACHER_LOG" >&2
    wait "$TEACHER_PID" || true
    exit 1
  fi

  if (( SECONDS - teacher_wait_start >= TEACHER_STARTUP_TIMEOUT_SECS )); then
    echo "Teacher did not become ready within ${TEACHER_STARTUP_TIMEOUT_SECS}s. See $TEACHER_LOG" >&2
    exit 1
  fi

  sleep 5
done

export TEACHER_ENDPOINT=http://127.0.0.1:8000/v1

DATASET_TAG=${DATASET_TAG:-physics}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/sciknoweval/${DATASET_TAG}/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/sciknoweval/${DATASET_TAG}/test.parquet"}
MODEL_PATH=${MODEL_PATH:-"/home/zrliu14/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"}

PROJECT_NAME=${PROJECT_NAME:-"tg_qwen3_4b"}
RUN_ID=${SLURM_JOB_ID:-local}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-reinforce_plus_plus_baseline}
GROUP_ADVANTAGE_BY_STEP=${GROUP_ADVANTAGE_BY_STEP:-True}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-16}
MAX_STEPS=${MAX_STEPS:-2}

case "$ADV_ESTIMATOR" in
  reinforce_plus_plus_baseline) ADV_ESTIMATOR_TAG=rppb ;;
  grpo) ADV_ESTIMATOR_TAG=grpo ;;
  reinforce_plus_plus) ADV_ESTIMATOR_TAG=rpp ;;
  reinforce) ADV_ESTIMATOR_TAG=reinf ;;
  *) ADV_ESTIMATOR_TAG=$ADV_ESTIMATOR ;;
esac

case "$GROUP_ADVANTAGE_BY_STEP" in
  True|true) GROUP_ADVANTAGE_BY_STEP_TAG=1 ;;
  False|false) GROUP_ADVANTAGE_BY_STEP_TAG=0 ;;
  *) GROUP_ADVANTAGE_BY_STEP_TAG=$GROUP_ADVANTAGE_BY_STEP ;;
esac

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"q34_tg_${DATASET_TAG}_${ADV_ESTIMATOR_TAG}_gs${GROUP_ADVANTAGE_BY_STEP_TAG}_n${ROLLOUT_N}_vn${VAL_N}_s${MAX_STEPS}_bs${TRAIN_BATCH_SIZE}_lr${LEARNING_RATE}_${RUN_ID}"}

python3 -m agent_r1.main_agent_ppo \
  algorithm.adv_estimator=$ADV_ESTIMATOR \
  +algorithm.use_verl_advantage=True \
  +algorithm.group_advantage_by_step=$GROUP_ADVANTAGE_BY_STEP \
  data.train_files=$TRAIN_FILE \
  data.val_files=$VAL_FILE \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.max_prompt_length=8192 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.prompt_length=8192 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.agent.default_agent_flow=teacher_guidance_agent \
  actor_rollout_ref.rollout.agent.agent_flow_config_path=examples/teacher_guidance_agent.yaml \
  actor_rollout_ref.rollout.agent.max_steps=$MAX_STEPS \
  critic.enable=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.total_epochs=100