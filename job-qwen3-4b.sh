#!/bin/bash
#SBATCH -J tg_qwen3_4b
#SBATCH -p A800
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --gres=gpu:A800:4
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH -w gpu5

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

PROJECT_NAME=${PROJECT_NAME:-"tg_qwen3_4b"}
RUN_ID=${SLURM_JOB_ID:-local}
DATASET_TAG=${DATASET_TAG:-biology}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/sciknoweval/${DATASET_TAG}/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/sciknoweval/${DATASET_TAG}/test.parquet"}
MODEL_PATH=${MODEL_PATH:-"/home/zrliu14/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-reinforce_plus_plus_baseline}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-16}

case "$ADV_ESTIMATOR" in
  reinforce_plus_plus_baseline) ADV_ESTIMATOR_TAG=rppb ;;
  grpo) ADV_ESTIMATOR_TAG=grpo ;;
  reinforce_plus_plus) ADV_ESTIMATOR_TAG=rpp ;;
  reinforce) ADV_ESTIMATOR_TAG=reinf ;;
  *) ADV_ESTIMATOR_TAG=$ADV_ESTIMATOR ;;
esac

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"q34_base_${DATASET_TAG}_${ADV_ESTIMATOR_TAG}_n${ROLLOUT_N}_vn${VAL_N}_bs${TRAIN_BATCH_SIZE}_lr${LEARNING_RATE}_${RUN_ID}"}

python3 -m agent_r1.main_agent_ppo \
  algorithm.adv_estimator=$ADV_ESTIMATOR \
  +algorithm.use_verl_advantage=True \
  +algorithm.group_advantage_by_step=True \
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
  actor_rollout_ref.rollout.agent.default_agent_flow=teacher_guidance_agent \
  actor_rollout_ref.rollout.agent.agent_flow_config_path=examples/teacher_guidance_agent.yaml \
  actor_rollout_ref.rollout.agent.max_steps=1 \
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