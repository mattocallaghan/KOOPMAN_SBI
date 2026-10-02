#!/bin/zsh
set -e

# SBIBM's Lotka--Volterra task invokes Julia through diffeqtorch.
export JULIA_PROJECT="/Users/ocallaghanm/.julia/environments/v1.10"
export JULIA_LOAD_PATH="/Users/ocallaghanm/.julia/environments/v1.10:@stdlib"

tasks=(
  lotka_volterra
  gaussian_mixture
  gaussian_linear_uniform
  two_moons
  bernoulli_glm
  sir
  gaussian_linear
  slcp
  slcp_distractors
  bernoulli_glm_raw
)

start_task=""
if [[ "$1" == "--from" ]]; then
  if [[ -z "$2" ]]; then
    echo "Usage: $0 [--from TASK]" >&2
    exit 2
  fi
  start_task="$2"
fi

found_start=false
if [[ -z "$start_task" ]]; then
  found_start=true
fi

for task in $tasks; do
  if [[ "$found_start" == false && "$task" != "$start_task" ]]; then
    continue
  fi
  found_start=true
  echo "===== Running $task ====="
  python -m koopman_sbi benchmark-suite \
    --task "$task" \
    --train-flow-matching \
    --generate-teacher \
    --force-retrain
done

if [[ -n "$start_task" && "$found_start" == false ]]; then
  echo "Unknown task: $start_task" >&2
  exit 2
fi
