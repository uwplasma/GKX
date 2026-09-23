#!/usr/bin/env bash
# Print the index of the first GPU whose only compute processes are the
# long-lived web workers listed in $IGNORE_PIDS, waiting until one is free.
set -u
ignore=" ${IGNORE_PIDS:-} "
while true; do
  for idx in 0 1; do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$idx")
    busy=0
    for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
      case "$ignore" in *" $pid "*) ;; *) busy=1 ;; esac
    done
    if [ "$busy" = 0 ]; then echo "$idx"; exit 0; fi
  done
  sleep 60
done
