#!/bin/bash

DATA_DIR="../data"
VIDEO_LIST="VIDEO_LIST.txt"

while read -r video_path
do
  if [[ -n "${video_path}" ]]; then
    if [[ ! -f "${DATA_DIR}/${video_path}" && ! -d "${DATA_DIR}/${video_path}" ]]; then
      echo "ERROR: ${DATA_DIR}/${video_path} does not exist"
    fi
  fi
done < "${VIDEO_LIST}"

echo "Done."
