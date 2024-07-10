#!/bin/bash

vlc_opt="$1"
vlc_bin="$2"

if [[ ${vlc_opt} == "--vlc" ]]; then
  VLC=${vlc_bin}
  shift 2
else
  module load misc/vlc/vlc 2> /dev/null
  if [[ ${?} == 0 ]]; then
    VLC="vlc"
  else
    VLC="/Applications/VLC.app/Contents/MacOS/VLC"
  fi
fi

output_opt="$1"
output_dir="$2"

if [[ ${output_opt} == "--output_dir" ]]; then
  shift 2
else
  output_dir=""
fi

if [[ ! -f ${VLC} && ${VLC} != "vlc" ]]; then
  echo "ERROR: ${VLC} does not exist. Specify a valid VLC binary, e.g.:"
  echo "$ bash transcode_video.sh --vlc $HOME/VideoLAN/VLC video1.asf video2.asf"
elif [[ -n ${output_dir} && ! -d ${output_dir} ]]; then
  echo "ERROR: Output directory ${output_dir} does not exist."
else
  for input in "$@"
  do
    output=$(echo ${input} | sed 's/\.[A-Za-z0-9]*$/.mp4/')
    if [[ -n ${output_dir} ]]; then
      output=${output_dir}/$(basename "${output}")
    elif [[ -f ${output} ]]; then
      output=$(echo ${input} | sed 's/\.[A-Za-z0-9]*$/_transcoded.mp4/')
    fi

    if [[ -f ${output} ]]; then
      echo "ERROR: ${output} already exists"
    else
      echo "INFO: Creating ${output}"
      ${VLC} --no-repeat --no-loop -I dummy "${input}" \
        --sout="#transcode{}:std{access=file, mux=mp4, dst=\"${output}\"}" vlc://quit
    fi
  done
fi
