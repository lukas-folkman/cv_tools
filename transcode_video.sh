#!/bin/bash

vlc_opt="$1"
vlc_bin="$2"

if [[ ${vlc_opt} == "--vlc" ]]; then
  VLC=${vlc_bin}
  shift 2
else
  VLC="/Applications/VLC.app/Contents/MacOS/VLC"
fi

if [[ ! -f ${VLC} ]]; then
  echo "ERROR: ${VLC} does not exist. Specify a valid VLC binary, e.g.:"
  echo "$ bash transcode_video.sh --vlc $HOME/VideoLAN/VLC video1.asf video2.asf"
else
  for input in "$@"
  do
    output=`echo $input | sed 's/\.[A-Za-z0-9]*$/.mp4/'`
    if [[ -f ${output} ]]; then
      echo "ERROR: ${output} already exists"
    else
      echo "INFO: Creating ${output}"
      ${VLC} --no-repeat --no-loop -I dummy "${input}" \
        --sout="#transcode{}:std{access=file, mux=mp4, dst=\"${output}\"}" vlc://quit
    fi
  done
fi
