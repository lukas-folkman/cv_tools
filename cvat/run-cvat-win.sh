#!/bin/bash

docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# for serverless support add
# -f components/serverless/docker-compose.serverless.yml

# for local storage add
# -f docker-compose.mount-storage.yml
# and run
# $ docker volume create --name cvat_share --opt type=none --opt device="$YOUR_DIRECTORY" --opt o=bind
