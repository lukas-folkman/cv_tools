#!/bin/bash

docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml -f docker-compose.mount-storage.yml -f docker-compose.fix-platform-mac.yml up -d
