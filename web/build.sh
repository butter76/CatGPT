#!/bin/bash
set -e
cd "$(dirname "$0")"
npm run build
sudo systemctl restart catgpt-web
