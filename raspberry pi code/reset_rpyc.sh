#!/bin/bash

server=$(ps aux | awk '/git\/rpyc/ {print}')
PID=$(ps aux | awk '/git\/ {print $2}'/)
kill $PID
nohup python /home/greenleaf/git/rpyc/bin/rpyc_classic.py --host 0.0.0.0&
nohup python /home/greenleaf/code/new_server.py &
echo "killed and reset $server"
