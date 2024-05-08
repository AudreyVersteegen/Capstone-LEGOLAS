#!/bin/bash
nohup python /home/greenleaf/git/rpyc/bin/rpyc_classic.py --host 0.0.0.0&
nohup python /home/greenleaf/code/new_server.py &
