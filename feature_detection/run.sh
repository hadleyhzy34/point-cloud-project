#!/bin/sh
rm /home/swarm/developments/point_cloud/feature_detection/frnn/*.so
python ./frnn/setup3.py build
cp ./frnn/build/lib.linux-x86_64-3.6/*.so ./frnn/
python ./test_frnn.py
