#!/usr/bin/env bash
spark-submit --master yarn --deploy-mode cluster --executor-memory 4G  --num-executors 4 spark-kmeans.py
