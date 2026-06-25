#!/usr/bin/env bash

COMPETITION="hyperspectral-object-tracking-challenge-2026"
SUBMISSION_FILE="notebooks/results/submission.csv"
MESSAGE="submission"

kaggle competitions submit \
    -c "$COMPETITION" \
    -f "$SUBMISSION_FILE" \
    -m "$MESSAGE"