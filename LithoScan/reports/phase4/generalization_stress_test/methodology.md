# Phase 4D Generalization Stress Test Methodology

## Objective
To measure the generalization sensitivity to perceptual similarity by evaluating model performance on successively filtered subsets of the official test benchmark. 

## Approach
1. Utilizing the perceptual dHash matrix computed in Phase 3.
2. Filtering the official DCID-7 test set to produce evaluation subsets (`TEST_FILTERED_k`) where NO image in the subset has a corresponding training image with a dHash distance <= k.
3. Thresholds evaluated: k in {2, 5, 10}.
4. The official test directory on disk remains absolutely unchanged. The official test set (`TEST_ALL`) remains the primary benchmark.

## Terminology
The differences in performance across subsets indicate the model's *generalization sensitivity to perceptual similarity*. Filtering the datasets changes the evaluation population and class distributions, acting purely as a diagnostic stress test.
