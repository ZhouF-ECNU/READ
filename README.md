# READ: Robust and Efficient Anomaly Detection under Data Contamination and Limited Supervision

# READ
Implementation of ["READ: Robust and Efficient Anomaly Detection under Data Contamination and Limited Supervision"]. (Accepted by SIGKDD 2025)

## Paper abstract
Existing anomaly detection methods tend to utilize a large amount of training data to learn patterns of normal data for effective anomaly identification, but such methods typically incur substantial training time overhead. Considering that unlabeled data often contains a lot of redundant information, selecting and utilizing a small yet representative subset instead of the entire dataset can significantly improve training efficiency while maintaining detection performance. To this end, we introduce an end-to-end reinforcement learning
framework with a balanced sampling strategy that targets both normal and abnormal instances. This framework identifies and exploits potential anomalies in the unlabeled data while sampling peripheral normal instances (often difficult to detect), thereby enhancing the overall anomaly detection performance without requiring excessive time for the sampling process. Additionally, we present a joint reward mechanism, combined with inconsistency penalties, which optimizes both an agent’s action space and the representation space, ultimately improving the quality of the sampling process. Extensive experiments on four public datasets from different domains demonstrate the effectiveness and efficiency of our framework.

## Usage
* main.py is all the codes of the TargAD model.
* The data folder is used to store experimental data.

## Running environment
Python version 3.9.7

Create suitable conda environment:
```
conda env create -f environment.yml
```

## Full paper source:
https://ieeexplore.ieee.org/document/10597675

## Citation
>Lu G., Zhou F., Pavlovski M., Zhou C., Jin C., “A Robust Prioritized Anomaly Detection when Not All Anomalies are of Primary Interest”, Proc. 40th International Conference on Data Engineering (ICDE), 2024, 775-788.
