# READ
Implementation of ["READ: Robust and Efficient Anomaly Detection under Data Contamination and Limited Supervision"]. (Accepted by SIGKDD 2025)

## Paper abstract
Existing anomaly detection methods tend to utilize a large amount of training data to learn patterns of normal data for effective anomaly identification, but such methods typically incur substantial training time overhead. Considering that unlabeled data often contains a lot of redundant information, selecting and utilizing a small yet representative subset instead of the entire dataset can significantly improve training efficiency while maintaining detection performance. To this end, we introduce an end-to-end reinforcement learning framework with a balanced sampling strategy that targets both normal and abnormal instances. This framework identifies and exploits potential anomalies in the unlabeled data while sampling peripheral normal instances (often difficult to detect), thereby enhancing the overall anomaly detection performance without requiring excessive time for the sampling process. Additionally, we present a joint reward mechanism, combined with inconsistency penalties, which optimizes both an agent’s action space and the representation space, ultimately improving the quality of the sampling process. Extensive experiments on four public datasets from different domains demonstrate the effectiveness and efficiency of our framework.

## Usage
* main.py is the executable file.
* The data folder is used to store experimental data.

## Running environment
Python version 3.9.18

Create suitable conda environment:
```
pip install -r requirements.txt
```

## Full paper source:
https://doi.org/10.1145/3711896.3737100

## Citation:
Shou H., Lu G., Pavlovski M., Zhou F., "READ: Robust and Efficient Anomaly Detection under Data Contamination and Limited Supervision", Proc. 31th Knowledge Discovery and Data Mining (KDD'25), 2025.


