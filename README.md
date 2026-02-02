# F_POI
### Paper validation demo

## Scope
This repository currently provides a simplified demo for paper validation. It includes four test groups and a minimal runnable flow; additional modules and full experiments will be added later.

## Test Groups
The demo supports four group sizes(P = 1,2,3,4). To switch between groups, update both values consistently in `FPOI.py`:

```python
classN = 2  # 2 / 4 / 8 / 16
range(2)    # 2 / 4 / 8 / 16
```

## Data Note
Due to the large size of the original dataset, the uploaded dataset currently keeps only 500 leakage points per trace that meet the F-test threshold.
