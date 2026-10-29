# One-clip format example

`gold.jsonl` contains one canonical gold record. `prediction.jsonl` expresses the
same graph in prediction format, so it scores 1.0 for the components present.

```bash
python -m evaluation.score --gold examples/gold.jsonl --predictions examples/prediction.jsonl
```

This example checks installation and the record format. It is not a model result.
Use the official 1,024-clip corpus or 197-clip test split for benchmark comparisons.
