# Paper figures

These are the editable generators and frozen inputs for the current paper's four main figures. The example stills and annotation screenshot come from real media and the actual annotation interface; graph labels and edges are drawn from the released gold records.

Install optional plotting dependencies and run from the repository root:

```bash
python -m pip install -r requirements-figures.txt
python figures/make_examples.py
python figures/make_workflow.py
python figures/make_dataset_statistics.py
python figures/make_figures.py
```

Outputs go to `figures/figs/`. The final command creates the compact paired-score audio-removal plot plus a pipeline diagram. All 40 endpoint scores and 20 differences in the paired plot use unrounded frozen values; its two panels share a linear scale.

`example_data.json` records complete gold graphs, frame filenames, timestamps, and SHA256 values. `dataset_statistics.json` contains the frozen dataset summary; the generator recomputes it from the released gold when available. `figure_data.csv` preserves the paper-matched estimates for the paired-score figure. The PNG previews in `assets/` are rendered from the approved vector figures.

The source-media rights are unchanged by inclusion of illustrative stills in the research figures. See the repository license and dataset provenance.
