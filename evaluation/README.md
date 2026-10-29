# Evaluation

Run commands from the repository root with Python 3.10 or later. Scoring and these
tests use only the Python standard library.

```bash
python -m evaluation.score --predictions results/predictions/main/gemini-3.1-pro_raw.jsonl
python -m evaluation.score --predictions results/predictions/finetuned/qwen2.5-omni-7b-ft_raw.jsonl --split test
python -m evaluation.reproduce
python -m unittest discover -s tests -v
```

The first command evaluates **all 1,024 clips**. The second evaluates **197 held-out
clips**. `--split train` selects the 827 training clips. The wrapper requires every
clip in the selected cohort: missing predictions must be supplied as explicit
empty graphs, rather than silently removing difficult clips from evaluation.
Duplicate filenames and filenames absent from gold are errors. A full-corpus
prediction file can also be evaluated with `--split test`.

`--predictions` accepts a file or a flat directory of JSONL files. `--output PATH`
writes unrounded JSON containing TP/gold/predicted counts, micro precision/recall/F1,
macro F1 and sample sizes. `--gold PATH` and `--split-file PATH` select custom inputs.
Use full evaluation for the small example:

```bash
python -m evaluation.score --gold examples/gold.jsonl --predictions examples/prediction.jsonl
```

## Metric and parser

`avsg_metric.py` is the unchanged archived scorer. Nodes match by modality and class;
edges use the class-respecting alignment maximizing total edge matches. Attributes
are ignored. Symmetric relations match in either direction. Other edges are
directional. OOV labels remain and cost precision. The original handling of dangling
edges, duplicate node IDs, alignment ties and the 200,000-combination fallback is
preserved. Consequently, this scorer does not establish instance-level source
binding: same-class speaker swaps can score perfectly.

The headline is corpus **micro-F1**. Macro F1 excludes clips where both the gold and
prediction component are empty. Invalid records are scored as empty graphs. All
eight invalid main-benchmark records already contain empty graphs in the archive.
The release wrapper adds coverage checks around the core metric; use this wrapper
instead of the archived metric's overlap-only two-file CLI.

The additional audio-class recall diagnostic follows the paper's separate analysis:
it uses the class multiset of all `a`-prefixed nodes without deduplicating IDs. The
core graph scorer instead collapses duplicate IDs and treats every non-`v` prefix
as audio. Both historical definitions are retained, including on malformed outputs.

`parse_output.py` is the final campaign parser, including multi-object merging,
bare-key repair and prefix salvage. `results/source_manifest.json` records its hash
and the older software-archive parser hash that it supersedes. The saved parsed
prediction files are unchanged; rescoring them does not run JSON repair again.
Raw text records (`filename` plus `raw`) are supported through this parser.

## Prompt and calibration baselines

`prompt.py` is the exact historical `prompt_as_run.py`, including the original
shared-edge wording. It provides `SYSTEM_PROMPT`, `build_system_prompt()` and
`build_user_text(caption=None)`. Both raw and caption settings use the same system
prompt with three worked examples. Supplying a caption changes only the user turn.
`vocab.py` is the archived machine vocabulary.

```bash
python -m evaluation.prompt
python -m evaluation.baselines data/gold/1k_result.jsonl outputs/baselines
python -m evaluation.score --predictions outputs/baselines
```

The baseline generator uses its original algorithms; only default paths have been
adapted. The five archived baseline files remain under `results/predictions/baselines`.

## Original graph validation

```bash
python -m evaluation.score_validation
```

This reads the anonymized four-validator study in `data/validation`. It reproduces
224/236 Agree judgments (94.9%). No deliberately corrupted catch trials were
deployed; `catch_key.json` contains the historical real-item keys. This study
judges the acceptability of shown graphs. It is separate from the later blinded
edge audit and is not a human graph-generation ceiling.
