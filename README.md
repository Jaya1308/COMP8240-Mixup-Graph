# Reproducing "Mixup for Node and Graph Classification" (Wang et al., WWW '21)

COMP8240 project – Jaya Sharma, Macquarie University, 2026.

This repository reproduces the node-classification results of
**Wang, Wang, Liang, Cai and Hooi (2021), "Mixup for Node and Graph Classification"**, *Proceedings of the Web Conference 2021*, pp. 3663–3674 ([DOI 10.1145/3442381.3449796](https://doi.org/10.1145/3442381.3449796)).
It uses the authors' released code ([vanoracai/MixupForGraph](https://github.com/vanoracai/MixupForGraph), MIT licence), and then tests the method on new data:
- two existing graphs the paper did not use (Amazon Computers and Photo);
- a new citation graph of 2,813 arXiv papers, built for this project.

All experiments run in Google Colab. Every notebook can be read on GitHub with its outputs.

## Main findings

| Question | Result |
|---|---|
| Does the authors' released code reproduce a Mixup gain? | Yes, in its own configuration (GraphConv, α = 4, 3 layers, 1 split × 5 seeds): Cora +1.51, Pubmed +0.52, Citeseer +0.33 points |
| Does the gain hold in the setting the paper describes (GCN/JKNet, α = 1, 10 random 60/20/20 splits × 3 seeds)? | Not consistently: Citeseer −0.13, Cora +0.11, Pubmed −0.08 (1,710 runs). Baselines are broadly comparable to the paper |
| Does it transfer to new data? | No. Amazon: Computers −3.46, Photo −0.66. arXiv graph: no backbone improves (all 95% CIs include 0); Mixup+GCN is slightly worse (macro-F1 −0.54, p = 0.02) |
| What explains the arXiv result? | Text features are strong (a feature-only MLP beats GCN and GAT). Homophily is moderate (0.636 vs 0.81 for Cora). Errors concentrate on papers with mixed-area neighbourhoods and on cross-listed papers |

<!-- TODO after the final run: add the α × layer result (Section 8.1), the Amazon convergence check (8.2), the kappa values (4.3/5.3) and the LLM-judge share (7.4). -->

## Repository layout

```
notebooks/
  00_original_code_run.ipynb       authors' mixup.py and graph_conv.py run unchanged (CPU, PyTorch 2.6)
  01_table3_reproduction.ipynb     Table 3 replication, released-code configuration, Amazon datasets
  02_arxiv_graph_construction.ipynb  building the new arXiv citation graph (OpenAlex + arXiv API)
  03_final_experiments.ipynb       annotation agreement, LLM-as-a-judge, extended tests, explanation experiments
data/arxiv_graph/
  paper_graph.pt                   features, labels and edges, loaded by notebook 03 (about 4 MB)
  papers.csv                       one row per node: arXiv id, title, label, year, citations, degree
  edges.csv                        undirected citation edges between node ids
  dataset_card.json                collection settings and statistics
  annotation_sheet_filled.csv      first label check (50 papers, arXiv label visible)
  blind_sheet_you.xlsx             blind annotation (52 papers, labels hidden)
  blind_key.csv                    maps blind items back to arXiv ids
results/
  tables/                          every table in the report (CSV + LaTeX)
  figures/                         every figure in the report (PNG + PDF)
  step4_results.jsonl              one line per training run (all splits and seeds)
  llm_cache.jsonl                  every Gemini answer used in the report
```

## How to run

1. Open a notebook in Google Colab (File → Open notebook → GitHub, or upload it).
2. Choose **Runtime → Change runtime type → T4 GPU**. Notebook 00 runs on CPU.
3. The notebooks save to Google Drive under `MyDrive/mixup_reproduction/` and pick up where they stopped after a disconnect.
4. Run them in order. Each notebook explains its settings at the top.
5. **To run only notebook 03 on the saved dataset,** copy the files in `data/arxiv_graph/` to `MyDrive/mixup_reproduction/paper_graph/`, then run it. Notebook 02 rebuilds the same files from the APIs.

| Notebook | Produces | Time |
|---|---|---|
| 00 | Authors' code output on Pubmed, Cora and Citeseer | ~30 min (CPU, including building `torch-sparse`) |
| 01 | Table 3 replication, released-code configuration, Amazon results | ~3 h on a T4 (`RUN_MODE = 'medium'`) |
| 02 | `paper_graph.pt`, `papers.csv`, `edges.csv`, `dataset_card.json`, annotation sheet | ~15 min (API calls are cached) |
| 03 | All tables and figures for the final report | ~2.5–3 h on a T4, plus Gemini time |

**Gemini (notebook 03 only).** Add a free Google AI Studio key to Colab Secrets as `GEMINI_API_KEY`. The key is never stored in the notebook. The free tier allows about 20 requests per model per day, and the notebook needs about 30. It caches every answer and continues when the quota runs out, so re-run it the next day to finish. Without a key, the notebook reuses the cached answers in `results/llm_cache.jsonl`.

**Quick check.** In notebook 03, set `QUICK = True` for a 10-minute smoke test with tiny settings. Its results are kept in a separate file.

## The arXiv citation graph

| Property | Value |
|---|---|
| Nodes / edges | 2,813 papers / 19,936 undirected citation links (largest connected component) |
| Labels | arXiv primary category, 6 classes: cs.CV 35.6%, cs.LG 29.7%, cs.CL 18.9%, cs.CR 7.3%, cs.RO 5.5%, cs.IR 3.0% |
| Features | 384-d `all-MiniLM-L6-v2` embedding of title + abstract |
| Edge homophily | 0.636 |
| Sources | [OpenAlex](https://openalex.org) (citations, CC0) and the [arXiv API](https://info.arxiv.org/help/api/) (metadata, CC0) |
| Collection | Snowball sample: 121 seed papers, 1 hop of references and citing papers, kept if on arXiv in one of the 6 categories |

Labels were checked by hand in two rounds: 50 papers with the arXiv label visible, then 52 blind. Agreement is reported as Cohen's kappa in notebook 03, Section 4.3. Gemini was used as an independent third rater.

## Environment

| | Notebooks 01–03 | Notebook 00 |
|---|---|---|
| Python | 3.13 (Colab) | 3.13 (Colab) |
| PyTorch | 2.11.0 + CUDA 13.0, Tesla T4 | 2.6.0 CPU |
| PyTorch Geometric | 2.8.1 | 2.8.x with `torch-scatter` 2.1.2 / `torch-sparse` 0.6.18 |
| LLM | `gemini-3.8-flash` (Google AI Studio, free tier, November 2026) | – |

Notebooks 01 and 03 do all graph operations in plain PyTorch, so they do not need `torch-scatter` or `torch-sparse`. `requirements.txt` lists the packages for running outside Colab.

## Reproducibility

- Split *s* is always the same random 60/20/20 partition (seed `1000 + s`). Run seeds are `100 * s + trial`, so a method and its Mixup version always see the same split and initial seed. The significance tests are paired over these runs.
- The model code was checked against the authors' `GraphConv`/`Net`. The outputs match to about 1e-7 for both the plain and the Mixup forward pass (notebook 01, Section 5).
- Results are selected by validation accuracy, and the test score at the best validation epoch is reported.

## Licence and credit

- The code in this repository is released under the MIT licence (see `LICENSE`).
- `notebooks/00_original_code_run.ipynb` runs, and notebook 01 ports, code from the authors' repository, which is © its authors under the MIT licence.
- Dataset sources: Planetoid (Yang et al., 2016), Amazon Computers/Photo (Shchur et al., 2018), OpenAlex (Priem et al., 2022) and arXiv.

If you use the arXiv graph, please cite the original paper and this repository.
