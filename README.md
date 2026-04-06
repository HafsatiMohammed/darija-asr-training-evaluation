# Darija ASR Training & Evaluation (Whisper)

This project builds and evaluates Moroccan Darija ASR models under compute constraints using Whisper.

It includes:
- dataset preparation + filtering + text normalization,
- local fine-tuning for Whisper models,
- detailed evaluation with global and bucketed metrics.

---

## 1) Project Structure

- `data_preparation_with_text_normalization.py`  
  Downloads/merges source datasets, normalizes Arabic text, applies usability filtering, and exports a Whisper-ready dataset.

- `train_whisper.py`  
  Fine-tunes a Whisper checkpoint (commonly `openai/whisper-small` or `openai/whisper-base`) on the prepared dataset.

- `evaluation_whisper.py`  
  Runs evaluation and exports:
  - global metrics (WER/CER/exact match),
  - per-sample details (JSONL + CSV),
  - grouped bucket summaries,
  - listening sets.

- `requirements.txt`  
  Minimal dependencies for training/evaluation.

---

## 2) Environment Setup

```bash
cd /home/mohammed/Documents/darija-asr-training-evaluation
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
My exact conda env is given 

### Hugging Face token (required)

> **Important:** Export your HF token before data prep / model access.

```bash
export HF_TOKEN="your_huggingface_token_here"
```

(Optional, but useful when pulling private assets/checkpoints):

```bash
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

---

## 3) Data Preparation

```bash
python data_preparation_with_text_normalization.py
```

By default, outputs are written under:

- `./darija_merged_analysis/`
- Whisper-ready dataset path used by training/eval:
  `./darija_merged_analysis/whisper_ready_dataset_16k_paths_success_only`

---

## 4) Train a Local Model

Edit constants in `train_whisper.py` as needed:
- `MODEL_NAME` (e.g., `openai/whisper-small` or `openai/whisper-base`)
- `OUTPUT_DIR` (e.g., `./whisper-small-darija`)
- batch size, max steps, etc.

Then run:

```bash
python train_whisper.py
```

Saved artifacts include:
- fine-tuned model directory (`OUTPUT_DIR`),
- `train_result.json`,
- `final_metrics.json`.

---

## 5) Evaluate Models

`evaluation_whisper.py` is environment-variable driven.

### Evaluate locally tuned checkpoint (example: small)

```bash
CHECKPOINT_PATH=./whisper-small-darija \
DATASET_PATH=./darija_merged_analysis/whisper_ready_dataset_16k_paths_success_only \
SPLIT=test \
OUTPUT_DIR=./eval_whisper-small-local \
python evaluation_whisper.py
```

### Evaluate locally tuned checkpoint (example: base)

```bash
CHECKPOINT_PATH=./whisper-base-darija \
DATASET_PATH=./darija_merged_analysis/whisper_ready_dataset_16k_paths_success_only \
SPLIT=test \
OUTPUT_DIR=./eval_whisper-base-local \
python evaluation_whisper.py
```

### Evaluate a public HF Whisper model

You can evaluate any HF Whisper checkpoint, for example:

```bash
CHECKPOINT_PATH=ychafiqui/whisper-small-darija \
DATASET_PATH=./darija_merged_analysis/whisper_ready_dataset_16k_paths_success_only \
SPLIT=test \
OUTPUT_DIR=./eval_hf_ychafiqui_whisper-small-darija \
python evaluation_whisper.py
```

### Evaluation outputs

For each run, the script writes (inside `OUTPUT_DIR`):
- `test_metrics.json`
- `test_predictions_detailed.jsonl`
- `test_predictions_detailed.csv`
- `test_group_summaries.json`
- `test_listening_sets.json`
- `test_predictions_preview.json`

---

## 6) Main Results (same curated test protocol)

### Global comparison

| Model | WER (%) | CER (%) | Exact match (%) |
|---|---:|---:|---:|
| Local Whisper Base | 53.15 | 25.75 | 1.70 |
| Local Whisper Small | **40.52** | **19.37** | **6.18** |
| `ychafiqui/whisper-small-darija` | 49.49 | 23.41 | 0.27 |

### Main global conclusion

The **locally fine-tuned Whisper Small** checkpoint is the best model in this study.

Compared with local Whisper Base, local Small improves by:
- **12.63 WER points**,
- **6.38 CER points**.

Compared with `ychafiqui/whisper-small-darija`, local Small improves by:
- **8.97 WER points**,
- **4.04 CER points**.

---

## 7) Bucketed Findings (summary)

- **Duration buckets:** Small is better across major duration buckets; 8–15s bucket is too small to trust.
- **Speaker/overlap buckets:** filtering appears effective; heavy overlap remains hardest but has very small sample count.
- **Turn-rate buckets:** Small is consistently better; low-turn-rate bucket is too small.
- **Speech-rate buckets:** slow speech is hardest (likely long-context/alignment/segmentation effects, not just articulation speed).
- **Text-length buckets:** very short transcripts are hardest (often insertion-heavy or mismatch-prone. I think it is more ground truth transcription was not good).
- **Source datasets:** speech-to-text source appears easier than TTS-clean source in this setup.

---

## 8) Notes / Caveats

- These results show superiority of local Small **under the same preprocessing/filtering/evaluation protocol on this curated benchmark**.
- This does **not** imply universal superiority on all Darija ASR tasks/domains.
- Small buckets with very low sample counts should not be over-interpreted.

---

## 9) Quick Repro Checklist

1. `export HF_TOKEN=...`
2. `pip install -r requirements.txt`
3. `python data_preparation_with_text_normalization.py`
4. Train (`python train_whisper.py`) or use an existing local checkpoint
5. Evaluate local small/base and HF models with `evaluation_whisper.py`
6. Compare `*_metrics.json` and `*_group_summaries.json`

## Contact
Mohammed HAFSATI
email: hafsati.mohammed@gmail.com

