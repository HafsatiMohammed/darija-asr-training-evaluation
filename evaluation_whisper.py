import os
import re
import json
import csv
import math
import statistics
import warnings

import numpy as np
import librosa
import torch
import evaluate
from tqdm.auto import tqdm

from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
)
from transformers.utils import logging as hf_logging



# CONFIG

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./whisper-small-darija") # or base or any model from HF sapce

DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "./darija_merged_analysis/whisper_ready_dataset_16k_paths_success_only",
)
SPLIT = os.getenv("SPLIT", "test")

LANGUAGE = os.getenv("LANGUAGE", "arabic")
TASK = os.getenv("TASK", "transcribe")
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))

AUDIO_PATH_COL = os.getenv("AUDIO_PATH_COL", "audio_path_16k")
TEXT_COL = os.getenv("TEXT_COL", "text")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "225"))
LISTENING_LIMIT = int(os.getenv("LISTENING_LIMIT", "50"))

RUN_NAME = os.getenv(
    "RUN_NAME",
    CHECKPOINT_PATH.rstrip("/").split("/")[-1] or "run",
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    f"./eval_{CHECKPOINT_PATH.replace('/', '__')}"
)

# Skip raw or binary-heavy columns in detailed exports
EXCLUDED_DETAIL_COLUMNS = {
    "audio",
    "audio_array",
    "array",
    "bytes",
}



# WARNING TO IGNORE

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings(
    "ignore",
    message=r".*Both `max_new_tokens`.*and `max_length`.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*A custom logits processor of type <class 'transformers\.generation\.logits_process\.SuppressTokensLogitsProcessor'>.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*A custom logits processor of type <class 'transformers\.generation\.logits_process\.SuppressTokensAtBeginLogitsProcessor'>.*",
)

hf_logging.set_verbosity_error()



# TEXT NORMALIZATION (not needed if text already normalized)

def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text



# JSON / CSV STUFFS

def to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_jsonable(value):
    value = to_python_scalar(value)

    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        clean = {}
        for k, v in value.items():
            key = str(k)

            # Special handling for HF audio-style dicts
            if key == "array":
                continue
            if key == "bytes":
                b = v
                if isinstance(b, (bytes, bytearray)):
                    clean["bytes_length"] = len(b)
                else:
                    clean["bytes"] = make_jsonable(b)
                continue

            clean[key] = make_jsonable(v)
        return clean

    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return {
                "__type__": "bytes",
                "length": len(value),
            }

    return value


def csv_safe(value):
    value = make_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value



# TYPING CONVERSION

def first_present(row, keys):
    for key in keys:
        if key in row and row[key] is not None and row[key] != "":
            return row[key]
    return None


def to_float(value):
    value = to_python_scalar(value)
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def to_int(value):
    value = to_python_scalar(value)
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def to_bool_or_none(value):
    value = to_python_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return None


def bool_label(value):
    b = to_bool_or_none(value)
    if b is None:
        return "unknown"
    return "yes" if b else "no"



# BUCKETING LOOK AT REPORT TO UNDERSTAND

def bucket_duration(seconds):
    x = to_float(seconds)
    if x is None:
        return "unknown"
    if x < 3:
        return "short_<3s"
    if x < 8:
        return "medium_3_8s"
    if x < 15:
        return "long_8_15s"
    return "very_long_>=15s"


def bucket_num_speakers(num_speakers, multiple_speakers=None):
    n = to_int(num_speakers)
    if n is not None:
        if n <= 1:
            return "single_speaker"
        if n == 2:
            return "two_speakers"
        return "multi_speaker_3plus"
    ms = to_bool_or_none(multiple_speakers)
    if ms is None:
        return "unknown"
    return "multi_speaker" if ms else "single_speaker"


def bucket_overlap(overlap_ratio=None, overlap_speech=None):
    ratio = to_float(overlap_ratio)
    if ratio is not None:
        if ratio <= 0.0:
            return "no_overlap"
        if ratio < 0.05:
            return "light_overlap"
        return "heavy_overlap"
    ov = to_bool_or_none(overlap_speech)
    if ov is None:
        return "unknown"
    return "overlap_present" if ov else "no_overlap"


def bucket_turn_rate(turns_per_minute):
    x = to_float(turns_per_minute)
    if x is None:
        return "unknown"
    if x < 2:
        return "low_turn_rate"
    if x < 6:
        return "medium_turn_rate"
    return "high_turn_rate"


def bucket_speech_rate(tokens_per_sec=None, chars_per_sec=None):
    tps = to_float(tokens_per_sec)
    cps = to_float(chars_per_sec)

    if tps is not None:
        if tps < 1.5:
            return "slow_speech"
        if tps < 3.0:
            return "medium_speech"
        if tps < 5.0:
            return "fast_speech"
        return "very_fast_speech"

    if cps is not None:
        if cps < 8:
            return "slow_speech"
        if cps < 15:
            return "medium_speech"
        if cps < 25:
            return "fast_speech"
        return "very_fast_speech"

    return "unknown"


def bucket_text_length(word_len_est=None, num_tokens_proxy=None, char_len=None):
    words = to_float(word_len_est)
    if words is not None:
        if words < 4:
            return "very_short_text"
        if words < 10:
            return "short_text"
        if words < 20:
            return "medium_text"
        return "long_text"

    tokens = to_float(num_tokens_proxy)
    if tokens is not None:
        if tokens < 4:
            return "very_short_text"
        if tokens < 10:
            return "short_text"
        if tokens < 20:
            return "medium_text"
        return "long_text"

    chars = to_float(char_len)
    if chars is not None:
        if chars < 15:
            return "very_short_text"
        if chars < 40:
            return "short_text"
        if chars < 80:
            return "medium_text"
        return "long_text"

    return "unknown"


def bucket_quality(quality_bin=None, asr_usability_score=None, usability_score_custom=None):
    if quality_bin is not None and str(quality_bin).strip() != "":
        return f"quality_{str(quality_bin)}"

    score = first_present(
        {
            "asr_usability_score": asr_usability_score,
            "usability_score_custom": usability_score_custom,
        },
        ["asr_usability_score", "usability_score_custom"],
    )
    score = to_float(score)
    if score is None:
        return "unknown"
    if score < 0.4:
        return "quality_low"
    if score < 0.7:
        return "quality_medium"
    return "quality_high"



# EDIT DISTANCE / PER-SAMPLE METRICS

def edit_distance(seq_a, seq_b):
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        current = [i]
        for j, b in enumerate(seq_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if a == b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_pair_error_stats(reference, prediction):
    ref = normalize_text(reference)
    pred = normalize_text(prediction)

    ref_words = ref.split() if ref else []
    pred_words = pred.split() if pred else []

    ref_chars = list(ref)
    pred_chars = list(pred)

    word_edits = edit_distance(ref_words, pred_words)
    char_edits = edit_distance(ref_chars, pred_chars)

    word_ref_len = len(ref_words)
    char_ref_len = len(ref_chars)

    if word_ref_len > 0:
        sample_wer = word_edits / word_ref_len
    else:
        sample_wer = 0.0 if len(pred_words) == 0 else 1.0

    if char_ref_len > 0:
        sample_cer = char_edits / char_ref_len
    else:
        sample_cer = 0.0 if len(pred_chars) == 0 else 1.0

    return {
        "word_edits": word_edits,
        "word_ref_len": word_ref_len,
        "char_edits": char_edits,
        "char_ref_len": char_ref_len,
        "sample_wer": sample_wer,
        "sample_cer": sample_cer,
        "exact_match": ref == pred,
    }



# RECORD ENRICHMENT

def build_analysis_fields(row):
    duration = first_present(row, ["duration_sec", "audio_num_seconds_16k"])
    num_speakers = first_present(row, ["num_speakers"])
    multiple_speakers = first_present(row, ["multiple_speakers"])
    overlap_ratio = first_present(row, ["overlap_ratio"])
    overlap_speech = first_present(row, ["overlap_speech"])
    turns_per_minute = first_present(row, ["turns_per_minute"])
    tokens_per_sec = first_present(row, ["tokens_per_sec"])
    chars_per_sec = first_present(row, ["chars_per_sec"])
    word_len_est = first_present(row, ["word_len_est"])
    num_tokens_proxy = first_present(row, ["num_tokens_proxy", "num_tokens"])
    char_len = first_present(row, ["char_len"])
    quality_bin = first_present(row, ["quality_bin"])
    asr_usability_score = first_present(row, ["asr_usability_score"])
    usability_score_custom = first_present(row, ["usability_score_custom"])

    analysis = {
        "duration_bucket": bucket_duration(duration),
        "speaker_bucket": bucket_num_speakers(num_speakers, multiple_speakers),
        "overlap_bucket": bucket_overlap(overlap_ratio, overlap_speech),
        "turn_rate_bucket": bucket_turn_rate(turns_per_minute),
        "speech_rate_bucket": bucket_speech_rate(tokens_per_sec, chars_per_sec),
        "text_length_bucket": bucket_text_length(word_len_est, num_tokens_proxy, char_len),
        "quality_bucket": bucket_quality(quality_bin, asr_usability_score, usability_score_custom),
        "source_dataset_bucket": str(first_present(row, ["source_dataset"]) or "unknown"),
        "source_split_bucket": str(first_present(row, ["source_split_original"]) or "unknown"),
        "multiple_speakers_flag": bool_label(first_present(row, ["multiple_speakers"])),
        "overlap_speech_flag": bool_label(first_present(row, ["overlap_speech"])),
        "token_count_outlier_flag": bool_label(first_present(row, ["token_count_outlier"])),
        "remove_short_audio_long_text_flag": bool_label(first_present(row, ["remove_short_audio_long_text"])),
        "remove_token_outlier_flag": bool_label(first_present(row, ["remove_token_outlier"])),
        "diarization_empty_output_flag": bool_label(first_present(row, ["diarization_empty_output"])),
        "processing_error_flag": bool_label(first_present(row, ["processing_error"])),
        "trainable_flag": bool_label(first_present(row, ["trainable"])),
    }
    return analysis



# GROUP SUMMARY

GROUP_KEYS = [
    "duration_bucket",
    "speaker_bucket",
    "overlap_bucket",
    "turn_rate_bucket",
    "speech_rate_bucket",
    "text_length_bucket",
    "quality_bucket",
    "source_dataset_bucket",
    "source_split_bucket",
    "multiple_speakers_flag",
    "overlap_speech_flag",
    "token_count_outlier_flag",
    "remove_short_audio_long_text_flag",
    "remove_token_outlier_flag",
    "diarization_empty_output_flag",
    "processing_error_flag",
    "trainable_flag",
]


def summarize_group(records):
    n = len(records)
    word_edits = sum(r["word_edits"] for r in records)
    word_ref_len = sum(r["word_ref_len"] for r in records)
    char_edits = sum(r["char_edits"] for r in records)
    char_ref_len = sum(r["char_ref_len"] for r in records)

    corpus_wer = 100.0 * word_edits / word_ref_len if word_ref_len > 0 else None
    corpus_cer = 100.0 * char_edits / char_ref_len if char_ref_len > 0 else None

    sample_wers = [r["sample_wer"] for r in records]
    sample_cers = [r["sample_cer"] for r in records]

    return {
        "n_samples": n,
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "mean_sample_wer": 100.0 * float(sum(sample_wers) / len(sample_wers)) if sample_wers else None,
        "median_sample_wer": 100.0 * float(statistics.median(sample_wers)) if sample_wers else None,
        "mean_sample_cer": 100.0 * float(sum(sample_cers) / len(sample_cers)) if sample_cers else None,
        "median_sample_cer": 100.0 * float(statistics.median(sample_cers)) if sample_cers else None,
        "exact_match_rate": 100.0 * sum(1 for r in records if r["exact_match"]) / n if n > 0 else None,
    }


def build_group_summaries(records):
    all_summaries = {}
    total_n = len(records)

    for group_key in GROUP_KEYS:
        groups = {}
        for row in records:
            value = row.get(group_key, "unknown")
            groups.setdefault(str(value), []).append(row)

        summary_rows = []
        for group_value, group_records in groups.items():
            row = {"group_value": group_value}
            row.update(summarize_group(group_records))
            row["share_of_dataset_pct"] = 100.0 * len(group_records) / total_n if total_n > 0 else None
            summary_rows.append(row)

        summary_rows.sort(
            key=lambda x: (
                -x["n_samples"],
                str(x["group_value"]),
            )
        )
        all_summaries[group_key] = summary_rows

    return all_summaries



# LISTENING SETS

LISTENING_FIELDS = [
    "sample_idx",
    "audio_path",
    "reference",
    "prediction",
    "sample_wer_pct",
    "sample_cer_pct",
    "exact_match",
    "duration_bucket",
    "speaker_bucket",
    "overlap_bucket",
    "turn_rate_bucket",
    "speech_rate_bucket",
    "text_length_bucket",
    "quality_bucket",
    "source_dataset_bucket",
    "source_split_bucket",
]


def compact_record_for_listening(row):
    return {k: make_jsonable(row.get(k)) for k in LISTENING_FIELDS}


def top_k(records, key_fn, limit=50, reverse=True, filter_fn=None):
    items = records
    if filter_fn is not None:
        items = [r for r in items if filter_fn(r)]
    items = sorted(items, key=key_fn, reverse=reverse)
    return [compact_record_for_listening(r) for r in items[:limit]]


def build_listening_sets(records, limit=50):
    listening_sets = {
        "hardest_by_wer": top_k(records, key_fn=lambda r: r["sample_wer"], limit=limit, reverse=True),
        "hardest_by_cer": top_k(records, key_fn=lambda r: r["sample_cer"], limit=limit, reverse=True),
        "easiest_non_exact_by_wer": top_k(
            records,
            key_fn=lambda r: r["sample_wer"],
            limit=limit,
            reverse=False,
            filter_fn=lambda r: not r["exact_match"],
        ),
        "exact_matches": top_k(
            records,
            key_fn=lambda r: r["sample_wer"],
            limit=limit,
            reverse=False,
            filter_fn=lambda r: r["exact_match"],
        ),
        "hard_long_audio": top_k(
            records,
            key_fn=lambda r: r["sample_wer"],
            limit=limit,
            reverse=True,
            filter_fn=lambda r: r["duration_bucket"] in {"long_8_15s", "very_long_>=15s"},
        ),
        "hard_overlap": top_k(
            records,
            key_fn=lambda r: r["sample_wer"],
            limit=limit,
            reverse=True,
            filter_fn=lambda r: r["overlap_bucket"] in {"light_overlap", "heavy_overlap", "overlap_present"},
        ),
        "hard_multi_speaker": top_k(
            records,
            key_fn=lambda r: r["sample_wer"],
            limit=limit,
            reverse=True,
            filter_fn=lambda r: r["speaker_bucket"] in {"two_speakers", "multi_speaker", "multi_speaker_3plus"},
        ),
        "hard_fast_speech": top_k(
            records,
            key_fn=lambda r: r["sample_wer"],
            limit=limit,
            reverse=True,
            filter_fn=lambda r: r["speech_rate_bucket"] in {"fast_speech", "very_fast_speech"},
        ),
        "hard_low_quality": top_k(
            records,
            key_fn=lambda r: r["sample_wer"],
            limit=limit,
            reverse=True,
            filter_fn=lambda r: r["quality_bucket"] == "quality_low",
        ),
        "possible_orthographic_or_tokenization_issues": top_k(
            records,
            key_fn=lambda r: (r["sample_wer"] - r["sample_cer"]),
            limit=limit,
            reverse=True,
            filter_fn=lambda r: r["sample_wer"] >= 0.4 and r["sample_cer"] <= 0.25,
        ),
    }
    return listening_sets



# LOAD DATASET

dataset = load_from_disk(DATASET_PATH)

if SPLIT not in dataset:
    raise ValueError(f"Split '{SPLIT}' not found. Available splits: {list(dataset.keys())}")

dataset = dataset[SPLIT]

required_columns = [AUDIO_PATH_COL, TEXT_COL]
missing = [c for c in required_columns if c not in dataset.column_names]
if missing:
    raise ValueError(f"Dataset split '{SPLIT}' is missing columns: {missing}")

print(f"Loaded split: {SPLIT}")
print(f"Number of samples: {len(dataset)}")
print(f"Checkpoint: {CHECKPOINT_PATH}")



# LOAD MODEL + PROCESSOR

processor = WhisperProcessor.from_pretrained(CHECKPOINT_PATH)
model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)



# CLEAN GENERATION CONFIG

gen_cfg = GenerationConfig.from_model_config(model.config)

if hasattr(model, "generation_config") and model.generation_config is not None:
    for k, v in model.generation_config.to_dict().items():
        setattr(gen_cfg, k, v)

gen_cfg.language = LANGUAGE
gen_cfg.task = TASK
gen_cfg.forced_decoder_ids = None
gen_cfg.max_new_tokens = MAX_NEW_TOKENS
gen_cfg.max_length = None
gen_cfg.min_length = 0







# METRICS

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")



# INFERENCE + PER-SAMPLE ANALYSIS

predictions = []
references = []
detailed_records = []

num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

for start in tqdm(
    range(0, len(dataset), BATCH_SIZE),
    total=num_batches,
    desc=f"Evaluating {SPLIT}",
    unit="batch",
):
    batch = dataset[start:start + BATCH_SIZE]
    batch_size = len(batch[AUDIO_PATH_COL])

    audio_arrays = []
    valid_indices = []

    for i, audio_path in enumerate(batch[AUDIO_PATH_COL]):
        try:
            wav, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            wav = np.asarray(wav, dtype=np.float32)
            audio_arrays.append(wav)
            valid_indices.append(i)
        except Exception as e:
            row = {col: batch[col][i] for col in batch.keys()}
            reference = normalize_text(row.get(TEXT_COL, ""))
            prediction = ""

            error_stats = compute_pair_error_stats(reference, prediction)
            analysis_fields = build_analysis_fields(row)

            record = {
                "run_name": RUN_NAME,
                "checkpoint_path": CHECKPOINT_PATH,
                "split": SPLIT,
                "sample_idx": start + i,
                "audio_path": row.get(AUDIO_PATH_COL),
                "reference": reference,
                "prediction": prediction,
                "sample_wer_pct": 100.0 * error_stats["sample_wer"],
                "sample_cer_pct": 100.0 * error_stats["sample_cer"],
                "audio_load_error": str(e),
                **error_stats,
                **analysis_fields,
            }

            for col in batch.keys():
                if col in {TEXT_COL, AUDIO_PATH_COL}:
                    continue

                if col == "audio":
                    audio_value = row[col]
                    if isinstance(audio_value, dict):
                        record["audio_path_original"] = make_jsonable(audio_value.get("path"))
                        record["audio_sampling_rate_original"] = make_jsonable(audio_value.get("sampling_rate"))
                    continue

                if col in EXCLUDED_DETAIL_COLUMNS:
                    continue

                record[col] = make_jsonable(row[col])

            predictions.append(prediction)
            references.append(reference)
            detailed_records.append(record)

    if not audio_arrays:
        continue

    inputs = processor(
        audio_arrays,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_features = inputs["input_features"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        pred_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

    pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    for j, i in enumerate(valid_indices):
        row = {col: batch[col][i] for col in batch.keys()}
        reference = normalize_text(row.get(TEXT_COL, ""))
        prediction = normalize_text(pred_texts[j])

        predictions.append(prediction)
        references.append(reference)

        error_stats = compute_pair_error_stats(reference, prediction)
        analysis_fields = build_analysis_fields(row)

        record = {
            "run_name": RUN_NAME,
            "checkpoint_path": CHECKPOINT_PATH,
            "split": SPLIT,
            "sample_idx": start + i,
            "audio_path": row.get(AUDIO_PATH_COL),
            "reference": reference,
            "prediction": prediction,
            "sample_wer_pct": 100.0 * error_stats["sample_wer"],
            "sample_cer_pct": 100.0 * error_stats["sample_cer"],
            **error_stats,
            **analysis_fields,
        }

        for col in batch.keys():
            if col in {TEXT_COL, AUDIO_PATH_COL}:
                continue

            if col == "audio":
                audio_value = row[col]
                if isinstance(audio_value, dict):
                    record["audio_path_original"] = make_jsonable(audio_value.get("path"))
                    record["audio_sampling_rate_original"] = make_jsonable(audio_value.get("sampling_rate"))
                continue

            if col in EXCLUDED_DETAIL_COLUMNS:
                continue

            record[col] = make_jsonable(row[col])

        detailed_records.append(record)








# FINAL GLOBAL METRICS

wer = 100.0 * wer_metric.compute(predictions=predictions, references=references)
cer = 100.0 * cer_metric.compute(predictions=predictions, references=references)

group_summaries = build_group_summaries(detailed_records)
listening_sets = build_listening_sets(detailed_records, limit=LISTENING_LIMIT)

corpus_word_edits = sum(r["word_edits"] for r in detailed_records)
corpus_word_ref_len = sum(r["word_ref_len"] for r in detailed_records)
corpus_char_edits = sum(r["char_edits"] for r in detailed_records)
corpus_char_ref_len = sum(r["char_ref_len"] for r in detailed_records)

metrics = {
    "run_name": RUN_NAME,
    "checkpoint_path": CHECKPOINT_PATH,
    "split": SPLIT,
    "n_samples": len(dataset),
    "n_records_written": len(detailed_records),
    "global_wer_evaluate_pct": wer,
    "global_cer_evaluate_pct": cer,
    "global_wer_from_sample_edits_pct": (
        100.0 * corpus_word_edits / corpus_word_ref_len if corpus_word_ref_len > 0 else None
    ),
    "global_cer_from_sample_edits_pct": (
        100.0 * corpus_char_edits / corpus_char_ref_len if corpus_char_ref_len > 0 else None
    ),
    "exact_match_rate_pct": (
        100.0 * sum(1 for r in detailed_records if r["exact_match"]) / len(detailed_records)
        if detailed_records
        else None
    ),
    "audio_load_error_count": sum(1 for r in detailed_records if r.get("audio_load_error")),
}






# SAVE OUTPUTS

os.makedirs(OUTPUT_DIR, exist_ok=True)

metrics_path = os.path.join(OUTPUT_DIR, f"{SPLIT}_metrics.json")
details_jsonl_path = os.path.join(OUTPUT_DIR, f"{SPLIT}_predictions_detailed.jsonl")
details_csv_path = os.path.join(OUTPUT_DIR, f"{SPLIT}_predictions_detailed.csv")
group_summary_path = os.path.join(OUTPUT_DIR, f"{SPLIT}_group_summaries.json")
listening_sets_path = os.path.join(OUTPUT_DIR, f"{SPLIT}_listening_sets.json")
preview_path = os.path.join(OUTPUT_DIR, f"{SPLIT}_predictions_preview.json")

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(make_jsonable(metrics), f, ensure_ascii=False, indent=2)

with open(details_jsonl_path, "w", encoding="utf-8") as f:
    for row in detailed_records:
        f.write(json.dumps(make_jsonable(row), ensure_ascii=False) + "\n")

csv_fieldnames = sorted({k for row in detailed_records for k in row.keys()})
with open(details_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
    writer.writeheader()
    for row in detailed_records:
        writer.writerow({k: csv_safe(row.get(k)) for k in csv_fieldnames})

with open(group_summary_path, "w", encoding="utf-8") as f:
    json.dump(make_jsonable(group_summaries), f, ensure_ascii=False, indent=2)

with open(listening_sets_path, "w", encoding="utf-8") as f:
    json.dump(make_jsonable(listening_sets), f, ensure_ascii=False, indent=2)

preview = []
for i in range(min(100, len(detailed_records))):
    preview.append(
        {
            "sample_idx": detailed_records[i]["sample_idx"],
            "reference": detailed_records[i]["reference"],
            "prediction": detailed_records[i]["prediction"],
            "sample_wer_pct": detailed_records[i]["sample_wer_pct"],
            "sample_cer_pct": detailed_records[i]["sample_cer_pct"],
            "duration_bucket": detailed_records[i]["duration_bucket"],
            "speaker_bucket": detailed_records[i]["speaker_bucket"],
            "overlap_bucket": detailed_records[i]["overlap_bucket"],
            "quality_bucket": detailed_records[i]["quality_bucket"],
            "source_dataset_bucket": detailed_records[i]["source_dataset_bucket"],
        }
    )

with open(preview_path, "w", encoding="utf-8") as f:
    json.dump(make_jsonable(preview), f, ensure_ascii=False, indent=2)







# PRINT REPORT

print("\nFinal metrics:")
print(json.dumps(make_jsonable(metrics), indent=2, ensure_ascii=False))

print("\nTop grouped summaries (first few groups):")
for group_key in GROUP_KEYS[:7]:
    rows = group_summaries.get(group_key, [])[:5]
    print(f"\n[{group_key}]")
    print(json.dumps(make_jsonable(rows), indent=2, ensure_ascii=False))

print(f"\nSaved metrics to: {metrics_path}")
print(f"Saved detailed JSONL to: {details_jsonl_path}")
print(f"Saved detailed CSV to: {details_csv_path}")
print(f"Saved group summaries to: {group_summary_path}")
print(f"Saved listening sets to: {listening_sets_path}")
print(f"Saved preview to: {preview_path}")