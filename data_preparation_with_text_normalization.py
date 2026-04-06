import os
import re
import io
import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import soundfile as sf

from tqdm.auto import tqdm
from scipy.stats import spearmanr, mannwhitneyu

from datasets import (
    load_dataset,
    concatenate_datasets,
    DatasetDict,
    Audio,
    Value,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG

HF_TOKEN = os.getenv("HF_TOKEN", "")


REPO_TTS = "EtMmohammedHafsati/darija_tts_clean_metadata_full"
REPO_STT = "EtMmohammedHafsati/darija_speech_to_text_metadata_full"

OUT_DIR = Path("./darija_merged_analysis")
PLOTS_DIR = OUT_DIR / "plots"
AUDIO_16K_DIR = OUT_DIR / "audio_16k"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_16K_DIR.mkdir(parents=True, exist_ok=True)

TURN_RATE_REF = 15.0
RANDOM_SEED = 42
TARGET_SR = 16000
EXPORT_16K_AUDIO = True
USABILITY_THRESHOLD = 0.85

PREFERRED_SPLIT_ORDER = ["train", "validation", "test"]

DEDUP_PRIORITY = {
    "test": 0,
    "validation": 1,
    "train": 2,
}


# TEXT NORMALIZATION CONFIG

REMOVE_PUNCTUATION = True
REMOVE_DIACRITICS = True
REMOVE_TATWEEL = True
NORMALIZE_ALEF = True
NORMALIZE_HAMZA_SEAT = True
NORMALIZE_ALEF_MAQSURA = True
NORMALIZE_TA_MARBUTA = False
NORMALIZE_DIGITS_TO_ASCII = False
LOWERCASE_LATIN = True


# EXPECTED SCHEMA / known with previous work

NUMERIC_COLS = [
    "duration_sec",
    "num_tokens",
    "median_tokens_reference",
    "num_speakers",
    "speaker_turns",
    "turns_per_minute",
    "dominant_speaker_ratio",
    "second_speaker_ratio",
    "non_dominant_speech_ratio",
    "speaker_balance_score",
    "speaker_entropy",
    "overlap_duration_sec",
    "overlap_ratio",
    "num_overlap_regions",
    "mean_overlap_region_sec",
    "max_overlap_region_sec",
    "max_concurrent_speakers",
    "asr_usability_score",
]

BOOL_COLS = [
    "token_count_outlier",
    "remove_short_audio_long_text",
    "remove_token_outlier",
    "multiple_speakers",
    "overlap_speech",
    "diarization_empty_output",
    "asr_usable_single_speaker",
]

STRING_COLS = [
    "text",
    "source_split_original",
    "metadata_generated_by",
    "processing_error",
    "source_dataset",
]

BASE_REQUIRED_COLS = ["audio"] + NUMERIC_COLS + BOOL_COLS + STRING_COLS


# ARABIC / DARIJA NORMALIZATION

ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

ASCII_PUNCT_RE = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]""")
ARABIC_PUNCT_EXTRA_RE = re.compile(r"[«»،؛؟…ـ]")
MULTISPACE_RE = re.compile(r"\s+")

ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EASTERN_ARABIC_INDIC_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

CHAR_REPLACEMENTS = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ؤ": "و",
    "ئ": "ي",
    "ى": "ي",
    "ک": "ك",
    "ی": "ي",
    "ے": "ي",
    "ہ": "ه",
    "ۃ": "ه",
    "ﻻ": "لا",
    "ﻷ": "لا",
    "ﻹ": "لا",
    "ﻵ": "لا",
    "ـ": "",
    "،": ",",
    "؛": ";",
    "؟": "?",
    "«": '"',
    "»": '"',
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "’": "'",
    "‘": "'",
    "‚": "'",
    "‛": "'",
    "`": "'",
    "´": "'",
    "…": "...",
}

def strip_unicode_control_chars(text: str) -> str:
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf"}
    )

def normalize_arabic_text(text):
    text = "" if text is None else str(text)

    text = strip_unicode_control_chars(text)
    text = text.strip()

    if LOWERCASE_LATIN:
        text = text.lower()

    # normalize compatibility forms
    text = unicodedata.normalize("NFKC", text)

    # char-level replacements
    for src, tgt in CHAR_REPLACEMENTS.items():
        text = text.replace(src, tgt)

    if REMOVE_TATWEEL:
        text = text.replace("ـ", "")

    if REMOVE_DIACRITICS:
        text = ARABIC_DIACRITICS_RE.sub("", text)

    if NORMALIZE_ALEF and not NORMALIZE_HAMZA_SEAT:
        text = (
            text.replace("أ", "ا")
                .replace("إ", "ا")
                .replace("آ", "ا")
                .replace("ٱ", "ا")
        )

    if NORMALIZE_HAMZA_SEAT:
        text = text.replace("ؤ", "و").replace("ئ", "ي")

    if NORMALIZE_ALEF_MAQSURA:
        text = text.replace("ى", "ي")

    if NORMALIZE_TA_MARBUTA:
        text = text.replace("ة", "ه")

    if NORMALIZE_DIGITS_TO_ASCII:
        text = text.translate(ARABIC_INDIC_DIGITS)
        text = text.translate(EASTERN_ARABIC_INDIC_DIGITS)

    # standardize whitespace first
    text = MULTISPACE_RE.sub(" ", text).strip()

    if REMOVE_PUNCTUATION:
        text = ARABIC_PUNCT_EXTRA_RE.sub(" ", text)
        text = ASCII_PUNCT_RE.sub(" ", text)

    # final whitespace cleanup
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text


def load_hf_dataset_dict(repo_id: str):
    try:
        return load_dataset(repo_id, token=HF_TOKEN)
    except TypeError:
        return load_dataset(repo_id, use_auth_token=HF_TOKEN)


def ordered_splits(split_names):
    split_names = list(split_names)
    preferred = [s for s in PREFERRED_SPLIT_ORDER if s in split_names]
    remaining = sorted([s for s in split_names if s not in PREFERRED_SPLIT_ORDER])
    return preferred + remaining


def detect_audio_col(ds):
    for col, feat in ds.features.items():
        if isinstance(feat, Audio):
            return col
    for cand in ["audio", "wav", "speech", "sound"]:
        if cand in ds.column_names:
            return cand
    raise ValueError(f"Could not detect audio column. Columns: {ds.column_names}")


def detect_text_col(ds):
    preferred = ["text", "sentence", "transcription", "transcript", "normalized_text", "utterance"]
    for cand in preferred:
        if cand in ds.column_names:
            return cand

    string_cols = []
    for col, feat in ds.features.items():
        if isinstance(feat, Value) and feat.dtype == "string":
            string_cols.append(col)

    banned = {
        "source_split_original",
        "metadata_generated_by",
        "processing_error",
        "id",
        "path",
        "file",
        "audio_path",
    }
    string_cols = [c for c in string_cols if c not in banned]

    if not string_cols:
        raise ValueError(f"Could not detect text column. Columns: {ds.column_names}")

    return string_cols[0]


def safe_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def cast_if_exists(ds, columns, dtype):
    for col in columns:
        if col in ds.column_names:
            ds = ds.cast_column(col, Value(dtype))
    return ds


def add_missing_columns(ds, required_columns, numeric_cols, bool_cols, string_cols):
    for col in required_columns:
        if col not in ds.column_names:
            if col in numeric_cols:
                fill = [float("nan")] * len(ds)
            elif col in bool_cols:
                fill = [False] * len(ds)
            elif col in string_cols:
                fill = [""] * len(ds)
            else:
                fill = [None] * len(ds)
            ds = ds.add_column(col, fill)
    return ds


def prepare_source_split(ds, source_name):
    audio_col = detect_audio_col(ds)
    text_col = detect_text_col(ds)

    if audio_col != "audio":
        ds = ds.rename_column(audio_col, "audio")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    ds = ds.cast_column("audio", Audio(decode=False))

    if "source_dataset" in ds.column_names:
        ds = ds.remove_columns(["source_dataset"])
    ds = ds.add_column("source_dataset", [source_name] * len(ds))

    ds = add_missing_columns(
        ds,
        BASE_REQUIRED_COLS,
        NUMERIC_COLS,
        BOOL_COLS,
        STRING_COLS,
    )

    ds = ds.select_columns(BASE_REQUIRED_COLS)

    ds = cast_if_exists(ds, NUMERIC_COLS, "float32")
    ds = cast_if_exists(ds, BOOL_COLS, "bool")
    ds = cast_if_exists(ds, STRING_COLS, "string")

    return ds


def np_nanpercentile_safe(values, q, default=0.0):
    arr = pd.Series(values).dropna().to_numpy()
    if len(arr) == 0:
        return default
    return float(np.nanpercentile(arr, q))


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def add_spearman_test(df, tests, x_col, y_col, name):
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 10:
        return
    rho, p = spearmanr(sub[x_col], sub[y_col])
    tests.append({
        "test_name": name,
        "test_type": "spearman",
        "x": x_col,
        "y": y_col,
        "n": int(len(sub)),
        "statistic": float(rho),
        "p_value": float(p),
    })


def add_mwu_test(df, tests, value_col, group_col, group_a, group_b, name):
    sub = df[[value_col, group_col]].dropna()
    a = sub.loc[sub[group_col] == group_a, value_col]
    b = sub.loc[sub[group_col] == group_b, value_col]
    if len(a) < 10 or len(b) < 10:
        return
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    tests.append({
        "test_name": name,
        "test_type": "mannwhitneyu",
        "value_col": value_col,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "statistic": float(stat),
        "p_value": float(p),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
    })


def to_pylist_string(series, fill=""):
    return series.fillna(fill).astype(str).tolist()


def to_pylist_bool(series):
    return series.fillna(False).map(bool).tolist()


def to_pylist_numeric(series):
    return pd.to_numeric(series, errors="coerce").astype("float32").tolist()


def load_audio_librosa_16k(audio_obj, target_sr=16000):
    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj.get("bytes", None)
        path = audio_obj.get("path", None)

        if audio_bytes is not None:
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            data = np.asarray(data, dtype=np.float32)
            return data, sr, "bytes"

        if path is not None:
            wav, sr = librosa.load(path, sr=target_sr, mono=True)
            wav = np.asarray(wav, dtype=np.float32)
            return wav, sr, path

        raise ValueError("audio object has neither bytes nor path")

    if isinstance(audio_obj, str):
        wav, sr = librosa.load(audio_obj, sr=target_sr, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        return wav, sr, audio_obj

    raise ValueError(f"Unsupported audio object type: {type(audio_obj)}")


def export_split_audio_to_16k(ds, split_name, out_root, target_sr=16000):
    split_out_dir = out_root / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    exported_seconds = []
    export_errors = []
    export_success = []

    total_samples_loaded = 0

    iterator = tqdm(
        range(len(ds)),
        desc=f"Exporting {split_name} to 16k",
        unit="audio",
        dynamic_ncols=True,
    )

    for i in iterator:
        ex = ds[i]
        try:
            wav, sr, resolved_src = load_audio_librosa_16k(
                ex["audio"],
                target_sr=target_sr,
            )
            out_path = split_out_dir / f"{i:07d}.wav"
            sf.write(str(out_path), wav, sr)

            wav_len = len(wav)
            total_samples_loaded += wav_len

            exported_paths.append(str(out_path))
            exported_seconds.append(float(wav_len / sr))
            export_errors.append("")
            export_success.append(True)

        except Exception as e:
            exported_paths.append("")
            exported_seconds.append(np.nan)
            export_errors.append(str(e))
            export_success.append(False)

        if (i + 1) % 200 == 0 or (i + 1) == len(ds):
            success_count = int(np.sum(export_success))
            fail_count = (i + 1) - success_count
            loaded_hours = total_samples_loaded / float(target_sr * 3600.0)
            iterator.set_postfix(
                done=i + 1,
                ok=success_count,
                fail=fail_count,
                hours=f"{loaded_hours:.2f}",
            )

    ds = ds.add_column("audio_path_16k", exported_paths)
    ds = ds.add_column("audio_num_seconds_16k", exported_seconds)
    ds = ds.add_column("audio_export_error", export_errors)
    ds = ds.add_column("audio_export_success", export_success)

    manifest_cols = [c for c in ds.column_names if c != "audio"]
    manifest_df = ds.select_columns(manifest_cols).to_pandas()

    final_hours_loaded = total_samples_loaded / float(target_sr * 3600.0)
    final_success = int(np.sum(export_success))
    final_fail = int(len(ds) - final_success)

    print(
        f"\nFinished split={split_name} | "
        f"n_total={len(ds)} | ok={final_success} | fail={final_fail} | "
        f"hours_loaded_from_waveforms={final_hours_loaded:.4f}"
    )

    return ds, manifest_df, final_hours_loaded, final_success, final_fail


def compute_exported_hours_from_files(split_dir: Path, target_sr=16000):
    wav_paths = sorted(split_dir.glob("*.wav"))
    total_samples = 0
    total_files = 0
    for p in tqdm(wav_paths, desc=f"Recount {split_dir.name}", unit="wav", dynamic_ncols=True):
        info = sf.info(str(p))
        total_samples += int(info.frames)
        total_files += 1
    total_hours = total_samples / float(target_sr * 3600.0)
    return total_files, total_hours


def summarize_state(df_in, state_name):
    out = (
        df_in.groupby("hf_split", dropna=False)
        .agg(
            n_samples=("text_normalized", "size"),
            total_hours=("duration_sec", lambda s: np.nansum(s) / 3600.0),
            median_duration_sec=("duration_sec", "median"),
        )
        .reset_index()
    )
    out["state"] = state_name
    return out


def summarize_state_by_source(df_in, state_name):
    out = (
        df_in.groupby(["hf_split", "source_dataset"], dropna=False)
        .agg(
            n_samples=("text_normalized", "size"),
            total_hours=("duration_sec", lambda s: np.nansum(s) / 3600.0),
            median_duration_sec=("duration_sec", "median"),
        )
        .reset_index()
    )
    out["state"] = state_name
    return out


def plot_state_summary(summary_df, out_dir):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary_df, x="hf_split", y="total_hours", hue="state")
    plt.title("Total hours by split and processing state")
    plt.ylabel("Hours")
    save_plot(out_dir / "09_hours_by_split_and_state.png")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary_df, x="hf_split", y="n_samples", hue="state")
    plt.title("Sample counts by split and processing state")
    plt.ylabel("Number of samples")
    save_plot(out_dir / "10_counts_by_split_and_state.png")


def plot_state_source_summary(summary_df, out_dir):
    plt.figure(figsize=(12, 5))
    sns.barplot(data=summary_df, x="hf_split", y="total_hours", hue="source_dataset")
    plt.title("Total hours by split and source")
    plt.ylabel("Hours")
    save_plot(out_dir / "11_hours_by_split_and_source.png")

    plt.figure(figsize=(12, 5))
    sns.barplot(data=summary_df, x="hf_split", y="n_samples", hue="source_dataset")
    plt.title("Counts by split and source")
    plt.ylabel("Number of samples")
    save_plot(out_dir / "12_counts_by_split_and_source.png")



# LOAD REPOS

print("Loading TTS repo...")
tts_dd = load_hf_dataset_dict(REPO_TTS)

print("Loading STT repo...")
stt_dd = load_hf_dataset_dict(REPO_STT)

all_splits = ordered_splits(set(tts_dd.keys()) | set(stt_dd.keys()))
if not all_splits:
    raise ValueError("No splits found in either dataset repo.")


# MERGE SPLIT BY SPLIT

merged_by_split = {}

for split in all_splits:
    split_parts = []

    if split in tts_dd:
        ds_tts = prepare_source_split(tts_dd[split], "darija_tts_clean_metadata_full")
        split_parts.append(ds_tts)

    if split in stt_dd:
        ds_stt = prepare_source_split(stt_dd[split], "darija_speech_to_text_metadata_full")
        split_parts.append(ds_stt)

    if not split_parts:
        continue

    merged_split = split_parts[0] if len(split_parts) == 1 else concatenate_datasets(split_parts)
    merged_split = merged_split.add_column("hf_split", [split] * len(merged_split))
    merged_split = merged_split.add_column("_merged_index", list(range(len(merged_split))))

    merged_by_split[split] = merged_split

if not merged_by_split:
    raise ValueError("No merged splits were created.")

raw_merged_dict = DatasetDict(merged_by_split)
raw_merged_dict.save_to_disk(str(OUT_DIR / "merged_raw_no_decode"))

print("\nMerged raw dataset by split:")
for split, ds in raw_merged_dict.items():
    print(f"  {split}: {len(ds)} rows")


# BUILD METADATA DATAFRAME

meta_datasets = []
for split, ds in merged_by_split.items():
    meta_cols = [c for c in ds.column_names if c != "audio"]
    meta_datasets.append(ds.select_columns(meta_cols))

combined_meta_ds = meta_datasets[0] if len(meta_datasets) == 1 else concatenate_datasets(meta_datasets)
df = combined_meta_ds.to_pandas()

# preserve raw text
df["text_raw"] = df["text"].fillna("").astype(str)

# normalized text used everywhere downstream
df["text_normalized"] = df["text_raw"].map(normalize_arabic_text)

# make text column the normalized one for downstream compatibility
df["text"] = df["text_normalized"]

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in BOOL_COLS:
    if col in df.columns:
        df[col] = df[col].map(safe_bool)

for col in STRING_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)

df["hf_split"] = df["hf_split"].astype(str)
df["_merged_index"] = pd.to_numeric(df["_merged_index"], errors="coerce").astype(int)

raw_df = df.copy()


# DERIVED STATS

df["char_len"] = df["text_normalized"].str.len().astype(float)
df["word_len_est"] = df["text_normalized"].str.split().str.len().fillna(0).astype(float)

if "num_tokens" not in df.columns or df["num_tokens"].isna().all():
    df["num_tokens_proxy"] = df["word_len_est"].astype(float)
else:
    df["num_tokens_proxy"] = df["num_tokens"].fillna(df["word_len_est"]).astype(float)

df["chars_per_sec"] = df["char_len"] / df["duration_sec"].replace(0, np.nan)
df["tokens_per_sec"] = df["num_tokens_proxy"] / df["duration_sec"].replace(0, np.nan)

df["turns_penalty_proxy"] = (
    df["turns_per_minute"].fillna(0).clip(lower=0) / TURN_RATE_REF
).clip(0, 1)

df["usability_score_custom"] = (
    0.6 * (1.0 - df["overlap_ratio"].fillna(0).clip(0, 1))
    + 0.3 * df["dominant_speaker_ratio"].fillna(1).clip(0, 1)
    + 0.1 * (1.0 - df["turns_penalty_proxy"])
)

df["quality_bin"] = pd.cut(
    df["usability_score_custom"],
    bins=[-np.inf, 0.50, 0.70, 0.85, np.inf],
    labels=["low", "medium", "high", "very_high"],
)


# GLOBAL DEDUPLICATION

df["dedup_key"] = (
    df["text_normalized"].str.lower()
    + "||"
    + df["duration_sec"].fillna(-1).round(2).astype(str)
)

df["split_keep_priority"] = df["hf_split"].map(DEDUP_PRIORITY).fillna(99).astype(int)

before_dedup = len(df)

df = (
    df.sort_values(
        by=["dedup_key", "split_keep_priority", "source_dataset", "_merged_index"],
        ascending=[True, True, True, True],
    )
    .drop_duplicates(subset=["dedup_key"], keep="first")
    .reset_index(drop=True)
)

after_dedup = len(df)
print(f"\nDeduplicated rows: {before_dedup} -> {after_dedup}")

dedup_df = df.copy()


# TRAINABLE MASK

processing_error_present = (
    df["processing_error"].fillna("").astype(str).str.len() > 0
    if "processing_error" in df.columns
    else pd.Series(False, index=df.index)
)

text_empty = df["text_normalized"].fillna("").str.len() == 0

remove_short_audio_long_text = (
    df["remove_short_audio_long_text"].map(safe_bool)
    if "remove_short_audio_long_text" in df.columns
    else pd.Series(False, index=df.index)
)

remove_token_outlier = (
    df["remove_token_outlier"].map(safe_bool)
    if "remove_token_outlier" in df.columns
    else pd.Series(False, index=df.index)
)

usability_too_low = df["usability_score_custom"].fillna(-1.0) <= USABILITY_THRESHOLD

print("\nReject counts:")
print("text_empty:", int(text_empty.sum()))
print("remove_short_audio_long_text:", int(remove_short_audio_long_text.sum()))
print("remove_token_outlier:", int(remove_token_outlier.sum()))
print(f"usability_too_low (<= {USABILITY_THRESHOLD}):", int(usability_too_low.sum()))

df["trainable"] = ~(
    text_empty
    | remove_short_audio_long_text
    | remove_token_outlier
    | usability_too_low
)

print("trainable:", int(df["trainable"].sum()))
print("\nTrainable summary by split:")
print(
    df.groupby("hf_split")["trainable"]
    .agg(n_total="size", n_trainable="sum")
    .reset_index()
    .to_string(index=False)
)

trainable_df = df.loc[df["trainable"]].copy()


# HOURS / COUNTS BY STATE

raw_state_summary = summarize_state(raw_df, "raw_merged")
dedup_state_summary = summarize_state(dedup_df, "deduplicated")
trainable_state_summary = summarize_state(trainable_df, "whisper_ready")

state_summary = pd.concat(
    [raw_state_summary, dedup_state_summary, trainable_state_summary],
    ignore_index=True,
)
state_summary.to_csv(OUT_DIR / "summary_by_state_and_split.csv", index=False)
plot_state_summary(state_summary, PLOTS_DIR)

raw_state_source_summary = summarize_state_by_source(raw_df, "raw_merged")
dedup_state_source_summary = summarize_state_by_source(dedup_df, "deduplicated")
trainable_state_source_summary = summarize_state_by_source(trainable_df, "whisper_ready")

state_source_summary = pd.concat(
    [raw_state_source_summary, dedup_state_source_summary, trainable_state_source_summary],
    ignore_index=True,
)
state_source_summary.to_csv(OUT_DIR / "summary_by_state_split_source.csv", index=False)
plot_state_source_summary(
    state_source_summary[state_source_summary["state"] == "whisper_ready"],
    PLOTS_DIR,
)

print("\nHours / counts by state and split:")
print(state_summary.to_string(index=False))


# SUMMARIES

summary_by_source = (
    df.groupby("source_dataset", dropna=False)
    .agg(
        n_samples=("text_normalized", "size"),
        n_trainable=("trainable", "sum"),
        total_hours=("duration_sec", lambda s: np.nansum(s) / 3600.0),
        median_duration_sec=("duration_sec", "median"),
        median_text_chars=("char_len", "median"),
        median_tokens=("num_tokens_proxy", "median"),
        median_usability_custom=("usability_score_custom", "median"),
        median_asr_usability=("asr_usability_score", "median"),
        overlap_rate=("overlap_speech", "mean"),
        multi_speaker_rate=("multiple_speakers", "mean"),
    )
    .reset_index()
)

summary_by_split = (
    df.groupby("hf_split", dropna=False)
    .agg(
        n_samples=("text_normalized", "size"),
        n_trainable=("trainable", "sum"),
        total_hours=("duration_sec", lambda s: np.nansum(s) / 3600.0),
        median_duration_sec=("duration_sec", "median"),
        median_usability_custom=("usability_score_custom", "median"),
    )
    .reset_index()
)

summary_by_source_and_split = (
    df.groupby(["hf_split", "source_dataset"], dropna=False)
    .agg(
        n_samples=("text_normalized", "size"),
        n_trainable=("trainable", "sum"),
        total_hours=("duration_sec", lambda s: np.nansum(s) / 3600.0),
        median_duration_sec=("duration_sec", "median"),
        median_usability_custom=("usability_score_custom", "median"),
    )
    .reset_index()
)

summary_by_source.to_csv(OUT_DIR / "summary_by_source.csv", index=False)
summary_by_split.to_csv(OUT_DIR / "summary_by_split.csv", index=False)
summary_by_source_and_split.to_csv(OUT_DIR / "summary_by_source_and_split.csv", index=False)

global_summary = {
    "n_total_after_dedup": int(len(df)),
    "n_trainable_after_dedup": int(df["trainable"].sum()),
    "hours_total_after_dedup": float(np.nansum(df["duration_sec"]) / 3600.0),
    "hours_trainable_after_dedup": float(np.nansum(df.loc[df["trainable"], "duration_sec"]) / 3600.0),
    "dedup_removed": int(before_dedup - after_dedup),
    "text_normalization_remove_punctuation": REMOVE_PUNCTUATION,
    "text_normalization_remove_diacritics": REMOVE_DIACRITICS,
    "text_normalization_normalize_alef": NORMALIZE_ALEF,
    "text_normalization_normalize_hamza_seat": NORMALIZE_HAMZA_SEAT,
    "text_normalization_normalize_alef_maqsura": NORMALIZE_ALEF_MAQSURA,
    "text_normalization_normalize_ta_marbuta": NORMALIZE_TA_MARBUTA,
}

with open(OUT_DIR / "global_summary.json", "w", encoding="utf-8") as f:
    json.dump(global_summary, f, ensure_ascii=False, indent=2)



# PLOTS

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="source_dataset")
plt.title("Sample count by source")
plt.xticks(rotation=15)
save_plot(PLOTS_DIR / "01_sample_count_by_source.png")

plt.figure(figsize=(8, 4))
plot_df = df.copy()
upper = np_nanpercentile_safe(plot_df["duration_sec"], 99, default=10.0)
plot_df["duration_sec_clip"] = plot_df["duration_sec"].clip(upper=upper)
sns.histplot(
    data=plot_df,
    x="duration_sec_clip",
    hue="source_dataset",
    bins=80,
    stat="density",
    common_norm=False,
    element="step",
)
plt.title("Duration distribution by source (clipped at p99)")
plt.xlabel("Duration (sec)")
save_plot(PLOTS_DIR / "02_duration_distribution.png")

plt.figure(figsize=(8, 4))
sns.histplot(
    data=df,
    x="usability_score_custom",
    hue="source_dataset",
    bins=50,
    stat="density",
    common_norm=False,
    element="step",
)
plt.title("Custom usability score distribution")
save_plot(PLOTS_DIR / "03_custom_usability_distribution.png")

if "asr_usability_score" in df.columns and df["asr_usability_score"].notna().sum() > 10:
    plt.figure(figsize=(6, 6))
    score_df = df[["asr_usability_score", "usability_score_custom"]].dropna()
    sample_n = min(len(score_df), 5000)
    if sample_n > 0:
        score_df = score_df.sample(sample_n, random_state=RANDOM_SEED)
    sns.scatterplot(
        data=score_df,
        x="asr_usability_score",
        y="usability_score_custom",
        s=15,
        alpha=0.5,
    )
    plt.title("Existing ASR usability vs custom usability")
    save_plot(PLOTS_DIR / "04_existing_vs_custom_score.png")

if {"overlap_ratio", "dominant_speaker_ratio"}.issubset(df.columns):
    scatter_df = df[["overlap_ratio", "dominant_speaker_ratio", "source_dataset"]].dropna()
    if len(scatter_df) > 0:
        plt.figure(figsize=(6, 6))
        sample_n = min(len(scatter_df), 8000)
        scatter_df = scatter_df.sample(sample_n, random_state=RANDOM_SEED)
        sns.scatterplot(
            data=scatter_df,
            x="overlap_ratio",
            y="dominant_speaker_ratio",
            hue="source_dataset",
            s=12,
            alpha=0.4,
        )
        plt.title("Overlap ratio vs dominant speaker ratio")
        save_plot(PLOTS_DIR / "05_overlap_vs_dominant_ratio.png")

plt.figure(figsize=(8, 4))
quality_ct = (
    df.groupby(["source_dataset", "quality_bin"], dropna=False)
    .size()
    .reset_index(name="count")
)
quality_ct["quality_bin"] = quality_ct["quality_bin"].astype(str)
sns.barplot(data=quality_ct, x="source_dataset", y="count", hue="quality_bin")
plt.title("Quality bins by source")
plt.xticks(rotation=15)
save_plot(PLOTS_DIR / "06_quality_bins_by_source.png")

plt.figure(figsize=(8, 4))
box_df = df[["source_dataset", "tokens_per_sec"]].dropna().copy()
if len(box_df) > 0:
    q_hi = np_nanpercentile_safe(box_df["tokens_per_sec"], 99, default=10.0)
    box_df["tokens_per_sec"] = box_df["tokens_per_sec"].clip(upper=q_hi)
    sns.boxplot(data=box_df, x="source_dataset", y="tokens_per_sec")
    plt.title("Tokens per second by source (clipped at p99)")
    plt.xticks(rotation=15)
    save_plot(PLOTS_DIR / "07_tokens_per_sec_by_source.png")
else:
    plt.close()

corr_cols = [
    c for c in [
        "duration_sec",
        "num_tokens_proxy",
        "chars_per_sec",
        "tokens_per_sec",
        "num_speakers",
        "speaker_turns",
        "turns_per_minute",
        "dominant_speaker_ratio",
        "second_speaker_ratio",
        "non_dominant_speech_ratio",
        "speaker_balance_score",
        "speaker_entropy",
        "overlap_ratio",
        "num_overlap_regions",
        "max_concurrent_speakers",
        "asr_usability_score",
        "usability_score_custom",
    ]
    if c in df.columns
]
if len(corr_cols) >= 2:
    corr = df[corr_cols].corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    save_plot(PLOTS_DIR / "08_correlation_heatmap.png")


# REBUILD DEDUP DATASETDICT

dedup_by_split = {}
whisper_ready_by_split = {}

for split in ordered_splits(df["hf_split"].unique().tolist()):
    if split not in merged_by_split:
        continue

    split_df = df.loc[df["hf_split"] == split].copy()
    split_df = split_df.sort_values("_merged_index").reset_index(drop=True)

    if len(split_df) == 0:
        continue

    keep_indices = split_df["_merged_index"].astype(int).tolist()
    split_ds = merged_by_split[split].select(keep_indices)

    # replace text column by normalized text for final training artifacts
    if "text" in split_ds.column_names:
        split_ds = split_ds.remove_columns(["text"])
    split_ds = split_ds.add_column("text", to_pylist_string(split_df["text_normalized"], fill=""))

    split_ds = split_ds.add_column("text_raw", to_pylist_string(split_df["text_raw"], fill=""))
    split_ds = split_ds.add_column("text_normalized", to_pylist_string(split_df["text_normalized"], fill=""))

    split_ds = split_ds.add_column("char_len", to_pylist_numeric(split_df["char_len"]))
    split_ds = split_ds.add_column("word_len_est", to_pylist_numeric(split_df["word_len_est"]))
    split_ds = split_ds.add_column("num_tokens_proxy", to_pylist_numeric(split_df["num_tokens_proxy"]))
    split_ds = split_ds.add_column("chars_per_sec", to_pylist_numeric(split_df["chars_per_sec"]))
    split_ds = split_ds.add_column("tokens_per_sec", to_pylist_numeric(split_df["tokens_per_sec"]))
    split_ds = split_ds.add_column("turns_penalty_proxy", to_pylist_numeric(split_df["turns_penalty_proxy"]))
    split_ds = split_ds.add_column("usability_score_custom", to_pylist_numeric(split_df["usability_score_custom"]))
    split_ds = split_ds.add_column("quality_bin", to_pylist_string(split_df["quality_bin"], fill="unknown"))
    split_ds = split_ds.add_column("trainable", to_pylist_bool(split_df["trainable"]))

    dedup_by_split[split] = split_ds

    trainable_positions = np.where(split_df["trainable"].to_numpy())[0].tolist()
    split_trainable_ds = split_ds.select(trainable_positions)

    drop_helper_cols = [c for c in ["hf_split", "_merged_index"] if c in split_trainable_ds.column_names]
    if drop_helper_cols:
        split_trainable_ds = split_trainable_ds.remove_columns(drop_helper_cols)

    whisper_ready_by_split[split] = split_trainable_ds

dedup_dict = DatasetDict(dedup_by_split)
dedup_dict.save_to_disk(str(OUT_DIR / "merged_dedup_official_splits"))

whisper_ready_dict = DatasetDict(whisper_ready_by_split)
print("\nWhisper-ready split sizes before save:")
for split, ds in whisper_ready_dict.items():
    print(f"  {split}: {len(ds)} rows")
whisper_ready_dict.save_to_disk(str(OUT_DIR / "whisper_ready_dataset"))

# =========================
# EXPORT 16K WAV FILES + DIAGNOSTICS
# =========================
if EXPORT_16K_AUDIO:
    whisper_ready_16k_by_split = {}
    manifest_frames = []
    export_summary_rows = []
    failed_exports_frames = []
    mismatch_frames = []

    print("\nExporting 16 kHz audio with librosa.load(..., sr=16000)...")
    split_names = list(whisper_ready_dict.keys())

    for split_idx, split in enumerate(split_names, start=1):
        ds = whisper_ready_dict[split]
        print(f"\n[{split_idx}/{len(split_names)}] Starting split={split} n={len(ds)}")

        ds_16k, manifest_df, final_hours_loaded, final_success, final_fail = export_split_audio_to_16k(
            ds=ds,
            split_name=split,
            out_root=AUDIO_16K_DIR,
            target_sr=TARGET_SR,
        )

        whisper_ready_16k_by_split[split] = ds_16k
        manifest_df["hf_split"] = split
        manifest_frames.append(manifest_df)
        manifest_df.to_csv(OUT_DIR / f"{split}_manifest_16k.csv", index=False)

        manifest_df["duration_sec"] = pd.to_numeric(manifest_df["duration_sec"], errors="coerce")
        manifest_df["loaded_duration_sec"] = pd.to_numeric(manifest_df["audio_num_seconds_16k"], errors="coerce")

        ok_df = manifest_df[manifest_df["audio_export_success"]].copy()
        bad_df = manifest_df[~manifest_df["audio_export_success"]].copy()

        if len(bad_df) > 0:
            bad_df.to_csv(OUT_DIR / f"{split}_failed_audio_exports.csv", index=False)
            failed_exports_frames.append(bad_df)
            print(f"Saved failed export list for {split}: {len(bad_df)} rows")
            print(bad_df["audio_export_error"].value_counts().head(20).to_string())

        if len(ok_df) > 0:
            ok_df["abs_diff_sec"] = (ok_df["loaded_duration_sec"] - ok_df["duration_sec"]).abs()
            ok_df["ratio"] = ok_df["loaded_duration_sec"] / ok_df["duration_sec"]

            print(f"\nDuration check for split={split}")
            print(ok_df[["duration_sec", "loaded_duration_sec", "abs_diff_sec", "ratio"]].describe().to_string())

            largest_mismatch = ok_df.sort_values("abs_diff_sec", ascending=False).head(200).copy()
            largest_mismatch.to_csv(OUT_DIR / f"{split}_largest_duration_mismatches.csv", index=False)
            mismatch_frames.append(largest_mismatch)

        metadata_hours = float(pd.to_numeric(manifest_df["duration_sec"], errors="coerce").sum() / 3600.0)

        export_summary_rows.append({
            "hf_split": split,
            "n_total": len(ds),
            "n_success": final_success,
            "n_failed": final_fail,
            "metadata_hours": metadata_hours,
            "loaded_hours_from_waveforms": final_hours_loaded,
            "hours_ratio_loaded_vs_metadata": (
                final_hours_loaded / metadata_hours if metadata_hours > 0 else np.nan
            ),
        })

    whisper_ready_16k_dict = DatasetDict(whisper_ready_16k_by_split)
    whisper_ready_16k_dict.save_to_disk(str(OUT_DIR / "whisper_ready_dataset_16k_paths"))

    full_manifest_df = pd.concat(manifest_frames, ignore_index=True)
    full_manifest_df.to_csv(OUT_DIR / "full_manifest_16k.csv", index=False)

    export_summary_df = pd.DataFrame(export_summary_rows)
    export_summary_df.to_csv(OUT_DIR / "export_audio_summary.csv", index=False)

    print("\nFinal exported audio summary:")
    print(export_summary_df.to_string(index=False))

    total_loaded_hours = export_summary_df["loaded_hours_from_waveforms"].sum()
    total_metadata_hours = export_summary_df["metadata_hours"].sum()
    print(
        f"\nTOTAL loaded hours from waveforms: {total_loaded_hours:.4f} h | "
        f"TOTAL metadata hours: {total_metadata_hours:.4f} h"
    )

    if failed_exports_frames:
        pd.concat(failed_exports_frames, ignore_index=True).to_csv(
            OUT_DIR / "failed_audio_exports_all.csv", index=False
        )

    if mismatch_frames:
        pd.concat(mismatch_frames, ignore_index=True).to_csv(
            OUT_DIR / "largest_duration_mismatches_all.csv", index=False
        )

    recount_rows = []
    print("\nRecounting exported WAV hours from disk...")
    for split in whisper_ready_16k_dict.keys():
        split_dir = AUDIO_16K_DIR / split
        n_files, hours = compute_exported_hours_from_files(split_dir, target_sr=TARGET_SR)
        recount_rows.append({
            "hf_split": split,
            "exported_wav_files": n_files,
            "exported_hours_from_disk": hours,
        })
        print(f"{split}: exported_wav_files={n_files}, exported_hours_from_disk={hours:.4f}")

    recount_df = pd.DataFrame(recount_rows)
    recount_df.to_csv(OUT_DIR / "exported_wav_recount.csv", index=False)

    clean_export_dict = {}
    for split, ds in whisper_ready_16k_dict.items():
        ok_idx = [i for i, x in enumerate(ds["audio_export_success"]) if x]
        clean_export_dict[split] = ds.select(ok_idx)

    clean_export_dict = DatasetDict(clean_export_dict)
    clean_export_dict.save_to_disk(
        str(OUT_DIR / "whisper_ready_dataset_16k_paths_success_only")
    )


# SAVE METADATA CSVs

df.to_csv(OUT_DIR / "full_dedup_metadata.csv", index=False)

for split in ordered_splits(df["hf_split"].unique().tolist()):
    split_df = df.loc[df["hf_split"] == split].copy().sort_values("_merged_index")
    if len(split_df) == 0:
        continue
    split_df.to_csv(OUT_DIR / f"{split}_metadata_dedup.csv", index=False)
    split_df.loc[split_df["trainable"]].to_csv(OUT_DIR / f"{split}_metadata_trainable.csv", index=False)




# FINAL LOG

print("\nDone.")
print(f"Artifacts saved under: {OUT_DIR.resolve()}")
print(f"Raw merged dataset: {(OUT_DIR / 'merged_raw_no_decode').resolve()}")
print(f"Dedup merged dataset: {(OUT_DIR / 'merged_dedup_official_splits').resolve()}")
print(f"Whisper-ready dataset: {(OUT_DIR / 'whisper_ready_dataset').resolve()}")
print(f"Whisper-ready 16k paths dataset: {(OUT_DIR / 'whisper_ready_dataset_16k_paths').resolve()}")
print(f"Whisper-ready 16k success-only dataset: {(OUT_DIR / 'whisper_ready_dataset_16k_paths_success_only').resolve()}")
print(f"Exported 16k wav root: {AUDIO_16K_DIR.resolve()}")

print("\nWhisper-ready split sizes:")
for split, ds in whisper_ready_dict.items():
    print(f"  {split}: {len(ds)} rows")