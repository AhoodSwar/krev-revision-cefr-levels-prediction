#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible KUPA-KEYS Task-2 revision-feature CEFR analysis.

Inputs are user-supplied KUPA-KEYS Task2.csv and DLLA.csv. The script does
not download or redistribute the original corpus.

Usage:
    python kupakeys_revision13.py --task2 /path/to/Task2.csv \
        --dlla /path/to/DLLA.csv --output-dir results

The output includes writing-level/episode-level data and participant IDs;


Dependencies: numpy, pandas, matplotlib, scikit-learn, mord, krev.
"""
from pathlib import Path
from difflib import SequenceMatcher
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from krev import KeystrokeText
from mord import LogisticIT
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)
RANDOM_STATE = 42
GT1000_MS = 1000.0
NON_PRINTABLE_KEYS = {
    'Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab', 'Escape',
    'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
    'PageUp', 'PageDown', 'Home', 'End', 'Insert',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
    'AltGraph', 'ContextMenu', 'Dead',
}
LEVELS = ['B1', 'B2', 'C1', 'C2']
YMAP = {'B1': 0, 'B2': 1, 'C1': 2, 'C2': 3}
NEW_GT1000 = 'n_backspace_sequences_with_gt1000ms_intervals'
REVISION13 = [
    'revisions_per_100_words',
    'total_revision_time_ratio',
    'revision_regularity_sec_sd',
    'ratio_backspace_seq_gt3_in_revisions',
    NEW_GT1000,
    'mean_time_to_edit_ms_in_revisions',
    'median_backspaces_per_revision',
    'median_chars_deleted_per_revision',
    'median_chars_inserted_per_revision',
    'median_time_from_start_process_sec',
    'median_pause_before_revision_sec',
    'n_backspacing_sequences_in_revisions',
    'mean_backspaces_per_sequence',
]
FEATURE_LABELS = {
    'revisions_per_100_words': 'Revision episodes per 100 words',
    'total_revision_time_ratio': 'Proportion of time spent revising',
    'revision_regularity_sec_sd': 'Revision interval variability (SD, s)',
    'ratio_backspace_seq_gt3_in_revisions': 'Backspace sequences >3 (proportion)',
    NEW_GT1000: 'Backspace sequences with >1 s interval',
    'mean_time_to_edit_ms_in_revisions': 'Time to edit (mean, ms)',
    'median_backspaces_per_revision': 'Backspaces per revision (median)',
    'median_chars_deleted_per_revision': 'Characters deleted per revision (median)',
    'median_chars_inserted_per_revision': 'Characters inserted per revision (median)',
    'median_time_from_start_process_sec': 'Time from writing start (median, s)',
    'median_pause_before_revision_sec': 'Pause before revision (median, s)',
    'n_backspacing_sequences_in_revisions': 'Backspace sequences in revisions',
    'mean_backspaces_per_sequence': 'Backspaces per sequence (mean)',
}
plt.rcParams.update({
    'font.size': 15, 'axes.titlesize': 19, 'axes.labelsize': 17,
    'xtick.labelsize': 14, 'ytick.labelsize': 15,
    'legend.fontsize': 14, 'figure.titlesize': 19,
})


def pretty(name):
    return FEATURE_LABELS.get(name, name)


def finite_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_lookup(values, idx):
    try:
        i = int(idx)
        return values[i] if 0 <= i < len(values) else np.nan
    except (TypeError, ValueError):
        return np.nan


def event_rank(series):
    return series.map({'click': 0, 'down': 1, 'input': 2,
                       'insert': 2, 'up': 3, 'capture': 4}).fillna(5)


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value) if pd.notna(value) else False


def sorted_events(user_df):
    df = user_df.copy()
    df['_rank'] = event_rank(df['type'])
    return df.sort_values(['time', '_rank'], kind='mergesort').reset_index(drop=True)


def process_single_participant(user_df):
    """Original reconstruction: one text/cursor/time state per key-down."""
    df = sorted_events(user_df)
    text = ''
    current_cursor = current_selection_end = 0
    text_states, cursor_positions, time_list = [], [], []
    for _, row in df.iterrows():
        typ = row['type']
        if pd.notna(row.get('range_start')) and pd.notna(row.get('range_end')):
            start = min(max(0, int(row['range_start'])), len(text))
            end = min(max(0, int(row['range_end'])), len(text))
            current_cursor, current_selection_end = sorted((start, end))

        if typ in ('input', 'insert'):
            inp = str(row['text']) if pd.notna(row.get('text')) else ''
            if len(inp) > 1:
                lo, hi = sorted((current_cursor, current_selection_end))
                text = text[:lo] + inp + text[hi:]
                current_cursor = lo + len(inp)
                current_selection_end = current_cursor
        elif typ == 'down':
            key = str(row['key']) if pd.notna(row.get('key')) else ''
            if not (as_bool(row.get('ctrl_key')) or as_bool(row.get('meta_key'))):
                lo, hi = sorted((current_cursor, current_selection_end))
                if key == 'Backspace':
                    if lo == hi:
                        if current_cursor > 0:
                            text = text[:current_cursor - 1] + text[current_cursor:]
                            current_cursor -= 1
                    else:
                        text = text[:lo] + text[hi:]
                        current_cursor = lo
                    current_selection_end = current_cursor
                elif key == 'Delete':
                    if lo == hi:
                        if current_cursor < len(text):
                            text = text[:current_cursor] + text[current_cursor + 1:]
                    else:
                        text = text[:lo] + text[hi:]
                        current_cursor = lo
                    current_selection_end = current_cursor
                elif key == 'Enter':
                    text = text[:lo] + '\n' + text[hi:]
                    current_cursor = lo + 1
                    current_selection_end = current_cursor
                elif len(key) == 1 and key not in NON_PRINTABLE_KEYS:
                    text = text[:lo] + key + text[hi:]
                    current_cursor = lo + 1
                    current_selection_end = current_cursor
        elif typ == 'capture':
            cap = row.get('text')
            if pd.notna(cap):
                text = str(cap)
                current_cursor = min(current_cursor, len(text))
                current_selection_end = min(current_selection_end, len(text))

        if typ == 'down':
            text_states.append(text)
            cursor_positions.append(current_cursor)
            time_list.append(row['time'])
    return text_states, cursor_positions, time_list


def sync_provenance_after_capture(old_text, old_origins, new_text):
    matcher = SequenceMatcher(None, str(old_text), str(new_text), autojunk=False)
    origins = [np.nan] * len(new_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            origins[j1:j2] = old_origins[i1:i2]
    return origins


def build_task2_trace(user_df):
    """Original physical Backspace sequence definition and >1000 ms criterion."""
    df = sorted_events(user_df)
    text, origin_times = '', []
    cursor = selection_end = 0
    down_rows = []
    previous_down_time = np.nan
    for _, row in df.iterrows():
        typ = row['type']
        t = finite_float(row.get('time'))
        if pd.notna(row.get('range_start')) and pd.notna(row.get('range_end')):
            a = min(max(0, int(row['range_start'])), len(text))
            b = min(max(0, int(row['range_end'])), len(text))
            cursor, selection_end = sorted((a, b))
        if typ in ('input', 'insert'):
            inp = str(row['text']) if pd.notna(row.get('text')) else ''
            if len(inp) > 1:
                lo, hi = sorted((cursor, selection_end))
                text = text[:lo] + inp + text[hi:]
                origin_times = origin_times[:lo] + [t] * len(inp) + origin_times[hi:]
                cursor = lo + len(inp)
                selection_end = cursor
        elif typ == 'down':
            key = str(row['key']) if pd.notna(row.get('key')) else ''
            pause = t - previous_down_time if np.isfinite(t) and np.isfinite(previous_down_time) else np.nan
            deleted_origin, deleted_count = np.nan, 0
            if not (as_bool(row.get('ctrl_key')) or as_bool(row.get('meta_key'))):
                lo, hi = sorted((cursor, selection_end))
                if key == 'Backspace':
                    if lo == hi:
                        if cursor > 0:
                            deleted_origin = origin_times[cursor - 1]
                            deleted_count = 1
                            text = text[:cursor - 1] + text[cursor:]
                            del origin_times[cursor - 1]
                            cursor -= 1
                    else:
                        deleted_count = hi - lo
                        text = text[:lo] + text[hi:]
                        del origin_times[lo:hi]
                        cursor = lo
                    selection_end = cursor
                elif key == 'Delete':
                    if lo == hi:
                        if cursor < len(text):
                            deleted_count = 1
                            text = text[:cursor] + text[cursor + 1:]
                            del origin_times[cursor]
                    else:
                        deleted_count = hi - lo
                        text = text[:lo] + text[hi:]
                        del origin_times[lo:hi]
                        cursor = lo
                    selection_end = cursor
                elif key == 'Enter':
                    text = text[:lo] + '\n' + text[hi:]
                    origin_times = origin_times[:lo] + [t] + origin_times[hi:]
                    cursor = lo + 1
                    selection_end = cursor
                elif len(key) == 1 and key not in NON_PRINTABLE_KEYS:
                    text = text[:lo] + key + text[hi:]
                    origin_times = origin_times[:lo] + [t] + origin_times[hi:]
                    cursor = lo + 1
                    selection_end = cursor
            down_rows.append({
                'down_index': len(down_rows), 'time_ms': t, 'key': key,
                'pause_2_before_ms': pause, 'deleted_origin_time_ms': deleted_origin,
                'deleted_char_count': deleted_count,
            })
            previous_down_time = t
        elif typ == 'capture':
            cap = row.get('text')
            if pd.notna(cap):
                cap = str(cap)
                if cap != text:
                    origin_times = sync_provenance_after_capture(text, origin_times, cap)
                text = cap
                cursor = min(cursor, len(text))
                selection_end = min(selection_end, len(text))

    trace = pd.DataFrame(down_rows)
    if trace.empty:
        return trace, pd.DataFrame()
    trace['is_backspace'] = trace['key'].eq('Backspace')
    trace['backspace_sequence_id'] = pd.Series(pd.NA, index=trace.index, dtype='Int64')
    seq_records, seq_id, i = [], 0, 0
    while i < len(trace):
        if not bool(trace.loc[i, 'is_backspace']):
            i += 1
            continue
        start = i
        while i + 1 < len(trace) and bool(trace.loc[i + 1, 'is_backspace']):
            i += 1
        end = i
        seq_id += 1
        trace.loc[start:end, 'backspace_sequence_id'] = seq_id
        seq = trace.loc[start:end]
        pauses = pd.to_numeric(seq['pause_2_before_ms'], errors='coerce')
        qualifies = bool((pauses > GT1000_MS).fillna(False).any())
        first_bs = finite_float(seq.iloc[0]['time_ms'])
        erased = finite_float(seq.iloc[0]['deleted_origin_time_ms'])
        tte = first_bs - erased if np.isfinite(first_bs) and np.isfinite(erased) and first_bs >= erased else np.nan
        seq_records.append({
            'backspace_sequence_id': seq_id,
            'sequence_start_down_index': int(trace.loc[start, 'down_index']),
            'sequence_end_down_index': int(trace.loc[end, 'down_index']),
            'n_backspaces': len(seq), 'is_gt3': len(seq) > 3,
            'gt1000ms_qualifies': qualifies, 'time_to_edit_ms': tte,
        })
        i += 1
    return trace, pd.DataFrame(seq_records)


def transition_edit_chunks(before, after):
    matcher = SequenceMatcher(None, '' if before is None else str(before),
                              '' if after is None else str(after), autojunk=False)
    deleted, inserted = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('delete', 'replace') and i2 > i1:
            deleted.append(before[i1:i2])
        if tag in ('insert', 'replace') and j2 > j1:
            inserted.append(after[j1:j2])
    return deleted, inserted


def episode_character_counts(text_list, s, e):
    try:
        s, e = int(s), int(e)
    except (ValueError, TypeError):
        return np.nan, np.nan
    if s < 0 or e <= s or e >= len(text_list):
        return np.nan, np.nan
    nd = ni = 0
    for i in range(s + 1, e + 1):
        d, ins = transition_edit_chunks(text_list[i - 1], text_list[i])
        nd += sum(map(len, d))
        ni += sum(map(len, ins))
    return float(nd), float(ni)


def union_duration_sec(starts, ends):
    pairs = []
    for a, b in zip(starts, ends):
        if pd.notna(a) and pd.notna(b):
            a, b = float(a), float(b)
            pairs.append((min(a, b), max(a, b)))
    if not pairs:
        return 0.0
    pairs.sort()
    cs, ce = pairs[0]
    total = 0.0
    for s, e in pairs[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return (total + ce - cs) / 1000.0


def make_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticIT(alpha=1.0)),
    ])


def metric_dict(y_true, y_pred, labels=None):
    yt, yp = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    return {
        'accuracy': accuracy_score(yt, yp),
        'balanced_accuracy': balanced_accuracy_score(yt, yp),
        'mae': mean_absolute_error(yt, yp),
        'adjacent_accuracy': float(np.mean(np.abs(yt - yp) <= 1)),
    }


def pseudo_r2(fitted, X):
    Xi = fitted.named_steps['imputer'].transform(X)
    Xs = fitted.named_steps['scaler'].transform(Xi)
    beta = np.asarray(fitted.named_steps['model'].coef_, dtype=float)
    latent = np.asarray(Xs).dot(beta)
    v = float(np.var(latent, ddof=0))
    return v / (v + np.pi ** 2 / 3)


def to_cefr(v):
    if pd.isna(v):
        return np.nan
    n = int(v)
    if n in (1, 2): return 'A1'
    if n in (3, 4): return 'A2'
    if n in (5, 6): return 'B1'
    if n in (7, 8): return 'B2'
    if n in (9, 10): return 'C1'
    if n in (11, 12, 13): return 'C2'
    return np.nan


def run_model_analysis(data, levels, out_dir, prefix):
    """Fit and evaluate the primary four-level CEFR model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = data.loc[data['CEFR_target'].isin(levels)].copy().reset_index(drop=True)
    level_map = {level: i for i, level in enumerate(levels)}
    subset['model_target'] = subset['CEFR_target'].map(level_map).astype(int)
    X = subset[REVISION13].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    y = subset['model_target']
    counts = subset['CEFR_target'].value_counts().reindex(levels, fill_value=0)
    print(f'\n{prefix}: N={len(subset)}; distribution={counts.to_dict()}')
    if counts.min() < 10:
        raise ValueError(f'{prefix}: fewer than 10 observations in at least one CEFR level.')
    missing_all = X.columns[X.isna().all()].tolist()
    if missing_all:
        raise ValueError(f'{prefix}: features entirely missing: {missing_all}')

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    fold_rows, oof = [], []
    for fold, (tri, tei) in enumerate(cv.split(X, y), 1):
        model = make_pipeline()
        model.fit(X.iloc[tri], y.iloc[tri])
        yp = np.asarray(model.predict(X.iloc[tei]), dtype=int)
        fold_rows.append({'fold': fold, **metric_dict(y.iloc[tei], yp)})
        z = subset.iloc[tei][['id', 'CEFR_target']].copy()
        z['fold'] = fold
        z['true_numeric'] = y.iloc[tei].to_numpy()
        z['pred_numeric'] = yp
        z['predicted_CEFR'] = [levels[int(v)] for v in yp]
        oof.append(z)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out_dir / f'{prefix}_CV_fold_results.csv', index=False)
    names = ['accuracy', 'balanced_accuracy', 'mae', 'adjacent_accuracy']
    summary = pd.DataFrame({
        'metric': names,
        'mean': [folds[c].mean() for c in names],
        'sd': [folds[c].std(ddof=1) for c in names],
    })
    summary.to_csv(out_dir / f'{prefix}_CV_summary.csv', index=False)
    oo = pd.concat(oof, ignore_index=True)
    oo.to_csv(out_dir / f'{prefix}_OOF_predictions.csv', index=False)
    pooled = metric_dict(oo['true_numeric'], oo['pred_numeric'])
    pd.DataFrame([pooled]).to_csv(out_dir / f'{prefix}_pooled_OOF_metrics.csv', index=False)

    cm = confusion_matrix(oo['true_numeric'], oo['pred_numeric'], labels=list(range(len(levels))))
    pd.DataFrame(cm, index=[f'true_{x}' for x in levels],
                 columns=[f'pred_{x}' for x in levels]).to_csv(
                     out_dir / f'{prefix}_confusion_matrix_counts.csv')
    report = classification_report(
        oo['true_numeric'], oo['pred_numeric'], labels=list(range(len(levels))),
        target_names=levels, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(out_dir / f'{prefix}_classification_report.csv')

    final = make_pipeline()
    final.fit(X, y)
    coef = pd.DataFrame({
        'feature': REVISION13,
        'standardized_coefficient': final.named_steps['model'].coef_,
    })
    coef['absolute_coefficient'] = coef['standardized_coefficient'].abs()
    coef.to_csv(out_dir / f'{prefix}_standardized_coefficients.csv', index=False)
    r2 = pseudo_r2(final, X)
    pd.DataFrame([{'McKelvey_Zavoina_pseudo_R2': r2, 'percent': 100 * r2}]).to_csv(
        out_dir / f'{prefix}_pseudo_R2.csv', index=False)

    # Figure 1: OOF confusion matrix.
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=levels)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title('Ordinal Regression Confusion Matrix', fontsize=18, pad=15)
    ax.set_xlabel('Predicted CEFR level', fontsize=16)
    ax.set_ylabel('True CEFR level', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    if disp.text_ is not None:
        for t in np.asarray(disp.text_).ravel():
            t.set_fontsize(15)
    fig.tight_layout()
    fig.savefig(out_dir / f'{prefix}_confusion_matrix.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: Top 10 absolute standardized coefficients.
    top10 = coef.nlargest(10, 'absolute_coefficient').sort_values('absolute_coefficient')
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.barh(top10['feature'].map(pretty), top10['absolute_coefficient'])
    ax.set_xlabel('Absolute standardized coefficient')
    ax.set_title('Top 10 Ordinal Regression Features', pad=15)
    fig.tight_layout()
    fig.savefig(out_dir / f'{prefix}_top10_features.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    # Figure 3: Signed coefficients; 13 features shown (no duplication).
    signed = coef.sort_values('standardized_coefficient')
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.barh(signed['feature'].map(pretty), signed['standardized_coefficient'], color='#1f77b4')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Signed standardized coefficient')
    ax.set_title('Features Associated with Lower vs Higher CEFR Levels', pad=15)
    fig.tight_layout()
    fig.savefig(out_dir / f'{prefix}_signed_coefficients.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    # Figure 4: Top 6 predictors, raw values from the model-ready dataset.
    top6 = coef.nlargest(6, 'absolute_coefficient')['feature'].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, feature in zip(axes.ravel(), top6):
        vals = [pd.to_numeric(subset.loc[subset['CEFR_target'].eq(level), feature],
                              errors='coerce').dropna().to_numpy() for level in levels]
        ax.boxplot(vals, tick_labels=levels)
        ax.set_title(pretty(feature), fontsize=14, pad=12)
        ax.set_xlabel('CEFR level', fontsize=14)
        ax.set_ylabel(pretty(feature), fontsize=12)
        ax.tick_params(axis='both', labelsize=13)
    for ax in axes.ravel()[len(top6):]:
        ax.set_visible(False)
    fig.tight_layout(pad=3.0)
    fig.savefig(out_dir / f'{prefix}_top6_boxplots.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    print(summary.to_string(index=False))
    print('Pooled OOF:', pooled)
    print(f'Full-sample McKelvey-Zavoina pseudo-R2: {r2:.6f}')
    return {'summary': summary, 'pooled': pooled, 'pseudo_r2': r2,
            'n': len(subset), 'levels': levels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task2', required=True)
    parser.add_argument('--dlla', required=True)
    parser.add_argument('--output-dir', default='KUPAKEYS_Revision13_updated')
    args = parser.parse_args()
    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    task2 = pd.read_csv(args.task2, low_memory=False)
    dlla = pd.read_csv(args.dlla, low_memory=False)
    for df in (task2, dlla):
        if 'id' not in df.columns:
            raise KeyError("Required column 'id' not found")
        df['id'] = df['id'].astype(str).str.strip()
    if dlla['id'].duplicated().any():
        raise RuntimeError('DLLA contains duplicate participant IDs; inspect before merging.')

    # 1. Reconstruct Task-2 only; retain all Task-2 validation passes.
    recon_all, validation_rows, failed_ids = {}, [], []
    for pid, g in task2.groupby('id', sort=False):
        text_list, cursor_list, time_list = process_single_participant(g)
        recon_all[pid] = {'text_list': text_list, 'cursor_list': cursor_list,
                          'time_list': time_list}
        passed, error = True, ''
        try:
            if len(text_list) != int(g['type'].eq('down').sum()):
                raise AssertionError('Number of reconstructed states != number of key-downs')
            for i, (txt, cur) in enumerate(zip(text_list, cursor_list)):
                if not 0 <= cur <= len(txt):
                    raise AssertionError(f'Keystroke {i}: cursor out of bounds')
            capture_events = g.sort_values('time')[g.sort_values('time')['type'].eq('capture')]
            if not capture_events.empty and text_list:
                actual = capture_events.iloc[-1]['text']
                actual = str(actual) if pd.notna(actual) else ''
                simulated = text_list[-1]
                match_exact = simulated == actual
                match_tolerant = (abs(len(simulated) - len(actual)) <= 2
                                  and simulated.replace(' ', '') == actual.replace(' ', ''))
                match_in_flight = ((simulated.startswith(actual) or actual.startswith(simulated))
                                   and abs(len(simulated) - len(actual)) <= 3)
                if not (match_exact or match_tolerant or match_in_flight):
                    raise AssertionError(f'Final text mismatch: expected tail={actual[-40:]!r}; got={simulated[-40:]!r}')
        except AssertionError as exc:
            passed, error = False, str(exc)
            failed_ids.append(pid)
        validation_rows.append({'id': pid, 'validation_passed': passed,
                                'error': error, 'n_text_states': len(text_list)})
    pd.DataFrame(validation_rows).to_csv(OUT / '00_Task2_reconstruction_validation.csv', index=False)
    failed_set = set(failed_ids)
    recon = {pid: v for pid, v in recon_all.items() if pid not in failed_set}
    print(f'Task-2 validation: passed={len(recon)}, failed={len(failed_ids)}')

    # 2. K-REV episodes.
    episode_rows, summary_rows = [], []
    for pid, kd in recon.items():
        texts, cursors, times = kd['text_list'], kd['cursor_list'], kd['time_list']
        kt = KeystrokeText(texts, cursors)
        revs = kt.extract_revisions()
        valid_times = [finite_float(t) for t in times if np.isfinite(finite_float(t))]
        summary_rows.append({
            'id': pid, 'n_text_states': len(texts), 'n_revisions_krev': len(revs),
            'session_start_time_ms': valid_times[0] if valid_times else np.nan,
            'session_end_time_ms': valid_times[-1] if valid_times else np.nan,
        })
        for rid, rev in enumerate(revs, 1):
            s, e = int(rev.index_start), int(rev.index_end)
            episode_rows.append({
                'id': pid, 'revision_id': rid, 'index_start': s, 'index_end': e,
                'revision_start_time_ms': safe_lookup(times, s),
                'revision_end_time_ms': safe_lookup(times, e),
                'sequencing': getattr(rev, 'sequencing', None),
            })
    episodes = pd.DataFrame(episode_rows, columns=[
        'id', 'revision_id', 'index_start', 'index_end',
        'revision_start_time_ms', 'revision_end_time_ms', 'sequencing'])
    pd.DataFrame(summary_rows).to_csv(OUT / '01_KREV_summary.csv', index=False)
    episodes.to_csv(OUT / '02_KREV_revision_episodes.csv', index=False)

    # 3. Backspace traces and alignment checks.
    traces, seqs = {}, {}
    groups = {pid: g for pid, g in task2.groupby('id', sort=False)}
    for pid in recon:
        traces[pid], seqs[pid] = build_task2_trace(groups[pid])
    align_rows = []
    for pid, kd in recon.items():
        tr = traces[pid]
        a = np.asarray([finite_float(t) for t in kd['time_list']], dtype=float)
        b = (pd.to_numeric(tr['time_ms'], errors='coerce').to_numpy(float)
             if not tr.empty else np.asarray([], dtype=float))
        same = len(a) == len(b)
        exact = False
        if same:
            mask = np.isfinite(a) & np.isfinite(b)
            exact = bool(np.array_equal(np.isfinite(a), np.isfinite(b))
                         and np.allclose(a[mask], b[mask], rtol=0, atol=1e-9))
        align_rows.append({'id': pid, 'n_krev': len(a), 'n_raw_down': len(b),
                           'same_n': same, 'exact_timestamp_alignment': exact})
    pd.DataFrame(align_rows).to_csv(OUT / '03_KREV_raw_alignment.csv', index=False)

    # 4. Episode and writing-level measures (original definitions retained).
    conijn_rows, writing_rows = [], []
    for pid, kd in recon.items():
        eps = episodes.loc[episodes['id'].eq(pid)].sort_values('revision_id')
        tr, sq = traces[pid], seqs[pid]
        tl, texts = kd['time_list'], kd['text_list']
        valid = [finite_float(t) for t in tl if np.isfinite(finite_float(t))]
        session_start = valid[0] if valid else np.nan
        session_end = valid[-1] if valid else np.nan
        aligned = len(tr) == len(tl)
        if aligned and len(tr):
            a = np.asarray([finite_float(t) for t in tl])
            b = pd.to_numeric(tr['time_ms'], errors='coerce').to_numpy(float)
            mask = np.isfinite(a) & np.isfinite(b)
            aligned = bool(np.array_equal(np.isfinite(a), np.isfinite(b))
                           and np.allclose(a[mask], b[mask], rtol=0, atol=1e-9))
        covered = set()
        crows = []
        for r in eps.itertuples(index=False):
            s, e = int(r.index_start), int(r.index_end)
            lo, hi = sorted((s, e))
            nd, ni = episode_character_counts(texts, s, e)
            nbs = np.nan
            if aligned and not tr.empty:
                nbs = int(tr.loc[
                    tr['down_index'].between(lo, hi, inclusive='both')
                    & tr['key'].astype(str).str.lower().eq('backspace')].shape[0])
            pause = tfs = np.nan
            first_edit = s + 1
            if (first_edit < len(texts) and len(str(texts[first_edit])) < len(str(texts[s]))
                    and aligned):
                tb, tf = finite_float(tl[s]), finite_float(tl[first_edit])
                if np.isfinite(tb) and np.isfinite(tf) and tf >= tb:
                    pause = (tf - tb) / 1000.0
                if np.isfinite(tf) and np.isfinite(session_start) and tf >= session_start:
                    tfs = (tf - session_start) / 1000.0
            rec = {
                'id': pid, 'revision_id': int(r.revision_id),
                'conijn_n_backspaces': nbs,
                'conijn_n_characters_deleted': nd,
                'conijn_n_characters_inserted': ni,
                'conijn_time_from_start_process_sec': tfs,
                'conijn_pause_before_revision_sec': pause,
            }
            crows.append(rec)
            conijn_rows.append(rec)
            if not tr.empty:
                ids = tr.loc[
                    tr['down_index'].between(lo, hi, inclusive='both')
                    & tr['backspace_sequence_id'].notna(),
                    'backspace_sequence_id'].astype(int).tolist()
                covered.update(ids)
        cg = pd.DataFrame(crows)
        def med(col):
            if cg.empty:
                return np.nan
            x = pd.to_numeric(cg[col], errors='coerce').dropna()
            return float(x.median()) if len(x) else np.nan

        revision_seq = (sq.loc[sq['backspace_sequence_id'].isin(covered)]
                        .drop_duplicates('backspace_sequence_id') if not sq.empty else pd.DataFrame())
        nseq = len(revision_seq)
        ratio_gt3 = float(revision_seq['is_gt3'].mean()) if nseq else np.nan
        ngt1000 = int(revision_seq['gt1000ms_qualifies'].sum()) if nseq else 0
        tte = (pd.to_numeric(revision_seq['time_to_edit_ms'], errors='coerce').dropna()
               if nseq else pd.Series(dtype=float))
        mean_tte = float(tte.mean()) if len(tte) else np.nan
        # The original script used an additional Pacquetet trace with the same
        # physical contiguous-Backspace segmentation, linked by sequence start.
        # Its two model features are computed here from the equivalent sequence
        # start indices and lengths, without altering their definitions.
        revision_sequence_ids = set()
        if not sq.empty:
            for r in eps.itertuples(index=False):
                lo, hi = sorted((int(r.index_start), int(r.index_end)))
                revision_sequence_ids.update(sq.loc[
                    sq['sequence_start_down_index'].between(lo, hi, inclusive='both'),
                    'backspace_sequence_id'].astype(int).tolist())
        pacquetet_seq = (sq.loc[sq['backspace_sequence_id'].isin(revision_sequence_ids)]
                         .drop_duplicates('backspace_sequence_id') if not sq.empty else pd.DataFrame())
        n_pacquetet_seq = len(pacquetet_seq)
        mean_bs = (float(pd.to_numeric(pacquetet_seq['n_backspaces'], errors='coerce').mean())
                   if n_pacquetet_seq else np.nan)
        nrev = len(eps)
        total_rev = (union_duration_sec(eps['revision_start_time_ms'],
                                        eps['revision_end_time_ms']) if nrev else 0.0)
        elapsed = ((session_end - session_start) / 1000.0
                   if np.isfinite(session_start) and np.isfinite(session_end)
                   and session_end > session_start else np.nan)
        ratio_time = total_rev / elapsed if pd.notna(elapsed) and elapsed > 0 else np.nan
        starts = pd.to_numeric(eps['revision_start_time_ms'], errors='coerce').dropna().sort_values()
        if len(starts) >= 2:
            intervals = starts.diff().dropna() / 1000.0
            regularity = float(intervals.std(ddof=1)) if len(intervals) >= 2 else np.nan
        else:
            regularity = np.nan
        writing_rows.append({
            'id': pid, 'n_revision_episodes': nrev,
            'total_revision_time_ratio': ratio_time,
            'revision_regularity_sec_sd': regularity,
            'ratio_backspace_seq_gt3_in_revisions': ratio_gt3,
            NEW_GT1000: ngt1000,
            'mean_time_to_edit_ms_in_revisions': mean_tte,
            'median_backspaces_per_revision': med('conijn_n_backspaces'),
            'median_chars_deleted_per_revision': med('conijn_n_characters_deleted'),
            'median_chars_inserted_per_revision': med('conijn_n_characters_inserted'),
            'median_time_from_start_process_sec': med('conijn_time_from_start_process_sec'),
            'median_pause_before_revision_sec': med('conijn_pause_before_revision_sec'),
            'n_backspacing_sequences_in_revisions': n_pacquetet_seq,
            'mean_backspaces_per_sequence': mean_bs,
        })
    pd.DataFrame(conijn_rows).to_csv(OUT / '04_Conijn_episode_features.csv', index=False)
    feat_all = pd.DataFrame(writing_rows)
    feat_all.to_csv(OUT / '05a_Revision13_all_validated_writings.csv', index=False)

    # No Task-1 validation exclusions. All Task-2 valid participants retained.
    feat = feat_all.copy().reset_index(drop=True)
    if 'answer' not in dlla.columns:
        raise KeyError("DLLA missing required final essay column 'answer'.")
    dlla['text_length_words'] = dlla['answer'].fillna('').astype(str).str.split().str.len()
    feat = feat.merge(dlla[['id', 'text_length_words']], on='id', how='left',
                      validate='one_to_one')
    feat['revisions_per_100_words'] = (
        feat['n_revision_episodes'] / feat['text_length_words'].replace(0, np.nan) * 100.0)
    feat.to_csv(OUT / '05_Revision13_writing_level_features.csv', index=False)

    # 5. Composite CEFR labels (unchanged).
    marks = ['mark_a0', 'mark_h1', 'mark_h2', 'mark_h3']
    for col in marks:
        if col not in dlla.columns:
            raise KeyError(f'DLLA missing {col}')
        dlla[col] = pd.to_numeric(dlla[col], errors='coerce')
    dlla['cefr_composite_mean'] = dlla[marks].mean(axis=1)
    dlla['cefr_composite_rounded'] = np.round(dlla['cefr_composite_mean'])
    dlla['CEFR_target'] = dlla['cefr_composite_rounded'].apply(to_cefr)
    original = dlla['CEFR_target'].value_counts().reindex(
        ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], fill_value=0)
    original.to_csv(OUT / '06_composite_CEFR_distribution_before_feature_merge.csv', header=['n'])
    data = feat.merge(dlla[['id', 'CEFR_target', 'cefr_composite_mean',
                            'cefr_composite_rounded']],
                      on='id', how='inner', validate='one_to_one')
    data = data.loc[data['CEFR_target'].isin(LEVELS)].reset_index(drop=True)
    data['cefr_numeric'] = data['CEFR_target'].map(YMAP).astype(int)
    data.to_csv(OUT / '07_model_ready_common_population.csv', index=False)

    audit = pd.DataFrame({
        'observed_original': original.astype(int),
        'expected_original_reference': pd.Series({
            'A1': 0, 'A2': 4, 'B1': 111, 'B2': 549, 'C1': 323, 'C2': 19}),
    })
    audit.to_csv(OUT / '08_fidelity_audit.csv')
    print('Original CEFR distribution:', original.to_dict())
    print('Final Task-2-only model population:', len(data))
    print('Model CEFR distribution:', data['CEFR_target'].value_counts().reindex(LEVELS, fill_value=0).to_dict())
    if data.empty or data['id'].duplicated().any():
        raise RuntimeError('Model data empty or duplicate IDs found.')
    bad_word_counts = data['text_length_words'].isna() | (data['text_length_words'] <= 0)
    if bad_word_counts.any():
        bad_ids = data.loc[bad_word_counts, 'id'].tolist()
        pd.DataFrame({'id': bad_ids}).to_csv(OUT / 'invalid_final_word_counts.csv', index=False)
        raise RuntimeError(f'{len(bad_ids)} model writings have missing/zero final essay word counts; inspect invalid_final_word_counts.csv.')

    # 6. Primary four-class model only (B1, B2, C1, C2).
    run_model_analysis(data, LEVELS, OUT / 'Primary_4_levels', 'primary')
    print(f'\nOutputs saved to: {OUT.resolve()}')


if __name__ == '__main__':
    main()
