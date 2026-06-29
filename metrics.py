import numpy as np


def frame_prf(pred_bin, gt):
    """
    Per-frame precision, recall, F1.
    pred_bin : binary numpy array (0/1), shape [H, W]
    gt       : same shape; -1 = ignore, 0 = bg, 1 = fg
    Returns (precision, recall, f1) floats for this frame.
    """
    valid = gt != -1
    p = pred_bin[valid].astype(bool)
    g = gt[valid].astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def prf_from_counts(tp, fp, fn):
    """Micro-average P/R/F1 from cumulative counts (kept for reference)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def empty_prf_seq():
    """Per-sequence P/R/F1 bucket: one entry per frame, nanmean at the end."""
    return {'precision': [], 'recall': [], 'f1': []}


def mean_prf(prf_all_sequences):
    """Compute overall macro-average P/R/F1 from a {seq_name: empty_prf_seq()} dict."""
    ps, rs, fs = [], [], []
    for v in prf_all_sequences.values():
        ps.extend(v['precision'])
        rs.extend(v['recall'])
        fs.extend(v['f1'])
    if not ps:
        return 0.0, 0.0, 0.0
    import numpy as np
    return float(np.nanmean(ps)), float(np.nanmean(rs)), float(np.nanmean(fs))


def empty_prf_bucket():
    """Top-level bucket for main_tissue val: {seq_name: empty_prf_seq()}"""
    return {}
