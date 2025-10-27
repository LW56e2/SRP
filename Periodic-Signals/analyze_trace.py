import argparse, os, json, csv, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except Exception: # pylint can go hug itself
    PIL_OK = False

DIVS_Y = 8   # 8 vertical divisions on the oscilloscope given to us
DIVS_X = 10  # 10 horizontal divisions

def load_points(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        xs = df["x_px"].to_numpy(float)
        ys = df["y_px"].to_numpy(float)
    elif path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        xs = np.array(data["x_px"], dtype=float)
        ys = np.array(data["y_px"], dtype=float)
    else:
        raise SystemExit("Input must be .csv or .json with x_px,y_px")
    # Sort by x (just in case)
    idx = np.argsort(xs)
    return xs[idx], ys[idx]



def moving_median(x, k):
    if k <= 1:
        return x.copy()

    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i+k])
    return out

def moving_mean(x, k):
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(xp, kernel, mode="valid")

def find_local_extrema(y, min_prominence=0.0, neighborhood=0):
    n = len(y)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    dy_prev = y[1:-1] - y[:-2]
    dy_next = y[1:-1] - y[2:]
    cand = np.arange(1, n-1)
    peaks = cand[(dy_prev >= 0) & (dy_next >= 0)]
    troughs = cand[(dy_prev <= 0) & (dy_next <= 0)]

    if min_prominence > 0:
        win = max(11, (n // 200) * 2 + 1)
        base = moving_median(y, win)
        prom = np.abs(y - base)

        span = np.percentile(y, 95) - np.percentile(y, 5)
        prom_floor = max(min_prominence, float(getattr(find_local_extrema, "_prom_frac", 0.25)) * span)
        peaks = np.array([i for i in peaks if prom[i] >= prom_floor], dtype=int)
        troughs = np.array([i for i in troughs if prom[i] >= prom_floor], dtype=int)

    if neighborhood and neighborhood > 0:
        peaks   = _nms_extrema(y, peaks,   neighborhood, keep="max")
        troughs = _nms_extrema(y, troughs, neighborhood, keep="min")

    return peaks, troughs


def robust_spacing(indices):
    if len(indices) < 3:
        return None, None
    d = np.diff(indices).astype(float)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) if len(d) > 1 else 0.0
    score = (mad / med) if med > 0 else np.inf
    return med, score

def _nms_extrema(y, idxs, neighborhood, keep="max"):
    if len(idxs) == 0 or neighborhood <= 0:
        return idxs
    idxs = np.array(idxs, dtype=int)
    order = np.argsort(y[idxs])  # ascending by value
    if keep == "max":
        order = order[::-1]       # process highest first
    kept = []
    taken = np.zeros(len(y), dtype=bool)
    half = int(neighborhood)
    for j in order:
        i = idxs[j]
        if taken[i]:
            continue
        kept.append(i)
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        taken[lo:hi] = True
        taken[i] = True
    return np.array(sorted(kept), dtype=int)


def estimate_period_from_extrema(xs_px, y_V, seconds_per_pixel, args, i_pk_hint=None, i_tr_hint=None):
    prom = np.std(y_V) * 0.1
    peaks, troughs = find_local_extrema(
        y_V, min_prominence=prom,
        neighborhood=args.extrema_neighborhood
    )

    candidates = []  # (period_s, method, score)

    # 1) p2p
    med_pp, score_pp = robust_spacing(peaks)
    if med_pp is not None and np.isfinite(score_pp):
        candidates.append((med_pp * seconds_per_pixel, "peak_to_peak", score_pp))

    # 2) t2t
    med_tt, score_tt = robust_spacing(troughs)
    if med_tt is not None and np.isfinite(score_tt):
        candidates.append((med_tt * seconds_per_pixel, "trough_to_trough", score_tt))

    # 3) peak to trough
    if len(peaks) >= 1 and len(troughs) >= 1:
        diffs = []
        ti = 0
        for ip in peaks:
            while ti < len(troughs) and troughs[ti] <= ip:
                ti += 1
            if ti < len(troughs):
                diffs.append(troughs[ti] - ip)
        if len(diffs) >= 2:
            diffs = np.array(diffs, dtype=float)
            med = np.median(diffs)
            mad = np.median(np.abs(diffs - med)) if len(diffs) > 1 else 0.0
            score = (mad / med) if med > 0 else np.inf
            candidates.append((2.0 * med * seconds_per_pixel, "peak_to_trough_x2", score))

    if candidates:
        period_s, method, _ = min(candidates, key=lambda c: c[2])
        return float(period_s), method

    if i_pk_hint is not None and i_tr_hint is not None:
        half_period_px = abs(i_pk_hint - i_tr_hint)
        period_s = 2.0 * half_period_px * seconds_per_pixel
        return float(period_s), "fallback_peak_trough_x2"

    period_s = (xs_px[-1] - xs_px[0]) * seconds_per_pixel
    return float(period_s), "fallback_full_width"

def estimate_period_from_edges(xs_px, y_V, seconds_per_pixel,
                               sign="auto", thresh_pctl=95.0, min_sep_px=100):
    # derivative per pixel
    dy = np.diff(y_V)
    mag = np.abs(dy)
    thr = np.percentile(mag, thresh_pctl)

    rises = np.where((dy > 0) & (mag >= thr))[0]
    falls = np.where((dy < 0) & (mag >= thr))[0]

    def nms_keep(idxs, min_sep):
        kept = []
        last = -1e9
        for i in idxs:
            if i - last >= min_sep:
                kept.append(i)
                last = i
        return np.array(kept, dtype=int)

    rises = nms_keep(rises, min_sep_px)
    falls = nms_keep(falls, min_sep_px)

    if sign == "rise":
        chosen, used = rises, "edge_rise"
    elif sign == "fall":
        chosen, used = falls, "edge_fall"
    else:
        if len(rises) >= len(falls) and len(rises) >= 2:
            chosen, used = rises, "edge_rise_auto"
        elif len(falls) >= 2:
            chosen, used = falls, "edge_fall_auto"
        else:
            both = np.sort(np.concatenate([rises, falls]))
            chosen, used = both, "edge_both_auto"

    # need at least two edges to get a spacing
    if len(chosen) >= 3:
        med_px, score = robust_spacing(chosen)
        if med_px is not None and np.isfinite(score):
            return float(med_px * seconds_per_pixel), used, chosen
    if len(chosen) >= 2:
        # just use first two as a single-spacing estimate
        return float((chosen[1] - chosen[0]) * seconds_per_pixel), used + "_2only", chosen[:2]

    return None, "edge_failed", np.array([], dtype=int)


def estimate_vpp_from_peak_trough(y, args):
    peaks, troughs = find_local_extrema(
        y, min_prominence=(np.std(y) * 0.1),
        neighborhood=args.extrema_neighborhood
    )

    if len(peaks) == 0 or len(troughs) == 0:
        i_max = int(np.argmax(y))
        i_min = int(np.argmin(y))
        vpp = float(np.abs(y[i_max] - y[i_min]))
        return vpp, (i_max, i_min)


    best = None
    best_drop = -np.inf
    tset = set(troughs.tolist())
    for ip in peaks:
        it_candidates = [it for it in troughs if it > ip]
        if not it_candidates:
            continue
        it = it_candidates[0]
        drop = y[ip] - y[it]
        if drop > best_drop:
            best_drop = drop
            best = (ip, it)
    if best is None:
        # fallback to global
        i_max = int(np.argmax(y))
        i_min = int(np.argmin(y))
        vpp = float(np.abs(y[i_max] - y[i_min]))
        return vpp, (i_max, i_min)

    ip, it = best
    vpp = float(abs(y[ip] - y[it]))
    return vpp, (ip, it)

def _first_crossing(t, v, level, rising=True, i0=0, i1=None):
    if i1 is None:
        i1 = len(v) - 1
    if i1 <= i0:
        return None, None, None
    rng = range(i0, i1)
    for i in rng:
        v0, v1 = v[i], v[i+1]
        if rising:
            hit = (v0 <= level) and (v1 >= level)
        else:
            hit = (v0 >= level) and (v1 <= level)
        if hit and v1 != v0:
            # linear interp
            a = (level - v0) / (v1 - v0)
            t_cross = t[i] + a * (t[i+1] - t[i])
            v_cross = level
            return t_cross, v_cross, i
    return None, None, None

def _pick_segment_for_tau(y_V, edge_indices, i_pk, i_tr):
    if i_pk is None or i_tr is None:
        return (0, len(y_V)-1)
    if i_tr < i_pk:
        return (int(i_tr), int(i_pk))   # charging (up)
    else:
        return (int(i_pk), int(i_tr))   # discharging (down)


def estimate_tau(t_s, y_V, i_start, i_end, method="63", sign="auto"):
    i_start = max(0, min(i_start, len(y_V)-1))
    i_end   = max(0, min(i_end,   len(y_V)-1))
    if i_end <= i_start:
        return {"tau_s": float("nan"), "tau_method": "tau_invalid", "tau_mark": None, "levels": {}}

    v0 = float(y_V[i_start])
    v1 = float(y_V[i_end])
    t0 = float(t_s[i_start])

    dv = v1 - v0
    if sign == "auto":
        rising = dv > 0
    elif sign == "charge":
        rising = True
    else:
        rising = False  # "discharge"

    i0, i1 = i_start, i_end
    # ensure [i0, i1] goes in the assumed direction
    if rising and y_V[i1] < y_V[i0]:
        i0, i1 = i1, i0
        v0, v1, t0, dv = float(y_V[i0]), float(y_V[i1]), float(t_s[i0]), float(y_V[i1]-y_V[i0])
    if (not rising) and y_V[i1] > y_V[i0]:
        i0, i1 = i1, i0
        v0, v1, t0, dv = float(y_V[i0]), float(y_V[i1]), float(t_s[i0]), float(y_V[i1]-y_V[i0])

    out = {"tau_s": float("nan"), "tau_method": None, "tau_mark": None, "levels": {}}

    if method == "63":
        # charging: 63.2% up from v0 toward v1; discharging: 36.8% down from v0 toward v1
        if rising:
            level = v0 + 0.632 * (v1 - v0)
            t_cross, v_cross, _ = _first_crossing(t_s, y_V, level, rising=True, i0=i0, i1=i1)
        else:
            level = v0 - 0.632 * (v0 - v1)  # equivalently v1 + 0.368*(v0 - v1)
            t_cross, v_cross, _ = _first_crossing(t_s, y_V, level, rising=False, i0=i0, i1=i1)
        if t_cross is not None:
            out["tau_s"] = float(t_cross - t0)
            out["tau_method"] = "63"
            out["tau_mark"] = (t_cross, v_cross)
            out["levels"] = {"level_63": level, "v0": v0, "v1": v1, "t0": t0}
        else:
            out["tau_method"] = "63_failed"
        return out

    elif method == "10-90":
        if rising:
            level10 = v0 + 0.10 * (v1 - v0)
            level90 = v0 + 0.90 * (v1 - v0)
            t10, _, _ = _first_crossing(t_s, y_V, level10, rising=True,  i0=i0, i1=i1)
            t90, _, _ = _first_crossing(t_s, y_V, level90, rising=True,  i0=i0, i1=i1)
        else:
            level10 = v0 - 0.10 * (v0 - v1)  # 90% of the way down
            level90 = v0 - 0.90 * (v0 - v1)  # 10% of the way down
            t10, _, _ = _first_crossing(t_s, y_V, level10, rising=False, i0=i0, i1=i1)
            t90, _, _ = _first_crossing(t_s, y_V, level90, rising=False, i0=i0, i1=i1)

        if (t10 is not None) and (t90 is not None) and (t90 > t10):
            tau = (t90 - t10) / 2.2
            out["tau_s"] = float(tau)
            out["tau_method"] = "10-90"
            out["tau_mark"] = ((t10, level10), (t90, level90))
            out["levels"] = {"level10": level10, "level90": level90, "v0": v0, "v1": v1}
        else:
            out["tau_method"] = "10-90_failed"
        return out

    else:
        out["tau_method"] = "tau_unknown_method"
        return out

def _pick_segment_for_tau_curvefit(t_s, y_V, edge_indices, i_pk, i_tr):
    def volts_from_extrema(rising: bool):
        if (i_pk is not None) and (i_tr is not None):
            if rising:
                return float(y_V[int(i_tr)]), float(y_V[int(i_pk)])  # trough -> peak
            else:
                return float(y_V[int(i_pk)]), float(y_V[int(i_tr)])  # peak -> trough
        vmin, vmax = float(np.min(y_V)), float(np.max(y_V))
        return (vmin, vmax) if rising else (vmax, vmin)

    # EDGES for time window
    if edge_indices is not None and len(edge_indices) >= 2:
        e0, e1 = int(edge_indices[0]), int(edge_indices[1])
        mid = e0 + (e1 - e0)//2
        i0, i1 = min(e0, mid), max(e0, mid)
        if i1 - i0 < 20:
            i1 = min(len(y_V) - 1, i0 + 20)
        rising = (y_V[i1] - y_V[i0]) >= 0
        V0_fix, Vinf_fix = volts_from_extrema(rising)
    else:
        # Fall back to extrema for time window
        if (i_pk is not None) and (i_tr is not None):
            if i_tr < i_pk:  # charging
                i0, i1 = int(i_tr), int(i_pk)
                V0_fix, Vinf_fix = volts_from_extrema(True)
            else:            # discharging
                i0, i1 = int(i_pk), int(i_tr)
                V0_fix, Vinf_fix = volts_from_extrema(False)
        else:
            i0, i1 = 0, len(y_V) - 1
            rising = (y_V[i1] - y_V[i0]) >= 0
            V0_fix, Vinf_fix = volts_from_extrema(rising)

    seg = y_V[i0:i1+1]
    j_rel = int(np.argmin(np.abs(seg - V0_fix)))
    j0 = i0 + j_rel
    return i0, i1, V0_fix, Vinf_fix, j0




def _exp_general(t, Vinf, V0, tau):

    return Vinf + (V0 - Vinf) * np.exp(-t / np.clip(tau, 1e-18, np.inf))

def estimate_tau_curvefit(t_s, y_V, i0, i1, V0_fix, Vinf_fix, j0):
    try:
        from scipy.optimize import least_squares
    except Exception:
        return {"tau_s": float("nan"), "tau_method": "curvefit_scipy_missing",
                "tau_mark": None, "levels": {}, "fit_segment": None}

    i0 = max(0, min(i0, len(y_V) - 2))
    i1 = max(i0 + 1, min(i1, len(y_V) - 1))
    j0 = int(np.clip(j0, i0, i1))

    t = t_s[i0:i1+1].astype(float)
    v = y_V[i0:i1+1].astype(float)

    t0 = float(t_s[j0])
    V0 = float(V0_fix)
    Vinf = float(Vinf_fix)
    def model(tau):
        # VERY IMPORTANT ACTUAL MODEL
        tau = np.maximum(tau, 1e-18)
        return Vinf + (V0 - Vinf) * np.exp(-(t - t0) / tau)

    def residual(tau):
        return model(tau) - v

    tspan = max(1e-18, float(t[-1] - t[0]))
    tau0 = tspan / 3.0

    res = least_squares(lambda x: residual(x[0]), x0=np.array([tau0]),
                        bounds=(1e-12, np.inf), loss="soft_l1")
    tau = float(res.x[0])

    v_fit = model(tau)
    return {
        "tau_s": tau,
        "tau_method": "curvefit_fixed_V_robust",
        "tau_mark": (t0, V0),  # mark where we anchored time to V0
        "levels": {"Vinf_fix": Vinf, "V0_fix": V0, "t0": t0},
        "fit_segment": {"t": t.copy(), "v_fit": v_fit.copy(), "i0": int(i0), "i1": int(i1), "j0": int(j0)}
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points_path", help=".CSV or .JSON with x_px,y_px")
    ap.add_argument("--vdiv", type=float, required=True, help="Volts per division")
    ap.add_argument("--tdiv", type=float, required=True, help="Seconds per division")
    ap.add_argument("--image", type=str, default=None, help="Optional: path to the adjusted screen image to get width/height")
    ap.add_argument("--height-px", type=int, default=None, help="screen height in pixels")
    ap.add_argument("--width-px", type=int, default=None, help="screen width in pixels")
    ap.add_argument("--median-k", type=int, default=5, help="Median filter window (odd)")
    ap.add_argument("--mean-k", type=int, default=5, help="Moving average window (odd)")

    ap.add_argument("--extrema-neighborhood", type=int, default=20,
                    help="peaks/troughs local for NMS")

    ap.add_argument("--edge-period", action="store_true",
                    help="Estimate period from steep-edge spacing instead of peaks/troughs")
    ap.add_argument("--edge-sign", choices=["rise", "fall", "auto"], default="auto",
                    help="which edges to use for spacing")
    ap.add_argument("--edge-thresh-pctl", type=float, default=95.0,
                    help="Percentile of dv/dx to define a steep edge")
    ap.add_argument("--edge-min-sep-px", type=int, default=20,
                    help="Minimum pixel separation")

    ap.add_argument("--tau", action="store_true",
                    help="Estimate RC time constant tau from a single rise/fall ")
    ap.add_argument("--tau-method", choices=["63", "10-90", "curvefit"], default="63",
                    help="tau estimation method")
    ap.add_argument("--tau-sign", choices=["auto", "charge", "discharge"], default="auto",
                    help="Up/down")

    ap.add_argument("--pt-periods", type=float, default=0.5,
                    help="Assume the detected peak to trough has this many periods")

    ap.add_argument("--detrend-slope", type=float, default=0.0,
                    help="Manually remove linear drift")
    ap.add_argument("--auto-detrend", action="store_true",
                    help="Estimate and remove best-fit linear drift")


    args = ap.parse_args()

    edge_indices = None

    xs_px, ys_px = load_points(args.points_path)

    W = args.width_px
    H = args.height_px
    if args.image:
        if not PIL_OK:
            raise SystemExit("Pillow not available")
        img = Image.open(args.image)
        W, H = img.size
    else:
        if W is None:
            W = int(np.nanmax(xs_px)) + 1
        if H is None:
            raise SystemExit("Need height and wid")

    # Pixels per division
    px_per_div_x = W / DIVS_X
    px_per_div_y = H / DIVS_Y

    # I hate units so much
    seconds_per_pixel = args.tdiv / px_per_div_x
    volts_per_pixel   = args.vdiv / px_per_div_y

    # Smooth y
    y_px = ys_px.copy()
    if args.median_k and args.median_k % 2 == 1 and args.median_k > 1:
        y_px = moving_median(y_px, args.median_k)
    if args.mean_k and args.mean_k % 2 == 1 and args.mean_k > 1:
        y_px = moving_mean(y_px, args.mean_k)

    # Convert to volts, centered around 0V
    y_V = -(y_px) * volts_per_pixel
    vmax, vmin = np.nanmax(y_V), np.nanmin(y_V)
    y_V -= (vmax + vmin) / 2.0

    # Time axis in seconds
    t_s = xs_px * seconds_per_pixel

    if args.detrend_slope != 0.0:
        y_V = y_V - args.detrend_slope * (t_s - t_s[0])
    elif args.auto_detrend:
        a, b = np.polyfit(t_s, y_V, 1)  # a=V/s
        print(a)
        y_V = y_V - a * (t_s - t_s[0])

    # get amplitude
    vpp, (i_pk, i_tr) = estimate_vpp_from_peak_trough(y_V, args=args)
    amplitude = 0.5 * vpp

    # Period estimation
    edge_indices = None
    if args.edge_period:
        period_s, period_method, edge_idx = estimate_period_from_edges(
            xs_px, y_V, seconds_per_pixel, args=args,
            sign=args.edge_sign,
            thresh_pctl=args.edge_thresh_pctl,
            min_sep_px=args.edge_min_sep_px
        )
        if period_s is None:
            period_s, period_method = estimate_period_from_extrema(
                xs_px, y_V, seconds_per_pixel, args=args, i_pk_hint=i_pk, i_tr_hint=i_tr
            )
        else:
            edge_indices = edge_idx
    else:
        period_s, period_method = estimate_period_from_extrema(
            xs_px, y_V, seconds_per_pixel, args=args, i_pk_hint=i_pk, i_tr_hint=i_tr
        )

    freq_hz = (1.0 / period_s) if period_s and period_s > 0 else float('nan')

    if args.pt_periods != 0.5 and (i_pk is not None) and (i_tr is not None):
        dt = abs(i_pk - i_tr) * seconds_per_pixel
        if args.pt_periods > 0:
            period_s = dt / args.pt_periods
            period_method = f"peak_trough_override_K={args.pt_periods}"
            freq_hz = (1.0 / period_s) if period_s > 0 else float('nan')

    tau_info = {"tau_s": float("nan"), "tau_method": None, "tau_mark": None, "levels": {}, "fit_segment": None}
    tau_segment = []
    if args.tau:
        if args.tau_method == "curvefit":
            i_start, i_end, V0_fix, Vinf_fix, j0 = _pick_segment_for_tau_curvefit(
                t_s, y_V, edge_indices, i_pk, i_tr
            )
            tau_info = estimate_tau_curvefit(t_s, y_V, i_start, i_end, V0_fix, Vinf_fix, j0)

        else:
            i_start, i_end = _pick_segment_for_tau(y_V, edge_indices, i_pk, i_tr)
            tau_info = estimate_tau(
                t_s, y_V, i_start, i_end,
                method=args.tau_method,
                sign=args.tau_sign
            )
            tau_segment = [int(i_start), int(i_end)]

    stem = Path(args.points_path).with_suffix("").name
    out_points = f"{stem}_points_time_volts.csv"
    pd.DataFrame({"t_s": t_s, "v": y_V}).to_csv(out_points, index=False)

    metrics = {
        "pixels_per_div_x": px_per_div_x,
        "pixels_per_div_y": px_per_div_y,
        "seconds_per_pixel": seconds_per_pixel,
        "volts_per_pixel": volts_per_pixel,
        "period_s": period_s,
        "frequency_hz": freq_hz,
        "period_method": period_method,
        "edge_indices": edge_indices.tolist() if edge_indices is not None else [],
        "vpp": vpp,
        "amplitude": amplitude,
        "peak_index": int(i_pk),
        "trough_index": int(i_tr),
        "tau_s": tau_info.get("tau_s"),
        "tau_method": tau_info.get("tau_method"),
        "tau_segment": tau_segment,
        "pt_periods": args.pt_periods,
        "detrend_slope_manual": args.detrend_slope,
        "auto_detrend_used": bool(args.auto_detrend and args.detrend_slope == 0.0),
    }
    out_metrics = f"{stem}_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    overlay_path = None
    if args.image and PIL_OK:
        img = Image.open(args.image).convert("RGB")
        draw = ImageDraw.Draw(img)
        pts = list(zip(xs_px.tolist(), ys_px.tolist()))
        for i in range(len(pts)-1):
            draw.line([pts[i], pts[i+1]], fill=(255,0,0), width=2)
        overlay_path = f"{stem}_overlay2.png"
        img.save(overlay_path)

    plt.figure()
    plt.plot(t_s, y_V, linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Extracted trace (time vs volts)")
    if 0 <= i_pk < len(t_s) and 0 <= i_tr < len(t_s):
        plt.scatter([t_s[i_pk], t_s[i_tr]],[y_V[i_pk], y_V[i_tr]])
    if edge_indices is not None and len(edge_indices) > 0:
        plt.scatter(t_s[edge_indices], y_V[edge_indices])
    # Tau marks
    if args.tau and tau_info.get("tau_mark") is not None and args.tau_method != "curvefit":
        mark = tau_info["tau_mark"]
        if isinstance(mark, tuple) and len(mark) == 2 and isinstance(mark[0], tuple):
            (t10, v10), (t90, v90) = mark
            plt.scatter([t10, t90], [v10, v90])
        elif isinstance(mark, tuple) and len(mark) == 2:
            t_cross, v_cross = mark
            plt.scatter([t_cross], [v_cross])

    if args.tau and args.tau_method.startswith("curvefit") and tau_info.get("fit_segment"):
        seg = tau_info["fit_segment"]
        plt.plot(seg["t"], seg["v_fit"], linewidth=1.5)
        # t0 (where it was aligned to V0)
        if "j0" in seg:
            plt.scatter([t_s[seg["j0"]]], [y_V[seg["j0"]]])

    plt.tight_layout()
    plt.savefig(f"{stem}_curve.png", dpi=150)
    plt.show()

    print(f"Saved:\n- {out_points}\n- {out_metrics}")
    if overlay_path:
        print(f"- {overlay_path}")
    print(f"- {stem}_curve.png")

if __name__ == "__main__":
    main()
