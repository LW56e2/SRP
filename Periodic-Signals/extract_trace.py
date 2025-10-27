# ONLY USE ON PERSPECTIVE-CORRECTED IMAGES
import sys, json, csv, os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def top_edge_from_mask(mask):
    h, w = mask.shape
    ys = np.full(w, np.nan, dtype=float)

    idxs = np.argmax(mask, axis=0)
    has  = (mask[idxs, np.arange(w)] > 0) # I spent 1 hour debugging this

    ys[has] = idxs[has].astype(float)
    return ys

def main(path):
    if len(sys.argv) > 1:
        path = sys.argv[1]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read: {path}")

    bgr = img.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # define green color range in HSV
    lower = np.array([40, 30, 30], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)


    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    cv2.imwrite("debug_mask_raw.png", mask)
    print("mask coverage:", np.mean(mask > 0))

    # Find top edge
    ys = top_edge_from_mask(mask)
    xs = np.arange(mask.shape[1], dtype=float)

    # drop NaNs
    valid = ~np.isnan(ys)
    xs_v = xs[valid]
    ys_v = ys[valid]

    if xs_v.size > 5:
        y32 = ys_v.astype(np.float32).reshape(-1, 1)
        ys_v = cv2.medianBlur(y32, 5).ravel().astype(float)

    stem = os.path.splitext(os.path.basename(path))[0]
    csv_path = f"{stem}_trace.csv"
    json_path = f"{stem}_trace.json"
    with open(csv_path, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["x_px", "y_px"])
        for x, y in zip(xs_v, ys_v):
            wri.writerow([float(x), float(y)])
    with open(json_path, "w") as f:
        json.dump({"x_px": xs_v.tolist(), "y_px": ys_v.tolist()}, f)

    overlay = bgr.copy()
    pts = np.stack([xs_v, ys_v], axis=1).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(overlay, [pts], isClosed=False, color=(0,0,255), thickness=2)  # red
    overlay_path = f"{stem}_overlay.png"
    cv2.imwrite(overlay_path, overlay)

    print(f"saved: {csv_path}, {json_path}, {overlay_path}")


    plt.figure()
    plt.plot(xs_v, ys_v, linewidth=1.5)
    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y, top=0)")
    plt.title("Extracted Wave (top edge)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_trace.py image.png")
        sys.exit(1)
    main(sys.argv[1])
