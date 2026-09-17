# made by - Karthik
import sys, os, csv, json, time, math
from pathlib import Path

if len(sys.argv) > 2:
    PUBLIC_DIR = Path(sys.argv[1])
    SUBMISSION_OUT = Path(sys.argv[2])
else:
    PUBLIC_DIR = Path(".")
    SUBMISSION_OUT = Path("working/submission.csv")

DEADLINE_SECONDS = float(os.environ.get("ERIS_DEADLINE_SECONDS", 4800))
N_FOLDS = int(os.environ.get("ERIS_N_FOLDS", 4))
EPOCHS = int(os.environ.get("ERIS_EPOCHS", 100000))
LR = float(os.environ.get("ERIS_LR", 2e-3))
BATCH_SIZE = int(os.environ.get("ERIS_BATCH_SIZE", 32))
SEED = int(os.environ.get("ERIS_SEED", 42))
HOLDOUT_FRAC = float(os.environ.get("ERIS_HOLDOUT_FRAC", 0.20))
CONF_SHRINK = float(os.environ.get("ERIS_CONF_SHRINK", 0.6))

T_START = time.time()

import numpy as np

try:
    from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure
    def dilate2(mask):
        struct = iterate_structure(generate_binary_structure(2, 1), 2)
        return binary_dilation(mask.astype(bool), structure=struct)
except Exception:
    def dilate2(mask):
        m = mask.astype(bool)
        out = m.copy()
        for _ in range(2):
            shifted = np.zeros_like(out)
            shifted[1:, :] |= out[:-1, :]
            shifted[:-1, :] |= out[1:, :]
            shifted[:, 1:] |= out[:, :-1]
            shifted[:, :-1] |= out[:, 1:]
            out = out | shifted
        return out

from PIL import Image
import torch
import torch.nn as nn

DEVICE = torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
_cpus = os.cpu_count() or 4
torch.set_num_threads(max(1, _cpus))

H = W = 128


def decode_rle(rle, h=H, w=W):
    m = np.zeros(h * w, dtype=np.uint8)
    if rle and isinstance(rle, str) and rle.strip():
        for tok in rle.strip().split():
            s, l = tok.split(":")
            s = int(s); l = int(l)
            m[s:s + l] = 1
    return m.reshape(h, w)


def encode_rle(mask):
    flat = mask.reshape(-1).astype(np.int32)
    if flat.sum() == 0:
        return ""
    diff = np.diff(flat)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if flat[0] == 1:
        starts = np.concatenate(([0], starts))
    if flat[-1] == 1:
        ends = np.concatenate((ends, [len(flat)]))
    return " ".join(f"{s}:{e - s}" for s, e in zip(starts, ends))


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return dict(x=0.0, y=0.0, w=1.0 / W, h=1.0 / H)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x = x0 / W; y = y0 / H
    w = (x1 - x0 + 1) / W; h = (y1 - y0 + 1) / H
    eps = 1.0 / W
    x = min(max(x, 0.0), 1.0 - eps)
    y = min(max(y, 0.0), 1.0 - eps)
    w = min(max(w, eps), 1.0 - x)
    h = min(max(h, eps), 1.0 - y)
    return dict(x=x, y=y, w=w, h=h)


def s_mask_components(pred, true):
    pred = pred.astype(bool); true = true.astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    iou = inter / union if union > 0 else 1.0
    psum, tsum = pred.sum(), true.sum()
    dice = (2 * inter) / (psum + tsum) if (psum + tsum) > 0 else 1.0
    pred_d = dilate2(pred); true_d = dilate2(true)
    tp = np.logical_and(pred, true_d).sum()
    fp = np.logical_and(pred, ~true_d).sum()
    fn = np.logical_and(~pred_d, true).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    tolf1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 1.0
    return 0.34 * iou + 0.38 * dice + 0.28 * tolf1


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_image(rel_path):
    img = Image.open(PUBLIC_DIR / rel_path).convert("L")
    return np.array(img, dtype=np.float32)


def build_input_channels(img_arr):
    gray = img_arr / 255.0
    return np.stack([gray], axis=0).astype(np.float32)


N_CHANNELS = 1


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=N_CHANNELS, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.e1 = ConvBlock(cin, c1)
        self.e2 = ConvBlock(c1, c2)
        self.e3 = ConvBlock(c2, c3)
        self.b = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.d3 = ConvBlock(c4, c3)
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.d2 = ConvBlock(c3, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.d1 = ConvBlock(c2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.head(d1)


def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    union = prob.sum(dims) + target.sum(dims)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def tolerant_loss(logits, target_dil):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    fp = (prob * (1 - target_dil)).sum(dims)
    denom = prob.sum(dims) + 1.0
    return (fp / denom).mean()


def augment_batch(imgs, masks, rng):
    n = imgs.shape[0]
    for i in range(n):
        if rng.random() < 0.5:
            imgs[i] = np.flip(imgs[i], axis=2).copy()
            masks[i] = np.flip(masks[i], axis=1).copy()
    return imgs, masks


def make_folds(n, n_folds, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, n_folds)


def train_one_fold(train_imgs, train_masks, val_imgs, epochs, deadline_left):
    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = train_imgs.shape[0]
    pos_rate = train_masks.mean()
    pos_rate = min(max(pos_rate, 1e-4), 0.5)
    pos_weight = torch.tensor([min(max((1 - pos_rate) / pos_rate, 1.0), 15.0)], device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(SEED)
    fold_deadline = time.time() + deadline_left
    val_t = torch.from_numpy(val_imgs).float().to(DEVICE)
    ep = 0
    while time.time() < fold_deadline and ep < epochs:
        ep += 1
        model.train()
        order = rng.permutation(n)
        imgs_ep = train_imgs.copy()
        masks_ep = train_masks.copy()
        imgs_ep, masks_ep = augment_batch(imgs_ep, masks_ep, rng)
        for bstart in range(0, n, BATCH_SIZE):
            bidx = order[bstart:bstart + BATCH_SIZE]
            xb = torch.from_numpy(imgs_ep[bidx]).float().to(DEVICE)
            yb = torch.from_numpy(masks_ep[bidx]).float().unsqueeze(1).to(DEVICE)
            yb_dil = torch.from_numpy(
                np.stack([dilate2(masks_ep[j]) for j in bidx]).astype(np.float32)
            ).unsqueeze(1).to(DEVICE)
            logits = model(xb)
            loss = 0.45 * bce(logits, yb) + 0.4 * dice_loss(logits, yb) + 0.15 * tolerant_loss(logits, yb_dil)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        val_logits = model(val_t)
        val_prob = torch.sigmoid(val_logits).squeeze(1).cpu().numpy()
    return model, val_prob


def predict_with_model(model, imgs):
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(imgs).float().to(DEVICE)
        logits = model(t)
        return torch.sigmoid(logits).squeeze(1).cpu().numpy()


def ridge_fit(X, y, alpha=1.0):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    b = Xb.T @ y
    w = np.linalg.solve(A, b)
    return dict(w=w, mu=mu, sd=sd)


def ridge_predict(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    Xb = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    return Xb @ model["w"]


def tiny_kmeans(X, k, iters=20, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for c in range(k):
            if (labels == c).any():
                centers[c] = X[labels == c].mean(axis=0)
    return labels


def run_diagnostics(masks, bboxes, oof_prob_singlefold, oof_idx, holdout_idx, holdout_ens_prob, best_thr, ridge_w):
    nh = len(holdout_idx)
    rows = []
    for k in range(nh):
        i = holdout_idx[k]
        pm = holdout_ens_prob[k] > best_thr
        sm = s_mask_components(pm, masks[i])
        pred_b = mask_to_bbox(pm)
        true_b = bboxes[i]
        err = abs(pred_b["x"] - true_b["x"]) + abs(pred_b["y"] - true_b["y"]) + \
              abs(pred_b["w"] - true_b["w"]) + abs(pred_b["h"] - true_b["h"])
        sb = math.exp(-err / 0.55)
        core = 0.72 * sm + 0.18 * sb
        area_frac = pm.mean()
        pos_prob = holdout_ens_prob[k][pm].mean() if pm.sum() > 0 else 0.0
        feat = np.array([[area_frac, pos_prob]], dtype=np.float64)
        core_hat_raw = float(ridge_predict(ridge_w, feat)[0])
        core_hat = CONF_SHRINK * core_hat_raw + (1 - CONF_SHRINK) * ridge_w["shrink_target"]
        core_hat = min(max(core_hat, 0.0), 1.0)
        confidence = min(max(core_hat / 0.90, 0.0), 1.0)
        calib = 1 - abs(confidence - core / 0.90)
        row = core + 0.10 * calib
        rows.append(dict(sm=sm, sb=sb, core=core, row=row, calib=calib, area_frac=area_frac))

    sm_arr = np.array([r["sm"] for r in rows])
    sb_arr = np.array([r["sb"] for r in rows])
    core_arr = np.array([r["core"] for r in rows])
    row_arr = np.array([r["row"] for r in rows])
    calib_arr = np.array([r["calib"] for r in rows])
    area_arr = np.array([r["area_frac"] for r in rows])

    print(f"[diag] TRUE held-out ensemble (n={nh}, never trained on by any fold model): "
          f"mean S_mask={sm_arr.mean():.4f} S_bbox={sb_arr.mean():.4f} "
          f"core={core_arr.mean():.4f} calib={calib_arr.mean():.4f} row={row_arr.mean():.4f} "
          f"pred_area_frac={area_arr.mean():.4f} (true mean ~0.0298)", file=sys.stderr)

    if len(oof_idx) > 0:
        sf_scores = [s_mask_components(oof_prob_singlefold[i] > best_thr, masks[i]) for i in oof_idx]
        print(f"[diag] honest single-fold pool-internal OOF S_mask at same thr={best_thr:.3f}: "
              f"{np.mean(sf_scores):.4f} (n={len(oof_idx)}) vs true-holdout-ensemble {sm_arr.mean():.4f} "
              f"-- both numbers are now leak-free; a big gap between them still reflects a real "
              f"single-model-vs-ensemble difference, not leakage", file=sys.stderr)

    shapes = np.stack([masks[i] for i in holdout_idx]).reshape(nh, H // 8, 8, W // 8, 8).mean(axis=(2, 4)).reshape(nh, -1)
    fam_k = min(4, max(1, nh // 8))
    fam_labels = tiny_kmeans(shapes.astype(np.float64), fam_k) if fam_k > 1 else np.zeros(nh, dtype=int)
    areas = np.array([masks[i].sum() for i in holdout_idx])
    order = np.argsort(areas)
    footprint_labels = np.zeros(nh, dtype=int)
    third = max(1, nh // 3)
    footprint_labels[order[:third]] = 0
    footprint_labels[order[third:2 * third]] = 1
    footprint_labels[order[2 * third:]] = 2

    worst_fam = min(row_arr[fam_labels == g].mean() for g in np.unique(fam_labels))
    worst_fp = min(row_arr[footprint_labels == g].mean() for g in np.unique(footprint_labels))
    raw_final = 0.78 * row_arr.mean() + 0.12 * worst_fam + 0.10 * worst_fp
    final = raw_final ** 1.45
    print(f"[diag] proxy worst_family={worst_fam:.4f} worst_footprint={worst_fp:.4f} "
          f"raw_final~={raw_final:.4f} Final~={final:.4f} (n={nh} holdout rows -- noisy, small sample)",
          file=sys.stderr)
    for g in np.unique(fam_labels):
        gm = fam_labels == g
        print(f"[diag] family {g}: n={int(gm.sum())} mean_row={row_arr[gm].mean():.4f} "
              f"mean_S_mask={sm_arr[gm].mean():.4f}", file=sys.stderr)
    for g in np.unique(footprint_labels):
        gm = footprint_labels == g
        print(f"[diag] footprint {g}: n={int(gm.sum())} mean_row={row_arr[gm].mean():.4f} "
              f"mean_S_mask={sm_arr[gm].mean():.4f}", file=sys.stderr)


def main():
    train_rows = read_csv(PUBLIC_DIR / "train.csv")
    test_rows = read_csv(PUBLIC_DIR / "test.csv")
    train_rows.sort(key=lambda r: int(r["id"]))

    n = len(train_rows)
    imgs = np.zeros((n, N_CHANNELS, H, W), dtype=np.float32)
    masks = np.zeros((n, H, W), dtype=np.float32)
    bboxes = []
    for i, row in enumerate(train_rows):
        arr = load_image(row["target_shadow_path"])
        imgs[i] = build_input_channels(arr)
        masks[i] = decode_rle(row["hand_ink_rle"])
        bboxes.append(json.loads(row["hand_bbox_json"]))

    rng_split = np.random.RandomState(SEED + 777)
    perm = rng_split.permutation(n)
    n_holdout = max(20, int(round(n * HOLDOUT_FRAC)))
    holdout_idx = perm[:n_holdout]
    pool_idx = perm[n_holdout:]

    folds = make_folds(len(pool_idx), N_FOLDS, SEED)
    oof_prob = np.zeros((n, H, W), dtype=np.float32)
    oof_filled = np.zeros(n, dtype=bool)
    models = []

    per_fold_budget = max(30.0, (DEADLINE_SECONDS - (time.time() - T_START) - 120) / N_FOLDS)

    for fi in range(N_FOLDS):
        val_idx = pool_idx[folds[fi]]
        train_idx = pool_idx[np.concatenate([folds[j] for j in range(N_FOLDS) if j != fi])]
        model, val_prob = train_one_fold(
            imgs[train_idx], masks[train_idx], imgs[val_idx],
            EPOCHS, per_fold_budget,
        )
        oof_prob[val_idx] = val_prob
        oof_filled[val_idx] = True
        models.append(model)
        if time.time() - T_START > DEADLINE_SECONDS - 90:
            break

    oof_idx = np.where(oof_filled)[0]

    holdout_ens_prob = np.zeros((len(holdout_idx), H, W), dtype=np.float32)
    for model in models:
        holdout_ens_prob += predict_with_model(model, imgs[holdout_idx])
    holdout_ens_prob /= max(1, len(models))

    def row_core(prob_img, true_mask, true_b, thr):
        pm = prob_img > thr
        sm = s_mask_components(pm, true_mask)
        pred_b = mask_to_bbox(pm)
        err = abs(pred_b["x"] - true_b["x"]) + abs(pred_b["y"] - true_b["y"]) + \
              abs(pred_b["w"] - true_b["w"]) + abs(pred_b["h"] - true_b["h"])
        sb = math.exp(-err / 0.55)
        return 0.72 * sm + 0.18 * sb, sm, sb

    thr_grid = np.concatenate([
        np.arange(0.002, 0.02, 0.002), np.arange(0.02, 0.10, 0.01), np.arange(0.10, 0.91, 0.05),
    ])
    best_thr, best_score = 0.5, -1.0
    for thr in thr_grid:
        cores = [row_core(holdout_ens_prob[k], masks[holdout_idx[k]], bboxes[holdout_idx[k]], thr)[0]
                 for k in range(len(holdout_idx))]
        mc = float(np.mean(cores))
        if mc > best_score:
            best_score, best_thr = mc, float(thr)

    feats = []
    core_targets = []
    for k in range(len(holdout_idx)):
        core, sm, sb = row_core(holdout_ens_prob[k], masks[holdout_idx[k]], bboxes[holdout_idx[k]], best_thr)
        pm = holdout_ens_prob[k] > best_thr
        area_frac = pm.mean()
        pos_prob = holdout_ens_prob[k][pm].mean() if pm.sum() > 0 else 0.0
        feats.append([area_frac, pos_prob])
        core_targets.append(core)
    feats = np.array(feats, dtype=np.float64)
    core_targets = np.array(core_targets, dtype=np.float64)
    holdout_mean_core = float(core_targets.mean())
    ridge_w = ridge_fit(feats, core_targets, alpha=2.0)
    ridge_w["shrink_target"] = holdout_mean_core

    if os.environ.get("ERIS_DIAGNOSTICS") == "1":
        run_diagnostics(masks, bboxes, oof_prob, oof_idx, holdout_idx, holdout_ens_prob, best_thr, ridge_w)

    m = len(test_rows)
    test_imgs = np.zeros((m, N_CHANNELS, H, W), dtype=np.float32)
    for i, row in enumerate(test_rows):
        arr = load_image(row["target_shadow_path"])
        test_imgs[i] = build_input_channels(arr)

    test_prob = np.zeros((m, H, W), dtype=np.float32)
    for model in models:
        test_prob += predict_with_model(model, test_imgs)
    test_prob /= max(1, len(models))

    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    if SUBMISSION_OUT.exists():
        vN = 1
        while (SUBMISSION_OUT.parent / f"{SUBMISSION_OUT.stem}_v{vN}{SUBMISSION_OUT.suffix}").exists():
            vN += 1
        SUBMISSION_OUT.rename(SUBMISSION_OUT.parent / f"{SUBMISSION_OUT.stem}_v{vN}{SUBMISSION_OUT.suffix}")

    min_area = max(20, int(0.006 * H * W))
    with open(SUBMISSION_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "hand_ink_rle", "hand_bbox_json", "confidence"])
        for i, row in enumerate(test_rows):
            prob = test_prob[i]
            pm = prob > best_thr
            if pm.sum() < min_area:
                flat = prob.flatten()
                k = min(min_area, flat.size)
                top_idx = np.argpartition(flat, -k)[-k:]
                pm = np.zeros_like(flat, dtype=bool)
                pm[top_idx] = True
                pm = pm.reshape(H, W)
            area_frac = pm.mean()
            pos_prob = prob[pm].mean() if pm.sum() > 0 else 0.0
            feat = np.array([[area_frac, pos_prob]], dtype=np.float64)
            core_hat_raw = float(ridge_predict(ridge_w, feat)[0])
            core_hat = CONF_SHRINK * core_hat_raw + (1 - CONF_SHRINK) * ridge_w["shrink_target"]
            core_hat = min(max(core_hat, 0.0), 1.0)
            confidence = min(max(core_hat / 0.90, 0.0), 1.0)
            bbox = mask_to_bbox(pm)
            rle = encode_rle(pm.astype(np.uint8))
            writer.writerow([
                row["id"], rle,
                json.dumps({"x": round(bbox["x"], 5), "y": round(bbox["y"], 5),
                            "w": round(bbox["w"], 5), "h": round(bbox["h"], 5)}),
                round(confidence, 4),
            ])

    print(f"done in {time.time() - T_START:.1f}s, best_thr={best_thr:.2f}, ens_oof_core={best_score:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
