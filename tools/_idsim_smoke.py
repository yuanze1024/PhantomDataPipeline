"""Smoke test: does ID-Sim load, and does it separate same-instance from different-instance?

Loading is the easy half. The half that decides whether ID-Sim is worth adopting is whether its
distances rank *this* data's pairs correctly, and the cheapest honest check is a same-vs-cross
comparison built from pairs the pipeline already has:

* same-instance   -- each subject's own (reference crop, target crop). These are the pairs the
  identity judge scores today; median 83 seconds apart, different clips, so they carry the real
  viewpoint/lighting variation rather than a synthetic augmentation.
* different-instance -- each subject's reference crop against *another* subject's target crop,
  paired by index shift so every subject contributes exactly one negative.

The separation number reported is the gap between the two medians and, more usefully, the AUROC
of "same" versus "cross". A metric that cannot separate a subject from an unrelated subject is
not going to separate a subject from a lookalike, so this is a floor test, not a validation --
the real question needs the human labels.

DINOv2 backbone, not DINOv3: Meta gates the DINOv3 weights behind a click-through page that
returns 403 to any scripted fetch, while ID-Sim ships a DINOv2 ViT-L variant whose backbone
comes from torch.hub. That also makes a cleaner comparison than the paper's headline config,
because the baseline it replaces (``redetect.Models.identity_cosine``) is *also* DINOv2 --
so any difference is attributable to ID-Sim's LoRA and projection heads rather than to a
stronger backbone.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CACHE = "/mnt/pfs/users/yuanze/models/id_sim_checkpoint"
DS = Path("/mnt/pfs/data/yuanze/phantom_koala_newbox_v1")
RT = "_redetect_trust"
LIMIT = int(os.environ.get("IDSIM_LIMIT", "24"))
ID_SIM_TYPE = os.environ.get("IDSIM_TYPE", "dinov2_vitl14_cls_patch")
DEVICE = os.environ.get("IDSIM_DEVICE", "cuda")

# torch.hub must find the DINOv2 *code* (github, via proxy) and the weights we pre-downloaded
# into <CACHE>/hub/checkpoints/, which is torch.hub's own layout -- create_model calls
# torch.hub.set_dir(load_dir) for dinov2, so load_dir is the hub root, not a checkpoint dir.
torch.hub.set_dir(CACHE + "/hub")

sys.path.insert(0, "/mnt/pfs/users/yuanze/projects/2026/BboxCondition/third_party/id_sim")
from id_sim import id_sim  # noqa: E402

sys.path.insert(0, "src")
from phantom_data.boxes import crop_box  # noqa: E402


def auroc(positive: list[float], negative: list[float]) -> float:
    """Rank-based AUROC. Distances, so *lower* should mean same-instance; we score -distance."""
    pairs = [(-value, 1) for value in positive] + [(-value, 0) for value in negative]
    pairs.sort(key=lambda item: item[0])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(pairs):
        stop = index
        while stop + 1 < len(pairs) and pairs[stop + 1][0] == pairs[index][0]:
            stop += 1
        average = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    positive_rank_sum = sum(ranks[i] for i, (_, label) in enumerate(pairs) if label == 1)
    n_pos, n_neg = len(positive), len(negative)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    report = json.loads((DS / RT / "gate_report.json").read_text())
    subjects = [s for s in report["subjects"]
                if s.get("chosen_box_ref") and s.get("chosen_box_seed")][:LIMIT]
    print(f"loading ID-Sim {ID_SIM_TYPE} ...", flush=True)
    model, preprocess = id_sim(pretrained=True, device=DEVICE, cache_dir=CACHE,
                              id_sim_type=ID_SIM_TYPE)
    print("loaded", flush=True)

    crops_ref, crops_seed, stored = [], [], []
    for subject in subjects:
        ref_path = DS / RT / str(subject["ref_frame"])
        seed_path = DS / RT / str(subject["seed_frame"])
        if not ref_path.is_file() or not seed_path.is_file():
            continue
        ref_frame = np.asarray(Image.open(ref_path).convert("RGB"))
        seed_frame = np.asarray(Image.open(seed_path).convert("RGB"))
        left = crop_box(ref_frame, subject["chosen_box_ref"])
        right = crop_box(seed_frame, subject["chosen_box_seed"])
        if left is None or right is None:
            continue
        # preprocess() returns a CPU tensor; the model lives on DEVICE. ID-Sim's own README
        # shows the .to(device) at the call site rather than inside preprocess.
        crops_ref.append(preprocess(Image.fromarray(left)).to(DEVICE))
        crops_seed.append(preprocess(Image.fromarray(right)).to(DEVICE))
        stored.append(subject.get("rule_identity"))
    print(f"{len(crops_ref)} usable subjects", flush=True)

    same, cross = [], []
    with torch.inference_mode():
        for i in range(len(crops_ref)):
            same.append(float(model(crops_ref[i], crops_seed[i], mode="cls")))
            j = (i + 1) % len(crops_ref)          # one negative per subject
            cross.append(float(model(crops_ref[i], crops_seed[j], mode="cls")))
            print(f"  [{i+1}/{len(crops_ref)}] same={same[-1]:.4f} cross={cross[-1]:.4f} "
                  f"(stored dinov2_cos={stored[i]})", flush=True)

    med = lambda v: float(np.median(v))
    print()
    print(f"ID-Sim distance  same-instance  median={med(same):.4f}  "
          f"mean={np.mean(same):.4f}")
    print(f"ID-Sim distance  cross-instance median={med(cross):.4f}  "
          f"mean={np.mean(cross):.4f}")
    print(f"separation (cross-median - same-median) = {med(cross)-med(same):+.4f}")
    print(f"AUROC same-vs-cross = {auroc(same, cross):.4f}   (1.0 = perfect, 0.5 = chance)")

    # The same floor test on the incumbent judge, so the two are read on one scale. Stored
    # cosines are same-instance only, so the negative side is recomputed here from the report's
    # own numbers being unavailable -- we just report the stored distribution for context.
    usable = [v for v in stored if v is not None]
    if usable:
        print(f"\nincumbent DINOv2 cosine (same-instance only, from report): "
              f"median={med(usable):.4f}  min={min(usable):.4f} max={max(usable):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
