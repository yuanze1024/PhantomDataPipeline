"""What does one ID-Sim comparison cost, and what does the candidate-ranking design cost at scale?

Two numbers decide whether "score every candidate box and let ID-Sim pick" is affordable:

* per-pair latency, split into embedding and comparison. This matters because the ranking design
  embeds the reference crop **once** and compares it against N candidates -- if the cost is
  dominated by the backbone forward pass, N candidates cost 1 + N embeddings rather than N pairs,
  which is a different scaling law.
* the batch behaviour. 126k subjects x N candidates is a lot of forward passes; if batching helps
  materially the runner should batch.

Reports both, plus the projected GPU-hours for the full target set at several candidate counts.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

CACHE = "/mnt/pfs/users/yuanze/models/id_sim_checkpoint"
REPO = "/mnt/pfs/users/yuanze/projects/2026/BboxCondition/third_party/id_sim"
ID_SIM_TYPE = os.environ.get("IDSIM_TYPE", "dinov2_vitl14_cls_patch")
DEVICE = os.environ.get("IDSIM_DEVICE", "cuda")

#: Full-scale target from the pilot's planning: ~126k subjects, two sides each.
TARGET_SUBJECTS = 126_000


def main() -> int:
    torch.hub.set_dir(CACHE + "/hub")
    sys.path.insert(0, REPO)
    from id_sim import id_sim

    t0 = time.time()
    model, preprocess = id_sim(pretrained=True, device=DEVICE, cache_dir=CACHE,
                              id_sim_type=ID_SIM_TYPE)
    load_sec = time.time() - t0
    print(f"model load: {load_sec:.1f}s  ({ID_SIM_TYPE} on {DEVICE})", flush=True)

    rng = np.random.default_rng(0)
    crops = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
             for _ in range(64)]
    tensors = [preprocess(c).to(DEVICE) for c in crops]

    def timed(fn, n, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        start = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        return (time.time() - start) / n

    with torch.inference_mode():
        # 1. Full pair comparison, the naive unit.
        pair_sec = timed(lambda: model(tensors[0], tensors[1], mode="cls"), 20)
        print(f"\nper pair (embed both + compare): {pair_sec*1000:.1f} ms", flush=True)

        # 2. Embedding alone -- the reusable half.
        embed_sec = timed(lambda: model.embed(tensors[0], mode="cls"), 20)
        print(f"per embed (one crop):            {embed_sec*1000:.1f} ms", flush=True)

        # 3. Batched embedding: does stacking help?
        for batch in (1, 4, 8, 16, 32):
            stack = torch.cat([t if t.dim() == 4 else t.unsqueeze(0)
                               for t in tensors[:batch]], dim=0)
            per = timed(lambda s=stack: model.embed(s, mode="cls"), 10) / batch
            print(f"  batch={batch:3d}: {per*1000:6.2f} ms/crop", flush=True)

        # 4. Comparison of precomputed embeddings -- what a candidate re-rank actually costs
        #    once the reference is embedded.
        # The public IDSimModel wrapper exposes only forward()/embed(); the distance-from-
        # embeddings entry point lives on the inner PerceptualModel. Reaching through .model is
        # what makes the "embed the reference once, compare against N candidates" design possible
        # at all -- without it every candidate would re-embed the reference.
        ref_embed = model.embed(tensors[0], mode="cls")
        cand_embed = model.embed(tensors[1], mode="cls")
        inner = model.model
        cmp_sec = timed(
            lambda: inner.compute_distance_from_embeddings(
                {"cls_embed": ref_embed["cls"]}, {"cls_embed": cand_embed["cls"]}), 50)
        print(f"\ncompare two cached embeddings:   {cmp_sec*1000:.3f} ms", flush=True)

    print("\n--- projected cost at scale (both sides of every subject) ---")
    print(f"{'candidates/side':>16} {'embeds/subject':>15} {'GPU-hours':>11} {'note':>34}")
    for n_cand in (1, 2, 3, 5, 8):
        # ref crop embedded once per side; each candidate embedded once; comparisons are free
        # relative to the forwards.
        embeds = 2 * (1 + n_cand)
        hours = TARGET_SUBJECTS * (embeds * embed_sec + 2 * n_cand * cmp_sec) / 3600
        note = "current design (1 box/side)" if n_cand == 1 else ""
        print(f"{n_cand:>16} {embeds:>15} {hours:>11.1f} {note:>34}")
    print(f"\nfor reference: SAM2 stage C measured at ~133 GPU-h for {TARGET_SUBJECTS} subjects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
