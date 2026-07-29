"""Does the cached-embedding path give the same score as ID-Sim's own forward()?

The candidate ranking embeds each crop once and compares cached embeddings, because comparison
costs 0.202 ms against 33.6 ms for an embedding -- that is what makes an exhaustive N x M pairing
affordable. But it reaches past the public ``IDSimModel`` wrapper into the inner
``PerceptualModel.compute_distance_from_embeddings``, so it could silently diverge from
``model(a, b, mode="cls")``: a different normalisation, a different projection head, or the wrong
dict key would all produce plausible-looking numbers that rank differently.

Every identity score in the candidate reports comes from that path, so the equivalence is checked
rather than assumed.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

CACHE = "/mnt/pfs/users/yuanze/models/id_sim_checkpoint"
REPO = "/mnt/pfs/users/yuanze/projects/2026/BboxCondition/third_party/id_sim"
ID_SIM_TYPE = os.environ.get("IDSIM_TYPE", "dinov2_vitl14_cls_patch")
DEVICE = os.environ.get("IDSIM_DEVICE", "cuda")


def main() -> int:
    torch.hub.set_dir(CACHE + "/hub")
    sys.path.insert(0, REPO)
    from id_sim import id_sim

    model, preprocess = id_sim(pretrained=True, device=DEVICE, cache_dir=CACHE,
                              id_sim_type=ID_SIM_TYPE)

    rng = np.random.default_rng(0)
    crops = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
             for _ in range(6)]
    tensors = [preprocess(c).to(DEVICE) for c in crops]

    worst = 0.0
    with torch.inference_mode():
        embeds = [model.embed(t, mode="cls") for t in tensors]
        for i in range(len(tensors)):
            for j in range(len(tensors)):
                direct = float(model(tensors[i], tensors[j], mode="cls"))
                cached = float(model.model.compute_distance_from_embeddings(
                    {"cls_embed": embeds[i]["cls"]},
                    {"cls_embed": embeds[j]["cls"]})["cls"].reshape(-1)[0])
                worst = max(worst, abs(direct - cached))
                if i == j:
                    # A crop against itself must be distance ~0 on both paths; a nonzero value
                    # here would mean the embedding is not deterministic.
                    assert direct < 1e-4, f"self-distance {direct} on the direct path"
    print(f"max |direct - cached| over 36 pairs: {worst:.3e}")
    print("EQUIVALENT" if worst < 1e-5 else "DIVERGENT -- do not use the cached path")
    return 0 if worst < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
