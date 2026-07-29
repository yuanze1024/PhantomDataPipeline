"""One-off: print per-token Grounding DINO scores for the ``dog doll`` subject and friends.

Throwaway driver for probe_dino_tokens.probe_frame; the reusable logic lives there.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path[:0] = ["src", "tools"]
import probe_dino_tokens as P  # noqa: E402
from phantom_data import redetect  # noqa: E402

DS = Path("/mnt/pfs/data/yuanze/phantom_koala_newbox_v1")
RT = "_redetect_trust"
INSPECT = Path("/mnt/pfs/data/yuanze/phantom_koala_inspect100_v1/_redetect100/gate_report.json")

subjects = json.loads((DS / RT / "gate_report.json").read_text())["subjects"]
llm = {(s["sample_id"], s["subject_id"]): s.get("dis")
       for s in json.loads(INSPECT.read_text())["subjects"]}

models = redetect.Models(device="cpu")

# The subject the user named, plus two multi-concept controls: a compound breed name (does the
# probe handle wordpiece splits) and a phrase whose modifier is a colour (often ungroundable).
wanted = ["dog doll", "French Bulldog", "small black dog with a red collar"]
for phrase in wanted:
    matches = [s for s in subjects if (s.get("dis") or "") == phrase]
    if not matches:
        print(f"!! no subject with dis == {phrase!r}")
        continue
    subject = matches[0]
    key = (subject["sample_id"], subject["subject_id"])
    frame_path = DS / RT / str(subject["ref_frame"])
    if not frame_path.is_file():
        print(f"!! missing frame {frame_path}")
        continue
    frame = np.asarray(Image.open(frame_path).convert("RGB"))

    print("=" * 78)
    print(f"subject {subject['sample_id']} subj{subject['subject_id']:02d}   frame {frame.shape}")
    print(f"  stored: det={subject['detector_score_ref_dis']} box={subject['box_ref_dis']}")
    print(f"  phantom box={subject['box_ref_phantom']}  "
          f"iou_ref={subject['iou_dis_vs_phantom']}")

    for query in (phrase, llm.get(key)):
        if not query:
            continue
        probe = P.probe_frame(models, frame, query)
        ranking = P.rank_disagreement(probe["candidates"])
        print()
        print(f"  QUERY {query!r}")
        print(f"    content words: {probe['content_words']}")
        for i, cand in enumerate(probe["candidates"]):
            print(f"    cand{i}  max={cand['max']:.4f}  "
                  f"min={-1 if cand['min'] is None else cand['min']:.4f}  "
                  f"mean={-1 if cand['mean'] is None else cand['mean']:.4f}  box={cand['box']}")
            print(f"          per_word: {cand['per_word']}")
        print(f"    agree={ranking.get('agree')}  "
              f"iou_between_picks={ranking.get('iou_between_picks')}")
