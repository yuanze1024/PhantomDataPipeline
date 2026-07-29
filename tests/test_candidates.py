"""Tests for the candidate pool and the mutual ID-Sim ranking.

The design's whole claim is that two box sources with different failure modes, plus a judge that
picks between them, beats either source alone. So the tests are about the *selection* being
faithful -- that a wrong-instance detector box loses to a right-instance SAM2 box, that ties and
absences are reported rather than papered over, and that the evidence trail lets a human overturn
the choice. Box geometry itself is tested in test_tighten.py.
"""
from __future__ import annotations

import numpy as np

from phantom_data import candidates


def candidate(source, box, **extra):
    return {"source": source, "box": box, "mask": None, **extra}


# ---------------------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------------------

def test_two_sources_proposing_the_same_box_collapse_to_one():
    pool = candidates.dedupe_candidates([
        candidate(candidates.SOURCE_PHANTOM_SAM2, [100, 100, 200, 200]),
        candidate(candidates.SOURCE_DETECTOR_SAM2, [101, 100, 200, 201]),
    ])
    assert len(pool) == 1
    # Agreement is recorded, not discarded: two independent sources landing on one box is
    # evidence about that box, and it is the kind of thing a reviewer asks about.
    assert pool[0]["also_proposed_by"] == [candidates.SOURCE_DETECTOR_SAM2]


def test_the_phantom_sourced_candidate_survives_a_duplicate_not_the_detector_one():
    # Ordering is load-bearing. When both sources agree geometrically, the surviving record must
    # credit the source whose *semantics* are guaranteed -- SAM2 from Phantom's box segments the
    # instance Phantom meant, whereas the detector merely happened to agree.
    pool = candidates.dedupe_candidates([
        candidate(candidates.SOURCE_PHANTOM_SAM2, [100, 100, 200, 200]),
        candidate(candidates.SOURCE_DETECTOR_SAM2, [100, 100, 200, 200]),
    ])
    assert pool[0]["source"] == candidates.SOURCE_PHANTOM_SAM2


def test_distinct_boxes_are_both_kept():
    pool = candidates.dedupe_candidates([
        candidate(candidates.SOURCE_PHANTOM_SAM2, [100, 100, 200, 200]),
        candidate(candidates.SOURCE_DETECTOR_SAM2, [600, 600, 700, 700]),
    ])
    assert len(pool) == 2


def test_a_candidate_without_a_box_is_dropped():
    pool = candidates.dedupe_candidates([
        candidate(candidates.SOURCE_PHANTOM_SAM2, None),
        candidate(candidates.SOURCE_DETECTOR_SAM2, [10, 10, 20, 20]),
    ])
    assert len(pool) == 1 and pool[0]["source"] == candidates.SOURCE_DETECTOR_SAM2


# ---------------------------------------------------------------------------------------
# matted crop
# ---------------------------------------------------------------------------------------

def test_matting_replaces_the_background_with_white_inside_the_crop():
    frame = np.full((50, 50, 3), 30, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    out = candidates.matted_crop(frame, mask, [10, 10, 20, 20])
    assert out.shape == (10, 10, 3)
    assert (out == 30).all(), "a crop exactly covering the mask keeps every pixel"

    wider = candidates.matted_crop(frame, mask, [5, 5, 25, 25])
    assert (wider[0, 0] == 255).all(), "outside the mask becomes white"
    assert (wider[7, 7] == 30).all(), "inside the mask is untouched"


def test_matting_returns_none_without_a_mask_or_box():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert candidates.matted_crop(frame, None, [0, 0, 5, 5]) is None
    assert candidates.matted_crop(frame, np.ones((10, 10), bool), None) is None


# ---------------------------------------------------------------------------------------
# best_pair
# ---------------------------------------------------------------------------------------

def fake_judge(scores):
    """An embed/compare pair whose similarity is looked up from ``scores[(ref_box, seed_box)]``.

    Embeddings are the candidate's own box tuple, so the comparison function can express any
    ranking the test needs -- including the adversarial ones a real model might produce.
    """
    def embed(crop):
        return crop  # the harness passes box tuples through matted_crop-free paths

    def compare(left, right):
        return scores.get((left, right))

    return embed, compare


def pool_with_boxes(pairs):
    """Candidates whose "crop" is their own box tuple, so fake_judge can key on them."""
    out = []
    for source, box in pairs:
        item = candidate(source, list(box))
        item["_key"] = tuple(box)
        out.append(item)
    return out


def rank(ref_pairs, seed_pairs, scores):
    ref_pool = pool_with_boxes(ref_pairs)
    seed_pool = pool_with_boxes(seed_pairs)

    def embed(_crop):
        return None  # replaced below; kept explicit so a real call would fail loudly

    # matted_crop needs a mask, which these synthetic candidates lack, so embed is driven off
    # the candidate list directly by index-free identity: patch in per-candidate keys.
    ref_keys = [tuple(c["box"]) for c in ref_pool]
    seed_keys = [tuple(c["box"]) for c in seed_pool]
    ref_iter, seed_iter = iter(ref_keys), iter(seed_keys)

    def embed_ref_then_seed(_crop):
        try:
            return next(ref_iter)
        except StopIteration:
            return next(seed_iter)

    def compare(left, right):
        return scores.get((left, right))

    return candidates.best_pair(embed_ref_then_seed, compare, ref_pool, seed_pool,
                                np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))


def test_the_highest_scoring_cross_side_pair_wins():
    ref = [(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10)),
           (candidates.SOURCE_DETECTOR_SAM2, (50, 50, 60, 60))]
    seed = [(candidates.SOURCE_PHANTOM_SAM2, (1, 1, 11, 11)),
            (candidates.SOURCE_DETECTOR_SAM2, (70, 70, 80, 80))]
    result = rank(ref, seed, {
        ((0, 0, 10, 10), (1, 1, 11, 11)): 0.9,
        ((0, 0, 10, 10), (70, 70, 80, 80)): 0.2,
        ((50, 50, 60, 60), (1, 1, 11, 11)): 0.3,
        ((50, 50, 60, 60), (70, 70, 80, 80)): 0.5,
    })
    assert result["identity"] == 0.9
    assert result["chosen"]["ref_index"] == 0 and result["chosen"]["seed_index"] == 0
    assert result["pairs_scored"] == 4


def test_a_detector_box_can_beat_the_phantom_sourced_one():
    # The design's reason to exist: when Phantom's box clipped the subject, the detector's
    # unclipped box should win on identity. Nothing in the ranking privileges the source.
    ref = [(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10)),
           (candidates.SOURCE_DETECTOR_SAM2, (0, 0, 20, 20))]
    seed = [(candidates.SOURCE_PHANTOM_SAM2, (1, 1, 21, 21))]
    result = rank(ref, seed, {
        ((0, 0, 10, 10), (1, 1, 21, 21)): 0.4,
        ((0, 0, 20, 20), (1, 1, 21, 21)): 0.85,
    })
    assert result["chosen"]["ref_source"] == candidates.SOURCE_DETECTOR_SAM2
    assert result["used_detector"] is True


def test_used_detector_is_false_when_both_winning_sides_came_from_phantom():
    # The headline question about this design -- does adding the detector change anything? -- is
    # answered by counting this flag, so it must not be true merely because a detector box was
    # *considered*.
    ref = [(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10)),
           (candidates.SOURCE_DETECTOR_SAM2, (50, 50, 60, 60))]
    seed = [(candidates.SOURCE_PHANTOM_SAM2, (1, 1, 11, 11))]
    result = rank(ref, seed, {
        ((0, 0, 10, 10), (1, 1, 11, 11)): 0.9,
        ((50, 50, 60, 60), (1, 1, 11, 11)): 0.1,
    })
    assert result["used_detector"] is False


def test_the_margin_over_the_runner_up_is_recorded():
    # The named residual risk of mutual ranking: a distractor present in *both* clips can be
    # picked on both sides and score high while describing the wrong subject. That failure tends
    # to leave a small margin, so the margin has to survive into the report.
    ref = [(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10))]
    seed = [(candidates.SOURCE_PHANTOM_SAM2, (1, 1, 11, 11)),
            (candidates.SOURCE_DETECTOR_SAM2, (2, 2, 12, 12))]
    result = rank(ref, seed, {
        ((0, 0, 10, 10), (1, 1, 11, 11)): 0.80,
        ((0, 0, 10, 10), (2, 2, 12, 12)): 0.78,
    })
    assert result["margin"] == 0.02
    assert result["runner_up"]["similarity"] == 0.78


def test_an_unopposed_winner_has_no_margin_rather_than_a_zero_one():
    # None and 0.0 are different claims. "There was no alternative" must not read as
    # "the alternative was equally good", which is what a threshold on margin would see.
    result = rank([(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10))],
                  [(candidates.SOURCE_PHANTOM_SAM2, (1, 1, 11, 11))],
                  {((0, 0, 10, 10), (1, 1, 11, 11)): 0.7})
    assert result["margin"] is None
    assert result["pairs_scored"] == 1


def test_every_pair_is_kept_for_review_sorted_best_first():
    ref = [(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10))]
    seed = [(candidates.SOURCE_PHANTOM_SAM2, (1, 1, 11, 11)),
            (candidates.SOURCE_DETECTOR_SAM2, (2, 2, 12, 12))]
    result = rank(ref, seed, {
        ((0, 0, 10, 10), (1, 1, 11, 11)): 0.3,
        ((0, 0, 10, 10), (2, 2, 12, 12)): 0.6,
    })
    scores = [row["similarity"] for row in result["all_pairs"]]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) == 2, "the losing pair is on the record, not just the winner"


def test_an_empty_pool_yields_no_choice_rather_than_raising():
    result = rank([], [(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10))], {})
    assert result["chosen"] is None
    assert result["pairs_scored"] == 0
    assert "no comparable" in result["reason"]


def test_pairs_that_cannot_be_compared_are_skipped_not_scored_as_zero():
    # A None from the judge means "not measured". Scoring it as 0.0 would make an unmeasurable
    # pair compete -- and win, if every other pair also failed.
    ref = [(candidates.SOURCE_PHANTOM_SAM2, (0, 0, 10, 10))]
    seed = [(candidates.SOURCE_PHANTOM_SAM2, (1, 1, 11, 11)),
            (candidates.SOURCE_DETECTOR_SAM2, (2, 2, 12, 12))]
    result = rank(ref, seed, {((0, 0, 10, 10), (2, 2, 12, 12)): 0.5})
    assert result["pairs_scored"] == 1
    assert result["chosen"]["seed_index"] == 1


# ---------------------------------------------------------------------------------------
# subject_noun -- keeping part words out of the detector query
# ---------------------------------------------------------------------------------------

def test_the_head_noun_survives_and_the_attributes_are_cut():
    # The measured failure: a query's confidence is a max over its tokens, so "glasses" wins its
    # own box. Cutting at the joiner removes the attributes that can win boxes.
    assert candidates.subject_noun("man with short dark hair and glasses") == "man"
    assert candidates.subject_noun("young woman with long, wavy hair") == "young woman"
    assert candidates.subject_noun(
        "person in a dark blue uniform with a badge") == "person"
    assert candidates.subject_noun("dog doll with tan coat and pink collar") == "dog doll"


def test_a_phrase_with_no_joiner_is_left_alone():
    assert candidates.subject_noun("French Bulldog") == "French Bulldog"
    assert candidates.subject_noun("butterfly") == "butterfly"


def test_a_leading_joiner_does_not_produce_an_empty_query():
    # An empty query silently disables the detector for that subject, which is worse than an
    # imperfect one: the candidate pool would quietly shrink to Phantom's box alone.
    assert candidates.subject_noun("with a hat") == "with a hat"
    assert candidates.subject_noun("") == ""
    assert candidates.subject_noun(None) == ""


def test_trailing_punctuation_on_a_joiner_still_cuts():
    assert candidates.subject_noun("woman, with a bag") == "woman"


# ---------------------------------------------------------------------------------------
# plausible_instance -- the area floor
# ---------------------------------------------------------------------------------------

def test_a_part_sized_box_is_rejected():
    # Real measured part boxes ran 0.02-0.23 of Phantom's area.
    phantom = [0, 0, 100, 100]
    assert candidates.plausible_instance([0, 0, 20, 20], phantom, 0.3) is False
    assert candidates.plausible_instance([0, 0, 15, 15], phantom, 0.3) is False


def test_an_instance_sized_box_is_accepted_even_if_smaller_than_phantoms():
    # Phantom's boxes are loose -- median mask share of the prompt was 0.517 -- so a genuine
    # instance box is routinely well under Phantom's area and must not be filtered.
    assert candidates.plausible_instance([0, 0, 70, 70], [0, 0, 100, 100], 0.3) is True


def test_a_box_larger_than_phantoms_is_accepted():
    # The clipping case this whole design exists for: the detector box exceeds Phantom's because
    # Phantom cut off a limb.
    assert candidates.plausible_instance([0, 0, 150, 150], [0, 0, 100, 100], 0.3) is True


def test_without_a_reference_box_nothing_is_filtered():
    # No Phantom box means no scale to judge against; filtering on a guess would drop candidates
    # for a reason that was never measured.
    assert candidates.plausible_instance([0, 0, 5, 5], None, 0.3) is True


# ---------------------------------------------------------------------------------------
# crop_for -- plain crops for rank-first, matted when a mask is available
# ---------------------------------------------------------------------------------------

def test_a_candidate_without_a_mask_yields_a_plain_crop():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop = candidates.crop_for(frame, {"box": [10, 10, 20, 20]})
    assert crop.shape == (10, 10, 3)
    assert (crop == frame[10:20, 10:20]).all(), "no matting, no modification"


def test_a_candidate_with_a_mask_yields_a_matted_crop():
    frame = np.full((50, 50, 3), 30, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:15, 10:15] = True
    crop = candidates.crop_for(frame, {"box": [5, 5, 25, 25], "mask": mask})
    assert (crop[0, 0] == 255).all(), "background outside the mask is whitened"


def test_crop_for_declines_an_empty_or_missing_box():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert candidates.crop_for(frame, {"box": None}) is None
    assert candidates.crop_for(frame, {"box": [5, 5, 5, 5]}) is None
