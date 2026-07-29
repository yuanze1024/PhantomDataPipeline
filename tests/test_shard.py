"""Tests for source sharding in :mod:`phantom_data.build.plan`.

Sharding exists so several pods can plan and build disjoint slices of the 183k-source pool
in parallel. Three properties make that safe, and each one is a failure mode if it breaks:

* **Disjoint and complete.** Two pods must never build the same source (wasted GPU hours,
  and two pods writing the same ``sample_id``), and no source may fall through the cracks.
  The union of all shards must equal the unsharded set exactly.
* **Independently reproducible.** A shard must be recomputable from its own arguments alone.
  Re-running shard 3 after shard 5 crashed must yield the same sources as before, with no
  knowledge of what the other shards did.
* **Byte-identical by default.** ``--num-shards 1`` is the current behaviour, so the
  pre-sharding output must be reproduced exactly -- including the stats file, which must not
  grow a ``shard`` key when sharding is off.

The functions under test are pure (a list of row dicts in, a filtered list out), so none of
this needs the 460MB parquet, the network, or a GPU.
"""
from __future__ import annotations

import pytest

from phantom_data.build import plan


def rows(count: int, windows_per_source: int = 1) -> list[dict[str, str]]:
    """Synthetic filtered-table rows shaped like Phantom's ``<uuid>_<start>_<end>`` ids.

    ``windows_per_source`` > 1 produces several rows per source uuid, which is the real
    shape: :func:`plan.pick_one_row_per_source` collapses them before sharding, and sharding
    must key on the same uuid that collapse used.
    """
    out: list[dict[str, str]] = []
    for index in range(count):
        uuid = f"{index:032x}"
        for window in range(windows_per_source):
            start = 10.0 * window
            out.append({
                "video_id": f"{uuid}_{start}_{start + 8.0}",
                "video_caption": f"caption {index}",
                "cross_pair": "{}",
            })
    return out


def ids(selected: list[dict[str, str]]) -> list[str]:
    return [row["video_id"] for row in selected]


# --------------------------------------------------------------------------------------
# disjointness and completeness
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_shards", [2, 3, 4, 8, 16])
def test_shards_partition_the_pool_exactly(num_shards):
    """Union of all shards == the unsharded set, with no source in two shards.

    The property the whole feature rests on: a source built twice is wasted work and a
    ``sample_id`` collision, a source built zero times is silently missing data.
    """
    pool = plan.pick_one_row_per_source(rows(500))
    unsharded = set(ids(pool))
    shards = [set(ids(plan.select_shard(pool, shard_id=index, num_shards=num_shards)))
              for index in range(num_shards)]

    union: set[str] = set()
    for index, shard in enumerate(shards):
        overlap = union & shard
        assert not overlap, f"shard {index} overlaps an earlier shard: {sorted(overlap)[:3]}"
        union |= shard
    assert union == unsharded
    # And the sizes add up, which catches a shard that silently duplicated its own rows.
    assert sum(len(shard) for shard in shards) == len(unsharded)


@pytest.mark.parametrize("num_shards", [2, 4, 8])
def test_every_shard_gets_work(num_shards):
    """A hash that starved a shard would leave a pod idle while others ran long."""
    pool = plan.pick_one_row_per_source(rows(500))
    sizes = [len(plan.select_shard(pool, shard_id=index, num_shards=num_shards))
             for index in range(num_shards)]
    assert all(size > 0 for size in sizes), sizes
    # Not a uniformity proof, just a sanity bound: 500 sources over <=8 shards should not
    # land more than ~2x the mean anywhere.
    assert max(sizes) < 2 * (len(pool) / num_shards)


def test_sharding_keys_on_the_source_not_the_window():
    """All windows of one source must land in one shard.

    They cannot actually collide after ``pick_one_row_per_source`` (which keeps one row per
    source), but keying on the full ``video_id`` instead of the uuid would silently break the
    guarantee if that order ever changed -- and would split a source across pods.
    """
    multi = rows(200, windows_per_source=3)
    by_source: dict[str, set[int]] = {}
    for row in multi:
        uuid = plan.source_uuid_of(row["video_id"])
        by_source.setdefault(uuid, set()).add(plan.shard_of(uuid, 8))
    assert all(len(shards) == 1 for shards in by_source.values())


def test_source_uuid_matches_what_the_dedup_uses():
    """The shard hash and the one-row-per-source dedup must key on the same string."""
    video_id = "0047e41c47cc69d1f81f9322bf391cca_35.4_43.7"
    assert plan.source_uuid_of(video_id) == "0047e41c47cc69d1f81f9322bf391cca"
    # Exactly the expression pick_one_row_per_source applies.
    assert plan.source_uuid_of(video_id) == video_id.rsplit("_", 2)[0]


# --------------------------------------------------------------------------------------
# determinism / independence
# --------------------------------------------------------------------------------------


def test_shard_assignment_is_deterministic_across_calls():
    pool = plan.pick_one_row_per_source(rows(200))
    first = ids(plan.select_shard(pool, shard_id=2, num_shards=5))
    second = ids(plan.select_shard(pool, shard_id=2, num_shards=5))
    assert first == second and first, "an empty result would make this vacuous"


def test_a_shard_is_reproducible_without_the_other_shards():
    """Shard membership depends only on the uuid, so a re-run needs no global state.

    Deliberately *not* ``index % num_shards``: positional assignment depends on the whole
    input, so a new parquet or a different ``--seed`` would reshuffle every shard and force
    pods to rebuild slices they had already finished.
    """
    pool = plan.pick_one_row_per_source(rows(200))
    expected = ids(plan.select_shard(pool, shard_id=1, num_shards=4))
    # Same sources arriving in a different order, and with unrelated sources removed.
    shuffled = list(reversed(pool))
    assert sorted(ids(plan.select_shard(shuffled, shard_id=1, num_shards=4))) == sorted(expected)
    subset = [row for row in pool if plan.shard_of(
        plan.source_uuid_of(row["video_id"]), 4) in (1, 2)]
    assert ids(plan.select_shard(subset, shard_id=1, num_shards=4)) == expected


def test_shard_assignment_is_independent_of_the_order_seed():
    """``--seed`` orders sources within a shard; it must not migrate them between shards.

    Folding the seed into the shard hash would mix two unrelated knobs: changing the sampling
    order would also change which pod owns which source.
    """
    pool_a = plan.pick_one_row_per_source(rows(200), seed=1)
    pool_b = plan.pick_one_row_per_source(rows(200), seed=99999)
    shard_a = set(ids(plan.select_shard(pool_a, shard_id=3, num_shards=6)))
    shard_b = set(ids(plan.select_shard(pool_b, shard_id=3, num_shards=6)))
    assert shard_a == shard_b


def test_shard_membership_is_stable_when_the_pool_grows():
    """Adding sources must not move existing ones between shards.

    Phantom shipping a bigger parquet must not invalidate slices already built.
    """
    small = plan.pick_one_row_per_source(rows(100))
    grown = plan.pick_one_row_per_source(rows(400))
    known = {row["video_id"] for row in small}
    before = set(ids(plan.select_shard(small, shard_id=1, num_shards=4)))
    after = {vid for vid in ids(plan.select_shard(grown, shard_id=1, num_shards=4))
             if vid in known}
    assert before == after


# --------------------------------------------------------------------------------------
# the default path stays exactly as it was
# --------------------------------------------------------------------------------------


def test_num_shards_one_is_the_identity():
    """``--num-shards 1`` must reproduce today's behaviour, list and order included."""
    pool = plan.pick_one_row_per_source(rows(120))
    assert plan.select_shard(pool, shard_id=0, num_shards=1) == pool
    assert ids(plan.select_shard(pool)) == ids(pool)


def test_num_shards_one_does_not_even_hash():
    """Short-circuit, so the unsharded path is unchanged rather than merely equivalent."""
    assert plan.shard_of("anything", 1) == 0
    assert plan.shard_of("anything", 0) == 0


def test_select_shard_preserves_the_seeded_order():
    """Shards inherit the pool's order, so ``--num-sources`` still takes a stable prefix."""
    pool = plan.pick_one_row_per_source(rows(300))
    shard = ids(plan.select_shard(pool, shard_id=2, num_shards=5))
    assert shard == [vid for vid in ids(pool) if vid in set(shard)]


# --------------------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("shard_id,num_shards", [(0, 0), (0, -1), (-1, 4), (4, 4), (9, 4)])
def test_bad_shard_arguments_are_rejected_loudly(shard_id, num_shards):
    """An out-of-range shard id would silently build nothing; a launcher typo must raise."""
    with pytest.raises(ValueError):
        plan.select_shard(rows(10), shard_id=shard_id, num_shards=num_shards)


def test_shard_of_returns_an_in_range_shard():
    for index in range(200):
        assert 0 <= plan.shard_of(f"{index:032x}", 7) < 7


# --------------------------------------------------------------------------------------
# sample ids cannot collide across shards (the downstream no-change claim)
# --------------------------------------------------------------------------------------


def test_sample_ids_cannot_collide_across_shards():
    """Why downstream needs no sharding support at all.

    Marker files and BOS keys are both derived from ``sample_id``, which is
    ``<source uuid>_w<window ms>``. Shards are disjoint *by source uuid*, so two shards can
    never produce the same ``sample_id`` -- meaning ``_stages/<stage>/<sample_id>.json`` and
    the BOS key layout are already safe for parallel pods writing into one dataset root.
    """
    from phantom_data.build.window import sample_id_for

    pool = plan.pick_one_row_per_source(rows(300))
    seen: dict[str, int] = {}
    for shard_id in range(6):
        for row in plan.select_shard(pool, shard_id=shard_id, num_shards=6):
            uuid = plan.source_uuid_of(row["video_id"])
            # Same window start for every source: the worst case for collisions, since only
            # the uuid can distinguish the ids.
            sample_id = sample_id_for(uuid, 12.5)
            assert sample_id not in seen, (
                f"{sample_id} produced by shard {seen.get(sample_id)} and {shard_id}")
            seen[sample_id] = shard_id
    assert len(seen) == len(pool)
