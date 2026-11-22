import warnings
import zlib

import numpy
import polars
import spells
import spells.cache

import card_curves
import utils

DRAFT_BOOT_REPLICATES = 200
DRAFT_BOOT_SEED = 1234


def replicate_curves(set_df, raw_counts, set_code, event_type, keep):
  packs, picks = utils.max_packs(), utils.max_picks()
  key = ["name", "set_code", "is_bonus_sheet"]
  cards = set_df.select(key)
  n = cards.height

  broken_names = {(drafted_in, name): broken for (drafted_in, broken), (name, _, _) in utils.BROKEN_17LANDS_CARDS.items()}
  columns = [f"pack_card_{broken_names.get((set_code, name), name)}" for name in cards["name"]]
  assert len(set(columns)) == n, f"{set_code}: two cards share one 17lands log column"

  path = spells.cache.data_file_path(set_code, "draft", event_type)
  logs = polars.scan_parquet(path)
  missing = sorted(set(columns) - set(logs.collect_schema().names()))
  assert not missing, f"{set_code}: no column in {path} for {missing[:5]}"

  drafts = logs.select("draft_id").unique().collect().sort("draft_id").with_row_index("draft_index")
  seed = numpy.random.SeedSequence([DRAFT_BOOT_SEED, zlib.crc32(f"{set_code}/{event_type.value}".encode())])
  weights = numpy.random.default_rng(seed).poisson(1.0, size=(DRAFT_BOOT_REPLICATES, drafts.height)).astype(float)

  counts = numpy.zeros((DRAFT_BOOT_REPLICATES, n, packs, picks))
  unit = numpy.zeros((n, packs, picks))
  for pack in range(packs):
    rows = (logs.filter((polars.col("pack_number") == pack) & (polars.col("pick_number") < picks))
                .select("draft_id", "pick_number", *[polars.col(column).fill_null(0) for column in columns])
                .collect()
                .join(drafts, on="draft_id", how="left"))
    for pick in range(picks):
      at_pick = rows.filter(polars.col("pick_number") == pick)
      if at_pick.is_empty():
        # 13-card formats have no 14th pick
        continue
      copies = at_pick.select(columns).to_numpy().astype(float)
      counts[:, :, pack, pick] = weights[:, at_pick["draft_index"].to_numpy()] @ copies
      unit[:, pack, pick] = copies.sum(axis=0)
    del rows

  worst = numpy.abs(unit - card_curves.pick_counts(cards.join(raw_counts, on=key, how="left", maintain_order="left"))).max()
  assert worst == 0, f"{set_code}: the logs with every weight 1 miss the spells counts by up to {worst}"

  scales, usable, pools = card_curves.anchor_parts(set_df)
  point = card_curves.sanitised_curves(unit, scales, usable, pools)
  stored = card_curves.pick_counts(set_df)
  worst = (numpy.abs(point - stored) / numpy.maximum(stored, 1.0)).max()
  assert worst < 1e-9, f"{set_code}: the bootstrap's pipeline misses the stored curves by up to {worst:.3g} (relative)"

  replicates = card_curves.sanitised_curves(counts, scales, usable, pools)
  rows = keep.select(key).join(cards.with_row_index("__row"), on=key, how="left", maintain_order="left")["__row"]
  assert rows.null_count() == 0, f"{set_code}: {rows.null_count()} kept cards are not in the frame the anchor was formed on"
  rows = rows.to_numpy()
  return point[rows], replicates[:, rows], drafts.height


def curve_sigmas(point, replicates):
  """Calculate the bootstrap standard deviation (ddof=1) of:
     - every cleaned P{p}P{k}. size = [card count, packs, picks]
     - every consecutive ratio P{p}P{k} / P{p}P{k-1}. size = [card count, packs, picks - 1]
     - every share P{p}P{k} / P{p}P1. size = [card count, packs, picks]
  """
  degenerate = point[..., 0] <= 0
  count_sd = replicates.std(axis=0, ddof=1)

  with numpy.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    ratios = numpy.where(replicates[..., :-1] > 0, replicates[..., 1:] / replicates[..., :-1], numpy.nan)
    ratio_sd = numpy.nan_to_num(numpy.nanstd(ratios, axis=0, ddof=1), nan=0.0)
    shares = numpy.where(replicates[..., :1] > 0, replicates / replicates[..., :1], numpy.nan)
    share_sd = numpy.nan_to_num(numpy.nanstd(shares, axis=0, ddof=1), nan=0.0)

  for sd in (count_sd, ratio_sd, share_sd):
    # Degenerate curve with all-zero has no spread
    sd[degenerate] = 0.0

  return count_sd, ratio_sd, share_sd
