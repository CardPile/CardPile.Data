import numpy
import polars
import spells

import card_frequencies
import mtgjson
import utils


def missing_p1p1_sets():
  return ["TLA", "TMT", "ECL", "STX", "AFR"]


def pack_pick_filter_spec(pack, pick):
  return {"$and": [{spells.ColName.PACK_NUM: pack}, {spells.ColName.PICK_NUM: pick}]}


picksext = {
  f'P{pack}P{pick}': spells.ColSpec(
    col_type = spells.ColType.AGG,
    expr = polars.col('pack_card')
  ) for pack in range(1, utils.max_packs() + 1) for pick in range(1, utils.max_picks() + 1)
}


def pick_count_frames(set_code, event_type):
  frames = []
  for pack in range(1, utils.max_packs() + 1):
    for pick in range(1, utils.max_picks() + 1):
      new_frame = spells.summon(set_code,
                                columns=[spells.ColName.SET_CODE, spells.ColName.IS_BONUS_SHEET, f"P{pack}P{pick}"],
                                filter_spec=pack_pick_filter_spec(pack, pick),
                                extensions=picksext,
                                event_type=event_type)
      new_frame = utils.repair_broken_cards(new_frame, set_code)
      new_frame = utils.filter_basic_lands(new_frame)
      new_frame = utils.sanitize_data_frame(new_frame)
      frames.append(new_frame)
  return frames


MEAN3_COLUMNS = [*[f"anchor_P{p}P1" for p in range(1, 4)],
                 *[f"anchor_raw_P{p}P1" for p in range(1, 4)],
                 *[f"anchor_scale_p{p}" for p in range(1, 4)]]

POOL_Z = 3.0
POOL_PHI = 0.94


ANCHOR_COLUMNS = MEAN3_COLUMNS + ["anchor_pool"]


def card_pool(name, card_set_code, is_bonus_sheet, set_code, set_jsons):
  try:
    f = card_frequencies.get_card_frequency({"name": name, "set_code": card_set_code, "is_bonus_sheet": is_bonus_sheet}, set_code, set_jsons)
  except ValueError:
    return None

  printings = [card for card in set_jsons[card_set_code]["data"]["cards"]
               if card_frequencies.disambiguate_reversible_card_printings(card) == name or card.get("faceName") == name]
  front = ([card for card in printings if card.get("side") in (None, "a")] or printings)[0]
  sheets = set_jsons[set_code]["data"]["booster"]["play"]["sheets"].values()
  on_sheets = {card["number"] for card in printings if any(card["uuid"] in sheet["cards"] for sheet in sheets)}
  return f"{card_set_code}|{is_bonus_sheet}|{front['rarity']}|{f:.9f}|{len(on_sheets)}|{'Land' in front.get('types', [])}"


def with_pooled_anchor(set_df, set_code):
  packs = range(1, utils.max_packs() + 1)
  scales = numpy.array([set_df[f"anchor_scale_p{p}"][0] for p in packs])
  raw = set_df.select([f"anchor_raw_P{p}P1" for p in packs]).to_numpy().astype(float)
  usable = ~numpy.isnan(raw)
  mean3 = numpy.where(usable, raw * scales, 0.0).sum(axis=1) / usable.sum(axis=1)
  var3 = POOL_PHI * numpy.where(usable, raw * scales ** 2, 0.0).sum(axis=1) / usable.sum(axis=1) ** 2

  set_jsons = {}
  pools = []
  for name, card_set_code, is_bonus_sheet in set_df.select("name", "set_code", "is_bonus_sheet").iter_rows():
    for code in (set_code, card_set_code):
      if code not in set_jsons:
        set_jsons[code] = mtgjson.get_set_json(code)
    pools.append(card_pool(name, card_set_code, is_bonus_sheet, set_code, set_jsons))

  pools = numpy.array(pools, dtype=object)

  anchor, pooled_in = mean3.copy(), numpy.full(len(pools), None, dtype=object)
  for pool in {p for p in pools if p is not None}:
    rows = numpy.flatnonzero(pools == pool)
    if len(rows) < 2:
      continue

    members = numpy.ones(len(rows), dtype=bool)

    for _ in range(20):
      z = numpy.full(len(rows), numpy.inf)
      for a in range(len(rows)):
        others = members.copy()
        others[a] = False
        if others.any():
          z[a] = (mean3[rows[a]] - mean3[rows[others]].mean()) / numpy.sqrt(var3[rows[a]] + var3[rows[others]].sum() / others.sum() ** 2)
      settled = numpy.abs(z) <= POOL_Z
      if (settled == members).all():
        break
      members = settled

    if members.sum() < 2:
      continue

    anchor[rows[members]] = mean3[rows[members]].mean()
    pooled_in[rows[members]] = pool

  n_pooled = int((pooled_in != None).sum())
  print(f"{set_code} -> pooled anchor: {n_pooled} of {len(pools)} cards in {len({p for p in pooled_in if p is not None})} pools, "
        f"{int((pools == None).sum())} without an f, {len(pools) - n_pooled - int((pools == None).sum())} on their own mean3")
  return set_df.with_columns(*[polars.Series(f"anchor_P{p}P1", anchor / scales[p - 1]) for p in packs],
                             polars.Series("anchor_pool", pooled_in.tolist(), dtype=polars.String))


def with_mean3_anchor(set_df, set_code):
  raw = [polars.col(f"P{p}P1").fill_null(0).cast(polars.Float64) for p in range(1, utils.max_packs() + 1)]
  sums = set_df.select(*[r.sum().alias(f"s{i}") for i, r in enumerate(raw)]).row(0)

  if set_code in missing_p1p1_sets():
    a = sums[1] / sums[2]
    assert 0.9 < a < 1.1, f"{set_code}: impossible pack 2 -> 3 attrition factor {a}"
    scales, usable = [1.0, a, a * a], [False, True, True]
  else:
    scales, usable = [1.0, sums[0] / sums[1], sums[0] / sums[2]], [True, True, True]

  pieces = [raw[i] * scales[i] for i in range(len(raw)) if usable[i]]
  mean = polars.sum_horizontal(pieces) / len(pieces)

  print(f"{set_code} -> mean3 anchor: pack scales {[round(v, 4) for v in scales]}, packs used {[i + 1 for i in range(3) if usable[i]]}")

  return set_df.with_columns(
    *[(mean / scales[i]).alias(f"anchor_P{i + 1}P1") for i in range(len(raw))],
    *[(raw[i] if usable[i] else polars.lit(None, dtype=polars.Float64)).alias(f"anchor_raw_P{i + 1}P1") for i in range(len(raw))],
    *[polars.lit(scales[i], dtype=polars.Float64).alias(f"anchor_scale_p{i + 1}") for i in range(len(raw))])


def sanitised_curves(raw, scales, usable, pools):
  anchor = numpy.where(usable, raw[..., 0] * scales, 0.0).sum(axis=-1) / usable.sum(axis=-1)
  for members in pools:
    anchor[..., members] = anchor[..., members].mean(axis=-1, keepdims=True)
  curves = numpy.empty_like(raw)
  curves[..., 0] = numpy.maximum(anchor[..., None] / scales, raw[..., 1:].max(axis=-1))
  curves[..., 1:] = numpy.minimum.accumulate(raw[..., 1:], axis=-1)
  return curves


def anchor_parts(set_df):
  packs = range(1, utils.max_packs() + 1)
  scales = numpy.array([set_df[f"anchor_scale_p{p}"][0] for p in packs])
  usable = ~numpy.isnan(set_df.select([f"anchor_raw_P{p}P1" for p in packs]).to_numpy().astype(float))
  pool_of = set_df["anchor_pool"]
  pools = [numpy.flatnonzero((pool_of == pool).fill_null(False).to_numpy()) for pool in pool_of.drop_nulls().unique().to_list()]
  return scales, usable, pools


def pick_counts(set_df):
  return numpy.stack([numpy.stack([set_df[f"P{p}P{k}"].cast(polars.Float64).fill_null(0).to_numpy()
                                   for k in range(1, utils.max_picks() + 1)], axis=1) for p in range(1, utils.max_packs() + 1)], axis=1)


def with_card_curves(set_df, set_code):
  set_df = with_pooled_anchor(with_mean3_anchor(set_df, set_code), set_code)
  curves = sanitised_curves(pick_counts(set_df), *anchor_parts(set_df))
  return set_df.with_columns(*[polars.Series(f"P{p}P{k}", curves[:, p - 1, k - 1])
                               for p in range(1, utils.max_packs() + 1) for k in range(1, utils.max_picks() + 1)])
