"""Expected copies of a card per Arena draft pack, from MTGJSON's play-booster sheets (the paper collation).

MTGJSON's sheets list PRINTINGS: a card with alternate art (showcase, borderless, scene, ...) has several entries on a
sheet, which on paper either add to the card's weight or split it. Arena gives a CARD one weight per slot whatever its
printings (the alternate arts are cosmetic styles there), so the printings are collapsed per card:
  rarity sheets (not foil, not lands, main-set printings only)   each rarity's weight is split equally among the
                                                                  sheet's cards of that rarity
  foil and wildcard sheets                                        the same, among the cards that sit on a rarity sheet;
                                                                  a card on none (MKM's surveil lands) keeps the summed
                                                                  weight of its printings
  land and bonus sheets                                           the summed weight of the card's printings
A split goes to the cards that are in Arena's packs (seen in a pick-1 pack of the set's 17lands draft logs): a card
missing there leaves its share to the others.

The land slot. Arena's land slot holds exactly one card per pack, so a card from it never shares a pack with a basic land.
In the set's 17lands draft logs a nonbasic card seen in at least LAND_SLOT_MIN_PACKS pick-1 packs is a land-slot card when
it shares them with a basic less than LAND_SLOT_MAX_SHARE times as often as independence predicts (0 for a land-slot
card, ~1 for any other). This finds exactly the cards that the Arena card database files under the land slot's rarity of
its own (Cards.Rarity 1) in all 16 sets (checked 2026-11-18), TMT's Escape Tunnel included, which paper's land sheets
alone miss. A set whose Arena packs carry no basics (basics in under LAND_SLOT_MIN_BASIC_SHARE of its pick-1 packs: BLB,
ECL, EOE, MKM) has no land slot to find, and the card database files no nonbasic card under it there either. 95 of the 96
main-set land-slot cards sit on paper's land sheets, but on paper some also sit on the foil / wildcard sheets (FDN's gain
lands) and TMT's Escape Tunnel sits on the common sheet instead. On Arena such a card comes from the land slot only, so it
takes no share of the rarity, foil or wildcard sheets (its weight there stays with the other cards), and a card that is
on no paper land sheet joins every land sheet's nonbasic cards, whose weight is then spread equally over them all.

Both used to be read off the Arena card database (Raw_CardDatabase_*.mtga in the Arena install), whose file name changes
with every Arena update. The draft logs give the same f for all 4,458 Premier cards that have one, and no f for the same
6 (2026-11-18).

The earlier rule summed the weights of the printings found on Arena one by one. That counted cosmetic Arena styles as
extra printings (TLA scene / ECL showcase uncommons ~1.8x too high) and dropped paper printings Arena does not have,
whose weight Arena gives to the card (DSK / OTJ cards 1.4-1.5x too low). Measured 2026-10-26 against the 17lands pick-1
counts of 16 sets (real / predicted per card): main-sheet mean z^2 29.8 -> 6.7, TLA 48 -> 5.0, ECL 109 -> 9.3, DSK 85 ->
2.6, OTJ 56 -> 1.4, FDN 23.6 -> 3.1 (gain lands 0.85 -> 1.00), TMT 30 -> 11 (Escape Tunnel 0.68 -> 0.92), MKM 1.3 -> 1.0.
What is left is Arena's own collation, which MTGJSON does not describe: its basic / nonbasic split
of the land slot (FIN 1/3 basics vs paper 0.45, TDM 1/6 vs 0.13, TMT 0.45 vs 0.40), its mythic share and its bonus-sheet
rates.
"""
import functools

import polars
import spells.cache

import mtgjson
import utils

DRAFT_LOGS_TYPE = spells.cache.EventType.PREMIER
BASIC_LAND_NAMES = ("Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes")
LAND_SLOT_MIN_BASIC_SHARE = 0.05   # Cutoff share of basics lands in a pack. Below this we assume no land slot.
LAND_SLOT_MIN_PACKS = 100          # Cutoff raw count of a nonbasic land in a land slot that is considered.
LAND_SLOT_MAX_SHARE = 0.02         # Cutoff max share of land slot.


def front_face(card_name):
  return card_name.split(" // ")[0]


@functools.cache
def get_arena_cards_and_land_slot_cards_for_set(set_code):
  logs = polars.scan_parquet(spells.cache.data_file_path(set_code, "draft", DRAFT_LOGS_TYPE))

  repaired = {broken: name for (drafted_in, broken), (name, _, _) in utils.BROKEN_17LANDS_CARDS.items() if drafted_in == set_code}

  names = {}
  for column in logs.collect_schema().names():
    if column.startswith("pack_card_"):
      name = column.removeprefix("pack_card_")
      names[column] = repaired.get(name, name)

  in_pack = {column: polars.col(column).fill_null(0) > 0 for column in names}
  basics = [in_pack[column] for column, name in names.items() if name in BASIC_LAND_NAMES]
  with_basic = polars.any_horizontal(basics) if basics else polars.lit(False)
  counts = (logs.filter(polars.col("pick_number") == 0)
                .select(polars.len().alias("packs"), with_basic.sum().alias("packs_with_basic"),
                        *[in_pack[column].sum().alias(f"seen|{column}") for column in names],
                        *[(in_pack[column] & with_basic).sum().alias(f"with_basic|{column}") for column in names])
                .collect()
                .row(0, named=True))

  basic_share = counts["packs_with_basic"] / counts["packs"]
  cards, land_slot = set(), set()
  for column, name in names.items():
    seen = counts[f"seen|{column}"]
    if not seen:
      continue

    cards.add(front_face(name))

    if basic_share < LAND_SLOT_MIN_BASIC_SHARE:
      continue

    if name in BASIC_LAND_NAMES or (seen >= LAND_SLOT_MIN_PACKS and
                                    counts[f"with_basic|{column}"] < LAND_SLOT_MAX_SHARE * seen * basic_share):
      land_slot.add(front_face(name))

  return frozenset(cards), frozenset(land_slot)


def is_in_arena_draft_packs(card_name, set_code):
  return front_face(card_name) in get_arena_cards_and_land_slot_cards_for_set(set_code)[0]


def is_in_arena_draft_pack_land_slot(card_name, set_code):
  return front_face(card_name) in get_arena_cards_and_land_slot_cards_for_set(set_code)[1]


def disambiguate_reversible_card_printings(card_json):
  # MTGJSON names card with layout "reversible_card" as "X // X". Fix this.
  name = card_json["name"]

  if card_json.get("layout") == "reversible_card":
    halves = name.split(" // ")
    half = len(halves) // 2
    if len(halves) % 2 == 0 and halves[:half] == halves[half:]:
      return " // ".join(halves[:half])

  return name


def sheet_kind(sheet_name, sheet, set_of_uuid, main_set_code):
  lowered = sheet_name.lower()
  if "land" in lowered or "basic" in lowered:
    return "land"
  if sheet.get("foil") or "wildcard" in lowered:
    return "foil_or_wildcard"
  if all(set_of_uuid.get(uuid) == main_set_code for uuid in sheet["cards"]):
    return "rarity"
  return "bonus"


def get_card_frequency(card_desc, main_set_code, set_jsons):
  # Expected number of copies of the card per main_set_code play booster

  card_name = card_desc["name"]
  set_code = card_desc["set_code"]

  printing = {}  # Maps uuid -> (set code, card name, rarity)
  basic = set()  # Names of basic lands
  for code in {main_set_code, set_code}:
    for card_json in set_jsons[code]["data"]["cards"]:
      printing[card_json["uuid"]] = (code, disambiguate_reversible_card_printings(card_json), card_json["rarity"])

      if "Basic" in card_json.get("supertypes", []):
        basic.add(disambiguate_reversible_card_printings(card_json))

  own = {card_json["uuid"] for card_json in set_jsons[set_code]["data"]["cards"]
         if disambiguate_reversible_card_printings(card_json) == card_name or card_json.get("faceName") == card_name}
  if not own:
    raise ValueError(f"No MTGJSON printing of {card_name} in {set_code}")

  own_names = {printing[uuid][1] for uuid in own}

  play_booster_json = set_jsons[main_set_code]["data"]["booster"]["play"]
  sheets = play_booster_json["sheets"]
  set_of_uuid = {uuid: info[0] for uuid, info in printing.items()}
  kinds = {name: sheet_kind(name, sheet, set_of_uuid, main_set_code) for name, sheet in sheets.items()}
  on_rarity_sheet = {printing[uuid][1] for name, sheet in sheets.items() if kinds[name] == "rarity"
                     for uuid in sheet["cards"] if uuid in printing}

  land_slot = {name for code, name, _ in printing.values() if code == main_set_code and is_in_arena_draft_pack_land_slot(name, code)}
  joins_land_slot = land_slot & on_rarity_sheet

  draw_probability = {}  # Maps sheet -> probability of drawing the requested card from it
  for sheet_name, sheet in sheets.items():
    total_weight = sheet["totalWeight"]
    own_weights = {uuid: weight for uuid, weight in sheet["cards"].items() if uuid in own}

    if kinds[sheet_name] == "land" and joins_land_slot and not own_names & basic:
      nonbasic_weight = sum(weight for uuid, weight in sheet["cards"].items()
                            if uuid in printing and printing[uuid][1] not in basic)
      nonbasic = {printing[uuid][1] for uuid in sheet["cards"] if uuid in printing and printing[uuid][1] not in basic}
      nonbasic |= joins_land_slot

      if own_names & nonbasic and nonbasic_weight:
        draw_probability[sheet_name] = nonbasic_weight / len(nonbasic) / total_weight

      continue

    if not own_weights:
      continue

    if kinds[sheet_name] in ("land", "bonus"):
      draw_probability[sheet_name] = sum(own_weights.values()) / total_weight
      continue

    if own_names & land_slot:
      # It comes from the land slot only
      continue

    probability = 0.0
    for rarity in {printing[uuid][2] for uuid in own_weights}:
      if kinds[sheet_name] == "foil_or_wildcard" and not own_names & on_rarity_sheet:
        probability += sum(w for uuid, w in own_weights.items() if printing[uuid][2] == rarity) / total_weight
        continue

      mass, members = 0.0, set()
      for uuid, weight in sheet["cards"].items():
        if uuid not in printing or printing[uuid][2] != rarity:
          continue

        name = printing[uuid][1]
        if kinds[sheet_name] == "foil_or_wildcard" and name not in on_rarity_sheet:
          continue

        mass += weight
        if is_in_arena_draft_packs(name, main_set_code) and name not in land_slot:
          members.add(name)

      if not own_names & members:
        raise ValueError(f"{card_name} ({set_code}) is on the {main_set_code} play booster's {sheet_name} sheet but "
                         f"not in its Arena draft packs")

      probability += mass / len(members) / total_weight

    draw_probability[sheet_name] = probability

  if not draw_probability:
    raise ValueError(f"{card_name} ({set_code}) gets no draw from any sheet of the {main_set_code} play booster")

  total_booster_weight = play_booster_json["boostersTotalWeight"]
  result = 0.0
  for booster_layout in play_booster_json["boosters"]:
    for sheet_name, count in booster_layout["contents"].items():
      result += count * booster_layout["weight"] * draw_probability.get(sheet_name, 0.0) / total_booster_weight

  return result


if __name__ == '__main__':
    # For testing
    # set_jsons = {"TLA": mtgjson.get_set_json("TLA"), "TLE": mtgjson.get_set_json("TLE")}
    # print(get_card_frequency({'name': "Black Sun's Zenith", 'set_code': 'TLE', 'is_bonus_sheet': True}, "TLA", set_jsons))
    # set_jsons = {"EOE": mtgjson.get_set_json("EOE"), "EOS": mtgjson.get_set_json("EOS"),"SPG": mtgjson.get_set_json("SPG")}
    # print(get_card_frequency({'name': "Honor", 'set_code': 'EOE', 'is_bonus_sheet': False}, "EOE", set_jsons))
    # print(get_card_frequency({'name': "Kavaron, Memorial World", 'set_code': 'EOE', 'is_bonus_sheet': False}, "EOE", set_jsons))
    # print(get_card_frequency({'name': "Brightspear Zealot", 'set_code': 'EOE', 'is_bonus_sheet': False}, "EOE", set_jsons))
    # print(get_card_frequency({'name': "Green Sun's Zenith", 'set_code': 'SPG', 'is_bonus_sheet': True}, "EOE", set_jsons))
    # set_jsons = {"FIN": mtgjson.get_set_json("FIN"), "FCA": mtgjson.get_set_json("FCA")}
    # print(get_card_frequency({'name': "Thrasios, Triton Hero", 'set_code': 'FCA', 'is_bonus_sheet': True}, "FIN", set_jsons))
    # set_jsons = {"DSK": mtgjson.get_set_json("DSK"), "SPG": mtgjson.get_set_json("SPG")}
    # print(get_card_frequency({'name': "Painter's Studio // Defaced Gallery", 'set_code': 'DSK', 'is_bonus_sheet': False}, "DSK", set_jsons))
    set_jsons = {"MKM": mtgjson.get_set_json("MKM"), "PLST": mtgjson.get_set_json("PLST")}
    print(get_card_frequency({'name': "Quintorius, Field Historian", 'set_code': 'PLST', 'is_bonus_sheet': True}, "MKM", set_jsons))
    print(get_card_frequency({'name': "Consign // Oblivion", 'set_code': 'PLST', 'is_bonus_sheet': True}, "MKM", set_jsons))
