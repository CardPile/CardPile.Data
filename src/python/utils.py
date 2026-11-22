import polars

# 17lands ships a few cards with bad data. Used to give them a real name, set code and bonus-sheet flag back in repair_broken_cards.
BROKEN_17LANDS_CARDS = {
  ("TMT", "Bespoke B?"): ("Bespoke Bō", "TMT", False),
  ("MKM", "Evolutionary Leap"): ("Evolutionary Leap", "ORI", True),
  ("MKM", "Possibility Storm"): ("Possibility Storm", "DGM", True),
}


MIN_NUM_GIH = 450


def filter_basic_lands(df):
  return df.filter(polars.col("name").is_in(["Plains", "Island", "Swamp", "Mountain", "Forest"]).not_())


def sanitize_data_frame(df):
  return df.filter(polars.col("set_code").is_not_null() & (polars.col("set_code").str.len_chars() > 0))


def repair_broken_cards(df, set_code):
  import mtgjson

  for (drafted_in, broken_name), (name, card_set_code, is_bonus_sheet) in BROKEN_17LANDS_CARDS.items():
    if drafted_in != set_code:
      continue

    broken = polars.col("name") == broken_name

    if "set_code" in df.columns:
      broken = broken & (polars.col("set_code").is_null() | (polars.col("set_code") == ""))

    if df.filter(broken).is_empty():
      continue

    front = next(card for card in mtgjson.get_set_json(card_set_code)["data"]["cards"]
                 if card["name"] == name and card.get("side", "a") == "a")

    values = {"name": name,
              "set_code": card_set_code,
              "is_bonus_sheet": is_bonus_sheet,
              "rarity": front["rarity"],
              "card_type": " ".join(front.get("types", [])),
              "mana_value": float(front.get("manaValue", 0.0)),
              "mana_cost": front.get("manaCost", "")
    }

    df = df.with_columns(*[polars.when(broken).then(polars.lit(value, dtype=df.schema[column])).otherwise(polars.col(column)).alias(column)
                           for column, value in values.items() if column in df.columns])
  return df


def default_card_population_filter():
  return polars.col("alsa").is_not_null() & (polars.col("num_gih") > MIN_NUM_GIH)


def max_packs():
  return 3


def max_picks():
  return 14
