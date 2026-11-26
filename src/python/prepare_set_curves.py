import argparse
import datetime
import functools
import hashlib
import json
import os
import pathlib
import subprocess

import numpy
import polars
import requests
import spells
from spells import cards, external
from spells.cache import EventType
from spells.enums import View

import card_curve_noise
import card_curves
import utils

OUTPUT_DIR = "set_curve_data"
KEY = ["name", "set_code", "is_bonus_sheet"]


def card_stats(set_code):
  df = spells.summon(set_code,
                     columns=[spells.ColName.SET_CODE, spells.ColName.IS_BONUS_SHEET, spells.ColName.ALSA],
                     event_type=EventType.PREMIER)
  df = utils.repair_broken_cards(df, set_code)
  df = utils.filter_basic_lands(df)
  df = utils.sanitize_data_frame(df)
  return df


def set_curves(set_code):
  counts = card_curves.pick_count_frames(set_code, EventType.PREMIER)
  every_card = polars.concat([card_stats(set_code), *counts], how="align_left")

  pooled = card_curves.with_card_curves(every_card, set_code)
  curves = pooled.filter(utils.default_card_population_filter())
  cleaned = card_curves.pick_counts(curves)

  first = cleaned[..., :1]
  with numpy.errstate(divide="ignore", invalid="ignore"):
    shares = numpy.where(first > 0, cleaned / first, numpy.nan)

  point, replicates, _ = card_curve_noise.replicate_curves(pooled, every_card, set_code, EventType.PREMIER, keep=curves)
  _, _, share_sd = card_curve_noise.curve_sigmas(point, replicates)

  packs, picks = range(1, utils.max_packs() + 1), range(1, utils.max_picks() + 1)
  return curves.select(KEY).with_columns(
    *[polars.Series(f"P{p}P1", cleaned[:, p - 1, 0]) for p in packs],
    *[polars.Series(f"P{p}P{k}N", shares[:, p - 1, k - 1]).fill_nan(None) for p in packs for k in picks],
    *[polars.Series(f"sigma_P{p}P{k}N", share_sd[:, p - 1, k - 1]) for p in packs for k in picks[1:]],
  ).sort(KEY)


@functools.cache
def get_public_datasets():
    url = "https://www.17lands.com/api/public_datasets"
    cutoff = datetime.datetime(2024, 2, 9)

    print(f"Fetching public dataset list from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    data = json.loads(response.content)

    result = {}
    for entry in data:
      set_code = entry['expansion']
      set_type = entry['event_type']
      last_updated = datetime.datetime.strptime(entry['last_updated'], "%Y-%m-%d")

      if set_code == "Powered Cube":
        # Special case for Powered Cube
        continue

      if set_code == "PIO":
        # Special case for PIO
        continue

      if set_type != "PremierDraft":
        continue

      if last_updated <= cutoff:
        continue

      result[set_code] = last_updated

    return result


@functools.cache
def load_manifest(data_dir):
  file_path = data_dir / "manifest.json"

  if not file_path.is_file():
    return {}

  with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

  result = {}
  for entry in data:
    set_code = entry['set_code']
    last_updated = datetime.datetime.strptime(entry['last_updated'], "%Y-%m-%d")
    digest = entry['digest']
    result[set_code] = (last_updated, digest)

  return result


def filter_valid_datasets(public_datasets, manifest_datasets):
  removed_sets = {k: manifest_datasets[k] for k in manifest_datasets if k not in public_datasets}

  common_sets = public_datasets.keys() & manifest_datasets.keys()
  up_to_date_sets = {k: manifest_datasets[k] for k in common_sets if manifest_datasets[k][0] >= public_datasets[k]}

  not_up_to_date_sets = {k: (public_datasets[k], manifest_datasets[k][1]) for k in common_sets if manifest_datasets[k][0] < public_datasets[k]}
  new_sets = {k: (public_datasets[k], 0) for k in public_datasets if k not in manifest_datasets}

  return (new_sets | not_up_to_date_sets, up_to_date_sets, removed_sets)


def download_set(set_code):
  external.download_data_set(set_code, View.DRAFT, event_type=EventType.PREMIER)
  cards.write_card_file(set_code, cards.names_from_parquet(set_code, EventType.PREMIER))


def check_set_curves(frame, set_code):
  packs, picks = range(1, utils.max_packs() + 1), range(1, utils.max_picks() + 1)
  values = frame.drop(KEY).to_numpy().astype(float)
  assert numpy.isfinite(values).all(), f"{set_code}: missing or infinite values"
  assert (frame.select([f"P{p}P1" for p in packs]).to_numpy() > 0).all(), f"{set_code}: a card with no pick-1 copies"

  shares = numpy.stack([frame.select([f"P{p}P{k}N" for k in picks]).to_numpy() for p in packs], axis=1)
  assert (numpy.abs(shares[..., 0] - 1) < 1e-9).all() and (shares >= 0).all() and (numpy.diff(shares, axis=-1) <= 1e-9).all(), f"{set_code}: shares that do not start at 1 and fall"


def fetch_sets(output_dir, sets):
  if len(sets) == 0:
    return {}

  result = {}
  for set_code in sets:
    print(f"Fetching data for set {set_code}")

    download_set(set_code)

    print(f"Calculating curves for {set_code}")
    frame = set_curves(set_code)

    check_set_curves(frame, set_code)

    frame_path = output_dir / f"{set_code}.parquet"
    frame.write_parquet(frame_path)

    with open(frame_path, "rb") as f:
      digest = hashlib.file_digest(f, "sha256").hexdigest()

    print(f"Removing data for set {set_code}")
    try:
      proces_result = subprocess.run(["spells", "remove", "--yes", set_code], capture_output=True, text=True, check=True)
      print(proces_result.stdout)
    except subprocess.CalledProcessError as e:
      print(f"'spells remove' command failed with exit code {e.returncode}")
      print(f"Error message: {e.stderr}")

    print(f"Data for set {set_code} ready with last updated {sets[set_code][0]} and digest {digest}")

    result[set_code] = (sets[set_code][0], digest)

  return result


def remove_sets(output_dir, sets):
  for set_code in sets:
    print(f"Removing data for set {set_code}")

    frame_path = output_dir / f"{set_code}.parquet"
    frame_path.unlink()


def write_manifest(output_dir, sets):
  data = []

  for (set_code, (last_updated, digest)) in sets.items():
    set_data = {}
    set_data['set_code'] = set_code
    set_data['last_updated'] = last_updated.strftime("%Y-%m-%d")
    set_data['digest'] = digest
    data.append(set_data)

  data.sort(key=lambda s: s['set_code'])

  file_path = output_dir / "manifest.json"
  with open(file_path, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)


def main():
  parser = argparse.ArgumentParser(description="Check if any expansion datasets have been updated and recompute their card decay curves.")
  parser.add_argument("output_dir", type=pathlib.Path, help="Output directory")
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)

  public_datasets = get_public_datasets()
  # print(public_datasets)

  manifest_datasets = load_manifest(args.output_dir)
  # print(manifest_datasets)

  (not_up_to_date_sets, up_to_date_sets, removed_sets) = filter_valid_datasets(public_datasets, manifest_datasets)

  not_up_to_date_sets = fetch_sets(args.output_dir, not_up_to_date_sets)

  remove_sets(args.output_dir, removed_sets)

  write_manifest(args.output_dir, not_up_to_date_sets | up_to_date_sets)


if __name__ == "__main__":
  main()
