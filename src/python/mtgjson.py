import functools
import gzip
import io
import json

import requests


@functools.cache
def get_set_json(set_code):
    url = f"https://mtgjson.com/api/v5/{set_code}.json.gz"
    
    print(f"Fetching MTGJSON data from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    gzipped_data = io.BytesIO(response.content)
    with gzip.open(gzipped_data, 'rb') as f:
        decompressed_data = f.read()

    json_string = decompressed_data.decode('utf-8')

    data = json.loads(json_string)

    return data
