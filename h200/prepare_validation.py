"""Explicitly generate the synthetic near-1M fixture used by the capacity test.

This contacts only the selected API's tokenizer. It downloads no dataset, runs no
generation requests, and does nothing on import. The final compressed fixture is
several megabytes; choose a local output directory outside the source checkout.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/dsv41"))
from kit_bench import Client, WORDS, load_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Server base URL, without /v1")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017-H200")
    parser.add_argument("--key-file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    client = Client(args.base, args.model, load_key(args.key_file))
    rng = random.Random(20261018)
    facts = [
        "案件名は春風計画。担当者は佐藤葵。予算は350万円。納期は2027年2月10日。",
        "案件名は青空計画。担当者は鈴木蓮。予算は240万円。納期は2027年1月20日。",
        "案件名は星海計画。担当者は高橋凛。予算は510万円。納期は2027年3月5日。",
    ]
    records = []
    # Preserve the original RNG history: the recorded campaign generated 32K
    # and 128K fixtures before the near-1M fixture. Only the last text is saved.
    for length in (32768, 131072, 1047000):
        phrases = [f'{rng.choice(["amber", "cobalt", "falcon", "granite"])}-'
                   f'{rng.choice(["harbor", "juniper", "lantern", "quartz"])}-{rng.randrange(1000,9999)}'
                   for _ in range(3)]
        words = rng.choices(WORDS, k=int(length * .75))

        def build():
            pieces = ["Read the notes below.\n"]
            previous = 0
            for index, depth in enumerate((.1, .5, .9)):
                cut = int(len(words) * depth)
                pieces.extend([" ".join(words[previous:cut]),
                               f"\nThe secret passphrase for vault {chr(65+index)} is {phrases[index]}. Remember it.\n",
                               facts[index], "\n"])
                previous = cut
            pieces.append(" ".join(words[previous:]))
            return " ".join(pieces)

        while True:
            text = build()
            count = len(client.text_ids(text))
            if count >= length - 150:
                break
            words.extend(rng.choices(WORDS, k=max(1, int((length - count) * .8))))
        if count >= 1048000:
            raise ValueError("Tokenizer produced too many input tokens for the 1M test")
        raw = text.encode()
        records.append({"case": "prompt_prepared", "target_length": length,
                        "raw_tokens": count, "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw), "phrases": phrases, "facts": facts,
                        "depths_by_word_position": [.1, .5, .9]})
        print(json.dumps({"target_length": length, "raw_tokens": count,
                          "sha256": records[-1]["sha256"]}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / "prompt-1047000.txt.gz", "wb") as handle:
        handle.write(raw)
    result = {"seed": 20261018, "synthetic_fixture": True,
              "expected_campaign_sha256": "bde79cc15b8d7f07802fb84ed6e3cd0753e9be9d4d0fd94a25eff3eb79c82fae",
              "rows": records}
    result["matches_recorded_campaign"] = records[-1]["sha256"] == result["expected_campaign_sha256"]
    (args.out / "validation-fixture.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"fixture_ready": True, "matches_recorded_campaign": result["matches_recorded_campaign"]}))


if __name__ == "__main__":
    main()
