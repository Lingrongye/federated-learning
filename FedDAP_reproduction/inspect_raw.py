"""Read verified originals; preserve labeled source thumbnails for human review."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from raw_data import DOMAINS, read_arrays, verify_sources


def inspect(root, output):
    provenance = verify_sources(root)
    output = Path(output)
    canvas = Image.new("RGB", (760, 4 * 80), "white")
    draw = ImageDraw.Draw(canvas)
    samples = {}
    for row, domain in enumerate(DOMAINS):
        images, labels = read_arrays(root, domain, True)
        draw.text((4, row * 80 + 28), domain, fill="black")
        for label in range(10):
            positions = np.flatnonzero(labels == label)
            if not len(positions):
                raise ValueError(f"Published split is missing class {label}: {domain}")
            index = int(positions[0])
            image = Image.fromarray(images[index]).convert("RGB").resize((48, 48))
            left = 82 + label * 66
            canvas.paste(image, (left, row * 80 + 20))
            draw.text((left, row * 80 + 3), str(label), fill="black")
            samples[f"{domain}_class{label}_index"] = np.array(index)
            samples[f"{domain}_class{label}_pixels"] = images[index]
    # Source previews are immutable like the experiment's other artifacts.
    with open(output / "source_preview.png", "xb") as stream:
        canvas.save(stream, format="PNG")
    with open(output / "source_samples.npz", "xb") as stream:
        np.savez_compressed(stream, **samples)
    with open(output / "download_manifest.json", "x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps({"source_inspection": "PASS", "splits": provenance["splits"],
                      "preview": str(output / "source_preview.png")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inspect(args.root, args.output)
