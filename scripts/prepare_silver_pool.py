"""Build a shared 1,050-page silver pool for RF-DETR and TrOCR."""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from htr import ROOT, SILVER_TEST_REVISION, clean_text, is_htr_line, load_pages, split_root

OUTPUT = ROOT / 'data/silver_pool'
QUOTAS = dict(archive=350, school=350, university=350)
SEED = 42


def fingerprint(path):
    with Image.open(path) as im:
        im.draft('L', (64, 64))
        small = np.asarray(im.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')


def phrases(page):
    words = re.findall(r'[а-яіїєґ]+', ' '.join(r['text'] for r in page['regions'] if is_htr_line(r)).lower())
    return {tuple(words[i:i + 12]) for i in range(len(words) - 11)}


def candidates(page, image):
    for i, region in enumerate(page['regions']):
        if not is_htr_line(region):
            continue
        target = clean_text(region['text'])
        x, y, right, bottom = region['bbox']
        w, h = right - x, bottom - y
        if not target or not re.search(r'[А-Яа-яІіЇїЄєҐґ]', target):
            continue
        if not (4 <= len(target) <= 140 and 2 <= w / h <= 80 and h >= 10):
            continue
        crop = image.crop(region['bbox']).convert('RGB')
        crop.thumbnail((1024, 1024))
        if ImageStat.Stat(crop.convert('L')).stddev[0] >= 12:
            yield i, region, target, crop


class PageScreen:
    def __init__(self):
        self.image_root = split_root('silver')
        self.blocked = set()
        self.hashes, self.text = [], set()
        for split in ('train', 'test'):
            images = split_root(split)
            for page in load_pages(split):
                name = page['file_name']
                self.blocked.add(name)
                self.hashes.append(fingerprint(images / name))
                if split == 'train':
                    self.text.update(phrases(page))

    def check(self, page, *, export=False):
        name = page['file_name']
        if name in self.blocked:
            return None
        boxes = [tuple(r['bbox']) for r in page['regions']]
        width, height = page['image_width'], page['image_height']
        if len(set(boxes)) != len(boxes) or not all(0 <= x < r <= width and 0 <= y < b <= height for x, y, r, b in boxes):
            return None
        words = phrases(page)
        if words and len(words & self.text) / len(words) >= 0.25:
            return None
        path = self.image_root / name
        signature = fingerprint(path)
        if any((signature ^ h).bit_count() <= 5 for h in self.hashes):
            return None
        with Image.open(path) as image:
            if image.size != (width, height) or image.getexif().get(274, 1) not in (0, 1):
                return None
            lines = list(candidates(page, image))
        if len(lines) < 3:
            return None
        self.blocked.add(name)
        self.hashes.append(signature)
        self.text.update(words)
        return lines if export else True

    def reserve(self, page):
        name = page['file_name']
        self.blocked.add(name)
        self.hashes.append(fingerprint(self.image_root / name))
        self.text.update(phrases(page))


def approved_pages(silver, manifest):
    by_id = {Path(p['file_name']).stem: p for p in silver}
    with manifest.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['page_id', 'source']:
            raise ValueError('Approved manifest must have page_id,source columns')
        rows = list(reader)
    if Counter(r['source'] for r in rows) != QUOTAS or len({r['page_id'] for r in rows}) != len(rows):
        raise ValueError('Approved manifest must have unique IDs and respect the source quotas')
    pages = []
    for row in rows:
        page = by_id.get(row['page_id'])
        if page is None or page['source'] != row['source']:
            raise ValueError(f"Approved page missing from its v3.0 silver source: {row['page_id']}")
        pages.append(page)
    return pages


def build(out, manifest=None):
    silver = load_pages('silver')
    if manifest is None:
        random.Random(SEED).shuffle(silver)
    else:
        silver = approved_pages(silver, manifest)
    screen = PageScreen()
    counts, rows, pages = Counter(), [], []
    (out / 'crops').mkdir()
    for page in silver:
        source, name = page['source'], page['file_name']
        if source not in QUOTAS or counts[source] >= QUOTAS[source]:
            continue
        lines = screen.check(page, export=True)
        if lines is None:
            if manifest is not None:
                raise ValueError(f'Approved page failed the eligibility or overlap check: {name}')
            continue
        page_id = Path(name).stem
        for i, region, target, crop in lines:
            rel = f'crops/{page_id}_{i:03}.jpg'
            crop.save(out / rel, quality=95)
            rows.append(dict(page_id=page_id, region=i, source=source, split='silver', type='handwritten',
                             annotation_source=page['annotation_source'], bbox=json.dumps(region['bbox']),
                             crop_path=rel, text=region['text'], target=target))
        counts[source] += 1
        pages.append(page)
    if counts != QUOTAS:
        raise ValueError(f'Could not reach the source quotas: {dict(counts)}')
    with (out / 'crops.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'pages.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pages))
    crop_counts = Counter(r['source'] for r in rows)
    table = '\n'.join(f'| {s} | {counts[s]} | {crop_counts[s]:,} |' for s in QUOTAS)
    selection_note = (
        'Pages passed a visual layout review with transcript spot checks when the layout was usable.'
        if manifest is not None else
        f'Selection seed {SEED}. Automatic labels await full visual review.'
    )
    (out / 'README.md').write_text(
        f'# RUKOPYS silver training pool\n\n'
        f'{len(pages):,} pages and {len(rows):,} handwritten crops. {selection_note}\n\n'
        '| Source | Pages | Line crops |\n| --- | ---: | ---: |\n'
        f'{table}\n\n'
        f'Images and automatic annotations come from [RUKOPYS](https://huggingface.co/datasets/'
        f'UkrainianCatholicUniversity/rukopys), Ukrainian Catholic University, CC BY 4.0. '
        f'Revision `{SILVER_TEST_REVISION}` (v3.0). Line images were cropped and resized.\n\n'
        '`pages.jsonl` lists original boxes and region types for RF-DETR. Resolve each `file_name` '
        'against the v3.0 snapshot’s `silver/` directory.\n\n'
        '`crops.csv` lists the TrOCR inputs. Resolve `crop_path` relative to this folder. '
        '`target` contains the cleaned supplied transcript. Join on `page_id` and `region`.\n\n'
        'The build script checks image dimensions, orientation, box geometry, duplicate signals '
        'and line eligibility.\n',
        encoding='utf-8',
    )
    print(f'Wrote {len(pages):,} pages and {len(rows):,} crops', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Rebuild from exact reviewed page IDs in CSV order')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Pool output directory')
    parser.add_argument('--replace', action='store_true', help='Replace the existing generated pool')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.replace:
        raise SystemExit(f'Pool already exists: {output}. Use --replace to regenerate it.')
    with TemporaryDirectory(dir=output.parent) as tmp:
        pool = Path(tmp) / 'silver_pool'
        pool.mkdir()
        build(pool, args.manifest)
        previous = Path(tmp) / 'previous'
        if output.exists():
            output.rename(previous)
        try:
            pool.rename(output)
        except BaseException:
            if previous.exists():
                previous.rename(output)
            raise
