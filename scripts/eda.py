import argparse
import hashlib
import json
import statistics
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('snapshot', type=Path)
parser.add_argument('--output', type=Path, default=Path('data/eda.md'))
args = parser.parse_args()
root = args.snapshot.resolve()
images_dir = args.output.parent / 'images'
images_dir.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
stats, examples, empty_labels = {}, {}, []
hashes = defaultdict(list)
classes = ('handwritten', 'printed', 'formula', 'table', 'annotation', 'image', 'graph')
sources = ('dictation', 'archive', 'university', 'school')

for split in ('train', 'silver', 'test'):
    counts, source_counts, type_counts, issues, lengths = Counter(), Counter(), Counter(), Counter(), []
    rows = [json.loads(line) for line in (root / split / 'metadata.jsonl').read_text().splitlines()]
    for row in rows:
        counts['pages'] += 1
        source_counts[row['source']] += 1
        path = root / split / row['file_name']
        if path.is_file():
            with path.open('rb') as stream:
                hashes[hashlib.file_digest(stream, 'sha256').hexdigest()].append((split, row['file_name']))
            with Image.open(path) as image:
                issues['Image/metadata size mismatch'] += image.size != (row['image_width'], row['image_height'])
                issues['Non-default EXIF orientation'] += image.getexif().get(274, 1) not in (1, None)
        else:
            issues['Missing images'] += 1
        for index, region in enumerate(row['regions']):
            kind, text = region['type'], region['text']
            type_counts[kind] += 1
            x1, y1, x2, y2 = region['bbox']
            issues['Invalid boxes'] += not (0 <= x1 < x2 <= row['image_width'] and 0 <= y1 < y2 <= row['image_height'])
            issues['Image/graph regions with text'] += kind in ('image', 'graph') and bool(text.strip())
            if kind not in ('image', 'graph') and region['legibility'] == 'legible' and not text.strip():
                issues['Empty legible labels'] += 1
                if split == 'train':
                    empty_labels.append(f'{row["file_name"]}, region {index} ({kind})')
            if kind == 'handwritten' and region['language'] == 'uk' and region['legibility'] == 'legible' and text.strip():
                lengths.append(len(text))
                if split == 'train' and row['source'] not in examples and 20 <= len(text) <= 80:
                    examples[row['source']] = (path, region['bbox'])
    stats[split] = dict(counts=counts, sources=source_counts, types=type_counts, issues=issues, lengths=lengths)
    print(f'{split}: {len(rows)} pages', flush=True)

font = ImageFont.load_default(size=18)
colors = ('#2563eb', '#d97706', '#16805d', '#9333ea', '#be185d', '#64748b', '#0891b2')


def chart(field, labels, title, filename):
    canvas = Image.new('RGB', (1000, 350), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), title, fill='black', font=font)
    for row_index, (split, data) in enumerate(stats.items()):
        y, x = 65 + row_index * 55, 150
        total = sum(data[field].values())
        draw.text((20, y + 7), split, fill='black', font=font)
        for label, color in zip(labels, colors):
            width = 820 * data[field][label] / total
            if width:
                draw.rectangle((x, y, x + width, y + 35), fill=color)
                if width > 50:
                    draw.text((x + 5, y + 7), f'{data[field][label] / total:.0%}', fill='white', font=font)
            x += width
    for index, (label, color) in enumerate(zip(labels, colors)):
        x, y = 20 + index % 4 * 245, 255 + index // 4 * 40
        draw.rectangle((x, y, x + 15, y + 15), fill=color)
        draw.text((x + 22, y - 2), label, fill='black', font=font)
    canvas.save(images_dir / filename)


chart('sources', sources, 'Page sources within each split (%)', 'sources.png')
chart('types', classes, 'Region types within each split (%)', 'regions.png')
canvas = Image.new('RGB', (1000, 600), 'white')
draw = ImageDraw.Draw(canvas)
for index, source in enumerate(sources):
    path, box = examples[source]
    with Image.open(path) as image:
        crop = image.crop(tuple(box)).convert('RGB')
    crop.thumbnail((950, 105))
    y = index * 150
    draw.text((20, y + 5), source, fill='black', font=font)
    canvas.paste(crop, (20, y + 35))
canvas.save(images_dir / 'examples.png')
lines = ['# RUKOPYS EDA', '', f'Snapshot: `{root.name}`. Descriptive EDA includes train, silver, and test; use validation for tuning.', '']


def table(headers, rows):
    lines.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'])
    lines.extend('| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows)
    lines.append('')


table(['Split', 'Pages', 'Regions', 'HTR candidates', 'Median text length'],
      [(split, f'{data["counts"]["pages"]:,}', f'{sum(data["types"].values()):,}', f'{len(data["lengths"]):,}',
        int(statistics.median(data['lengths']))) for split, data in stats.items()])
lines += ['HTR candidates are non-empty, legible Ukrainian handwritten labels; lengths count characters, not tokens.', '',
          '## Examples and distributions', '', 'Four training-line crops, one per source. Source: RUKOPYS/UCU, CC BY 4.0; cropped/resized.', '',
          '![Handwritten line examples](images/examples.png)', '', '![Page source distribution](images/sources.png)', '',
          'Silver is archive-heavy and has no dictation pages. The test source mixture differs from training.', '',
          '![Region type distribution](images/regions.png)', '', 'Handwriting dominates; tables and graphs are rare.', '', '## Checks that affect training', '']
checks = ('Missing images', 'Invalid boxes', 'Image/metadata size mismatch', 'Non-default EXIF orientation', 'Empty legible labels', 'Image/graph regions with text')
table(['Check', 'Train', 'Silver', 'Test'], [(check, *(stats[split]['issues'][check] for split in stats)) for check in checks])
within = Counter()
cross = 0
for entries in hashes.values():
    counts = Counter(split for split, _ in entries)
    within.update({split: count - 1 for split, count in counts.items()})
    cross += len(counts) > 1
lines += [f'Exact duplicate extra copies: train {within["train"]}, silver {within["silver"]}, test {within["test"]}; {cross} shared hash groups across splits. Hashes do not detect near-duplicates.', '',
          'Keep boxes aligned when handling EXIF orientation. Image/graph targets should be empty. Keep official test annotations unchanged for comparable evaluation.', '',
          '## Empty training transcripts', '', 'These three regions are marked legible but have no text. Review them before using recognition targets; their boxes may still be useful for detection. Region indices start at zero.', '']
lines += [f'- `{label}`' for label in empty_labels]
args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Wrote {args.output} and three figures', flush=True)
