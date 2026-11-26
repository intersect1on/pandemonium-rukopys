"""Page -> detector -> line crops -> TrOCR -> text in reading order."""
import gc
import math
from pathlib import Path

import jiwer
import torch
from PIL import Image

from cv import CLASSES, predict_pages, suppress_duplicates
from detectors import load_variant
from htr import clean_text, is_htr_line, snapshot

THRESHOLD = 0.55
NMS_IOU = 0.5


def detect(pages, checkpoint, threshold=THRESHOLD, device='gpu', *, image_root=None):
    """Detected regions per page (score >= threshold), top to bottom. Returns {file_name: [region, ...]}."""
    processor, model = load_variant('rf-small', checkpoint=checkpoint, device=device)
    raw = predict_pages(model, processor, pages, image_root=image_root)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    found = {}
    for p in suppress_duplicates(raw, NMS_IOU):
        regions = [dict(box=box, type=CLASSES[label], score=score)
                   for box, label, score in zip(p['boxes'], p['labels'], p['scores']) if score >= threshold]
        found[p['file_name']] = sorted(regions, key=lambda r: ((r['box'][1] + r['box'][3]) / 2, r['box'][0]))
    return found


@torch.no_grad()
def recognize(model, processor, images, batch_size=16, **decoding):
    device = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        pixels = processor(images=images[i:i + batch_size], return_tensors='pt').pixel_values
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ids = model.generate(pixels.to(device), use_cache=True, **decoding)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return [' '.join(t.split()) for t in out]


def transcribe(page, regions, model, processor, *, image_root=None, **decoding):
    """Adds `text` to every handwritten region (None for other types). Crops come from the raw image."""
    handwritten = [r for r in regions if r['type'] == 'handwritten']
    crops = []
    image_root = Path(image_root) if image_root is not None else snapshot() / 'train'
    with Image.open(image_root / page['file_name']) as source:
        for r in handwritten:
            x1, y1, x2, y2 = r['box']
            crop = source.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))).convert('RGB')
            crop.thumbnail((1024, 1024))
            crops.append(crop)
    texts = iter(recognize(model, processor, crops, **decoding) if crops else [])
    return [r | dict(text=next(texts) if r['type'] == 'handwritten' else None) for r in regions]


def reference_lines(page):
    """Eligible ground-truth lines in the same top-to-bottom order as the detections."""
    lines = [(r['bbox'], clean_text(r['text'])) for r in page['regions'] if is_htr_line(r)]
    lines = [(box, text) for box, text in lines if text]
    return [text for box, text in sorted(lines, key=lambda l: ((l[0][1] + l[0][3]) / 2, l[0][0]))]


def page_errors(reference, hypothesis):
    """Character edit operations: missed lines are deletions, extra predictions are insertions."""
    if not reference or not hypothesis:
        return len(reference) + len(hypothesis)
    m = jiwer.process_characters(reference, hypothesis)
    return m.substitutions + m.deletions + m.insertions
