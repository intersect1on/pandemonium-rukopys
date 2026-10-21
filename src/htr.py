"""Shared helpers for the handwritten text recognition (HTR) notebooks."""
import json
import re
import unicodedata
from pathlib import Path

import jiwer
import torch
from huggingface_hub import snapshot_download
from PIL import Image

REVISION = '78697007cfa5f77aae5d3b9fcedfac3169f84fb6'
ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / 'data/splits/train_val.csv'
CROP_DIR = ROOT / 'data/htr_crops'
CROP_FILE = CROP_DIR / 'crops.csv'
SOURCES = ('dictation', 'archive', 'university', 'school')

Image.MAX_IMAGE_PIXELS = None


def snapshot():
    return Path(snapshot_download('UkrainianCatholicUniversity/rukopys', repo_type='dataset',
                                  revision=REVISION, local_files_only=True))


def load_pages(split):
    with open(snapshot() / split / 'metadata.jsonl', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def is_htr_line(region):
    return (region['type'] == 'handwritten' and region['language'] == 'uk'
            and region['legibility'] == 'legible' and bool(region['text'].strip()))


STRIKE = re.compile(r'~~(.*?)~~(?:\{(.*?)\})?', re.S)
APOSTROPHES = str.maketrans({'’': "'", 'ʼ': "'", '‘': "'", 'ʹ': "'"})


def clean_text(text):
    """Recognition target for a raw transcript, or None if the line should be skipped.

    Struck-out text is dropped and its correction kept (`~~old~~{new}` -> `new`).
    Lines with `[illegible]`, unmatched `~` markers, or nothing left after cleaning are skipped.
    """
    if '[illegible]' in text:
        return None
    text = STRIKE.sub(lambda m: m.group(2) or '', text)
    if '~' in text:
        return None
    text = unicodedata.normalize('NFC', text).translate(APOSTROPHES).replace('\u200b', '')
    text = ' '.join(text.split())
    return text or None


def cer(refs, hyps):
    return jiwer.cer(list(refs), list(hyps))


def wer(refs, hyps):
    return jiwer.wer(list(refs), list(hyps))


class LineDataset(torch.utils.data.Dataset):
    def __init__(self, crops, processor, max_tokens=128):
        self.paths = [CROP_DIR / p for p in crops['crop_path']]
        self.texts = list(crops['target'])
        self.processor = processor
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            pixels = self.processor(images=image.convert('RGB'), return_tensors='pt').pixel_values[0]
        labels = self.processor.tokenizer(self.texts[i], max_length=self.max_tokens, truncation=True).input_ids
        return pixels, torch.tensor(labels)


def collate(batch, pad_id):
    pixels, labels = zip(*batch)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_id)
    labels[labels == pad_id] = -100
    return torch.stack(pixels), labels


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    device = next(model.parameters()).device
    losses = []
    for pixels, labels in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(pixel_values=pixels.to(device), labels=labels.to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def predict(model, processor, crops, batch_size=32, **generate_kwargs):
    model.eval()
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(LineDataset(crops, processor), batch_size=batch_size, num_workers=4,
                                         collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id))
    out = []
    for pixels, _ in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ids = model.generate(pixels.to(device), use_cache=True, **generate_kwargs)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return [' '.join(t.split()) for t in out]
