"""Data preparation, DETR training, and validation for the CV notebooks."""
import os

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import contextlib
import hashlib
import io
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from huggingface_hub import hf_hub_download
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessorPil

from htr import CROP_DIR, REVISION, ROOT, SOURCES, SPLIT_FILE, clean_text, is_htr_line, load_pages, snapshot

CLASSES = ('handwritten', 'printed', 'formula', 'table', 'annotation', 'image', 'graph')
BASE_MODEL = 'facebook/detr-resnet-50'
BASE_REVISION = '70120ba84d68ca1211e007c4fb61d0cd5424be54'
MODEL_DIR = ROOT / 'models/detr-baseline'


def device_for(requested='auto'):
    if requested in ('auto', 'gpu'):
        requested = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(requested)


def pages_with_split():
    pages = load_pages('train')
    split = pd.read_csv(SPLIT_FILE)
    names = {page['file_name'] for page in pages}
    if split.file_name.duplicated().any() or set(split.file_name) != names:
        raise ValueError('The split must contain each official train page exactly once.')
    if set(split['split']) != {'train', 'val'}:
        raise ValueError('Expected train and val assignments only.')
    assignments = split.set_index('file_name')['split'].to_dict()
    return [page | {'split': assignments[page['file_name']]} for page in pages]


def check_pages(pages):
    counts = Counter()
    orientations = Counter()
    for index, page in enumerate(pages, 1):
        with Image.open(snapshot() / 'train' / page['file_name']) as image:
            orientations[str(image.getexif().get(274, 1))] += 1
            if image.size != (page['image_width'], page['image_height']):
                raise ValueError(f'Image/metadata size mismatch: {page["file_name"]}')
            if image.getexif().get(274, 1) not in (0, 1, None):
                raise ValueError(f'Review image/box rotation before training: {page["file_name"]}')
        for region in page['regions']:
            x1, y1, x2, y2 = region['bbox']
            if not (0 <= x1 < x2 <= page['image_width'] and 0 <= y1 < y2 <= page['image_height']):
                raise ValueError(f'Invalid box: {page["file_name"]}')
            counts[region['type']] += 1
        if index % 200 == 0 or index == len(pages):
            print(f'Checked {index}/{len(pages)} pages', flush=True)
    return {'pages': len(pages), 'regions': dict(counts), 'exif_orientations': dict(orientations),
            'max_regions_per_page': max(len(page['regions']) for page in pages)}


def prepare_gt_crops(pages, output_dir=CROP_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for number, page in enumerate(pages, 1):
        page_id = Path(page['file_name']).stem
        regions = [(i, region) for i, region in enumerate(page['regions']) if is_htr_line(region)]
        missing = any(not (output_dir / f'{page_id}_{i:03d}.jpg').exists() for i, _ in regions)
        image = None
        if missing:
            with Image.open(snapshot() / 'train' / page['file_name']) as source:
                image = source.convert('RGB')
        for index, region in regions:
            name = f'{page_id}_{index:03d}.jpg'
            if not (output_dir / name).exists():
                crop = image.crop(region['bbox'])
                crop.thumbnail((1024, 1024))
                crop.save(output_dir / name, quality=95)
            records.append(dict(page_id=page_id, region=index, bbox=region['bbox'], type=region['type'],
                                crop_path=name, text=region['text'], target=clean_text(region['text']),
                                source=page['source'], annotation_source=page['annotation_source'], split=page['split']))
        if image is not None:
            image.close()
        if number % 100 == 0 or number == len(pages):
            print(f'GT crops: {number}/{len(pages)} pages, {len(records)} lines', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / 'crops.csv', index=False)
    return frame


def subset(pages, role, limit=None, seed=42):
    selected = [page for page in pages if page['split'] == role]
    if limit and limit < len(selected):
        rng = np.random.default_rng(seed)
        groups = [[page for page in selected if page['source'] == source] for source in SOURCES]
        for group in groups:
            rng.shuffle(group)
        selected = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)][:limit]
    return selected


class PageDataset(torch.utils.data.Dataset):
    def __init__(self, pages, processor):
        self.pages, self.processor = pages, processor

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        page = self.pages[index]
        size = self.processor.size
        longest = size.get('longest_edge') or max(size.get('height', 0), size.get('width', 0))
        with Image.open(snapshot() / 'train' / page['file_name']) as original:
            original.thumbnail((longest,) * 2)
            image = original.convert('RGB')
        sx, sy = image.width / page['image_width'], image.height / page['image_height']
        annotations = []
        for region in page['regions']:
            x1, y1, x2, y2 = region['bbox']
            box = [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]
            annotations.append(dict(bbox=box, category_id=CLASSES.index(region['type']), area=box[2] * box[3], iscrowd=0))
        inputs = self.processor(images=image, annotations={'image_id': index, 'annotations': annotations}, return_tensors='pt')
        return inputs.pixel_values[0], inputs.labels[0], page


def collate_pages(batch):
    pixels, labels, pages = zip(*batch)
    height, width = max(p.shape[1] for p in pixels), max(p.shape[2] for p in pixels)
    values, masks, targets = [], [], []
    for pixel, label in zip(pixels, labels):
        h, w = pixel.shape[1:]
        values.append(torch.nn.functional.pad(pixel, (0, width - w, 0, height - h)))
        mask = torch.zeros((height, width), dtype=torch.long)
        mask[:h, :w] = 1
        masks.append(mask)
        targets.append(label | {'boxes': label['boxes'] * torch.tensor([w / width, h / height] * 2),
                                'size': torch.tensor([height, width])})
    return dict(pixel_values=torch.stack(values), pixel_mask=torch.stack(masks), labels=targets, pages=list(pages))


def loader_for(pages, processor, batch_size=1, shuffle=False):
    return torch.utils.data.DataLoader(PageDataset(pages, processor), batch_size=batch_size, shuffle=shuffle,
                                      num_workers=0, collate_fn=collate_pages)


def load_detector(checkpoint=None):
    location = str(checkpoint or BASE_MODEL)
    kwargs = {} if checkpoint else {'revision': BASE_REVISION}
    size = {} if checkpoint else {'size': {'shortest_edge': 640, 'longest_edge': 960}}
    processor = DetrImageProcessorPil.from_pretrained(location, **kwargs, **size)
    labels = dict(enumerate(CLASSES))
    config_path = Path(checkpoint) / 'config.json' if checkpoint else Path(hf_hub_download(BASE_MODEL, 'config.json', revision=BASE_REVISION))
    settings = json.loads(config_path.read_text())
    settings.update(dilation=bool(settings.get('dilation')), auxiliary_loss=False,
                    id2label=labels, label2id={v: k for k, v in labels.items()})
    model = DetrForObjectDetection.from_pretrained(location, **kwargs, config=DetrConfig(**settings),
                                                 ignore_mismatched_sizes=not checkpoint, attn_implementation='eager')
    return processor, model


@torch.no_grad()
def predict_pages(model, processor, pages, batch_size=1):
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    for index, batch in enumerate(loader_for(pages, processor, batch_size), 1):
        output = model(pixel_values=batch['pixel_values'].to(device), pixel_mask=batch['pixel_mask'].to(device))
        padded_height, padded_width = batch['pixel_values'].shape[-2:]
        sizes = torch.tensor([(p['image_height'] * padded_height / mask[:, 0].sum().item(),
                               p['image_width'] * padded_width / mask[0].sum().item())
                              for p, mask in zip(batch['pages'], batch['pixel_mask'])], device=device)
        decoded = processor.post_process_object_detection(output, threshold=0.0, target_sizes=sizes)
        for page, prediction in zip(batch['pages'], decoded):
            boxes = prediction['boxes'].cpu().numpy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, page['image_width'])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, page['image_height'])
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            predictions.append(dict(file_name=page['file_name'], source=page['source'], boxes=boxes[valid].tolist(),
                                    labels=prediction['labels'].cpu().numpy()[valid].tolist(),
                                    scores=prediction['scores'].cpu().numpy()[valid].tolist()))
        if index % 25 == 0:
            print(f'Validation: {len(predictions)}/{len(pages)} pages', flush=True)
    return predictions


def suppress_duplicates(predictions, iou_threshold=0.5):
    if iou_threshold is None:
        return predictions
    filtered = []
    for prediction in predictions:
        boxes = np.asarray(prediction['boxes']).reshape(-1, 4)
        labels = np.asarray(prediction['labels'])
        keep = []
        for index in np.argsort(-np.asarray(prediction['scores']), kind='stable'):
            same_class = [i for i in keep if labels[i] == labels[index]]
            if same_class:
                intersection = np.maximum(0, np.minimum(boxes[index, 2:], boxes[same_class, 2:]) -
                                          np.maximum(boxes[index, :2], boxes[same_class, :2])).prod(1)
                union = (boxes[index, 2:] - boxes[index, :2]).prod() + (boxes[same_class, 2:] - boxes[same_class, :2]).prod(1) - intersection
                if np.max(intersection / np.maximum(union, 1e-9)) > iou_threshold:
                    continue
            keep.append(int(index))
        filtered.append(prediction | {key: [prediction[key][i] for i in keep] for key in ('boxes', 'labels', 'scores')})
    return filtered


def match_boxes(predicted, truth, threshold=0.5):
    a, b = np.asarray(predicted).reshape(-1, 4), np.asarray(truth).reshape(-1, 4)
    if not len(a) or not len(b):
        return []
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    union = (a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)
    cost = np.where(iou >= threshold, -1 - iou / (min(len(a), len(b)) + 1), 0)
    pi, gi = linear_sum_assignment(cost)
    return [(int(p), int(g), float(iou[p, g])) for p, g in zip(pi, gi) if iou[p, g] >= threshold]


def detection_metrics(pages, predictions, score_threshold=0.25):
    annotations, detections, images = [], [], []
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    matched = total_pred = total_gt = 0
    by_name = {p['file_name']: p for p in predictions}
    for image_id, page in enumerate(pages):
        pred = by_name[page['file_name']]
        images.append(dict(id=image_id, width=page['image_width'], height=page['image_height']))
        for region in page['regions']:
            x1, y1, x2, y2 = region['bbox']
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id, category_id=CLASSES.index(region['type']) + 1,
                                    bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0))
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            x1, y1, x2, y2 = box
            detections.append(dict(image_id=image_id, category_id=label + 1, bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
        kept = [i for i, score in enumerate(pred['scores']) if score >= score_threshold]
        matches = match_boxes([pred['boxes'][i] for i in kept], [r['bbox'] for r in page['regions']])
        matched += len(matches)
        total_pred += len(kept)
        total_gt += len(page['regions'])
        for pi, gi, _ in matches:
            confusion[CLASSES.index(page['regions'][gi]['type']), pred['labels'][kept[pi]]] += 1
    per_class = {name: None for name in CLASSES}
    mean_ap = ap50 = 0.0
    if detections:
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO()
            coco.dataset = dict(images=images, annotations=annotations,
                                categories=[dict(id=i + 1, name=name) for i, name in enumerate(CLASSES)])
            coco.createIndex()
            evaluator = COCOeval(coco, coco.loadRes(detections), 'bbox')
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        mean_ap, ap50 = map(float, evaluator.stats[:2])
        for i, name in enumerate(CLASSES):
            precision = evaluator.eval['precision'][:, :, i, 0, -1]
            per_class[name] = float(precision[precision >= 0].mean()) if (precision >= 0).any() else None
    precision, recall = matched / max(1, total_pred), matched / max(1, total_gt)
    denominator = confusion.sum(0) + confusion.sum(1)
    macro_f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(len(CLASSES)), where=denominator > 0).mean()
    return dict(pages=len(pages), mAP=mean_ap, AP50=ap50, AP_by_type=per_class, score_threshold=score_threshold,
                detection_precision=precision, detection_recall=recall,
                detection_F1=2 * precision * recall / max(1e-12, precision + recall),
                matched_regions=matched, gt_regions=total_gt, matched_coverage=recall,
                type_accuracy=float(confusion.trace() / max(1, matched)), type_macro_F1=float(macro_f1),
                type_confusion=confusion.tolist())


def train_detector(config):
    config = dict(seed=42, epochs=10, batch_size=1, accumulation=4, lr=1e-4, shortest_edge=640,
                  longest_edge=960, train_pages=None, val_pages=None, log_every=20,
                  score_threshold=.25, freeze_backbone=True, device='gpu') | config
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    pages = pages_with_split()
    train = subset(pages, 'train', config.get('train_pages'), config['seed'])
    val = subset(pages, 'val', config.get('val_pages'), config['seed'])
    directory = Path(config.get('output', MODEL_DIR))
    resume = config.get('resume', False)
    if not resume and (directory / 'last/training_state.pt').exists():
        raise ValueError('A checkpoint already exists. Use --resume or a new output directory.')
    processor, model = load_detector(directory / 'last' if resume else None)
    model.to(device_for(config['device']))
    model.model.backbone.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'], weight_decay=1e-4)
    loader = loader_for(train, processor, config['batch_size'], shuffle=True)
    total_steps = config['epochs'] * math.ceil(len(loader) / config['accumulation'])
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, min(int(total_steps * 0.1), math.ceil(len(loader) / config['accumulation']))), total_steps)
    history, start_epoch, best = [], 0, -1.0
    if resume:
        state = torch.load(directory / 'last/training_state.pt', map_location='cpu', weights_only=False)
        comparable = ('seed', 'epochs', 'train_pages', 'val_pages', 'batch_size', 'accumulation', 'lr', 'shortest_edge', 'longest_edge', 'freeze_backbone')
        if any(config.get(key) != state['config'].get(key) for key in comparable):
            raise ValueError('Resume settings must match the original training configuration.')
        if state['split_sha256'] != hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest():
            raise ValueError('The page split changed since the checkpoint.')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, start_epoch, best = state['history'], state['epoch'], state['best_mAP']
        torch.set_rng_state(state['rng_state'])
        if next(model.parameters()).device.type == 'mps' and state.get('mps_rng_state') is not None:
            torch.mps.set_rng_state(state['mps_rng_state'])
        if next(model.parameters()).device.type == 'cuda' and state.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])
    device = next(model.parameters()).device
    print(f'Device: {device}; train/val pages: {len(train)}/{len(val)}; backbone frozen: {config["freeze_backbone"]}', flush=True)
    print(f'{len(loader)} batches/epoch, effective batch {config["batch_size"] * config["accumulation"]}; output: {directory}', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(start_epoch + 1, config['epochs'] + 1):
            model.train()
            model.model.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            started, losses = time.time(), []
            print(f'Epoch {epoch}/{config["epochs"]} started', flush=True)
            for step, batch in enumerate(loader, 1):
                labels = [{k: v.to(device) for k, v in target.items()} for target in batch['labels']]
                loss = model(pixel_values=batch['pixel_values'].to(device), pixel_mask=batch['pixel_mask'].to(device), labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss in epoch {epoch}, step {step}')
                group_size = min(config['accumulation'], len(loader) - ((step - 1) // config['accumulation']) * config['accumulation'])
                (loss / group_size).backward()
                if step % config['accumulation'] == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
                if step == 1 or step % config['log_every'] == 0 or step == len(loader):
                    rate = (time.time() - started) / step
                    print(f'Epoch {epoch}/{config["epochs"]}, batch {step}/{len(loader)}, loss {np.mean(losses):.3f}, {rate:.2f}s/batch, train ETA {(len(loader)-step)*rate/60:.1f} min', flush=True)
            train_seconds = time.time() - started
            predictions = predict_pages(model, processor, val, config['batch_size'])
            metrics = detection_metrics(val, predictions, config['score_threshold'])
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)), minutes=(time.time() - started) / 60,
                       train_seconds=train_seconds, seconds_per_batch=train_seconds / len(loader), **metrics)
            history.append(row)
            improved = metrics['mAP'] > best
            best = max(best, metrics['mAP'])
            model.save_pretrained(directory / 'last')
            processor.save_pretrained(directory / 'last')
            state = dict(epoch=epoch, config=config, history=history, best_mAP=best, optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), rng_state=torch.get_rng_state(),
                         mps_rng_state=torch.mps.get_rng_state() if device.type == 'mps' else None,
                         cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                         split_sha256=hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest())
            torch.save(state, directory / 'last/training_state.pt')
            if improved:
                model.save_pretrained(directory / 'best')
                processor.save_pretrained(directory / 'best')
                (directory / 'best/selection.json').write_text(json.dumps(dict(epoch=epoch, mAP=best), indent=2))
            (directory / 'training.json').write_text(json.dumps(dict(config=config, dataset_revision=REVISION,
                base_model=BASE_MODEL, base_revision=BASE_REVISION, device=str(device), torch_version=torch.__version__,
                split_sha256=state['split_sha256'], git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                history=history), indent=2))
            print(f'Epoch {epoch} done: mAP {metrics["mAP"]:.4f}, AP50 {metrics["AP50"]:.4f}, {row["minutes"]:.1f} min; best checkpoint saved: {improved}', flush=True)
    except KeyboardInterrupt:
        print('Stopped. Use --resume to restart from the last completed epoch, if one was saved.', flush=True)
    return history


def export_predicted_crops(pages, predictions, score_threshold=0.25, output_dir=None):
    output_dir = Path(output_dir or ROOT / 'data/detector_crops')
    output_dir.mkdir(parents=True, exist_ok=True)
    by_name = {p['file_name']: p for p in predictions}
    records, eligible_count, matched_count = [], 0, 0
    for page in pages:
        prediction = by_name[page['file_name']]
        selected = [i for i, (label, score) in enumerate(zip(prediction['labels'], prediction['scores']))
                    if CLASSES[label] == 'handwritten' and score >= score_threshold]
        truth = [(i, r) for i, r in enumerate(page['regions']) if is_htr_line(r) and clean_text(r['text'])]
        matches = {pi: (gi, overlap) for pi, gi, overlap in match_boxes([prediction['boxes'][i] for i in selected], [r['bbox'] for _, r in truth])}
        eligible_count += len(truth)
        matched_count += len(matches)
        page_id = Path(page['file_name']).stem
        with Image.open(snapshot() / 'train' / page['file_name']) as source:
            for region, pi in enumerate(selected):
                box = prediction['boxes'][pi]
                bounds = (math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3]))
                crop = source.crop(bounds).convert('RGB')
                crop.thumbnail((1024, 1024))
                path = output_dir / f'{page_id}_{region:03d}.jpg'
                crop.save(path, quality=95)
                match = matches.get(region)
                original_index, gt = truth[match[0]] if match else (None, None)
                records.append(dict(page_id=page_id, region=region, bbox=box, type='handwritten', crop_path=str(path.resolve()),
                                    text=gt['text'] if gt else '', target=clean_text(gt['text']) if gt else None,
                                    source=page['source'], annotation_source=page['annotation_source'], split=page['split'],
                                    score=prediction['scores'][pi], gt_region=original_index, iou=match[1] if match else 0))
    columns = ['page_id', 'region', 'bbox', 'type', 'crop_path', 'text', 'target', 'source', 'annotation_source', 'split', 'score', 'gt_region', 'iou']
    frame = pd.DataFrame(records, columns=columns)
    frame.to_csv(output_dir / 'crops.csv', index=False)
    coverage = dict(eligible_gt_lines=eligible_count, matched_lines=matched_count, predicted_lines=len(frame),
                    missed_lines=eligible_count - matched_count, unmatched_predictions=len(frame) - matched_count,
                    matched_coverage=matched_count / max(1, eligible_count), score_threshold=score_threshold,
                    note='CER/WER on matched rows is conditional on successful detection.')
    (output_dir / 'coverage.json').write_text(json.dumps(coverage, indent=2))
    return frame, coverage
