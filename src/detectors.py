"""Time-budgeted detector comparisons on the shared page split."""
import os

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import hashlib
import json
import math
import re
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from cv import CLASSES, detection_metrics, device_for, export_predicted_crops, loader_for, pages_with_split, predict_pages, subset, suppress_duplicates
from htr import REVISION, ROOT, SPLIT_FILE

VARIANTS = {
    'rf-nano': ('Roboflow/rf-detr-nano', '9f201577d9415f0378084748d27e68586e6b3600'),
    'rf-small': ('Roboflow/rf-detr-small', '3bdc465063270f99769da5a1b4c00c68bd2d439d'),
    'rtv2-r18': ('PekingU/rtdetr_v2_r18vd', '5650961749fa93567c0d46fc7f43ea4f9e914107'),
    'dfine-small': ('ustc-community/dfine-small-coco', 'f79e65b5fbb33ceb9d3ebba042955d7410c608f8'),
    'deimv2-nano': ('harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers', 'b23bfb92684c98e1adf60939fdef28e409991f68'),
}
BENCH_DIR = ROOT / 'results/cv/benchmark'



def compact_training(training):
    keys = ('dataset_revision', 'split_sha256', 'device', 'torch_version', 'parameters', 'checkpoint',
            'pretrained_sha256', 'base_model', 'base_revision', 'charged_seconds', 'status', 'optimizer')
    result = {key: training[key] for key in keys if key in training}
    result['config'] = {key: value for key, value in training['config'].items() if key in
        ('model', 'minutes', 'size', 'seed', 'epochs', 'batch_size', 'accumulation', 'lr', 'lr_backbone',
         'shortest_edge', 'longest_edge', 'freeze_backbone', 'refine', 'lr_scale', 'patience', 'min_delta', 'init_from')}
    result['history'] = [{key: row[key] for key in ('epoch', 'pages_in_epoch', 'images_seen', 'seconds',
        'minutes', 'train_loss', 'mAP', 'AP50', 'epoch_complete') if key in row} for row in training['history']]
    if 'native_args' in training:
        native = training['native_args']
        result['optimizer'] = {key: native[key] for key in
            ('optimizer', 'lr0', 'lrf', 'warmup_epochs', 'weight_decay', 'momentum', 'batch', 'nbs', 'amp', 'cos_lr')}
        result['preprocessing'] = '640 letterboxing; augmentation disabled; EMA evaluation'
    elif 'model' in result['config']:
        result['preprocessing'] = '640 square; batch 1, accumulation 4; FP32; augmentation disabled'
    return result


def compact_metrics(report):
    return {key: compact_metrics(value) if isinstance(value, dict) else value
            for key, value in report.items() if key not in ('type_confusion', 'note')}


def save_run(name, training=None, metrics=None):
    path = ROOT / 'results/cv/rf-small-refined.json' if name == 'rf-small-refined' else BENCH_DIR / 'runs.json'
    runs = json.loads(path.read_text()) if path.exists() else {}
    record = runs.setdefault(name, {})
    if training is not None:
        record['training'] = compact_training(training)
    if metrics is not None:
        record['metrics'] = compact_metrics(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(runs, indent=2) + '\n')
    temporary.replace(path)


class BudgetSchedule:
    def __init__(self, optimizer, seconds):
        self.optimizer = optimizer
        self.budget_seconds = seconds
        self.warmup_seconds = min(150., seconds * .1)
        self.elapsed_seconds = 0.

    def step(self, seconds):
        self.elapsed_seconds = seconds
        progress = min(1., seconds / self.budget_seconds)
        factor = min(1., seconds / self.warmup_seconds) * (.01 + .99 * (1 + math.cos(math.pi * progress)) / 2)
        for group in self.optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor

    def state_dict(self):
        return dict(budget_seconds=self.budget_seconds, warmup_seconds=self.warmup_seconds,
                    elapsed_seconds=self.elapsed_seconds, base_lrs=[g['base_lr'] for g in self.optimizer.param_groups])

    def load_state_dict(self, state):
        self.warmup_seconds = state['warmup_seconds']
        self.elapsed_seconds = state['elapsed_seconds']


def early_stopping(history, patience=5, min_delta=.002):
    reference, waiting = history[0]['mAP'], 0
    for row in history[1:]:
        if not row['epoch_complete']:
            continue
        if row['mAP'] >= reference + min_delta:
            reference, waiting = row['mAP'], 0
        else:
            waiting += 1
    return waiting, waiting >= patience


class MPSIntegral(torch.nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins

    def forward(self, corners, project):
        batch, queries, _ = corners.shape
        probabilities = corners.reshape(-1, self.bins + 1).softmax(1)
        return torch.nn.functional.linear(probabilities, project.to(corners.device).reshape(1, -1)).reshape(batch, queries, 4)


def load_variant(name, checkpoint=None, size=640, device='gpu'):
    location, revision = VARIANTS[name]
    kwargs = dict(revision=revision) if checkpoint is None else {}
    location = str(checkpoint or location)
    processor = AutoImageProcessor.from_pretrained(location, **kwargs, size=dict(height=size, width=size))
    labels = dict(enumerate(CLASSES))
    model = AutoModelForObjectDetection.from_pretrained(location, **kwargs, id2label=labels,
        label2id={label: index for index, label in labels.items()}, ignore_mismatched_sizes=checkpoint is None,
        attn_implementation='eager')
    target = device_for(device)
    if target.type == 'mps':
        for module in model.modules():
            integral = getattr(module, 'integral', None)
            if integral is not None and type(integral).__name__ in ('DFineIntegral', 'Deimv2Integral'):
                module.integral = MPSIntegral(integral.max_num_bins)
    return processor, model.to(target)


def batch_loss(model, batch):
    device = next(model.parameters()).device
    labels = [{key: value.to(device) for key, value in target.items()} for target in batch['labels']]
    return model(pixel_values=batch['pixel_values'].to(device), pixel_mask=batch['pixel_mask'].to(device), labels=labels).loss


def optimizer_groups(name, model):
    grouped = {}
    for key, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rate = 1e-5 if 'backbone' in key else 1e-4
        if name.startswith('rf-') and 'backbone' in key:
            layer = re.search(r'encoder\.layer\.(\d+)\.', key)
            depth = model.config.backbone_config.num_hidden_layers
            distance = depth - 1 - int(layer[1]) if layer else depth if 'embeddings' in key else 0
            rate = 1.5e-4 * .8 ** distance
        grouped.setdefault(rate, []).append(parameter)
    return [dict(params=parameters, lr=rate, base_lr=rate) for rate, parameters in grouped.items()]


def summarize(name, model, processor, pages, directory, predictions=None):
    if predictions is None:
        predictions = predict_pages(model, processor, pages)
    raw = detection_metrics(pages, predictions)
    filtered = suppress_duplicates(predictions)
    search = [detection_metrics(pages, filtered, threshold) for threshold in (0.1, 0.25, 0.4, 0.55, 0.7)]
    chosen = max(search, key=lambda row: row['detection_F1'])
    by_source = {source: detection_metrics([p for p in pages if p['source'] == source],
        [p for p in filtered if p['source'] == source], chosen['score_threshold']) for source in sorted({p['source'] for p in pages})}
    report = dict(model=name, raw=raw, filtered=chosen, per_source=by_source, nms_iou=0.5,
        threshold_selection='validation geometry F1', threshold_search=[{k: r[k] for k in
        ('score_threshold', 'detection_precision', 'detection_recall', 'detection_F1')} for r in search])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'val_predictions.json').write_text(json.dumps(predictions))
    _, coverage = export_predicted_crops(pages, filtered, chosen['score_threshold'], ROOT / 'data/detector_crops' / name)
    report['handwritten_coverage'] = coverage
    save_run(name, metrics=report)
    return report


def train_budget(config):
    name = config['model']
    refine = config.get('refine', False)
    if refine:
        if name != 'rf-small':
            raise ValueError('Refinement is only available for RF-DETR Small.')
        config = config | dict(lr_scale=.3, patience=5, min_delta=.002, init_from='models/benchmark-rf-small/best')
    run_name = 'rf-small-refined' if refine else name
    directory = ROOT / 'models' / ('rf-small-refined' if refine else f'benchmark-{name}')
    resume = config.get('resume', False)
    if (directory / 'last/state.pt').exists() and not resume:
        raise ValueError(f'{directory} already has a run. Use --resume-existing.')
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    initial = ROOT / config['init_from'] if refine else None
    processor, model = load_variant(name, directory / 'last' if resume else initial, config['size'], config['device'])
    groups = optimizer_groups(name, model)
    for group in groups:
        group['lr'] *= config.get('lr_scale', 1.)
        group['base_lr'] = group['lr']
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = BudgetSchedule(optimizer, config['minutes'] * 60)
    pages = pages_with_split()
    train, val = subset(pages, 'train'), subset(pages, 'val')
    split_hash = hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest()
    if refine:
        source = json.loads((BENCH_DIR / 'runs.json').read_text())['rf-small']['training']
        if source['split_sha256'] != split_hash or source['dataset_revision'] != REVISION:
            raise ValueError('Refinement must use the benchmark dataset and page split.')
    epoch, offset, spent, images_seen, best, history = 1, 0, 0., 0, -1., []
    if resume:
        state = torch.load(directory / 'last/state.pt', map_location='cpu', weights_only=False)
        for key in ('model', 'size', 'seed', 'train_pages', 'val_pages'):
            if config.get(key) != state['config'].get(key):
                raise ValueError(f'Resume setting changed: {key}')
        if refine and any(config[key] != state['config'].get(key) for key in
                          ('refine', 'minutes', 'lr_scale', 'patience', 'min_delta', 'init_from')):
            raise ValueError('Refinement resume settings must match its saved configuration.')
        if state['split_sha256'] != split_hash:
            raise ValueError('The shared split changed.')
        optimizer.load_state_dict(state['optimizer'])
        if state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])
        epoch, offset, spent, images_seen, best, history = (state[k] for k in ('epoch', 'offset', 'seconds', 'images_seen', 'best', 'history'))
        torch.set_rng_state(state['rng'])
        if state.get('numpy_rng'):
            np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng'])
        if next(model.parameters()).device.type == 'mps':
            torch.mps.set_rng_state(state['device_rng'])
    started = time.monotonic()
    budget = config['minutes'] * 60
    elapsed = lambda: spent + time.monotonic() - started
    last_eval = elapsed()
    print(f'{name}: {next(model.parameters()).device}; {len(train)}/{len(val)} pages; {budget/60:g} min budget; starting epoch {epoch}, page {offset}', flush=True)
    print(f'{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters; input {config["size"]} square; batch 1, accumulation 4', flush=True)
    optimizer.zero_grad(set_to_none=True)
    losses = []

    def checkpoint(metrics):
        nonlocal best
        improved = metrics['mAP'] > best
        best = max(best, metrics['mAP'])
        model.save_pretrained(directory / 'last')
        processor.save_pretrained(directory / 'last')
        if improved:
            model.save_pretrained(directory / 'best')
            processor.save_pretrained(directory / 'best')
        next_epoch, next_offset = (epoch + 1, 0) if offset == len(train) else (epoch, offset)
        state = dict(config=config, epoch=next_epoch, offset=next_offset, seconds=elapsed(), images_seen=images_seen,
            best=best, history=history, optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
            scheduler=scheduler.state_dict(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            device_rng=torch.mps.get_rng_state() if next(model.parameters()).device.type == 'mps' else None,
            split_sha256=split_hash)
        temporary = directory / 'last/state.tmp.pt'
        torch.save(state, temporary)
        temporary.replace(directory / 'last/state.pt')
        stopped = refine and early_stopping(history, config['patience'], config['min_delta'])[1]
        metadata = dict(config=config, checkpoint=dict(zip(('id', 'revision'), VARIANTS[name])),
            dataset_revision=REVISION, split_sha256=split_hash, device=str(next(model.parameters()).device),
            torch_version=torch.__version__, parameters=sum(p.numel() for p in model.parameters()), history=history,
            charged_seconds=state['seconds'], status='early_stopped' if stopped else 'complete' if elapsed() >= budget else 'paused',
            optimizer=dict(name='AdamW', weight_decay=1e-4,
                base_learning_rates=[group['base_lr'] for group in optimizer.param_groups],
                schedule='cosine over elapsed budget, linear warmup up to 150 seconds', effective_batch=4))
        (directory / 'training.json').write_text(json.dumps(metadata, indent=2))
        save_run(run_name, training=metadata)
        print(f'{run_name}: validation mAP {metrics["mAP"]:.4f}, AP50 {metrics["AP50"]:.4f}; best {best:.4f}; saved at {elapsed()/60:.1f} min', flush=True)
        if refine:
            waiting = early_stopping(history, config['patience'], config['min_delta'])[0]
            print(f'Early stopping: {waiting}/{config["patience"]} epochs without a cumulative mAP gain of {config["min_delta"]}', flush=True)
        return stopped

    try:
        if refine and not resume:
            metrics = detection_metrics(val, predict_pages(model, processor, val))
            history.append(dict(epoch=0, pages_in_epoch=0, images_seen=0, seconds=elapsed(),
                                train_loss=None, epoch_complete=False, **metrics))
            checkpoint(metrics)
        stopped = refine and early_stopping(history, config['patience'], config['min_delta'])[1]
        while elapsed() < budget and not stopped:
            order = np.random.default_rng(config['seed'] + epoch).permutation(len(train))
            remaining = [train[i] for i in order[offset:]]
            loader = loader_for(remaining, processor)
            print(f'{name}: epoch {epoch} started ({len(remaining)} pages remaining)', flush=True)
            model.train()
            for step, batch in enumerate(loader, 1):
                loss = batch_loss(model, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                group_size = min(4, len(loader) - ((step - 1) // 4) * 4)
                (loss / group_size).backward()
                losses.append(loss.item())
                offset += 1
                images_seen += 1
                boundary = step % 4 == 0 or step == len(loader)
                if boundary:
                    scheduler.step(elapsed())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 20 == 0:
                    print(f'{name}: epoch {epoch}, page {offset}/{len(train)}, loss {np.mean(losses[-100:]):.3f}, elapsed {elapsed()/60:.1f}/{budget/60:g} min', flush=True)
                if boundary and ((not refine and elapsed() - last_eval >= 300) or step == len(loader) or elapsed() >= budget):
                    metrics = detection_metrics(val, predict_pages(model, processor, val))
                    history.append(dict(epoch=epoch, pages_in_epoch=offset, images_seen=images_seen, seconds=elapsed(),
                        train_loss=float(np.mean(losses[-100:])), epoch_complete=offset == len(train), **metrics))
                    stopped = checkpoint(metrics)
                    model.train()
                    last_eval = elapsed()
                if boundary and (elapsed() >= budget or stopped):
                    break
            if offset == len(train):
                epoch, offset = epoch + 1, 0
    except KeyboardInterrupt:
        print('Interrupted; resume from the last saved validation checkpoint.', flush=True)
        return
    del optimizer, model
    if device_for(config['device']).type == 'mps':
        torch.mps.empty_cache()
    processor, model = load_variant(name, directory / 'best', config['size'], config['device'])
    report = summarize(run_name, model, processor, val, directory)
    print(f'{run_name} completed: raw mAP {report["raw"]["mAP"]:.4f}; filtered F1 {report["filtered"]["detection_F1"]:.4f}', flush=True)
    return report


def prepare_yolo(pages):
    directory = ROOT / 'data/detector_benchmark'
    from htr import snapshot
    for page in pages:
        role = page['split']
        name = Path(page['file_name']).name
        images, labels = directory / 'images' / role, directory / 'labels' / role
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        image_path = images / name
        if not image_path.exists():
            image_path.symlink_to(snapshot() / 'train' / page['file_name'])
        rows = []
        w, h = page['image_width'], page['image_height']
        for region in page['regions']:
            x1, y1, x2, y2 = region['bbox']
            rows.append(f'{CLASSES.index(region["type"])} {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
        (labels / (Path(name).stem + '.txt')).write_text('\n'.join(rows) + '\n')
    configuration = dict(path=str(directory), train='images/train', val='images/val', names=list(CLASSES))
    path = directory / 'dataset.yaml'
    path.write_text(json.dumps(configuration, indent=2))
    return path


def yolo_predictions(model, pages, device, size):
    from htr import snapshot
    predictions = []
    for page in pages:
        result = model.predict(str(snapshot() / 'train' / page['file_name']), device=str(device), imgsz=size,
                               conf=0.0001, max_det=300, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, page['image_width'])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, page['image_height'])
        valid = np.isfinite(boxes).all(1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        predictions.append(dict(file_name=page['file_name'], source=page['source'], boxes=boxes[valid].tolist(),
            labels=result.boxes.cls.int().cpu().numpy()[valid].tolist(), scores=result.boxes.conf.cpu().numpy()[valid].tolist()))
    return predictions


def train_yolo(config):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
    from ultralytics import YOLO, settings
    settings.update({key: False for key in ('sync', 'hub', 'wandb', 'mlflow', 'comet', 'clearml') if key in settings})
    name = config['model']
    directory = ROOT / 'models' / f'benchmark-{name}'
    if directory.exists() and not config.get('resume'):
        raise ValueError('YOLO output directory already exists. Use --resume-existing.')
    pages = pages_with_split()
    train, val = subset(pages, 'train'), subset(pages, 'val')
    data = prepare_yolo(train + val)
    device = device_for(config['device'])
    base = ROOT / 'models/pretrained/yolo26s.pt'
    base.parent.mkdir(parents=True, exist_ok=True)
    resume_path = directory / 'weights/resumable.pt'
    resume_state = json.loads((directory / 'resume.json').read_text()) if config.get('resume') else {}
    if resume_state:
        for key in ('size', 'seed', 'train_pages', 'val_pages', 'pilot'):
            if config.get(key) != resume_state['config'].get(key):
                raise ValueError(f'Resume setting changed: {key}')
        if resume_state['split_sha256'] != hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest():
            raise ValueError('The shared split changed.')
    model = YOLO(str(resume_path) if config.get('resume') else str(base))
    started = None
    previous_seconds = resume_state.get('seconds', 0.)
    history = resume_state.get('history', [])
    common_best = max((r['mAP'] for r in history), default=-1.)
    checkpoint_epoch = -1
    runtime = lambda: previous_seconds + time.monotonic() - started
    def restore_training(trainer):
        nonlocal started
        if not config.get('resume'):
            started = time.monotonic()
            return
        saved = torch.load(directory / 'weights/training_state.pt', map_location='cpu', weights_only=False)
        trainer.model.load_state_dict(saved['model'])
        trainer.optimizer.load_state_dict(saved['optimizer'])
        trainer.epochs = trainer.args.epochs = saved['planned_epochs']
        trainer._setup_scheduler()
        trainer.scheduler.load_state_dict(saved['scheduler'])
        for group, previous in zip(trainer.optimizer.param_groups, saved['optimizer']['param_groups']):
            group['lr'] = previous['lr']
        trainer.scaler.load_state_dict(saved['scaler'])
        trainer.ema.ema.load_state_dict(saved['ema'])
        trainer.ema.updates = saved['ema_updates']
        torch.set_rng_state(saved['rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        if device.type == 'mps':
            torch.mps.set_rng_state(saved['device_rng'])
        started = time.monotonic()
    def epoch_order(trainer):
        trainer.train_loader.generator.manual_seed(config['seed'] + trainer.epoch + 1)
        trainer.train_loader.reset()
    def save_training(trainer):
        if trainer.epoch != checkpoint_epoch:
            return
        saved = dict(model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
            scheduler=trainer.scheduler.state_dict(), scaler=trainer.scaler.state_dict(), planned_epochs=trainer.epochs,
            ema=trainer.ema.ema.state_dict(), ema_updates=trainer.ema.updates, rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            device_rng=torch.mps.get_rng_state() if device.type == 'mps' else None)
        temporary = directory / 'weights/training_state.tmp.pt'
        torch.save(saved, temporary)
        temporary.replace(directory / 'weights/training_state.pt')
        state = dict(seconds=runtime(), history=history, config=config, completed_epochs=trainer.epoch + 1,
                     split_sha256=hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest())
        (directory / 'resume.json').write_text(json.dumps(state, indent=2))
    def enforce_budget(trainer):
        if runtime() >= config['minutes'] * 60:
            trainer.stop = True
    def checkpoint(trainer):
        nonlocal common_best, checkpoint_epoch
        checkpoint_epoch = trainer.epoch
        elapsed = runtime()
        shutil.copy2(trainer.last, directory / 'weights/resumable.pt')
        saved = dict(seconds=elapsed, history=history, config=config,
                     split_sha256=hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest())
        (directory / 'resume.json').write_text(json.dumps(saved, indent=2))
        if history and elapsed - history[-1]['seconds'] < 300 and elapsed < config['minutes'] * 60 - 20:
            return
        evaluator = YOLO(str(trainer.last))
        metrics = detection_metrics(val, yolo_predictions(evaluator, val, device, config['size']))
        history.append(dict(epoch=trainer.epoch + 1, images_seen=(trainer.epoch+1)*len(train),
                            seconds=runtime(), **metrics))
        if metrics['mAP'] > common_best:
            common_best = metrics['mAP']
            shutil.copy2(trainer.last, directory / 'weights/common_best.pt')
        saved['seconds'] = runtime()
        (directory / 'resume.json').write_text(json.dumps(saved, indent=2))
        print(f'{name}: common validation mAP {metrics["mAP"]:.4f}; best {common_best:.4f}', flush=True)
        del evaluator
        if device.type == 'mps':
            torch.mps.empty_cache()
    def progress(trainer):
        print(f'{name}: epoch {trainer.epoch + 1} finished; elapsed {runtime()/60:.1f} min', flush=True)
    model.add_callback('on_train_epoch_end', progress)
    model.add_callback('on_model_save', checkpoint)
    model.add_callback('on_train_start', restore_training)
    model.add_callback('on_train_epoch_start', epoch_order)
    model.add_callback('on_fit_epoch_end', save_training)
    model.add_callback('on_train_batch_end', enforce_budget)
    remaining_hours = max(0., config['minutes'] * 60 - previous_seconds) / 3600
    if remaining_hours > 0:
        model.train(data=str(data), device=str(device), imgsz=config['size'], batch=4, nbs=4,
        epochs=10000, time=config['minutes']/60, workers=0, amp=False,
        optimizer='AdamW', lr0=1e-3, lrf=.01, warmup_epochs=1., weight_decay=1e-4,
        seed=config['seed'], patience=10000, project=str(directory.parent), name=directory.name,
        exist_ok=True, resume=config.get('resume', False), plots=False, cache=False,
        mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=0., scale=0., shear=0., perspective=0.,
        flipud=0., fliplr=0., hsv_h=0., hsv_s=0., hsv_v=0., close_mosaic=0)
    seconds = json.loads((directory / 'resume.json').read_text())['seconds']
    model = YOLO(str(directory / 'weights/common_best.pt'))
    predictions = yolo_predictions(model, val, device, config['size'])
    report = summarize(name, None, None, val, directory, predictions)
    metadata = dict(config=config, seconds=seconds, dataset_revision=REVISION,
        charged_seconds=seconds, status='paused' if seconds < config['minutes'] * 60 else 'complete',
        split_sha256=hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(), device=str(device),
        pretrained_sha256=hashlib.sha256(base.read_bytes()).hexdigest(), torch_version=torch.__version__, history=history)
    import yaml
    metadata['native_args'] = yaml.safe_load((directory / 'args.yaml').read_text())
    metadata['parameters'] = sum(p.numel() for p in model.model.parameters())
    (directory / 'training.json').write_text(json.dumps(metadata, indent=2))
    save_run(name, training=metadata)
    print(f'{name} completed: raw mAP {report["raw"]["mAP"]:.4f}; filtered F1 {report["filtered"]["detection_F1"]:.4f}', flush=True)
    return report


def comparison_table():
    import pandas as pd
    rows = []
    baseline = ROOT / 'results/cv/baseline.json'
    if baseline.exists():
        record = json.loads(baseline.read_text())
        raw = record['metrics']['raw']
        rows.append(dict(model='DETR baseline', status='reference',
            minutes=sum(r.get('minutes', 0) for r in record['training']['history']),
            mAP=raw['mAP'], AP50=raw['AP50'], handwritten_AP=raw['AP_by_type']['handwritten'],
            F1=record['metrics']['filtered']['detection_F1'], handwritten_coverage=None))
    path = BENCH_DIR / 'runs.json'
    for name, record in (json.loads(path.read_text()) if path.exists() else {}).items():
        training = record['training']
        report = record.get('metrics')
        raw = report['raw'] if report else max(training['history'], key=lambda row: row['mAP'])
        rows.append(dict(model=name, status=training['status'], minutes=training['charged_seconds']/60,
            mAP=raw['mAP'], AP50=raw['AP50'], handwritten_AP=raw['AP_by_type']['handwritten'] if report else None,
            F1=report['filtered']['detection_F1'] if report else None,
            handwritten_coverage=report['handwritten_coverage']['matched_coverage'] if report else None))
    frame = pd.DataFrame(rows).sort_values('mAP', ascending=False)
    frame.to_csv(BENCH_DIR / 'comparison.csv', index=False)
    return frame


def comparison_figures():
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.patches import Rectangle, Patch
    from htr import snapshot
    frame = comparison_table()
    fig, ax = plt.subplots(figsize=(8, 4))
    runs = json.loads((BENCH_DIR / 'runs.json').read_text())
    for name, record in runs.items():
        history = record['training']['history']
        ax.plot([r['seconds']/60 for r in history], [r['mAP'] for r in history], '.-', label=name)
    ax.axhline(float(frame[frame.model == 'DETR baseline'].mAP.iloc[0]), color='gray', linestyle='--', label='DETR reference')
    ax.set(xlabel='Elapsed training + validation minutes', ylabel='Shared validation mAP', title='Quality within the time budget')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BENCH_DIR / 'learning_curves.png', dpi=140)
    plt.close(fig)
    complete = frame[frame.model.map(lambda name: 'metrics' in runs.get(name, {}))]
    if complete.empty:
        return
    winner = complete.iloc[0].model
    report = runs[winner]['metrics']
    predictions = suppress_duplicates(json.loads((ROOT / 'models' / f'benchmark-{winner}' / 'val_predictions.json').read_text()))
    pages = subset(pages_with_split(), 'val')
    colors = {name: f'C{i}' for i, name in enumerate(CLASSES)}
    fig, axes = plt.subplots(4, 2, figsize=(12, 22))
    for row, source in zip(axes, ('dictation', 'archive', 'university', 'school')):
        page = next(p for p in pages if p['source'] == source)
        pred = next(p for p in predictions if p['file_name'] == page['file_name'])
        with Image.open(snapshot() / 'train' / page['file_name']) as original:
            width, height = original.size
            image = original.copy()
            image.thumbnail((700, 850))
        truth = [(r['bbox'], r['type']) for r in page['regions']]
        detected = [(b, CLASSES[label]) for b, label, score in zip(pred['boxes'], pred['labels'], pred['scores']) if score >= report['filtered']['score_threshold']]
        for axis, regions, title in zip(row, (truth, detected), ('GT', winner)):
            axis.imshow(image)
            for (x1, y1, x2, y2), kind in regions:
                sx, sy = image.width/width, image.height/height
                axis.add_patch(Rectangle((x1*sx, y1*sy), (x2-x1)*sx, (y2-y1)*sy, fill=False, edgecolor=colors[kind], linewidth=.9))
                if kind != 'handwritten':
                    axis.text(x1*sx, y1*sy, kind, fontsize=6, color=colors[kind])
            axis.set_title(f'{source}: {title} ({len(regions)})')
            axis.axis('off')
    fig.legend(handles=[Patch(color=colors[name], label=name) for name in CLASSES], loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(BENCH_DIR / 'best_examples.png', dpi=100)
    plt.close(fig)


def write_comparison_report():
    import pandas as pd
    frame = comparison_table()
    names = {'rf-small': 'RF-DETR Small', 'rf-nano': 'RF-DETR Nano', 'rtv2-r18': 'RT-DETRv2 R18',
             'dfine-small': 'D-FINE Small', 'yolo26s': 'YOLO26 Small', 'deimv2-nano': 'DEIMv2 Nano'}
    columns = ['mAP', 'AP50', 'handwritten_AP', 'F1', 'handwritten_coverage']
    lines = ['# Detector comparison', '',
        '50 minutes per model on M3 Pro; 1,197 train / 133 validation pages, seven classes, seed 42. Scores use the best raw-mAP checkpoint.', '',
        '| Model | mAP | AP50 | Handwritten AP | Box F1 | Line coverage |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for _, row in frame[frame.status != 'reference'].iterrows():
        values = [('—' if pd.isna(row[key]) else f'{row[key]:.1%}' if key == 'handwritten_coverage' else f'{row[key]:.3f}') for key in columns]
        lines.append('| ' + ' | '.join([names.get(row.model, row.model)] + values) + ' |')
    reference = frame[frame.status == 'reference']
    if not reference.empty:
        row = reference.iloc[0]
        lines.extend(['', f'DETR reference: mAP **{row.mAP:.4f}**, 10 frozen-backbone epochs ({row.minutes:.1f} minutes).'])
    lines.extend(['', '**RF-DETR Small** is selected for longer fine-tuning. **YOLO26 Small** has the highest line coverage and remains an HTR alternative.', '',
        'mAP averages IoU 0.50–0.95. F1 and line coverage use IoU 0.5 with validation-tuned filtering. One seed, different training recipes; rare classes have few examples. Official test is unused.', '',
        '![Validation mAP over time](learning_curves.png)', '',
        '![RF-DETR Small: ground truth and predictions](best_examples.png)'])
    (BENCH_DIR / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return frame


def refinement_table():
    import pandas as pd
    benchmark = json.loads((BENCH_DIR / 'runs.json').read_text())['rf-small']
    refined = json.loads((ROOT / 'results/cv/rf-small-refined.json').read_text())['rf-small-refined']
    rows = []
    for name, record in (('50-minute benchmark', benchmark), ('RF-DETR Small refinement', refined)):
        training, metrics = record['training'], record.get('metrics')
        raw = metrics['raw'] if metrics else max(training['history'], key=lambda row: row['mAP'])
        rows.append(dict(run=name, status=training['status'], minutes=training['charged_seconds']/60,
            mAP=raw['mAP'], AP50=raw['AP50'], handwritten_AP=raw['AP_by_type']['handwritten'] if metrics else None,
            F1=metrics['filtered']['detection_F1'] if metrics else None,
            coverage=metrics['handwritten_coverage']['matched_coverage'] if metrics else None))
    return pd.DataFrame(rows)


def evaluate_variant(config):
    name = config['model']
    run_name = 'rf-small-refined' if config.get('refine') else name
    directory = ROOT / 'models' / ('rf-small-refined' if config.get('refine') else f'benchmark-{name}')
    val = subset(pages_with_split(), 'val')
    if name == 'yolo26s':
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics'))
        from ultralytics import YOLO
        model = YOLO(str(directory / 'weights/common_best.pt'))
        predictions = yolo_predictions(model, val, device_for(config['device']), config['size'])
        report = summarize(run_name, None, None, val, directory, predictions)
    else:
        processor, model = load_variant(name, directory / 'best', config['size'], config['device'])
        report = summarize(run_name, model, processor, val, directory)
    print(f'{run_name}: evaluated saved best checkpoint; raw mAP {report["raw"]["mAP"]:.4f}', flush=True)
    return report
