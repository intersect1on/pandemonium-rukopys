"""Silver fine-tuning of TrOCR: teacher labels (C), training (B, C, H) and validation scoring.

A  = saved human-trained baseline (`models/trocr-uk-baseline`), never modified.
B  = A + silver crops with the supplied (cleaned) v3.0 transcripts.
C  = A + the same silver crops relabelled by frozen A, filtered by the declared rule below.
H  = optional control: A + human v2.2 training lines, same step budget (isolates extra steps from data).
M  = exploratory, added after B/C were scored: A + supplied silver and human training lines shuffled together
     (human replay against the dictation forgetting seen in B). Not part of the pre-declared B/C comparison.

B, C and H start independently from A's weights with fresh optimizers and share every optimisation
and evaluation setting in `CONFIG`. Only the human v2.2 validation split is scored; the v3.0 test is not touched.

    uv run python scripts/silver_htr.py teacher
    uv run python scripts/silver_htr.py train B        # also C, H
    uv run python scripts/silver_htr.py evaluate A     # also B, C, H
    uv run python scripts/silver_htr.py pages A B C    # page CER with the midterm RF-DETR Small
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup
from transformers.utils import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from htr import CROP_FILE, SILVER_TEST_REVISION, LineDataset, cer, cer_interval, collate, predict, wer  # noqa: E402

logging.set_verbosity_error()
logging.disable_progress_bar()

POOL = ROOT / 'data/silver_pool'
OUT = ROOT / 'results/htr/silver'
MODEL_A = ROOT / 'models/trocr-uk-baseline'
MODELS = dict(A=MODEL_A, B=ROOT / 'models/trocr-uk-silver-B', C=ROOT / 'models/trocr-uk-silver-C',
              H=ROOT / 'models/trocr-uk-human-H', M=ROOT / 'models/trocr-uk-silver-human-M')
EVAL_DECODING = dict(num_beams=4, no_repeat_ngram_size=0, max_length=130)      # as notebooks 03, 08, 09
SELECTION_DECODING = dict(num_beams=1, no_repeat_ngram_size=0, max_length=130)  # as A's epoch selection

# Shared by B, C and H. Optimiser settings repeat A's recipe (notebook 02); the budget is fixed in steps.
CONFIG = dict(
    init=str(MODEL_A.relative_to(ROOT)),
    silver_revision=SILVER_TEST_REVISION,
    seed=42,
    steps=2800,            # about 3 passes over B's 14,918 lines at batch 16
    eval_every=400,
    batch_size=16,
    lr=3e-5,
    weight_decay=0.01,
    warmup_fraction=0.1,
    max_target_tokens=128,
    selection='lowest greedy validation CER among evaluations at steps 400, 800, ..., 2800',
)

# Declared filter for C, fixed before C is trained. The teacher is frozen A with EVAL_DECODING; its
# confidence is the beam's length-normalised log-probability. The supplied silver text is never used.
FILTER = dict(
    min_chars=4, max_chars=140,        # same length window as the pool builder
    needs_cyrillic=True,
    max_word_repeat=2,                 # reject a word repeated 3+ times in a row (decoding loop)
    target_val_cer=0.05,               # tau: lowest score whose accepted validation lines have CER <= 5%
)
# H and M ran after the laptop GPU began hanging (WSL2: GPU 0%, process stuck at 100% CPU, never recovers).
# They use an explicit seeded batch order and save full training state at every evaluation, so a killed run
# resumes where it stopped. B and C finished before this and used torch's shuffling with the same seed.
RESUMABLE = {'H', 'M'}
CYRILLIC = re.compile(r'[А-Яа-яІіЇїЄєҐґ]')


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def silver_crops():
    crops = pd.read_csv(POOL / 'crops.csv')
    crops['crop_path'] = [str(POOL / p) for p in crops.crop_path]
    return crops


def human_crops(split):
    crops = pd.read_csv(CROP_FILE).dropna(subset=['target'])
    return crops[crops.split == split].copy()


def within_token_limit(crops, processor):
    ok = crops.target.map(lambda t: len(processor.tokenizer(t).input_ids) <= CONFIG['max_target_tokens'])
    return crops[ok].copy(), int((~ok).sum())


@torch.no_grad()
def predict_scored(model, processor, crops, batch_size=16, **decoding):
    """Beam decoding plus each line's mean token log-probability under the same model (eos included).

    The score comes from one teacher-forced pass over the decoded tokens. Asking `generate` for
    `output_scores` keeps a full-vocabulary tensor per step and beam, which overflows 8 GB.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(LineDataset(crops, processor), batch_size=batch_size, num_workers=4,
                                         collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id))
    eos = model.generation_config.eos_token_id
    texts, scores = [], []
    for i, (pixels, _) in enumerate(loader):
        pixels = pixels.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            seqs = model.generate(pixels, use_cache=True, **decoding)
            encoded = model.encoder(pixel_values=pixels)
        targets = seqs[:, 1:]
        # keep tokens up to and including the first eos
        is_eos = (targets == eos).int()
        valid = ~(torch.cumsum(is_eos, 1) - is_eos > 0)
        for j in range(0, len(seqs), 4):  # small chunks keep the vocab-sized logits well under 8 GB
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(encoder_outputs=(encoded.last_hidden_state[j:j + 4],),
                               decoder_input_ids=seqs[j:j + 4, :-1]).logits
            logp = -torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), targets[j:j + 4],
                                                      reduction='none')
            v = valid[j:j + 4].float()
            scores += ((logp * v).sum(1) / v.sum(1)).cpu().tolist()
            del logits, logp
        texts += [' '.join(t.split()) for t in processor.batch_decode(seqs, skip_special_tokens=True)]
        if i % 25 == 0:
            log(f'  decoded {len(texts):,}/{len(crops):,}')
    return texts, scores


def looping(text, limit):
    words = text.split()
    return any(len(set(words[i:i + limit + 1])) == 1 for i in range(len(words) - limit))


def passes_shape(text):
    return (FILTER['min_chars'] <= len(text) <= FILTER['max_chars']
            and (not FILTER['needs_cyrillic'] or bool(CYRILLIC.search(text)))
            and not looping(text, FILTER['max_word_repeat']))


def calibrate_tau(val):
    """Lowest score threshold whose accepted validation lines (shape filter + score >= tau) have CER <= target."""
    val = val[val.teacher.map(passes_shape)].sort_values('score', ascending=False)
    measures = [jiwer.process_characters(r, h) for r, h in zip(val.target, val.teacher)]
    errors = np.cumsum([m.substitutions + m.deletions + m.insertions for m in measures])
    chars = np.cumsum(val.target.str.len().to_numpy())
    ok = np.nonzero(errors / chars <= FILTER['target_val_cer'])[0]
    if not len(ok):
        raise ValueError('No threshold reaches the target validation CER')
    k = ok.max()
    return float(val.score.iloc[k]), dict(val_lines_accepted=int(k + 1), val_lines=int(len(val)),
                                         val_accepted_CER=float(errors[k] / chars[k]))


def teacher():
    OUT.mkdir(parents=True, exist_ok=True)
    processor = TrOCRProcessor.from_pretrained(MODEL_A)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_A).cuda()
    results = {}
    for name, crops in [('val', human_crops('val')), ('silver', silver_crops())]:
        path = OUT / f'teacher_{name}.csv'
        if path.exists():
            results[name] = pd.read_csv(path, keep_default_na=False)
            log(f'teacher {name}: reusing {path.name}')
            continue
        start = time.time()
        log(f'teacher {name}: {len(crops):,} crops')
        crops['teacher'], crops['score'] = predict_scored(model, processor, crops, **EVAL_DECODING)
        crops[['page_id', 'region', 'source', 'crop_path', 'target', 'teacher', 'score']].assign(
            crop_path=lambda d: d.crop_path.map(lambda p: Path(p).name)).to_csv(path, index=False)
        log(f'teacher {name}: {1000 * (time.time() - start) / len(crops):.0f} ms per line')
        results[name] = crops

    tau, calibration = calibrate_tau(results['val'])
    silver = results['silver']
    silver['shape_ok'] = silver.teacher.map(passes_shape)
    silver['accepted'] = silver.shape_ok & (silver.score >= tau)
    acc = silver[silver.accepted]
    summary = dict(
        filter=FILTER, tau=tau, calibration=calibration,
        silver_lines=len(silver), shape_rejected=int((~silver.shape_ok).sum()),
        accepted=int(len(acc)), accepted_by_source=acc.groupby('source').size().to_dict(),
        lines_by_source=silver.groupby('source').size().to_dict(),
        teacher_vs_supplied_CER_all=cer(silver.target, silver.teacher),
        teacher_vs_supplied_CER_accepted=cer(acc.target, acc.teacher),
        teacher_vs_supplied_CER_rejected=cer(silver[~silver.accepted].target, silver[~silver.accepted].teacher),
        teacher_vs_supplied_CER_by_source={s: cer(g.target, g.teacher) for s, g in silver.groupby('source')},
        teacher_val_CER=cer(results['val'].target, results['val'].teacher),
    )
    (OUT / 'teacher_filter.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log(json.dumps(summary, indent=2, ensure_ascii=False))


def training_lines(run, processor):
    if run == 'B':
        crops = silver_crops()
    elif run == 'C':
        labels = pd.read_csv(OUT / 'teacher_silver.csv', keep_default_na=False)
        tau = json.loads((OUT / 'teacher_filter.json').read_text())['tau']
        labels = labels[labels.teacher.map(passes_shape) & (labels.score >= tau)]
        crops = silver_crops().merge(labels[['page_id', 'region', 'teacher']], on=['page_id', 'region'],
                                     validate='one_to_one')
        crops['target'] = crops.teacher
    elif run == 'H':
        crops = human_crops('train')
    elif run == 'M':
        crops = pd.concat([silver_crops(), human_crops('train')], ignore_index=True)
    else:
        raise ValueError(run)
    return within_token_limit(crops, processor)


def batch_order(n, seed, steps, batch_size):
    """Full batches drawn from consecutive seeded permutations of the n lines (epoch e uses rng([seed, e]))."""
    order, epoch = [], 0
    while len(order) < steps * batch_size:
        order += np.random.default_rng([seed, epoch]).permutation(n).tolist()
        epoch += 1
    return [order[i * batch_size:(i + 1) * batch_size] for i in range(steps)]


def train(run):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(CONFIG['seed'])
    resumable = run in RESUMABLE
    resume_dir = MODELS[run].with_name(f'_resume_{run}')
    state_file = resume_dir / 'trainer.pt'
    state = None
    if resumable and state_file.exists():
        state = torch.load(state_file, weights_only=False)
        if state['config'] != CONFIG:
            log(f'{run}: resume state has a different config, starting fresh')
            state = None
    processor = TrOCRProcessor.from_pretrained(MODEL_A)
    lines, dropped = training_lines(run, processor)
    val = human_crops('val')
    log(f'{run}: {len(lines):,} training lines from {lines.page_id.nunique()} pages ({dropped} over-long dropped)')

    model = VisionEncoderDecoderModel.from_pretrained(resume_dir if state else MODEL_A).cuda()
    model.gradient_checkpointing_enable()
    dataset = LineDataset(lines, processor, CONFIG['max_target_tokens'])
    pad_collate = lambda b: collate(b, processor.tokenizer.pad_token_id)  # noqa: E731
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], fused=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(CONFIG['warmup_fraction'] * CONFIG['steps']),
                                                CONFIG['steps'])
    resumed_at, batches = [], None
    if state:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        resumed_at = state['resumed_at'] + [state['step']]
        log(f"{run}: resuming at step {state['step']}")
    step = state['step'] if state else 0
    if resumable:
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_order(len(lines), CONFIG['seed'], CONFIG['steps'], CONFIG['batch_size'])[step:],
            num_workers=4, collate_fn=pad_collate)
        if state:
            # start the workers (which draws a seed from the CPU RNG) before restoring the saved RNG state
            batches = iter(loader)
            torch.set_rng_state(state['cpu_rng'])
            torch.cuda.set_rng_state(state['cuda_rng'])
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, persistent_workers=True,
            generator=torch.Generator().manual_seed(CONFIG['seed']), collate_fn=pad_collate)

    def save_state():
        tmp = resume_dir.with_name(resume_dir.name + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        torch.save(dict(config=CONFIG, step=step, history=history, best=best, minutes=(time.time() - start) / 60,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), resumed_at=resumed_at,
                        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), tmp / 'trainer.pt')
        shutil.rmtree(resume_dir, ignore_errors=True)
        tmp.rename(resume_dir)

    def evaluate(step, losses, start):
        # Return the training allocator's cached blocks first: on this 8 GB WSL GPU, training reserve plus
        # decoding can spill into shared system memory, which stalls instead of raising OOM.
        torch.cuda.empty_cache()
        hyps = predict(model, processor, val, **SELECTION_DECODING)
        torch.cuda.empty_cache()
        row = dict(step=step, epochs_seen=step * CONFIG['batch_size'] / len(lines),
                   train_loss=float(np.mean(losses)) if losses else None,
                   val_CER=cer(val.target, hyps), val_WER=wer(val.target, hyps), minutes=(time.time() - start) / 60)
        log(run, row)
        return row

    start = time.time() - (state['minutes'] * 60 if state else 0)
    history = state['history'] if state else [evaluate(0, [], start)]
    best, losses = (state['best'] if state else None), []
    model.train()
    while step < CONFIG['steps']:
        epoch_batches, batches = (batches if batches is not None else loader), None
        for pixels, labels in epoch_batches:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(pixel_values=pixels.cuda(), labels=labels.cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                log(f'{run} step {step} loss {np.mean(losses[-100:]):.4f} '
                    f'peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GB {(time.time() - start) / 60:.1f} min')
            if step % CONFIG['eval_every'] == 0 or step == CONFIG['steps']:
                history.append(evaluate(step, losses, start))
                losses = []
                if best is None or history[-1]['val_CER'] < best['val_CER']:
                    best = history[-1]
                    model.save_pretrained(MODELS[run])
                    processor.save_pretrained(MODELS[run])
                if resumable and step < CONFIG['steps']:
                    save_state()
                model.train()
            if step >= CONFIG['steps']:
                break

    by_source = lines.groupby('source').agg(lines=('target', 'size'), pages=('page_id', 'nunique')).to_dict('index')
    (OUT / f'train_{run}.json').write_text(json.dumps(dict(
        run=run, config=CONFIG, filter=FILTER if run == 'C' else None, training_lines=len(lines),
        sampler=('seeded per-epoch numpy permutations, full batches, resumable' if resumable
                 else 'torch DataLoader shuffle with a seeded generator'), resumed_at=resumed_at,
        distinct_lines_by_source=by_source, over_long_dropped=dropped, optimizer_steps=step,
        selected_step=best['step'], selected=best, checkpoint=str(MODELS[run].relative_to(ROOT)),
        history=history, minutes=(time.time() - start) / 60,
    ), indent=2, ensure_ascii=False), encoding='utf-8')
    shutil.rmtree(resume_dir, ignore_errors=True)
    log(f'{run}: done, best step {best["step"]} greedy val CER {best["val_CER"]:.4f}')


def evaluate(run):
    OUT.mkdir(parents=True, exist_ok=True)
    processor = TrOCRProcessor.from_pretrained(MODELS[run])
    model = VisionEncoderDecoderModel.from_pretrained(MODELS[run]).cuda()
    val = human_crops('val')
    start = time.time()
    val['prediction'] = predict(model, processor, val, **EVAL_DECODING)
    ms = 1000 * (time.time() - start) / len(val)
    measures = [jiwer.process_characters(r, h) for r, h in zip(val.target, val.prediction)]
    val['char_errors'] = [m.substitutions + m.deletions + m.insertions for m in measures]
    val['chars'] = val.target.str.len()
    rows = []
    for source, g in [('all', val)] + list(val.groupby('source')):
        low, high = cer_interval(g)
        rows.append(dict(source=source, pages=g.page_id.nunique(), lines=len(g), CER=cer(g.target, g.prediction),
                         CER_95_low=low, CER_95_high=high, WER=wer(g.target, g.prediction)))
    val[['page_id', 'region', 'source', 'annotation_source', 'target', 'prediction', 'char_errors', 'chars']].to_csv(
        OUT / f'val_predictions_{run}.csv', index=False)
    (OUT / f'val_metrics_{run}.json').write_text(json.dumps(dict(
        run=run, model=str(MODELS[run].relative_to(ROOT)), decoding=EVAL_DECODING, ms_per_line=ms, results=rows),
        indent=2), encoding='utf-8')
    log(run, pd.DataFrame(rows).round(4).to_string(index=False))


def pages(runs):
    from cv import pages_with_split, subset
    from pipeline import THRESHOLD, detect, page_errors, reference_lines, transcribe

    detector = ROOT / 'models/rf-small-refined/best'
    val = subset(pages_with_split(), 'val')
    detections = detect(val, detector)
    out = []
    for run in runs:
        processor = TrOCRProcessor.from_pretrained(MODELS[run])
        model = VisionEncoderDecoderModel.from_pretrained(MODELS[run]).cuda()
        start = time.time()
        for p in val:
            regions = transcribe(p, detections[p['file_name']], model, processor, **EVAL_DECODING)
            ref = ' '.join(reference_lines(p))
            hyp = ' '.join(r['text'] for r in regions if r['text'])
            out.append(dict(run=run, page_id=Path(p['file_name']).stem, source=p['source'], chars=len(ref),
                            char_errors=page_errors(ref, hyp), reference=ref, prediction=hyp))
        log(f'{run}: pages done in {time.time() - start:.0f} s')
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(out)
    rows = []
    for (run, source), g in [((r, 'all'), g) for r, g in df.groupby('run')] + list(df.groupby(['run', 'source'])):
        low, high = cer_interval(g)
        rows.append(dict(run=run, source=source, pages=len(g), CER=g.char_errors.sum() / g.chars.sum(),
                         CER_95_low=low, CER_95_high=high))
    df.to_csv(OUT / 'page_predictions.csv', index=False)
    (OUT / 'page_metrics.json').write_text(json.dumps(dict(
        detector=str(detector.relative_to(ROOT)), score_threshold=THRESHOLD, decoding=EVAL_DECODING,
        results=rows), indent=2), encoding='utf-8')
    log(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['teacher', 'train', 'evaluate', 'pages'])
    parser.add_argument('runs', nargs='*')
    args = parser.parse_args()
    if args.command == 'teacher':
        teacher()
    elif args.command == 'pages':
        pages(args.runs or ['A', 'B', 'C'])
    else:
        for run in args.runs:
            (train if args.command == 'train' else evaluate)(run)
