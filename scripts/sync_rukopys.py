"""Fetch the pinned RUKOPYS splits while reusing cached images."""
import json
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_XET_CACHE', str(ROOT / '.cache/huggingface/xet'))

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.hf_api import RepoFile

sys.path.insert(0, str(ROOT / 'src'))
from htr import DATASET_ID, REVISION, SILVER_TEST_REVISION

CACHE_NAME = 'datasets--' + DATASET_ID.replace('/', '--')
LOCAL_CACHE = ROOT / 'data/rukopys'


def token_from_env():
    token = os.environ.get('HF_TOKEN')
    env_file = ROOT / '.env'
    if token or not env_file.is_file():
        return token
    for line in env_file.read_text().splitlines():
        key, separator, value = line.strip().removeprefix('export ').partition('=')
        if separator and key.strip() == 'HF_TOKEN':
            return next(iter(shlex.split(value, comments=True)), None)
    return None


def split_complete(snapshot, split, expected):
    metadata = snapshot / split / 'metadata.jsonl'
    if not metadata.is_file():
        return False
    pages = [json.loads(line) for line in metadata.open(encoding='utf-8')]
    return len(pages) == expected and all((snapshot / split / p['file_name']).is_file() for p in pages)


def sync_split(api, cache, download_cache, revision, split, expected):
    snapshot = cache / 'snapshots' / revision
    if split_complete(snapshot, split, expected):
        print(f'{split}: {expected:,} pages already present at {revision[:8]}', flush=True)
        return

    if cache != LOCAL_CACHE:
        snapshot_download(DATASET_ID, repo_type='dataset', revision=revision, token=api.token,
                          allow_patterns=[f'{split}/images/*', f'{split}/metadata.jsonl'])
        if not split_complete(snapshot, split, expected):
            raise ValueError(f'{split}: pinned snapshot is incomplete')
        print(f'{split}: {expected:,} pages ready at {revision[:8]}', flush=True)
        return

    available = api.list_repo_tree(DATASET_ID, path_in_repo=split, recursive=True,
                                   repo_type='dataset', revision=revision)
    files = [f for f in available if isinstance(f, RepoFile)
             and (f.path == f'{split}/metadata.jsonl' or f.path.startswith(f'{split}/images/'))]
    if len(files) != expected + 1:
        raise ValueError(f'{split}: expected {expected} images and metadata, got {len(files)} files')

    missing = []
    for file in files:
        blob = cache / 'blobs' / (file.lfs.sha256 if file.lfs else file.blob_id)
        path = snapshot / file.path
        if path.is_file() and path.stat().st_size == file.size:
            continue
        if blob.is_file() and blob.stat().st_size == file.size:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
            path.symlink_to(os.path.relpath(blob, path.parent))
        else:
            missing.append((file, path, blob))

    size = sum(f.size for f, _, _ in missing)
    print(f'{split}: downloading {len(missing):,} files ({size / 2**20:,.1f} MiB)', flush=True)

    def fetch(entry):
        file, path, blob = entry
        downloaded = Path(hf_hub_download(DATASET_ID, file.path, repo_type='dataset', revision=revision,
                                         cache_dir=download_cache, token=api.token))
        if downloaded.stat().st_size != file.size:
            raise ValueError(f'Unexpected file size: {file.path}')
        if blob.is_symlink() and not blob.exists():
            blob.unlink(missing_ok=True)
        try:
            os.link(downloaded.resolve(), blob)
        except FileExistsError:
            if blob.stat().st_size != file.size:
                raise ValueError(f'Unexpected cached file size: {file.path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        path.symlink_to(os.path.relpath(blob, path.parent))

    with ThreadPoolExecutor(max_workers=8) as workers:
        for index, _ in enumerate(workers.map(fetch, missing), 1):
            if index % 100 == 0:
                print(f'{split}: downloaded {index:,}/{len(missing):,} files', flush=True)
    if not split_complete(snapshot, split, expected):
        raise ValueError(f'{split}: pinned snapshot is incomplete')
    print(f'{split}: {expected:,} pages ready at {revision[:8]}', flush=True)


if __name__ == '__main__':
    api = HfApi(token=token_from_env())
    with TemporaryDirectory(dir=ROOT / 'data') as temporary:
        cache = LOCAL_CACHE if LOCAL_CACHE.is_dir() else Path(HF_HUB_CACHE) / CACHE_NAME
        sync_split(api, cache, temporary, REVISION, 'train', 1330)
        sync_split(api, cache, temporary, SILVER_TEST_REVISION, 'silver', 8930)
        sync_split(api, cache, temporary, SILVER_TEST_REVISION, 'test', 382)
