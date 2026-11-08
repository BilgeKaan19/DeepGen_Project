'''
Pre-generate a SynthDistill training set: StyleGAN latents, aligned faces and teacher embeddings.

StyleGAN synthesis is ~64% of SynthDistill's training time and the teacher ~23%. Caching both
lets train.py --dataset train the student ~12-15x faster per sample, and every experiment sees
exactly the same data.

Faces come from the same pipeline as train.py (z ~ N(0,I) -> mapping, truncation_psi=1 ->
synthesis -> clamp -> (0,1) -> ArcFace 112x112 crop). Images are stored as uint8; the teacher
embedding is computed on the stored uint8 image, so image and target match exactly.

Output folder:
    meta.json          settings, progress (`done`), provenance
    w.npy              float32 (N, 512)          W latent of each face (all 14 style inputs equal)
    images.npy         uint8   (N, 3, 112, 112)  RGB faces
    teacher_emb.npy    float32 (N, 512)          ArcFace IResNet-100 embedding of the stored image
    w_stats.npz        mean/cov of W (for the `wcov` re-sampling strategy)

Each batch k is seeded with (seed, k), so an interrupted run resumes and produces identical data.

    python generate_dataset.py --out data/synth_1M --n 1000000
'''
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from src.stylegan_utils import load_stylegan, synthesize, to_uint8, compute_w_stats

FORMAT_VERSION = 1


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--n', type=int, required=True, help='number of faces')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--teacher_fp16', action='store_true', help='compute teacher embeddings in fp16 (~40%% faster teacher)')
    p.add_argument('--teacher_checkpoint', default=None)
    p.add_argument('--stylegan_repo', default=os.path.join(SCRIPT_DIR, 'stylegan3'))
    p.add_argument('--stylegan_pkl', default=os.path.join(SCRIPT_DIR, 'stylegan2-ffhq-256x256.pkl'))
    p.add_argument('--stylegan_ref_ops', action='store_true')
    p.add_argument('--save_every', type=int, default=50, help='flush + record progress every k batches')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, 'meta.json')
    paths = {k: os.path.join(args.out, f'{k}.npy') for k in ('w', 'images', 'teacher_emb')}

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    G = load_stylegan(args.stylegan_repo, args.stylegan_pkl, device, args.stylegan_ref_ops)
    from src.ArcFace import get_FaceRecognition_transformer
    teacher = get_FaceRecognition_transformer(device, checkpoint_path=args.teacher_checkpoint)

    settings = {'format_version': FORMAT_VERSION, 'n': args.n, 'batch_size': args.batch_size, 'seed': args.seed,
                'teacher_fp16': args.teacher_fp16, 'truncation_psi': 1.0, 'noise_mode': 'random',
                'image_size': 112, 'w_dim': G.w_dim, 'z_dim': G.z_dim, 'num_ws': G.num_ws}

    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if {k: meta.get(k) for k in settings} != settings:
            sys.exit(f'{args.out} was created with different settings:\n  {meta}\n'
                     'Use another --out or the same options to resume.')
        mode = 'r+'
        print(f'Resuming at {meta["done"]}/{args.n}')
    else:
        meta = dict(settings, done=0, complete=False,
                    stylegan_pkl=os.path.abspath(args.stylegan_pkl), stylegan_sha256=sha256(args.stylegan_pkl),
                    teacher_checkpoint=os.path.abspath(teacher.checkpoint_path),
                    teacher_sha256=sha256(teacher.checkpoint_path), created=time.strftime('%Y-%m-%d %H:%M:%S'))
        mode = 'w+'
    if meta.get('complete'):
        print(f'{args.out} is already complete ({args.n} faces).')
        return

    open_mm = np.lib.format.open_memmap
    if mode == 'w+':
        W = open_mm(paths['w'], mode='w+', dtype=np.float32, shape=(args.n, G.w_dim))
        IMG = open_mm(paths['images'], mode='w+', dtype=np.uint8, shape=(args.n, 3, 112, 112))
        EMB = open_mm(paths['teacher_emb'], mode='w+', dtype=np.float32, shape=(args.n, 512))
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=1)
    else:
        W, IMG, EMB = (open_mm(paths[k], mode='r+') for k in ('w', 'images', 'teacher_emb'))

    def save_progress(done):
        for a in (W, IMG, EMB):
            a.flush()
        meta['done'] = done
        with open(meta_path + '.tmp', 'w') as f:
            json.dump(meta, f, indent=1)
        os.replace(meta_path + '.tmp', meta_path)

    B = args.batch_size
    n_batches = (args.n + B - 1) // B
    first = meta['done'] // B
    t0 = time.time()
    for k in tqdm(range(first, n_batches), initial=first, total=n_batches, unit='batch'):
        s, e = k * B, min((k + 1) * B, args.n)
        torch.manual_seed(args.seed * 1_000_003 + k)   # per-batch seed -> resumable, reproducible
        with torch.no_grad():
            z = torch.randn(e - s, G.z_dim, device=device)
            w = G.mapping(z=z, c=None, truncation_psi=1.0)
            img_u8 = to_uint8(synthesize(G.synthesis, w))
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.teacher_fp16 and device.type == 'cuda'):
                emb = teacher.transform(img_u8.float()).float()     # teacher sees exactly the stored image
        W[s:e] = w[:, 0].cpu().numpy()
        IMG[s:e] = img_u8.cpu().numpy()
        EMB[s:e] = emb.cpu().numpy()
        if (k + 1) % args.save_every == 0:
            save_progress(e)
    save_progress(args.n)

    print('Computing W statistics ...')
    mean, cov = compute_w_stats(G.mapping, G.z_dim, device)
    np.savez(os.path.join(args.out, 'w_stats.npz'), mean=mean, cov=cov)
    meta['complete'] = True
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    meta['generation_seconds'] = round(time.time() - t0, 1)
    save_progress(args.n)
    gb = sum(os.path.getsize(p) for p in paths.values()) / 1e9
    print(f'Done: {args.n} faces in {args.out} ({gb:.1f} GB)')


if __name__ == '__main__':
    main()
