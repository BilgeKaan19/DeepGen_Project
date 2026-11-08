'''
Download the face-verification benchmarks used in the paper and store them as
InsightFace-style `.bin` files (the format read by eval.py and by the official
benchmark.py / insightface `arcface_torch` verification code).

Source: HuggingFace dataset `cat-claws/face-verification` (aligned 112x112 pairs,
same pair counts as the InsightFace binaries: 6000 pairs, 7000 for CFP). The
image bytes are copied unchanged; pair order is preserved, which matters for the
10-fold protocol.

    python download_benchmarks.py                       # -> binaries/{lfw,calfw,cplfw,cfp_fp,agedb_30}.bin
    python download_benchmarks.py --targets lfw cfp_ff --out_dir binaries

If you already have the original InsightFace .bin files (e.g. from the
faces_emore / MS1M package), just put them in the output folder instead.
'''
import argparse
import os
import pickle
import shutil
import urllib.request

import pyarrow.parquet as pq

REPO = 'https://huggingface.co/datasets/cat-claws/face-verification/resolve/main/data/'
FILES = {
    'lfw':      'lfw-00000-of-00001-eedd8244e78ffd55.parquet',
    'calfw':    'calfw-00000-of-00001-494813e56bc84049.parquet',
    'cplfw':    'cplfw-00000-of-00001-f1a0092b98fb0677.parquet',
    'cfp_fp':   'cfp_fp-00000-of-00001-b24730458d082e8b.parquet',
    'agedb_30': 'agedb_30-00000-of-00001-a951a9443360b4cd.parquet',
    'cfp_ff':   'cfp_ff-00000-of-00001-699a051e1c50724b.parquet',
}
EXPECTED_PAIRS = {'lfw': 6000, 'calfw': 6000, 'cplfw': 6000, 'agedb_30': 6000, 'cfp_fp': 7000, 'cfp_ff': 7000}


def download(url, dest):
    if os.path.isfile(dest):
        return
    print(f'  downloading {url}', flush=True)
    tmp = dest + '.part'
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dest)


def convert(parquet_path, bin_path, name):
    table = pq.read_table(parquet_path)
    img1 = table.column('image1').to_pylist()
    img2 = table.column('image2').to_pylist()
    target = table.column('target').to_pylist()
    bins, issame = [], []
    for a, b, t in zip(img1, img2, target):
        bins.append(a['bytes'])
        bins.append(b['bytes'])
        issame.append(bool(t))
    n = len(issame)
    if n != EXPECTED_PAIRS[name]:
        print(f'  WARNING: {name} has {n} pairs, expected {EXPECTED_PAIRS[name]}')
    with open(bin_path + '.part', 'wb') as f:
        pickle.dump((bins, issame), f, protocol=4)
    os.replace(bin_path + '.part', bin_path)
    print(f'  {name}: {n} pairs ({sum(issame)} same / {n - sum(issame)} different) -> {bin_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'binaries'))
    p.add_argument('--targets', nargs='+', default=['lfw', 'calfw', 'cplfw', 'cfp_fp', 'agedb_30'],
                   choices=list(FILES))
    p.add_argument('--keep_parquet', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.targets:
        bin_path = os.path.join(args.out_dir, name + '.bin')
        if os.path.isfile(bin_path):
            print(f'  {name}: {bin_path} exists, skipping')
            continue
        pq_path = os.path.join(args.out_dir, FILES[name])
        download(REPO + FILES[name], pq_path)
        convert(pq_path, bin_path, name)
        if not args.keep_parquet:
            os.remove(pq_path)


if __name__ == '__main__':
    main()
