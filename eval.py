'''
Face verification benchmark for SynthDistill students (and the ArcFace teacher).

The authors' official evaluation (benchmark.py in gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill)
calls InsightFace's `arcface_torch` CallBackVerification. This script re-implements exactly that
protocol without the insightface / mxnet / torch.distributed dependencies:
  * InsightFace `.bin` verification files (aligned 112x112 RGB pairs): lfw, calfw, cplfw, cfp_fp, agedb_30
    (get them with download_benchmarks.py)
  * input ((x/255) - 0.5) / 0.5 ; embedding = emb(img) + emb(horizontal flip), L2-normalised
    ("Accuracy-Flip" in insightface; disable flip with --no_flip)
  * 10-fold cross validation (KFold, no shuffle); per fold the threshold on the squared L2 distance
    is picked on 9 folds from np.arange(0, 4, 0.01) and tested on the remaining one;
    reported value = mean +- std of the 10 fold accuracies.

Note: the .bin files are Python pickles, so only load files from a source you trust.

Examples
  python eval.py --model TinyFaR_A --checkpoint checkpoints/SynthDistill-TinyFaR-A.pt
  python eval.py --model TinyFaR_A --checkpoint results/TinyFaR_A_rc1/light_model_16.pt results/TinyFaR_A_rc1/light_model_20.pt
  python eval.py --teacher --teacher_checkpoint /path/to/iresnet100-73e07ba7.pth
'''
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import io
import pickle

import numpy as np
import torch
from PIL import Image

TARGETS = ['lfw', 'cplfw', 'calfw', 'cfp_fp', 'agedb_30']


def load_bin(path):
    with open(path, 'rb') as f:
        try:
            bins, issame = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            bins, issame = pickle.load(f, encoding='bytes')
    imgs = np.empty((len(bins), 112, 112, 3), dtype=np.uint8)
    for i, b in enumerate(bins):
        if not isinstance(b, (bytes, bytearray)):
            b = bytes(b)  # numpy buffer
        img = Image.open(io.BytesIO(b)).convert('RGB')
        if img.size != (112, 112):
            img = img.resize((112, 112), Image.BILINEAR)
        imgs[i] = np.asarray(img)
    return imgs, np.asarray(issame, dtype=bool)


@torch.no_grad()
def embed(fn, imgs, device, batch_size, flip):
    out = []
    for s in range(0, len(imgs), batch_size):
        x = torch.from_numpy(imgs[s:s+batch_size]).to(device).permute(0, 3, 1, 2).float() / 255.  # (0,1) RGB
        e = fn(x)
        if flip:
            e = e + fn(torch.flip(x, dims=[3]))
        out.append(e.float().cpu())
    e = torch.cat(out).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def _acc(threshold, dist, issame):
    pred = dist < threshold
    return np.mean(pred == issame)


def verification(emb, issame, n_folds=10):
    e1, e2 = emb[0::2], emb[1::2]
    dist = np.sum((e1 - e2) ** 2, axis=1)
    thresholds = np.arange(0, 4, 0.01)
    n = len(issame)
    folds = np.array_split(np.arange(n), n_folds)  # == sklearn KFold(shuffle=False)
    accs, best_ths = [], []
    for k in range(n_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        train_accs = [_acc(t, dist[train], issame[train]) for t in thresholds]
        th = thresholds[int(np.argmax(train_accs))]
        accs.append(_acc(th, dist[test], issame[test]))
        best_ths.append(th)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(best_ths))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default=os.path.join(SCRIPT_DIR, 'binaries'), help='folder with the <target>.bin files')
    p.add_argument('--targets', default=','.join(TARGETS), help='comma-separated, e.g. lfw,agedb_30')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint', nargs='+', help='one or more student state_dicts (light_model_<epoch>.pt, checkpoint_last.pt, SynthDistill-TinyFaR-*.pt)')
    g.add_argument('--teacher', action='store_true', help='evaluate the ArcFace IResNet-100 teacher')
    p.add_argument('--model', default='TinyFaR_A', choices=['TinyFaR_A', 'TinyFaR_B', 'TinyFaR_C'])
    p.add_argument('--teacher_checkpoint', default=None)
    p.add_argument('--teacher_fp16', action='store_true', help='evaluate the teacher in fp16 autocast (as train.py --teacher_fp16)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--no_flip', action='store_true')
    p.add_argument('--out', default=None, help='optional CSV to append results to')
    args = p.parse_args()

    if args.checkpoint:
        missing = [c for c in args.checkpoint if not os.path.isfile(c)]
        if missing:
            sys.exit('Checkpoint(s) not found:\n  ' + '\n  '.join(missing) +
                     '\nTrain first (train.py writes results/<model>_rc<coef>/light_model_<epoch>.pt) '
                     'or use an official one, e.g. checkpoints/SynthDistill-TinyFaR-A.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.teacher:
        from src.ArcFace import get_FaceRecognition_transformer
        teacher = get_FaceRecognition_transformer(device, checkpoint_path=args.teacher_checkpoint)
        def teacher_fn(x):  # same input as in training
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.teacher_fp16 and device.type == 'cuda'):
                return teacher.transform(x * 255.).float()
        models = [('ArcFace-IResNet100 (teacher' + (', fp16)' if args.teacher_fp16 else ')'), teacher_fn)]
    else:
        from src.Network import LightNetwork
        names = {'TinyFaR_A': 'tinynet_a', 'TinyFaR_B': 'tinynet_b', 'TinyFaR_C': 'tinynet_c'}
        models = []
        for ckpt in args.checkpoint:
            def make(ckpt=ckpt):
                model = LightNetwork(model_name=names[args.model], pretrained=False)
                sd = torch.load(ckpt, map_location='cpu')
                if isinstance(sd, dict) and 'model' in sd and 'optimizer' in sd:
                    sd = sd['model']
                model.load_state_dict(sd, strict=True)
                model.to(device).eval()
                return lambda x: model((x - 0.5) / 0.5)   # same input as in training
            models.append((f'{args.model} ({ckpt})', make))

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    data = {}
    for t in targets:
        path = os.path.join(args.data_dir, t + '.bin')
        if os.path.isfile(path):
            data[t] = load_bin(path)
        else:
            print(f'  {t:9s} missing ({path}) -- run download_benchmarks.py')
    if not data:
        sys.exit('no benchmark files found')

    for name, fn in models:
        if not args.teacher:
            fn = fn()
        print(f'\nEvaluating {name}, flip={not args.no_flip}')
        rows = []
        for t, (imgs, issame) in data.items():
            emb = embed(fn, imgs, device, args.batch_size, not args.no_flip)
            acc, std, th = verification(emb, issame)
            print(f'  {t:9s} acc = {100*acc:.2f} +- {100*std:.2f}   (threshold {th:.2f}, {len(issame)} pairs)')
            rows.append((t, acc, std))
        print(f'  {"mean":9s} acc = {100*np.mean([r[1] for r in rows]):.2f}')
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            new = not os.path.isfile(args.out)
            with open(args.out, 'a') as f:
                if new:
                    f.write('model, target, acc, std\n')
                for t, acc, std in rows:
                    f.write(f'"{name}", {t}, {acc:.6f}, {std:.6f}\n')


if __name__ == '__main__':
    main()
