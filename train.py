'''
Source code for training SynthDistill (IJCB 2023):
SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data

The training procedure (models, losses, optimiser, schedule, sampling and
re-sampling, order of random draws) is identical to the original script.
Only the plumbing around it changed:
  * paths are CLI options and resolved relative to this file, not the cwd
  * every run writes to its own results folder; existing logs are never truncated
  * full checkpoints (model + optimiser + scheduler + RNG state) allow --resume
  * the three copies of "generate face -> distill" are factored into functions
  * the validation CSV column that holds a cosine *similarity* is named as such
  * optional --deterministic mode

Without --dataset the run is the original algorithm. Experimental extensions (opt-in):
  * --dataset DIR (from generate_dataset.py): step 1 ("fresh faces") reads cached faces and
    teacher embeddings instead of generating them.
      --resample online  hybrid: step 2 (re-sampling around the cached w) is generated online
      --resample none    static: train on the cached faces only (cf. "static" rows, paper Table 4)
  * --noise NAME: re-sampling noise/direction strategy from src/sampling.py (default: isotropic = original)
'''
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import math
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser(description='SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data')
parser.add_argument('--model', metavar='<model>', type=str, default='TinyFaR_A',
                    choices=['TinyFaR_A', 'TinyFaR_B', 'TinyFaR_C'],
                    help='TinyFaR_A,TinyFaR_B,TinyFaR_C')
parser.add_argument('--resampling_coef', metavar='<resampling_coef>', type=float, default=1.0,
                    help='resampling coefficient')
# ---- paths (originally hard-coded relative to the cwd) ----
parser.add_argument('--stylegan_repo', type=str, default=os.path.join(SCRIPT_DIR, 'stylegan3'),
                    help='clone of https://github.com/NVlabs/stylegan3')
parser.add_argument('--stylegan_pkl', type=str, default=os.path.join(SCRIPT_DIR, 'stylegan2-ffhq-256x256.pkl'),
                    help='pretrained StyleGAN2 FFHQ-256 generator')
parser.add_argument('--stylegan_ref_ops', action='store_true',
                    help="use StyleGAN's pure-PyTorch ops instead of compiling its CUDA kernels (slower, no nvcc needed)")
parser.add_argument('--teacher_fp16', action='store_true',
                    help='run the frozen teacher in fp16 autocast (~9%% faster overall; embeddings differ by '
                         'MSE ~1e-6 from fp32). Off by default = original behaviour')
parser.add_argument('--teacher_checkpoint', type=str, default=None,
                    help='optional local path to iresnet100-73e07ba7.pth (downloaded automatically otherwise)')
parser.add_argument('--results_path', type=str, default=None,
                    help='output folder (default: results/<model>_rc<resampling_coef>)')
# ---- run control ----
parser.add_argument('--resume', action='store_true', help='resume from <results_path>/checkpoint_last.pt')
parser.add_argument('--overwrite', action='store_true', help='allow starting a new run in a non-empty results folder')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--deterministic', action='store_true',
                    help='torch.use_deterministic_algorithms (slower; the original only set cudnn flags)')
# ---- hyper-parameters (defaults = paper / original script) ----
parser.add_argument('--batch_size', type=int, default=2*64)
parser.add_argument('--num_epochs', type=int, default=21)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--iterations_per_epoch_train', type=int, default=None,
                    help='default int(1e6/batch_size); only lower it for smoke tests')
parser.add_argument('--iterations_per_epoch_val', type=int, default=None,
                    help='default int(1e4/batch_size); only lower it for smoke tests')
# ---- experimental extensions (defaults = original algorithm) ----
from src.sampling import STRATEGIES
parser.add_argument('--dataset', type=str, default=None,
                    help='cached dataset from generate_dataset.py used for step 1 (default: generate online)')
parser.add_argument('--resample', choices=['online', 'none'], default='online',
                    help='with --dataset: generate the re-sampled step online (hybrid) or skip it (static)')
parser.add_argument('--noise', choices=sorted(STRATEGIES), default='isotropic',
                    help='re-sampling noise/direction strategy (isotropic = original N(0, I))')
parser.add_argument('--num_workers', type=int, default=8, help='data-loading workers for --dataset')
args = parser.parse_args()
if args.resample == 'none' and not args.dataset:
    parser.error('--resample none requires --dataset')

if args.deterministic:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if args.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

resampling_coef = args.resampling_coef

batch_size = args.batch_size
num_epochs = args.num_epochs
cached = None
if args.dataset:
    from src.cached_data import CachedDataset
    try:
        cached = CachedDataset(args.dataset)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(str(e))
    default_iters = cached.batches_per_epoch(batch_size)   # one pass over the cache
else:
    default_iters = int(1e6/batch_size)
iterations_per_epoch_train = args.iterations_per_epoch_train or default_iters
iterations_per_epoch_val   = args.iterations_per_epoch_val   or int(1e4/batch_size)


# Save the results in the following folder.
run_name = f'{args.model}_rc{resampling_coef:g}'
if cached is not None:
    run_name += f'_{os.path.basename(os.path.normpath(args.dataset))}_{args.resample}'
if args.noise != 'isotropic':
    run_name += f'_{args.noise}'
results_path = args.results_path or os.path.join('results', run_name)
os.makedirs(results_path, exist_ok=True)
log_train = os.path.join(results_path, 'log_train.txt')
log_csv   = os.path.join(results_path, 'log.csv')
ckpt_last = os.path.join(results_path, 'checkpoint_last.pt')

# ---- check inputs before touching the results folder ----
if args.teacher_checkpoint and not os.path.isfile(args.teacher_checkpoint):
    sys.exit(f'Teacher checkpoint not found: {args.teacher_checkpoint}')
if not os.path.isdir(os.path.join(args.stylegan_repo, 'torch_utils')):
    sys.exit(f'StyleGAN repo not found at {args.stylegan_repo}. Run: git clone https://github.com/NVlabs/stylegan3')
if not os.path.isfile(args.stylegan_pkl):
    sys.exit(f'StyleGAN weights not found at {args.stylegan_pkl}. See README for the download link.')


def _has_trained_epochs():
    # a folder only "contains a run" if an epoch finished (a crash during start-up leaves just headers)
    if os.path.isfile(ckpt_last):
        return True
    if os.path.isfile(log_csv):
        with open(log_csv) as f:
            return sum(1 for line in f if line.strip()) > 1
    return False


if args.resume:
    if not os.path.isfile(ckpt_last):
        sys.exit(f'--resume given but {ckpt_last} does not exist')
elif _has_trained_epochs() and not args.overwrite:
    sys.exit(f'{results_path} already contains a run. Use --resume, --overwrite or another --results_path.')
#========================================================

#=================== import Network =====================
from src.Network import LightNetwork
model_names = {'TinyFaR_A': 'tinynet_a', 'TinyFaR_B': 'tinynet_b', 'TinyFaR_C': 'tinynet_c'}
light_model = LightNetwork(model_name=model_names[args.model])
print('light_model # params', sum(p.numel() for p in light_model.parameters()))
light_model.to(device)

from src.ArcFace import get_FaceRecognition_transformer
large_model = get_FaceRecognition_transformer(device, checkpoint_path=args.teacher_checkpoint)

#=================== StyleGAN
from src.stylegan_utils import load_stylegan, synthesize as _synthesize
StyleGAN = load_stylegan(args.stylegan_repo, args.stylegan_pkl, device, ref_ops=args.stylegan_ref_ops)
StyleGAN_synthesis = StyleGAN.synthesis
StyleGAN_mapping   = StyleGAN.mapping
z_dim_StyleGAN = StyleGAN.z_dim
#========================================================

#=================== Re-sampling noise strategy ==========
from src.sampling import make_strategy, load_w_stats
noise_ctx = {'device': device, 'w_dim': StyleGAN.w_dim}
if STRATEGIES[args.noise].needs_w_stats:
    if cached is not None and os.path.isfile(cached.w_stats_path):
        noise_ctx['w_mean'], noise_ctx['w_cov'] = load_w_stats(cached.w_stats_path)
    else:
        from src.stylegan_utils import compute_w_stats
        noise_ctx['w_mean'], noise_ctx['w_cov'] = compute_w_stats(StyleGAN_mapping, z_dim_StyleGAN, device)
resample_noise = make_strategy(args.noise, noise_ctx)
if args.noise != 'isotropic':
    print(f'************ NOTE: re-sampling noise strategy = {args.noise}')
#========================================================

#=================== Loss ===============================
MSELoss = torch.nn.MSELoss(reduction='mean')
CosSim = torch.nn.CosineSimilarity()
#========================================================

#=================== Optimizers =========================
lr = args.lr
optimizer = torch.optim.Adam(light_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#========================================================


use_teacher_fp16 = args.teacher_fp16 and device.type == 'cuda'
if use_teacher_fp16:
    print('************ NOTE: teacher runs in fp16 autocast')

#=================== helpers (same ops as the original inline code) ====
@torch.no_grad()
def sample_w():
    # Generate batch of latent vectors; generate w from noise(z)
    noise = torch.randn(batch_size, z_dim_StyleGAN, device=device)
    return StyleGAN_mapping(z=noise, c=None, truncation_psi=1.0).detach()


def synthesize(w):
    # synthesise images from w, clamp, map to (0,1), align to 112x112 ArcFace crop
    return _synthesize(StyleGAN_synthesis, w)


def embeddings(img):
    emb_light = light_model((img - 0.5) / 0.5)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=use_teacher_fp16):
        emb_large = large_model.transform(img*255.)
    return emb_light, emb_large.float()


def train_step(img):
    emb_light, emb_large = embeddings(img)
    MSE = MSELoss(emb_light, emb_large)
    loss = MSE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, emb_light.detach(), emb_large


def train_step_cached(img, emb_large):
    # same as train_step, but with the teacher embedding read from the cached dataset
    emb_light = light_model((img - 0.5) / 0.5)
    loss = MSELoss(emb_light, emb_large)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, emb_light.detach(), emb_large


def resample_w(w, emb_light, emb_large):
    # Faces the student already matches well get a larger perturbation (explore),
    # hard faces stay close (exploit).
    cos_sim = CosSim(emb_light, emb_large)
    coef = (cos_sim + 1)/2.0
    w_ = w[:, 0, :]
    w_ = w_ + coef.unsqueeze(1) * resampling_coef * resample_noise(w_)   # isotropic: torch.randn(batch, w_dim) as originally
    return w_.unsqueeze(1).repeat([1, StyleGAN.num_ws, 1])


def train_iterations_online(epoch):
    # original algorithm: both steps generated online
    for itr in range(iterations_per_epoch_train):
        # 1) fresh random faces
        w = sample_w()
        img = synthesize(w)
        loss, emb_light, emb_large = train_step(img)

        # 2) re-sampled faces around the previous latents
        w = resample_w(w, emb_light, emb_large)
        img = synthesize(w)
        loss, emb_light, emb_large = train_step(img)
        yield itr, loss, emb_light, emb_large


def train_iterations_cached(epoch):
    # step 1 from the cached dataset; step 2 online (hybrid) or skipped (static)
    loader = cached.loader(batch_size, epoch, args.seed, num_workers=args.num_workers)
    for itr, (w, img_u8, emb_t) in enumerate(loader):
        if itr >= iterations_per_epoch_train:
            break
        w = w.to(device, non_blocking=True)
        img = img_u8.to(device, non_blocking=True).float() / 255.
        emb_large = emb_t.to(device, non_blocking=True)
        loss, emb_light, emb_large = train_step_cached(img, emb_large)

        if args.resample == 'online':
            w = resample_w(w.unsqueeze(1).repeat([1, StyleGAN.num_ws, 1]), emb_light, emb_large)
            img = synthesize(w)
            loss, emb_light, emb_large = train_step(img)
        yield itr, loss, emb_light, emb_large


def rng_state():
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(), 'random': random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])
    np.random.set_state(s['numpy'])
    random.setstate(s['random'])


def log(msg):
    print(msg, flush=True)
    with open(log_train, 'a') as f:
        f.write(msg + '\n')
#========================================================


if not args.resume:
    # everything loaded fine -> start fresh logs
    with open(log_train, 'w') as f:
        f.write(f'# {vars(args)}\n')
    with open(log_csv, 'w') as f:
        f.write("epoch, loss_test_MSE, test_cos_similarity, loss_test\n")

start_epoch = 0
if args.resume:
    ckpt = torch.load(ckpt_last, map_location='cpu')
    light_model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    set_rng_state(ckpt['rng'])
    start_epoch = ckpt['epoch'] + 1
    log(f'# resumed from {ckpt_last} at epoch {start_epoch}')

######################################################################

print('learning rate = ', str(lr))

for epoch in tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs):
    light_model.train()
    train_iterations = train_iterations_cached if cached is not None else train_iterations_online
    for itr, loss, emb_light, emb_large in train_iterations(epoch):
        if itr % 500 == 0:
            cos = CosSim(emb_light, emb_large).mean().item()
            log('epoch = ' + str(epoch) + ', iteration = ' + str(itr) + ', loss = ' + str(loss.item()) + ', cos = ' + str(cos))

    light_model.eval()

    torch.save(light_model.state_dict(), os.path.join(results_path, 'light_model_' + str(epoch) + '.pt'))

    loss_test = loss_test_cos = loss_test_MSE = 0
    for itr in range(iterations_per_epoch_val):
        with torch.no_grad():
            img = synthesize(sample_w())
            emb_light, emb_large = embeddings(img)
            MSE = MSELoss(emb_light, emb_large)
            cos = CosSim(emb_light, emb_large).mean()
            loss = MSE

            loss_test_MSE += MSE.item()
            loss_test_cos += cos.item()
            loss_test     += loss.item()

    with open(log_csv, 'a') as f:
        f.write(f"{epoch}, {loss_test_MSE/iterations_per_epoch_val}, {loss_test_cos/iterations_per_epoch_val}, {loss_test/iterations_per_epoch_val}\n")

    # Update schedulers
    scheduler.step()

    torch.save({'epoch': epoch, 'model': light_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'args': vars(args)}, ckpt_last)
