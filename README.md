# SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data

	
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-lfw)](https://paperswithcode.com/sota/synthetic-face-recognition-on-lfw?p=synthdistill-face-recognition-with-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-cplfw)](https://paperswithcode.com/sota/synthetic-face-recognition-on-cplfw?p=synthdistill-face-recognition-with-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-calfw)](https://paperswithcode.com/sota/synthetic-face-recognition-on-calfw?p=synthdistill-face-recognition-with-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-cfp-fp)](https://paperswithcode.com/sota/synthetic-face-recognition-on-cfp-fp?p=synthdistill-face-recognition-with-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthdistill-face-recognition-with-knowledge/synthetic-face-recognition-on-agedb-30)](https://paperswithcode.com/sota/synthetic-face-recognition-on-agedb-30?p=synthdistill-face-recognition-with-knowledge)

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2308.14852-009d81.svg)](https://arxiv.org/abs/2308.14852)
	
This repository contains the source code to train **SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data** (IJCB 2023). You can access arxiv version [here](https://arxiv.org/pdf/2308.14852.pdf).

## Installation
The original setup pins `torch==1.13.0+cu117`, which needs **Python 3.8-3.10** (no wheels for newer Python).
The recommended route below uses [uv](https://docs.astral.sh/uv/) + a plain virtualenv (no conda); uv also fetches
Python 3.10 if your system only has a newer one:

```sh
$ cd synthdistill
$ curl -LsSf https://astral.sh/uv/install.sh | sh   # once
$ uv venv --python 3.10 .venv
$ source .venv/bin/activate
$ uv pip install --index-strategy unsafe-best-match -r requirements.txt
```
(`--index-strategy` lets uv take torch from the PyTorch index and everything else from PyPI.)
(Any Python 3.8-3.10 with `python -m venv .venv && pip install -r requirements.txt` works too.
A conda alternative is kept in `environment.yml`.)

The `+cu117` wheel bundles its own CUDA 11.7 runtime, so it runs on any recent NVIDIA driver
(a driver reporting "CUDA Version: 12.x/13.x" in `nvidia-smi` is fine).

**StyleGAN custom CUDA kernels.** On first use StyleGAN JIT-compiles small CUDA ops (`bias_act`,
`upfirdn2d`); if compilation fails, training stops with an error. Either pass `--stylegan_ref_ops` to use
StyleGAN's pure-PyTorch reference ops instead (same math, still on the GPU, slower: ~2.1 s/iteration for
TinyFaR-A on an RTX A4000), or set up the compiler. Compiling needs an `nvcc` matching torch (11.7) and GCC/G++ <= 11.
Without conda, install a user-local toolkit (no root, no driver change; the installer is ~3.5 GB) and GCC 11:
```sh
$ wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
$ sh cuda_11.7.1_515.65.01_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-11.7 --override --no-man-page
$ sudo apt install gcc-11 g++-11
# before training (e.g. add to .venv/bin/activate):
$ export CUDA_HOME=$HOME/cuda-11.7 PATH=$HOME/cuda-11.7/bin:$PATH CC=gcc-11 CXX=g++-11
```

In our knowledge distillation framework, we use StyleGAN as a pretrained face generator network. Therefore, you need to clone StyleGAN repository and download its model weights (both inside this folder):
```sh
$ git clone https://github.com/NVlabs/stylegan3
$ wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl
```
(official page: [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files))

The ArcFace IResNet-100 teacher (`iresnet100-73e07ba7.pth`) is downloaded automatically on first use into
`~/bob_data/data/pytorch/iresnet-91a5de61/` (same place `bob` used). If the download servers are unreachable,
put the file there yourself or pass `--teacher_checkpoint /path/to/iresnet100-73e07ba7.pth`.

## Training models
To train models, you can use the following command:
```sh
$ python train.py --model TinyFaR_A --resampling_coef 1.0
```
Results go to `results/<model>_rc<coef>/` (`log_train.txt`, `log.csv`, `light_model_<epoch>.pt`, `checkpoint_last.pt`).
Interrupted runs continue with `--resume`. Other options: `python train.py --help`.

**Epochs:** the code trains for 21 epochs (default kept), while the paper (Sec. 4.2) states 17 epochs.
A model is saved after every epoch, so evaluate both `light_model_16.pt` (17 epochs) and `light_model_20.pt`.

**Speed:** about 1.8 s/iteration on an RTX A4000 (~3.4 days for 21 epochs). StyleGAN generation is ~64% of the time,
the teacher ~23%, the student ~13%. Optional `--teacher_fp16` runs the frozen teacher in fp16 and is ~8% faster overall;
teacher embeddings differ from fp32 by MSE ~1.6e-6 (cos 0.999999) and its benchmark accuracies are identical.
It is off by default so the default run stays faithful to the original code.

## Experiments with a cached dataset (extension, not in the paper's code)
StyleGAN generation (~64% of training time) and the teacher (~23%) can be computed once and reused:
```sh
# ~1.5 h and ~40 GB per 1M faces on an RTX A4000 (resumable: re-run the same command after an interruption)
$ python generate_dataset.py --out data/synth_1M --n 1000000
```
It stores `w`, the aligned 112×112 face (uint8) and the teacher embedding of exactly that image; each batch is
seeded separately, so resumed and uninterrupted runs give byte-identical data.

Training modes (`train.py` without `--dataset` is unchanged and bit-identical to the faithful run):
| Mode | Command | Step 1 (fresh faces) | Step 2 (re-sampled faces) | Speed* |
|---|---|---|---|---|
| original | `python train.py` | online | online | 1× |
| hybrid | `python train.py --dataset data/synth_1M --resample online` | cached | online | ~1.6× |
| static | `python train.py --dataset data/synth_1M --resample none` | cached | none | ~7× or more |

\*measured at batch 64 with 20 iterations per epoch, including validation.

- With `--dataset`, one epoch = one pass over the cache (`--iterations_per_epoch_train` caps it).
  Validation still uses fresh online faces, as in the original.
- The static mode corresponds to the paper's "static" rows (Table 4, TinyFaR-A, 17 epochs: 1M →
  LFW 98.87 / CP-LFW 83.52 / AgeDB-30 91.25), which is a sanity check for the cached pipeline:
  `python train.py --dataset data/synth_1M --resample none --num_epochs 17`.
- Compare cached experiments with a baseline trained on the same cache, not with the paper's dynamic numbers.
- Re-sampling strategies live in `src/sampling.py` (`--noise`): `isotropic` (original N(0, I)) and
  `wcov` (N(0, Σ_W), i.e. noise shaped like W itself; W has std ~0.41 per dimension, while the original noise
  has std 1.0). New direction strategies (PCA, pose, age, ...) are added there with `@register`.

## Evaluation
`eval.py` re-implements the authors' official evaluation (`benchmark.py` in the
[official repository](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill), which calls InsightFace's
`arcface_torch` verification): flip test, L2-normalised embeddings, 10-fold accuracy (mean ± std), without needing
insightface, mxnet or torch.distributed.

Download the benchmarks (LFW, CA-LFW, CP-LFW, CFP-FP, AgeDB-30; ~1.1 GB) as InsightFace `.bin` files into `binaries/`:
```sh
$ python download_benchmarks.py
```
(source: HuggingFace `cat-claws/face-verification`; original InsightFace `.bin` files can be dropped into `binaries/` instead)

Evaluate the official pretrained checkpoints, your own runs, or the teacher:
```sh
$ mkdir -p checkpoints && wget -P checkpoints https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill/-/raw/master/checkpoints/SynthDistill-TinyFaR-A.pt
$ python eval.py --model TinyFaR_A --checkpoint checkpoints/SynthDistill-TinyFaR-A.pt
$ python eval.py --model TinyFaR_A --checkpoint results/TinyFaR_A_rc1/light_model_16.pt results/TinyFaR_A_rc1/light_model_20.pt --out results/eval.csv
$ python eval.py --teacher --teacher_checkpoint /path/to/iresnet100-73e07ba7.pth
```

### Reported results (paper, Table 3 and footnote 2; c = 1)
| Model | LFW | CA-LFW | CP-LFW | CFP-FP | AgeDB-30 |
|---|---|---|---|---|---|
| Teacher (ArcFace IResNet-100) | 99.77 ± 0.28 | 96.10 ± 1.10 | 92.88 ± 1.52 | 96.27 ± 1.10 | 98.25 ± 0.71 |
| TinyFaR-A | 99.52 ± 0.31 | 94.57 ± 1.01 | 87.00 ± 1.64 | 90.89 ± 1.54 | 94.93 ± 1.35 |
| TinyFaR-B | 99.20 ± 0.41 | 93.78 ± 0.78 | 84.93 ± 2.10 | 88.19 ± 1.34 | 93.02 ± 1.30 |
| TinyFaR-C | 98.58 ± 0.44 | 91.80 ± 1.04 | 82.00 ± 2.14 | 84.54 ± 1.57 | 88.98 ± 1.49 |

### Official checkpoints re-evaluated with `eval.py` (this repo)
| Model | LFW | CA-LFW | CP-LFW | CFP-FP | AgeDB-30 |
|---|---|---|---|---|---|
| Teacher (ArcFace IResNet-100) | 99.77 ± 0.28 | 96.10 ± 1.10 | 92.85 ± 1.53 | 96.20 ± 1.09 | 98.25 ± 0.71 |
| TinyFaR-A | 99.52 ± 0.31 | 94.57 ± 1.01 | 87.07 ± 1.81 | 90.90 ± 1.52 | 94.95 ± 1.36 |
| TinyFaR-B | 99.20 ± 0.41 | 93.78 ± 0.78 | 84.98 ± 2.09 | 88.19 ± 1.34 | 93.07 ± 1.20 |
| TinyFaR-C | 98.58 ± 0.44 | 91.80 ± 1.04 | 81.93 ± 2.02 | 84.61 ± 1.50 | 88.95 ± 1.48 |

LFW and CA-LFW match the paper exactly; the other benchmarks differ by at most 0.07 points (likely
tiny image-decoding differences between the original `.bin` files and the HuggingFace copy).
The evaluation pipeline is therefore validated; differences larger than this in your own runs come from training.

## Changes with respect to the original code
The training algorithm and all hyper-parameters are unchanged. Fixes:
- `environment.yml`: PyTorch CUDA wheel index added (`torch==1.13.0+cu117` could not be installed), Python pinned to 3.8, stale build strings and the Idiap/bob channel removed; `requirements.txt` added.
- Teacher loading no longer needs `bob.extension` or `imp` (removed in Python 3.12); same Idiap tarball/MD5 and cache folder, with fallbacks and `--teacher_checkpoint` (architecture vendored in `src/iresnet.py`).
- `train.py`: paths are options resolved relative to the script; one results folder per run, logs never silently truncated; full checkpoints and `--resume`; duplicated generation code factored out (same operations and order of random draws); optional `--deterministic`.
- `log.csv`: the column with the mean cosine *similarity* is now named `test_cos_similarity` (was `loss_test_cos`).
- `LightNetwork.get_embedding()` now returns the 512-d embedding; the backbone features are `get_features()`.
- `eval.py` added: same protocol as the official `benchmark.py`, without insightface/mxnet/distributed dependencies; `download_benchmarks.py` fetches the benchmark `.bin` files.
- `--stylegan_ref_ops`: run StyleGAN without compiling its CUDA kernels.
- `--teacher_fp16` (opt-in): fp16 teacher, ~8% faster training.
- Extensions: `generate_dataset.py` and the `--dataset/--resample/--noise` options of `train.py` (see above).

## Pre-trained models (SynthDistill)
Checkpoints of trained models (TinyFaR-A, TinyFaR-B, and TinyFaR-C) using SynthDistill are available in the [official repository](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill):
- [TinyFaR-A](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill/-/blob/master/checkpoints/SynthDistill-TinyFaR-A.pt)
- [TinyFaR-B](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill/-/blob/master/checkpoints/SynthDistill-TinyFaR-B.pt)
- [TinyFaR-C](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill/-/blob/master/checkpoints/SynthDistill-TinyFaR-C.pt)


## Reference
If you use this repository, please cite the following paper, which is published in the proceedings of 2023 IEEE International Joint Conference on Biometrics (IJCB 2023). The PDF version of the paper is available as [pre-print on arxiv](https://arxiv.org/pdf/2308.14852.pdf). The complete source code for reproducing all experiments in the paper (including evlauation instructions) is also publicly available in the [official repository](https://gitlab.idiap.ch/bob/bob.paper.ijcb2023_synthdistill).


```bibtex
@inproceedings{synthdistill_IJCB2023,
  title={SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data},
  author={Otroshi Shahreza, Hatef and George, Anjith and Marcel, S{\'e}bastien},
  booktitle={2023 IEEE International Joint Conference on Biometrics (IJCB)},
  year={2023},
  organization={IEEE}
}
```