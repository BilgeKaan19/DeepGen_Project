"""
Teacher network: ArcFace IResNet-100 (InsightFace, MS1M) ported to PyTorch.

Faithful to the original SynthDistill code (same checkpoint, same preprocessing
`(x - 127.5) / 128`, same lazy loading, frozen weights). Changes:

* `imp` (removed in Python 3.12) is replaced by `importlib`.
* `bob.extension` is no longer required. The same Idiap tarball (same URLs, same
  MD5) is downloaded into the same cache folder bob used (`~/bob_data`, or
  `$BOB_DATA_FOLDER`), so an existing bob cache is reused as-is.
* If the Idiap server is unreachable, the identical checkpoint
  (`iresnet100-73e07ba7.pth`) is fetched from its original author's host, or can
  be given explicitly via `checkpoint_path=` / `$SYNTHDISTILL_TEACHER_CKPT`.
  In that case the vendored architecture in `src/iresnet.py` is used.
"""
import hashlib
import importlib.util
import os
import shutil
import tarfile
import urllib.request
import warnings

import torch
from sklearn.base import TransformerMixin, BaseEstimator


class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    Base Transformer using pytorch models


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        device='cpu',
        **kwargs
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor_ = preprocessor
        self.device = device

    def preprocessor(self, X):
        X = self.preprocessor_(X)
        return X

    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        image : torch.Tensor, shape (N, 3, 112, 112), RGB in [0, 255]

        **Returns:**

        feature : torch.Tensor, shape (N, 512)
        """
        if self.model is None:
            self._load_model()

            self.model.eval()

            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False

        X = self.preprocessor(X)

        return self.model(X)

    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def to(self, device):
        self.device = device

        if self.model is not None:
            self.model.to(self.device)


# --------------------------------------------------------------------------
# Checkpoint resolution (replaces bob.extension.download.get_file)
# --------------------------------------------------------------------------
_IDIAP_TARBALL = "iresnet-91a5de61.tar.gz"
_IDIAP_URLS = [
    "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
    "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
]
_IDIAP_MD5 = "3976c0a539811d888ef5b6217e5de425"
_CKPT_NAME = "iresnet100-73e07ba7.pth"
# Same weights, hosted by the author of the PyTorch port (pytorch-insightface).
_FALLBACK_URLS = ["https://sota.nizhib.ai/pytorch-insightface/iresnet100-73e07ba7.pth"]
_CKPT_SHA256_PREFIX = "73e07ba7"


def _cache_dir():
    root = os.environ.get("BOB_DATA_FOLDER", os.path.join(os.path.expanduser("~"), "bob_data"))
    return os.path.join(root, "data", "pytorch", "iresnet-91a5de61")


def _hash_file(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest, timeout=60):
    tmp = dest + ".part"
    print(f"Downloading {url} ...", flush=True)
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)


def _safe_extract(tar_path, out_dir):
    out_dir = os.path.realpath(out_dir)
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            target = os.path.realpath(os.path.join(out_dir, m.name))
            if not target.startswith(out_dir + os.sep) and target != out_dir:
                raise RuntimeError(f"Unsafe path in tarball: {m.name}")
        tar.extractall(out_dir)


def _find(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    return None


def _resolve_teacher_files(checkpoint_path=None):
    """Return (checkpoint_path, config_path_or_None)."""
    checkpoint_path = checkpoint_path or os.environ.get("SYNTHDISTILL_TEACHER_CKPT")
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        return checkpoint_path, None

    cache = _cache_dir()
    os.makedirs(cache, exist_ok=True)

    # 1) already extracted (by bob or by a previous run)
    ckpt = _find(cache, _CKPT_NAME)
    if ckpt:
        return ckpt, _find(cache, "iresnet.py")

    # 2) original Idiap tarball
    tar_path = os.path.join(cache, _IDIAP_TARBALL)
    for url in _IDIAP_URLS:
        try:
            if not os.path.isfile(tar_path):
                _download(url, tar_path)
            if _hash_file(tar_path, "md5") != _IDIAP_MD5:
                os.remove(tar_path)
                raise RuntimeError("MD5 mismatch")
            _safe_extract(tar_path, cache)
            ckpt = _find(cache, _CKPT_NAME)
            if ckpt:
                return ckpt, _find(cache, "iresnet.py")
        except Exception as e:  # network errors, timeouts, bad hash
            warnings.warn(f"Could not get teacher from {url}: {e}")

    # 3) same checkpoint from the PyTorch port's author
    ckpt = os.path.join(cache, _CKPT_NAME)
    for url in _FALLBACK_URLS:
        try:
            _download(url, ckpt)
            return ckpt, None
        except Exception as e:
            warnings.warn(f"Could not get teacher from {url}: {e}")

    raise RuntimeError(
        "Could not download the ArcFace IResNet-100 teacher checkpoint "
        f"({_CKPT_NAME}). Download it manually and either place it in\n  {cache}\n"
        "or pass --teacher_checkpoint /path/to/iresnet100-73e07ba7.pth "
        "(or set $SYNTHDISTILL_TEACHER_CKPT)."
    )


class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(self,
                 preprocessor=lambda x: (x - 127.5) / 128.0,
                 device='cpu',
                 checkpoint_path=None,
                 ):

        self.device = device
        checkpoint_path, config = _resolve_teacher_files(checkpoint_path)

        sha = _hash_file(checkpoint_path, "sha256")
        if not sha.startswith(_CKPT_SHA256_PREFIX):
            warnings.warn(
                f"Teacher checkpoint sha256 {sha[:8]}... does not start with {_CKPT_SHA256_PREFIX}; "
                "it may not be the checkpoint used in the paper.")

        super(IResnet100, self).__init__(
            checkpoint_path, config, preprocessor=preprocessor, device=device
        )

    def _load_model(self):
        if self.config is not None:
            # Original behaviour: architecture file shipped in the Idiap tarball.
            spec = importlib.util.spec_from_file_location("iresnet_idiap", self.config)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            from src import iresnet as module
        self.model = module.iresnet100(self.checkpoint_path)


def get_FaceRecognition_transformer(device='cpu', checkpoint_path=None):
    FaceRecognition_transformer = IResnet100(device=device, checkpoint_path=checkpoint_path)
    return FaceRecognition_transformer
