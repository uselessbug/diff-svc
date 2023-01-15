import importlib
import os

from utils.hparams import set_hparams, hparams

os.environ["OMP_NUM_THREADS"] = "1"


def binarize():
    binarizer_cls = hparams.get("binarizer_cls", 'basics.base_binarizer.BaseBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()


if __name__ == '__main__':
    set_hparams()
    binarize()
