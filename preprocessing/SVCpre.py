import logging
from copy import deepcopy

from preprocessing.base_binarizer import BaseBinarizer
from preprocessing.process_pipeline import File2Batch
from utils.hparams import hparams

SVCSINGING_ITEM_ATTRIBUTES = ['wav_fn', 'spk_id']


class SVCBinarizer(BaseBinarizer):
    def __init__(self, item_attributes=None):
        super().__init__(item_attributes)
        print('spkers: ', set(self.speakers))
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def split_train_test_set(item_names):
        auto_test = item_names[-5:]
        item_names = set(deepcopy(item_names))
        if hparams['choose_test_manually']:
            prefixes = set([str(pr) for pr in hparams['test_prefixes']])
            test_item_names = set()
            # Add prefixes that specified speaker index and matches exactly item name to test set
            for prefix in deepcopy(prefixes):
                if prefix in item_names:
                    test_item_names.add(prefix)
                    prefixes.remove(prefix)
            # Add prefixes that exactly matches item name without speaker id to test set
            for prefix in deepcopy(prefixes):
                for name in item_names:
                    if name.split(':')[-1] == prefix:
                        test_item_names.add(name)
                        prefixes.remove(prefix)
            # Add names with one of the remaining prefixes to test set
            for prefix in deepcopy(prefixes):
                for name in item_names:
                    if name.startswith(prefix):
                        test_item_names.add(name)
                        prefixes.remove(prefix)
            for prefix in prefixes:
                for name in item_names:
                    if name.split(':')[-1].startswith(prefix):
                        test_item_names.add(name)
            test_item_names = sorted(list(test_item_names))
        else:
            test_item_names = auto_test
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def load_meta_data(self, raw_data_dir, ds_id):
        self.items.update(File2Batch.file2temporary_dict(raw_data_dir, ds_id))
