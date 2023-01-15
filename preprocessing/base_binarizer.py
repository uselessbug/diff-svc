import json
import os

import numpy as np
import yaml
from tqdm import tqdm

from infer_tools.data_static import collect_f0, static_time
from network.vocoders.nsf_hifigan import NsfHifiGAN
from preprocessing.data_gen_utils import get_pitch_parselmouth, get_pitch_crepe
from preprocessing.hubertinfer import Hubertencoder
from utils.hparams import set_hparams, hparams
from utils.indexed_datasets import IndexedDatasetBuilder

os.environ["OMP_NUM_THREADS"] = "1"
BASE_ITEM_ATTRIBUTES = ['wav_fn', 'spk_id']


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    '''
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    '''

    def __init__(self, item_attributes=BASE_ITEM_ATTRIBUTES):
        self.binarization_args = hparams['binarization_args']
        self.vocoder = NsfHifiGAN()
        self.phone_encoder = Hubertencoder(pt_path=hparams['hubert_path'])
        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        self.load_meta_data()
        # check program correctness 检查itemdict的key只能在给定的列表中取值
        assert all([attr in self.item_attributes for attr in list(self.items.values())[0].keys()])
        self.item_names = sorted(list(self.items.keys()))

        # set default get_pitch algorithm
        if hparams['use_crepe']:
            self.get_pitch_algorithm = get_pitch_crepe
        else:
            self.get_pitch_algorithm = get_pitch_parselmouth

    def load_meta_data(self):
        raise NotImplementedError

    @property
    def train_item_names(self):
        raise NotImplementedError

    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        raise NotImplementedError

    def build_spk_map(self):
        spk_map = set()
        for item_name in self.item_names:
            spk_name = self.items[item_name]['spk_id']
            spk_map.add(spk_name)
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
        assert len(spk_map) == 0 or len(spk_map) <= hparams['num_spk'], len(spk_map)
        return spk_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.items[item_name]['spk_id']]

    def meta_data_iterator(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))
        self.process_data_split('valid')
        self.process_data_split('test')
        self.process_data_split('train')

    def process_data_split(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths = []
        total_sec = 0

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])
        spec_min = []
        spec_max = []
        f0_dict = {}
        # code for single cpu processing
        for i in tqdm(reversed(range(len(args))), total=len(args)):
            a = args[i]
            item = self.process_item(*a)
            if item is None:
                continue
            spec_min.append(item['spec_min'])
            spec_max.append(item['spec_max'])
            f0_dict[item['wav_fn']] = item['f0']
            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
        if prefix == 'train':
            spec_max = np.max(spec_max, 0)
            spec_min = np.min(spec_min, 0)
            pitch_time = static_time(f0_dict)
            effective_time = round(sum(pitch_time.values()), 2)
            pitch_time['effective_time'] = effective_time
            print(f"dataset effective time: {effective_time}s")
            with open(hparams['config_path'], encoding='utf-8') as f:
                _hparams = yaml.safe_load(f)
                _hparams['spec_max'] = spec_max.tolist()
                _hparams['spec_min'] = spec_min.tolist()
                _hparams['f0_static'] = json.dumps(pitch_time)
            with open(hparams['config_path'], 'w', encoding='utf-8') as f:
                yaml.safe_dump(_hparams, f)
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_item(self, item_name, meta_data, binarization_args):
        from preprocessing.process_pipeline import File2Batch
        return File2Batch.temporary_dict2processed_input(item_name, meta_data, self.phone_encoder)


if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
