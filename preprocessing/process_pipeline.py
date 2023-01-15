'''
    file -> temporary_dict -> processed_input -> batch
'''
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import utils
from network.vocoders.nsf_hifigan import nsf_hifigan
from utils.hparams import hparams
from .base_binarizer import BinarizationError
from .data_gen_utils import get_pitch_parselmouth, get_pitch_crepe


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if int(time.time()) - int(data_dict[wav_hash]["time"]) > 14 * 24 * 3600:
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


f0_dict = read_temp("./infer_tools/f0_temp.json")


class File2Batch:
    '''
        pipeline: file -> temporary_dict -> processed_input -> batch
    '''

    @staticmethod
    def file2temporary_dict():
        '''
            read from file, store data in temporary dicts
        '''
        raw_data_dir = Path(hparams['raw_data_dir'])
        utterance_labels = []
        utterance_labels.extend(list(raw_data_dir.rglob(f"*.wav")))
        utterance_labels.extend(list(raw_data_dir.rglob(f"*.ogg")))

        all_temp_dict = {}
        for utterance_label in utterance_labels:
            item_name = str(utterance_label)
            temp_dict = {'wav_fn': str(utterance_label), 'spk_id': hparams['speaker_id']}
            all_temp_dict[item_name] = temp_dict
        return all_temp_dict

    @staticmethod
    def temporary_dict2processed_input(item_name, temp_dict, encoder, infer=False, **kwargs):
        '''
            process data in temporary_dicts
        '''

        def get_pitch(wav, mel):
            # get ground truth f0 by self.get_pitch_algorithm
            global f0_dict
            use_crepe = hparams['use_crepe'] if not infer else kwargs['use_crepe']
            if use_crepe:
                md5 = get_md5(wav)
                if infer and md5 in f0_dict.keys():
                    print("load temp crepe f0")
                    gt_f0 = np.array(f0_dict[md5]["f0"])
                    coarse_f0 = np.array(f0_dict[md5]["coarse"])
                else:
                    torch.cuda.is_available() and torch.cuda.empty_cache()
                    gt_f0, coarse_f0 = get_pitch_crepe(wav, mel, hparams, threshold=0.05)
                if infer:
                    f0_dict[md5] = {"f0": gt_f0.tolist(), "coarse": coarse_f0.tolist(), "time": int(time.time())}
                    write_temp("./infer_tools/f0_temp.json", f0_dict)
            else:
                gt_f0, coarse_f0 = get_pitch_parselmouth(wav, mel, hparams)
            if sum(gt_f0) == 0:
                raise BinarizationError("Empty **gt** f0")
            processed_input['f0'] = gt_f0
            processed_input['pitch'] = coarse_f0

        def get_align(mel, phone_encoded):
            mel2ph = np.zeros([mel.shape[0]], int)
            start_frame = 0
            ph_durs = mel.shape[0] / phone_encoded.shape[0]
            for i_ph in range(phone_encoded.shape[0]):
                end_frame = int(i_ph * ph_durs + ph_durs + 0.5)
                mel2ph[start_frame:end_frame + 1] = i_ph + 1
                start_frame = end_frame + 1

            processed_input['mel2ph'] = mel2ph

        wav, mel = nsf_hifigan.wav2spec(temp_dict['wav_fn'])
        processed_input = {
            'item_name': item_name, 'mel': mel,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]
        }
        processed_input = {**temp_dict, **processed_input,
                           'spec_min': np.min(mel, axis=0),
                           'spec_max': np.max(mel, axis=0)}  # merge two dicts
        try:
            get_pitch(wav, mel)
            try:
                hubert_encoded = processed_input['hubert'] = encoder.encode(temp_dict['wav_fn'])
            except:
                traceback.print_exc()
                raise Exception(f"hubert encode error")
            get_align(mel, hubert_encoded)
        except Exception as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {temp_dict['wav_fn']}")
            return None
        return processed_input

    @staticmethod
    def processed_input2batch(samples):
        '''
            Args:
                samples: one batch of processed_input
            NOTE:
                the batch size is controlled by hparams['max_sentences']
        '''
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        hubert = utils.collate_2d([s['hubert'] for s in samples], 0.0)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples])
        uv = utils.collate_1d([s['uv'] for s in samples])
        energy = utils.collate_1d([s['energy'] for s in samples], 0.0)
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'hubert': hubert,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'energy': energy,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }
        return batch
