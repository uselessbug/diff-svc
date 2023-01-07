import json
import os
import shutil
from functools import reduce
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import yaml
from pylab import xticks, np
from tqdm import tqdm

from network.vocoders.base_vocoder import VOCODERS
from preprocessing.data_gen_utils import get_pitch_parselmouth, get_pitch_crepe
from utils.hparams import set_hparams

head_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

hparams = {'audio_sample_rate': 44100, "f0_max": 1100, "f0_min": 40, "f0_bin": 256, "hop_size": 512}


def compare_pitch(f0_static_dict, pitch_time_temp, trans_key=0):
    return sum({k: v * f0_static_dict[str(k + trans_key)] for k, v in pitch_time_temp.items() if
                str(k + trans_key) in f0_static_dict}.values())


def evaluate_key(input_wav_path, f0_static_dict, scan_key=True, reverse=True):
    pitch_time_temp = static_time(collect_f0(get_f0(input_wav_path, crepe=False)))
    eval_dict = {}
    if scan_key:
        for trans_key in range(-12, 12):
            eval_dict[trans_key] = compare_pitch(f0_static_dict, pitch_time_temp, trans_key=trans_key)
    else:
        eval_dict[0] = compare_pitch(f0_static_dict, pitch_time_temp)
    sort_key = sorted(eval_dict, key=eval_dict.get, reverse=reverse)[:5]
    return sort_key


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return round(f0_pitch, 0)


def pitch_to_name(pitch):
    return f"{head_list[int(pitch % 12)]}{int(pitch / 12) - 1}"


def get_f0(audio_path, crepe=False):
    wav, mel = VOCODERS["NsfHifiGAN"].wav2spec(audio_path)
    if crepe:
        f0, pitch_coarse = get_pitch_crepe(wav, mel, hparams)
    else:
        f0, pitch_coarse = get_pitch_parselmouth(wav, mel, hparams)
    return f0


def merge_dict(dict_list):
    def sum_dict(a, b):
        temp = dict()
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp

    return reduce(sum_dict, dict_list)


def collect_f0(f0):
    pitch_num = {}
    pitch_list = [f0_to_pitch(x) for x in f0[f0 > 0]]
    for key in pitch_list:
        pitch_num[key] = pitch_num.get(key, 0) + 1
    return pitch_num


def static_time(pitch_num):
    pitch_time = {}
    sort_key = sorted(pitch_num.keys())
    for key in sort_key:
        pitch_time[key] = round(pitch_num[key] * hparams['hop_size'] / hparams['audio_sample_rate'], 2)
    return pitch_time


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


if __name__ == "__main__":
    # 给config文件增加f0_static统计音域
    config_path = "../training/config.yaml"
    hparams = set_hparams(config=config_path, exp_name='', infer=True, reset=True, hparams_str='', print_hparams=False)
    f0_dict = {}
    # 获取batch文件夹下所有wav文件
    wav_paths = get_end_file("../batch", "wav")
    # parselmouth获取f0
    with tqdm(total=len(wav_paths)) as p_bar:
        p_bar.set_description('Processing')
        for wav_path in wav_paths:
            f0_dict[wav_path] = collect_f0(get_f0(wav_path, crepe=False))
            p_bar.update(1)

    pitch_num = merge_dict(f0_dict.values())
    pitch_time = static_time(pitch_num)
    total_time = round(sum(pitch_time.values()), 2)
    pitch_time["total_time"] = total_time
    print(f"total time: {total_time}s")
    shutil.copy(config_path, f"{Path(config_path).parent}\\back_{Path(config_path).name}")
    with open(config_path, encoding='utf-8') as f:
        _hparams = yaml.safe_load(f)
        _hparams['f0_static'] = json.dumps(pitch_time)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(_hparams, f)
        print("原config文件已在原目录建立备份：back_config.yaml")
        print("音域统计已保存至config文件，此模型可使用自动变调功能")
    matplotlib.use('TkAgg')
    plt.title("数据集音域统计", fontproperties='SimHei')
    plt.xlabel("音高", fontproperties='SimHei')
    plt.ylabel("时长(s)", fontproperties='SimHei')
    xticks_labels = [pitch_to_name(i) for i in range(36, 96)]
    xticks(np.linspace(36, 96, 60, endpoint=True), xticks_labels)
    plt.plot(pitch_time.keys(), pitch_time.values(), color='dodgerblue')
    plt.show()
