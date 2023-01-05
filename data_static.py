import matplotlib
import matplotlib.pyplot as plt
from pylab import xticks, np
from tqdm import tqdm

from infer_tools import infer_tool
from network.vocoders.base_vocoder import VOCODERS
from preprocessing.data_gen_utils import get_pitch_parselmouth
from utils.hparams import set_hparams

head_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
hparams = set_hparams(config="./training/config_nsf.yaml", exp_name='', infer=True,
                      reset=True, hparams_str='', print_hparams=False)


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return round(f0_pitch, 0)


def pitch_to_name(pitch):
    return f"{head_list[int(pitch % 12)]}{int(pitch / 12) - 1}"


def get_f0(audio_path):
    wav, mel = VOCODERS["NsfHifiGAN"].wav2spec(audio_path)
    f0, pitch_coarse = get_pitch_parselmouth(wav, mel, hparams)
    return [f0_to_pitch(x) for x in f0[f0 > 0]]


pitch_dict = {}
# 获取batch文件夹下所有wav文件
wav_paths = infer_tool.get_end_file("./batch", "wav")
# parselmouth获取f0
with tqdm(total=len(wav_paths)) as p_bar:
    p_bar.set_description('Processing')
    for wav_path in wav_paths:
        for key in get_f0(wav_path):
            pitch_dict[key] = pitch_dict.get(key, 0) + 1
        p_bar.update(1)
# 转换为时长
pitch_dict = {k: v * hparams['hop_size'] / hparams['audio_sample_rate'] for k, v in pitch_dict.items() if
              v * hparams['hop_size'] / hparams['audio_sample_rate'] > 10}
sort_key = sorted(pitch_dict.keys())
matplotlib.use('TkAgg')
xticks_labels = [pitch_to_name(i) for i in range(36, 96)]
xticks(np.linspace(36, 96, 60, endpoint=True), xticks_labels)
total_time = round(sum(pitch_dict.values()), 2)
print(f"total time: {total_time}s")
sort_value = [pitch_dict[x] for x in sort_key]
plt.plot(sort_key, sort_value, color='dodgerblue')
plt.show()
