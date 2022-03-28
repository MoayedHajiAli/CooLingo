from mimetypes import init
import os.path
from os import path
import gdown
import os
import sys
import torch
import time
import IPython

from TTS.utils.io import load_config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.text.symbols import symbols, phonemes, make_symbols
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.io import load_checkpoint

from src.modules.TTS_repo.TTS.vocoder.utils.generic_utils import setup_generator



files_to_download = {
    "tts_model.pth.tar": "1NFsfhH8W8AgcfJ-BsL8CYAwQfZ5k4T-n",
    "config.json": "1IAROF3yy9qTK43vG_-R67y3Py9yYbD6t",
    "vocoder_model.pth.tar": "1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K",
    "config_vocoder.json": "1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu",
    "scale_stats_vocoder.npy": "11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU",
}


class GlowTts:

    def __init__(self):

        self.load_models_and_files()

        sys.path.append('TTS_repo')

        # model paths
        files_path = os.path.join(os.getcwd(), "pretrained_models")
        files_path = os.path.join(files_path, "tts")
        TTS_MODEL = os.path.join(files_path, "tts_model.pth.tar")
        TTS_CONFIG = os.path.join(files_path, "config.json")
        VOCODER_MODEL = os.path.join(files_path, "vocoder_model.pth.tar")
        VOCODER_CONFIG = os.path.join(files_path, "config_vocoder.json")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

        # load configs
        self.TTS_CONFIG = load_config(TTS_CONFIG)
        self.VOCODER_CONFIG = load_config(VOCODER_CONFIG)

        # self.TTS_CONFIG.audio['stats_path'] = os.path.join(files_path, "scale_stats.npy")
        self.VOCODER_CONFIG.audio['stats_path'] = os.path.join(
            files_path, "scale_stats_vocoder.npy")

        # load the audio processor
        self.ap = AudioProcessor(**self.TTS_CONFIG.audio)

        # LOAD TTS MODEL
        # multi speaker
        speakers = []
        self.speaker_id = None

        if 'characters' in self.TTS_CONFIG.keys():
            symbols, phonemes = make_symbols(**self.TTS_CONFIG.characters)

        # load the model
        num_chars = len(
            phonemes) if self.TTS_CONFIG.use_phonemes else len(symbols)
        model = setup_model(num_chars, len(speakers), self.TTS_CONFIG)

        # load model state
        model, _ = load_checkpoint(model, TTS_MODEL, use_cuda=torch.cuda.is_available())
        self.model = model
        self.model.eval()
        self.model.store_inverse()

        # LOAD VOCODER MODEL
        self.vocoder_model = setup_generator(self.VOCODER_CONFIG)
        self.vocoder_model.load_state_dict(torch.load(
            VOCODER_MODEL, map_location="cpu")["model"])
        self.vocoder_model.remove_weight_norm()
        self.vocoder_model.inference_padding = 0

        # scale factor for sampling rate difference
        self.scale_factor = [1,  self.VOCODER_CONFIG['audio']
                             ['sample_rate'] / self.ap.sample_rate]
        print(f"scale_factor: {self.scale_factor}")

        self.ap_vocoder = AudioProcessor(**self.VOCODER_CONFIG['audio'])
        self.vocoder_model.to(self.device)
        self.vocoder_model.eval()

    def generate_voice(self, sentence: str, length_scale=1.0, noise_scale=0.33, use_cuda=True, enable_figures=False):
        use_cuda = use_cuda and torch.cuda.is_available()

        self.model.length_scale = length_scale  # set speed of the speech.
        self.model.noise_scale = noise_scale  # set speech variationd

        align, spec, stop_tokedns, wav = self.tts(
            self.model, sentence, self.TTS_CONFIG, use_cuda=use_cuda, ap=self.ap, use_gl=False, figures=True
        )
        return align, spec, stop_tokedns, wav

    def interpolate_vocoder_input(scale_factor, spec):
        """Interpolation to tolarate the sampling rate difference
        btw tts model and vocoder"""
        print(" > before interpolation :", spec.shape)
        spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
        spec = torch.nn.functional.interpolate(
            spec, scale_factor=scale_factor, mode='bilinear').squeeze(0)
        print(" > after interpolation :", spec.shape)
        return spec

    def tts(self, model, text, CONFIG, use_cuda, ap, use_gl, figures=True):
        t_1 = time.time()
        # run tts
        target_sr = CONFIG.audio['sample_rate']
        waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs =\
            synthesis(model,
                      text,
                      CONFIG,
                      use_cuda,
                      ap,
                      self.speaker_id,
                      None,
                      False,
                      CONFIG.enable_eos_bos_chars,
                      use_gl)
        # run vocoder
        mel_postnet_spec = self.ap._denormalize(mel_postnet_spec.T).T
        if not use_gl:
            target_sr = self.VOCODER_CONFIG.audio['sample_rate']
            vocoder_input = self.ap_vocoder._normalize(mel_postnet_spec.T)
            if self.scale_factor[1] != 1:
                vocoder_input = interpolate_vocoder_input(
                    self.scale_factor, vocoder_input)
            else:
                vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
            waveform = self.vocoder_model.inference(vocoder_input)
        # format output
        if use_cuda and not use_gl:
            waveform = waveform.cpu()
        if not use_gl:
            waveform = waveform.numpy()
        waveform = waveform.squeeze()
        # compute run-time performance
        rtf = (time.time() - t_1) / (len(waveform) / self.ap.sample_rate)
        tps = (time.time() - t_1) / len(waveform)
        # running time info
        # print(waveform.shape)
        # print(" > Run-time: {}".format(time.time() - t_1))
        # print(" > Real-time factor: {}".format(rtf))
        # print(" > Time per step: {}".format(tps))

        # display audio
        IPython.display.display(
            IPython.display.Audio(waveform, rate=target_sr))
        return alignment, mel_postnet_spec, stop_tokens, waveform

    def load_models_and_files(self):
        ckpt_dir = 'pretrained_models/tts'
        os.makedirs(ckpt_dir, exist_ok=True)

        for file_name in list(files_to_download.keys()):
            if not path.exists(os.path.join(ckpt_dir, file_name)):
                gdown.download(
                    id=files_to_download[file_name], output=os.path.join(ckpt_dir, file_name))
        