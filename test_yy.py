from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch
import scipy
from audiocraft.models import MusicGen
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio
from audiocraft.data.audio import audio_write

model_trans = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")


model = musicgen.MusicGen.get_pretrained('melody', device='cuda')

model.set_generation_params(duration=28)

descriptions = ['For elise with electric guitar and heavy drums style']

#model.lm.load_state_dict(torch.load('models/lm_final.pt'))
model = MusicGen.get_pretrained('melody')

melody, sr = torchaudio.load('./dataset/segment_003.wav')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody, sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

'''
res = model.generate(['For Elise with electric guitar and heavy drums'], progress=True) # with electric guitar and heavy drums

#display_audio(res, 32000)

sampling_rate = model_trans.config.audio_encoder.sampling_rate

scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=res.cpu().numpy())
'''