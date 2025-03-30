from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder, load_model,
    preprocess_ref_audio_text, infer_process
)
from os import path
import torch

from utils import (
    read_text_file,
    read_json_file,
    convert_to_wave_io,
    wave_to_mp3,
    SpeechModel
)

assets = './assets'
models = f'{assets}/models'
f5tts_v1_base = f'{models}/f5tts_v1_base'

voices = f'{assets}/voices'
default_ref_audio = f'{voices}/man.wav'
default_ref_text = f'{voices}/man.wav.txt'

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

f5tts_model = load_model(
    model_cls=DiT,
    ckpt_path=f'{f5tts_v1_base}/model_1250000.safetensors',
    vocab_file=f'{f5tts_v1_base}/vocab.txt',
    model_cfg=read_json_file(f'{f5tts_v1_base}/config.json')
)
vocoder_model = load_vocoder(
    vocoder_name='vocos',
    is_local=True,
    local_path=f'{models}/vocos-mel-24khz'
)

app = FastAPI()


@app.post("/v1/audio/speech", tags=['audio'])
def speech(params: SpeechModel):
    voice = params.voice
    ref_audio = default_ref_audio
    ref_text = read_text_file(default_ref_text)

    if path.exists(f'{voices}/{voice}.wav') and path.exists(f'{voices}/{voice}.wav.txt'):
        ref_audio = f'{voices}/{voice}.wav'
        ref_text = read_text_file(f'{voices}/{voice}.wav.txt')

    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)

        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=params.input,
            model_obj=f5tts_model,
            vocoder=vocoder_model,
            cross_fade_duration=0.15,
            nfe_step=32,
            speed=params.speed,
            device=device,
        )

        bytes_io = convert_to_wave_io(final_wave, final_sample_rate)

        if params.response_format == 'mp3':
            bytes_io = wave_to_mp3(bytes_io, final_sample_rate)

        return StreamingResponse(
            bytes_io,
            media_type="audio/mp3" if params.response_format == 'mp3' else 'audio/wav',
        )
    except Exception as ex:
        print(ex)
        return JSONResponse(
            content={'error': str(ex)},
            status_code=500
        )
