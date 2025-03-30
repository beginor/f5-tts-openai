import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from io import BytesIO
import json
from pydantic import BaseModel


class SpeechModel(BaseModel):
    model: str = 'F5-TTS_v1'
    input: str = ''
    instructions: str = ''
    voice: str = ''
    response_format: str = 'mp3'
    speed: float = 1


def convert_to_wave_io(wave_data: np.ndarray, sample_rate: int) -> BytesIO:
    wave_io = BytesIO()
    wavfile.write(wave_io, sample_rate, wave_data)
    wave_io.seek(0)
    return wave_io


def wave_to_mp3(wave_io: BytesIO) -> BytesIO:
    mp3_io = BytesIO()
    audio = AudioSegment.from_wav(wave_io)
    audio.export(mp3_io, format='mp3')
    mp3_io.seek(0)
    return mp3_io


def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content


def read_json_file(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
