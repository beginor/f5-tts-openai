import numpy as np
from scipy.io import wavfile
from io import BytesIO
import json
from pydantic import BaseModel
import subprocess
import os
import tempfile


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


def wave_to_mp3(wave_io: BytesIO, sample_rate: int) -> BytesIO:
    # Save the wave data to a temporary WAV file
    tmp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    with open(tmp_wav_path, 'wb') as tmp_file:
        tmp_file.write(wave_io.getvalue())
    wave_io.close()

    # Define the temporary MP3 file path
    tmp_mp3_path = tmp_wav_path.replace('.wav', '.mp3')

    # Use ffmpeg to convert the WAV file to MP3
    try:
        subprocess.run(
            ['ffmpeg', '-i', tmp_wav_path, '-ar', '16000', '-ac', '1', tmp_mp3_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
        raise

    # Read the MP3 file into a BytesIO object
    mp3_io = BytesIO()
    with open(tmp_mp3_path, 'rb') as mp3_file:
        mp3_io.write(mp3_file.read())
    mp3_io.seek(0)

    # Clean up the temporary files
    os.remove(tmp_wav_path)
    os.remove(tmp_mp3_path)

    # Return the MP3 data as BytesIO
    return mp3_io


def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content


def read_json_file(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)
