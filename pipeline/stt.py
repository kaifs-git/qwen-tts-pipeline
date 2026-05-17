import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from groq import Groq


class GroqSTT:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            wavfile.write(tmp_path, sample_rate, audio_data)

        try:
            with open(tmp_path, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    file=("audio.wav", audio_file),
                    model="whisper-large-v3-turbo",
                    response_format="text",
                )
            return (result or "").strip()
        finally:
            os.unlink(tmp_path)
