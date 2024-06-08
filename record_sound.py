import pyaudio
import wave
import numpy as np
from scipy.io import wavfile
import os

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 0.01  # Adjust as needed

def is_silence(data):
    """Check if the recorded audio data is silence."""
    # Convert bytes data to numpy array
    audio_array = np.frombuffer(data, dtype=np.int16)
    # Calculate root mean square (RMS)
    rms = np.sqrt(np.mean(np.square(audio_array)))
    return rms < SILENCE_THRESHOLD

def record_sound(duration=5, filename="output.wav"):
    """Record audio for the specified duration and save to a WAV file."""
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    
    # Save recorded audio to WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("File saved as", filename)

    try:
        data=wavfile.read(filename)
        if is_silence(data):
            os.remove(filename)
        else:
            return filename
    except Exception as e:
        print(f"Error while reading audio file: {e}")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


