from faster_whisper import WhisperModel


DEFAULT_MODEL_SIZE = "medium"
TRANSCRIBE_FILE_PATH="transcribed_audio"

def transcribe_audio():
    model_size=DEFAULT_MODEL_SIZE + ".en"
    model=WhisperModel(model_size)
    segments,info=model.transcribe('output.wav')
    transcription= ''.join(segment.text for segment in segments)
    print(transcription)
    return transcription


