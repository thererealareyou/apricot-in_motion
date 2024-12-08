import whisper
model = whisper.load_model("base")


def audio_to_text(file):
    result = model.transcribe(file, fp16=False)
    print("Успешно прослушал файл")
    return result["text"]