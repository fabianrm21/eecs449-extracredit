import whisper

model = whisper.load_model("base")
result = model.transcribe("voice_input.wav")
print(result["text"])