
import openai
import numpy as np
api_key='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'
openai.api_key = api_key
audio_file= open("C:\\Users\\EZShifa\Desktop\\chatbot_api\\audio_files\\file_1.mp3", "rb")
transcript = openai.Audio.translate(
  model="whisper-1", 
  file=audio_file,
  response_format="text",
)
print(transcript)