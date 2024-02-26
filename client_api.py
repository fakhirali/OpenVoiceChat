from utils import record
import soundfile as sf
import requests
from playsound import playsound


url = 'https://f7d2-154-192-45-8.ngrok-free.app'
file_path = 'media/test.wav'
res = requests.get(f'{url}/reset')
print(res.json())
while True:
    audio = record(2)
    sf.write(file_path, audio, 16_000, 'PCM_24')
    files = {'file': open(file_path, 'rb')}
    response = requests.post(f'{url}/upload_audio', files=files)
    open('media/returned.wav', 'wb').write(response.content)
    playsound('media/returned.wav')