# flask api for ngrock
from werkzeug.utils import secure_filename
import os
from flask import Flask, jsonify, request, send_file, g, render_template
from main import Ear, Chatbot, Mouth
from preprompts import call_pre_prompt, advisor_pre_prompt, sales_pre_prompt
import torch
import librosa
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# log = ''
# past_kv = None
# next_id = None


print('loading models...')
ear = Ear(device=device, silence_seconds=2)
john = Chatbot(device=device)
mouth = Mouth(speaker_id=7, device=device)


app = Flask(__name__)

@app.before_request
def before_request():
    g.log = ''
    g.past_kv = None
    g.next_id = None
# a person clicks start chat.
# the models are loaded already
# the chat loop begins
# the client sends audio file
# the server sends back the text


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'wav'}  # Add more extensions if needed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser may submit an empty part without a filename
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the file to the server
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # You can perform additional processing on the file if needed
        print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        audio = librosa.load(os.path.join(app.config['UPLOAD_FOLDER'], filename), sr=16000)[0]
        user_input = ear.transcribe(audio)
        break_word = '[USER]'
        name = '[JOHN]'
        response, g.past_kv, g.next_id = john.generate_response_greedy(user_input, call_pre_prompt + g.log,
                                                                   break_word, max_length=100, name=name,
                                                                   past_key_vals=g.past_kv, next_id=g.next_id,
                                                                   verbose=True, temp=0.6)

        audio = mouth.run_tts(response.replace('[USER]', '').replace('[END]', '').replace('[START]', ''))
        # save audio to file
        sf.write('uploads/response.wav', audio, mouth.model.config.sampling_rate)
        g.log += ' ' + user_input + '\n' + name + response
        print(' ' + user_input + '\n' + name + response)    
        # return the audio file
        return send_file('uploads/response.wav', as_attachment=True)

    return jsonify({'message': 'Invalid file extension'}), 400

@app.route('/reset', methods=['GET'])
def reset():
    g.log = ''
    g.past_kv = None
    g.next_id = None
    return jsonify({'message': 'reset'}), 200


@app.route('/')
def index():
    g.log = ''
    g.past_kv = None
    g.next_id = None
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    break_word = '[USER]'
    name = '[JOHN]'
    response, g.past_kv, g.next_id = john.generate_response_greedy(user_message, advisor_pre_prompt + g.log,
                                                               break_word, max_length=100, name=name,
                                                               past_key_vals=g.past_kv, next_id=g.next_id,
                                                               verbose=True, temp=0.6)
    response = response.replace('[USER]', '').replace('[END]', '').replace('[START]', '')
    return {'bot_response': response}


if __name__ == '__main__':
    app.run(debug=False)
