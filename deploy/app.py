# app.py

from flask import Flask, render_template, request, redirect, url_for,jsonify
import os
import boto3
import os
from openai import OpenAI
import uuid
import random



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


session = boto3.Session(
     aws_access_key_id='AKIAVAHUPHL6XWW4X7PU',
    aws_secret_access_key='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')

s3 = session.resource('s3')

client = OpenAI(api_key='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')

s3_client = boto3.client(service_name='s3', 
    region_name='us-east-2',
    aws_access_key_id='AKIAVAHUPHL6XWW4X7PU',
    aws_secret_access_key='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')

bucket_name = 'prof.ezshifa.com'

def predict(audio_file):
    
           
    try:
        audio_file= open(audio_file, "rb")
    except:
         
        return "Audio File failed to open"
    
    try:
        transcript = client.audio.translations.create(
            model="whisper-1", 
            file=audio_file,
            response_format="text"
        )

        print("[INFO] Prediction",transcript)
        
        audio_file.close()
        if len(transcript)==0:
             transcript ="[ERROR IN DETECTION]Try again model failed to detect"
        return transcript
    except:
         
         return "Model Prediction Failed"


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method=="GET":
         return render_template('upload.html')
    
    elif request.method=="POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)# Saving File
            ret = predict(path)# Making Prediction
    #        s3_client.upload_file(path, bucket_name, path.split("/")[-1])#Pushing to bucket
            # Filename - File to upload
            # Bucket - Bucket to upload to (the top level directory under AWS S3)
            # Key - S3 object name (can contain subdirectories). If not specified then file_name is used
            print("[INFO] Filename: ",filename)
            print('[INFO] retunr data: ',ret)
            s3.meta.client.upload_file(Filename=path, Bucket=bucket_name, Key=filename)



            if os.path.exists(path):#Remove from server
                    os.remove(path)


            
            return ret
        else:
            print("Invalid")
            return 'Invalid file type! Allowed file types are mp3, wav, ogg.'

def Predict_2(name):

    
        path = os.path.join(app.config['UPLOAD_FOLDER'], name)
        

        ret = predict(path)# Making Prediction

#        s3_client.upload_file(path, bucket_name, path.split("/")[-1])#Pushing to bucket
        # Filename - File to upload
        # Bucket - Bucket to upload to (the top level directory under AWS S3)
        # Key - S3 object name (can contain subdirectories). If not specified then file_name is used
        print("[Info] return",ret)
        print(name)
        s3.meta.client.upload_file(Filename=path, Bucket=bucket_name, Key=name)



        if os.path.exists(path):#Remove from server
                
                os.remove(path)
                print("[INFO] Audio Removed")


        print("[RETURN AUDIO MICROPHONE]",ret)
        return ret



@app.route('/save-record', methods=['POST','GET'])
def save_record():
    # check if the post request has the file part

    if request.method =='GET':
    # Code to handle GET requests
        print("save GET")
        return render_template('save.html')
    
    elif request.method == 'POST':
        print("save POST")
        if 'file' not in request.files:
            
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        random_number1 = random.uniform(1, 1e100)
        random_number2 = random.uniform(1, 1e100)

        file_name = str(random_number1)+str(random_number2) + ".mp3"
        full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

        while True:
             
              if os.path.exists(full_file_name):#Remove from server
                    
                    file_name = str(uuid.uuid4()) + ".mp3"
                    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                    continue
              break
                



        file.save(full_file_name)
        #return '<h1>Success</h1>'
        return Predict_2(file_name)


if __name__ == '__main__':
    from werkzeug.utils import secure_filename
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=3011, debug=True)
