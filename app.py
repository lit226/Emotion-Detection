from keras.models import load_model,Sequential
from werkzeug.utils import secure_filename
from flask import  Flask, redirect, url_for, request, render_template
import numpy as np
import pandas as pd
from keras.layers import *
import librosa
from sklearn.preprocessing import StandardScaler
import os


app = Flask(__name__,template_folder = 'templates')
app.config['UPLOAD_FOLDER'] = 'upload/audio'



def noise(data):
    wn = np.random.randn(len(data))
    data_wn = data + 0.005*wn
    return data_wn

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
def shifting(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def stretch(data,rate= 1):
    return librosa.effects.time_stretch(data, rate)

def extract_features(data,sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) 
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) 
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) 
    sc = np.mean(librosa.feature.spectral_centroid(y=data,sr = sample_rate).T , axis=0)
    result = np.hstack((result,sc))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) 
    shift_data = shifting(data)
    res3 = extract_features(shift_data,sample_rate)
    result = np.vstack((result,res3))
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res4 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res4)) # stacking vertically
    return result

model = load_model("emotion_model.h5")


def predict(path,model):
    feature = get_features(path)
    X = []
    for ele in feature:
        X.append(ele)
    df = pd.DataFrame(X)
    ss = StandardScaler()
    x_train = ss.fit_transform(df)
    x_train = np.expand_dims(x_train , axis=2)
    preds = model.predict(x_train)
    mann = list(np.max(preds , axis = 1))
    ans = ""
    for i in range(len(preds)):
        mx = np.where(preds == mann[i])
        mx = mx[0].astype(int)[0]
        if(mx == 0):
            ans = 'Angry'
        elif(mx==1):
            ans = 'Calm'
        elif(mx ==2):
            ans = 'Disgust'
        elif(mx==3):
            ans = 'Fear'
        elif(mx==4):
            ans = 'Happy'
        elif(mx ==5):
            ans = 'Neutral'
        elif(mx==6):
            ans = 'Pleasant'
        elif(mx==7):
            ans = 'Sad'
        else:
            ans = 'Surprise'
    return ans

@app.route("/",methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route("/",methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        'save the file in upload/audio'
        if(f.filename == ""):
            print("no file selected")
            return redirect(request.url)
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        f.save(filepath)
        
        preds = predict(filepath ,model)
        result = preds
        os.remove(filepath)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
    