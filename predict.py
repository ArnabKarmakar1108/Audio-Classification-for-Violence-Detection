import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "training_mix_reajusted_input.h5"
sample_length = 22050


def preprocess(file_path, qty=13, fft=2048, hop=512):
        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= sample_length:
            # ensure consistency of the length of the signal
            signal = signal[:sample_length]

            # extract MFCCs
            mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=qty, n_fft=fft, hop_length=hop)

            return mfcc.T


loaded=tf.keras.models.load_model(SAVED_MODEL_PATH)

def predict(file_path):
       
        # extract MFCC
        mfcc = preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = loaded.predict(mfcc)

        return predictions


res=predict("./dataset/a.wav")
res

print("Violence Prediction: \n")
print("Dog Barking: "+str(res[0,0]*100)+"%")
print("Demoestic Violence: "+str(res[0,1]*100)+"%")
print("Explosion: "+str(res[0,2]*100)+"%")
print("Gun Shot: "+str(res[0,3]*100)+"%")
print("Lightning: "+str(res[0,4]*100)+"%")
print("Physical Violence: "+str(res[0,5]*100)+"%")
print("Sexual Violence: "+str(res[0,6]*100)+"%")


# print("Demoestic Violence: "+str(res[0,0]*100)+"%")
# print("Physical Violence: "+str(res[0,1]*100)+"%")
# print("Sexual Violence: "+str(res[0,2]*100)+"%")
