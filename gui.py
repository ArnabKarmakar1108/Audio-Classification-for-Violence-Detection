import sys
import librosa
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QMessageBox, QLabel, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtGui import QIcon, QPixmap

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


loaded = tf.keras.models.load_model(SAVED_MODEL_PATH)

def predict(file_path):
    # extract MFCC
    mfcc = preprocess(file_path)

    # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # get the predicted label
    predictions = loaded.predict(mfcc)

    return predictions

class AudioProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Violence Detection")
        self.setWindowIcon(QIcon("police.png"))
        self.setStyleSheet('''
            QMainWindow {
                background-color: #272727;
            }
            QPushButton {
                background-color: #fbca36;
                color: #272727;
                font-size: 18px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ffdf6b;
            }
            QLabel {
                color: #fff;
                font-size: 16px;
                margin-bottom: 10px;
            }
            QTextEdit {
                background-color: #fff;
                color: #272727;
                font-size: 14px;
                border-radius: 5px;
                padding: 5px;
            }
            QMessageBox {
                color: #000;
            }
        ''')

        self.init_ui()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setGeometry(400, 50, 400, 200)
        pixmap = QPixmap("police.png")
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 300, 100)
        self.label.setStyleSheet('''
            QLabel {
                color: #fbca36;
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
                qproperty-alignment: 'AlignTop';
            }
        ''')
        self.label.setText("Audio Violence Detection")

        self.button = QPushButton("Select Audio File", self)
        self.button.clicked.connect(self.open_file_dialog)
        self.button.setGeometry(50, 300, 200, 50)

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_dialog.setStyleSheet('''
            QFileDialog {
                background-color: #272727;
                color: #fff;
            }
            QFileDialog QLabel {
                color: #fbca36;
                font-size: 16px;
            }
            QFileDialog QPushButton {
                background-color: #fbca36;
                color: #272727;
                font-size: 14px;
                border-radius: 5px;
                padding: 5px;
            }
            QFileDialog QPushButton:hover {
                background-color: #ffdf6b;
            }
        ''')
        file_path, _ = file_dialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg)")

        if file_path:
            res = predict(file_path)
            result_text = (
                "Violence Prediction:\n\n\n"
                "Dog Barking: " + str(res[0, 0] * 100) + " %\n\n"
                "Domestic Violence: " + str(res[0, 1] * 100) + " %\n\n"
                "Explosion: " + str(res[0, 2] * 100) + " %\n\n"
                "Gun Shot: " + str(res[0, 3] * 100) + " %\n\n"
                "Lightning: " + str(res[0, 4] * 100) + " %\n\n"
                "Physical Violence: " + str(res[0, 5] * 100) + " %\n\n"
                "Sexual Violence: " + str(res[0, 6] * 100) + " %\n\n"
            )
            msg_box = QMessageBox()
            msg_box.setStyleSheet("color: #000;")
            msg_box.setText(result_text)
            msg_box.setWindowTitle("Prediction Result")
            msg_box.exec_()

    def resizeEvent(self, event):
        # Override the resize event to set a fixed window size
        self.setFixedSize(event.size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("police.png"))
    gui = AudioProcessingGUI()
    gui.setGeometry(400, 400, 1000, 600)  # Set the initial window size (width, height)
    gui.show()
    sys.exit(app.exec_())
