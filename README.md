# Audionet
EEA Winter Project 2024-25 <br />

Starting from fundamental signal processing concepts like **Fourier Transforms** and **sampling**, we progressively built up to feature engineering techniques and applied them in deep learning pipelines for real-world audio classification.

---

## Spoof Audio Detection (ASVspoof 2019)
Constructed a binary classifier to detect **spoofed** (synthesized/replayed) vs **bonafide** (real) speech.

- **Dataset:** ASVspoof 2019 LA Track (19 different attack types- including TTS and VC based attacks)
- **Preprocessing:** Extracted Mel-Spectrograms
- **Model:** CNN-LSTM based binary classifier
- **Result:**  
  - **F1 Score: 87%**
  - The dataset includes **19 different attack types**. The model was:
  - Trained on: Attacks A01–A06 (seen attacks), tested on: Attacks A07–A19 (unseen attacks)
  - A14, A15: Text-to-Speech + Voice Conversion (TTS+VC)
  - A06, A17, A18, A19: Voice Conversion (VC) based attacks<br />
  *(Model performance drops significantly on these, showing generalization challenge.)*
  - Achieved a good performance on identification of TTS based spoof attacks.
  - Metrics for each attack type:
    | Attack ID | EER     | F1 Score | Accuracy | Recall  | Precision |
    |-----------|---------|----------|----------|---------|-----------|
    | A07       | 0.0014  | 0.9648   | 0.9635   | 1.0000  | 0.9319    |
    | A08       | 0.1530  | 0.8664   | 0.8735   | 0.8201  | 0.9182    |
    | A09       | 0.0008  | 0.9648   | 0.9635   | 1.0000  | 0.9319    |
    | A10       | 0.0026  | 0.9629   | 0.9616   | 0.9963  | 0.9317    |
    | A11       | 0.0033  | 0.9627   | 0.9614   | 0.9959  | 0.9317    |
    | A12       | 0.0022  | 0.9648   | 0.9635   | 1.0000  | 0.9319    |
    | A13       | 0.0045  | 0.9648   | 0.9635   | 1.0000  | 0.9319    |
    | A14 (TTS+VC)| 0.0016 | 0.9647   | 0.9634   | 0.9998  | 0.9319    |
    | A15 (TTS+VC)| 0.0049 | 0.9634   | 0.9621   | 0.9974  | 0.9317    |
    | A16       | 0.0096  | 0.9620   | 0.9607   | 0.9945  | 0.9316    |
    | A17 (VC)  | 0.4796  | 0.1939   | 0.5211   | 0.1152  | 0.6119    |
    | A18 (VC)  | 0.1827  | 0.6786   | 0.7390   | 0.5511  | 0.8829    |
    | A19 (VC)  | 0.3067  | 0.5369   | 0.6604   | 0.3938  | 0.8435    |

## Environmental Sound Classifier  
Built a CNN-based model using the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset to classify urban environmental sounds like sirens, dog barks, engine idling, etc.

- **Features Used:** MFCCs
- **Model:** CNNs
- **Result:**  
  - **F1 Score: 96%**
  - Confusion Matrix:  
    ![Confusion Matrix](https://i.imgur.com/79sqkUn.png)

---

**Tech Stack:** Python, Librosa, NumPy, Pandas, Matplotlib, Keras

---

## Learnings

### Signal Processing Basics
- **Fourier Transforms** (DFT/FFT)
- **Spectrograms** and Short-Time Fourier Transform (STFT)
- **Aliasing and Sampling Theorem**

### Feature Extraction Techniques
Implemented and analyzed the following features:
- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Chroma Features**
- **Zero Crossing Rate**
- **Mel Spectrogram**

### Neural Network Architectures
- **CNN (Convolutional Neural Networks)**
- **RNN (Recurrent Neural Networks)**
- **LSTM (Long Short-Term Memory Networks)**
