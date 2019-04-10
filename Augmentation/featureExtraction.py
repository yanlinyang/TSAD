import numpy as np
from scipy.fftpack import dct
from sklearn.preprocessing import normalize
class FeatureExtraction:
    def __init__(self,frames,sampleRate):
        self.NFFT = 512
        self.frames = frames
        self.sampleRate = sampleRate
        self.mfccs = None

    def STFT(self):
          # 512
        #print(np.fft.rfft(self.frames, self.NFFT).shape)
        fft = np.absolute(np.fft.rfft(self.frames, self.NFFT))

        len_frame = len(self.frames[0])
        pow_frames = ((1.0 / len_frame) * ((fft) ** 2))  # 能量谱转为功率谱 Power Spectrum
        #print("fft.shape=", fft.shape)

        return pow_frames

    def mel(self):
        pow_frames = self.STFT()
        nfilt = 40  # 窗的数目
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sampleRate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((self.NFFT + 1) * hz_points / self.sampleRate)

        #print("bin.shape=", bin.shape)
        #print(bin)

        fbank = np.zeros((nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        #print(fbank.shape)
        #print(np.finfo(float).eps)
        M = np.dot(pow_frames, fbank.T)
        M = np.where(M == 0, np.finfo(float).eps, M)  # 数值稳定性 Numerical Stability  where中的三个元素 a条件，b,c  如果a为真，选b,否则选c
        M_log = np.log(M)  # dB;348*26

        num_ceps = 20
        M_log_dct = dct(M_log, type=2, axis=1, norm='ortho')
        self.mfccs = M_log_dct[:, 1: (num_ceps + 1)]
        # (nframes, ncoeff) = mfcc.shape

        # 可以将正弦升降1应用于MFCC以降低已被声称在噪声信号中改善语音识别的较高MFCC
        # n = np.arange(ncoeff)
        # cep_lifter = 22
        # lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        # mfcc *= lift  # *

        # 平均归一化MFCC
        # filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        #self.mfccs -= (np.mean(self.mfccs, axis=0) + 1e-8)
        self.mfccs = normalize(self.mfccs, axis=0, norm='l1')

    def getMFCC(self):
        self.mfccs = np.mean(self.mfccs,axis = 0)
