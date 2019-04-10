import numpy as np
import soundfile as sf
import scipy.signal as Signal

class Preprocessing:
    def __init__(self):
        self.N_CHANNEL = [1, 2]
        self.SAMPLE_RATE = [32000, 44100, 48000]
        self.ALPHA = [0.95, 0.97, 0.99]
        self.FRAME_LEGTH = [20, 40, 60, 80]
        self.FRAME_STEP = [10, 20, 30, 40]
        self.WINDOW_TYPE = ["rectangle", "hanning", "hamming"]
        self.signal = None
        self.sampleRate = None
        self.channels = None
        self.alpha = None
        self.frameLength = None
        self.frameStep = None
        self.windowType = None
        self.frames = None


    def readSoundFile(self,file, sr=48000, n_channels=2,subtype='PCM_16'):
        self.signal = sf.read(file, channels=n_channels, samplerate=sr, subtype=subtype)[0]
        #print(self.signal[:, 0], self.signal[:, 1])
        # return self.data[0]
        # data, sr = librosa.load("1.wav", sr=48000, mono=False)
        # print(data[0], data[1])

    def channelConversion(self,channels=1):
        self.channels = channels
        #print(len(self.signal))
        if channels == 1:
            mono_signal = np.zeros(len(self.signal))
            for t in range(len(self.signal)):
                Symbol = -1 if self.signal[t, 0] < 0 and self.signal[t, 1] < 0 else 1
                mono_signal[t] = self.signal[t, 0] + self.signal[t, 1]-(self.signal[t, 0]*self.signal[t, 1]/(Symbol * (np.power(2,15) - 1)))
            self.signal = mono_signal
            #print(self.signal)
    def resampling(self,sr=48000):
        self.sampleRate = sr
        self.signal = Signal.resample(self.signal,sr)

    def pre_emphasis(self,alpha=0.97):
        self.alpha = alpha
        self.signal = self.signal[1:] - alpha * self.signal[:-1]
        self.signal = np.append(self.signal[0], self.signal)
        #print(self.signal)

    def framing(self, frameLength=20):
        self.frameLength = frameLength
        self.frameStep = int(frameLength/2)
        frameLength_sample = int(self.frameLength / 1000 * self.sampleRate)
        frameStep_sample = int(self.frameStep / 1000 * self.sampleRate)
        signal_length = len(self.signal)
        if signal_length <= frameLength: # 若信号长度小于一个帧的长度，则帧数定义为1
            n_frame = 1
        else:  # 否则，计算帧的总长度
            n_frame = int(np.ceil((1.0 * signal_length - frameLength_sample + frameStep_sample) / frameStep_sample))
        pad_length = int((n_frame - 1) * frameStep_sample + frameLength_sample)  # 所有帧加起来总的铺平后的长度
        zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
        pad_signal = np.concatenate((self.signal, zeros))  # 填补后的信号记为pad_signal
        indices = np.tile(np.arange(0, frameLength_sample), (n_frame, 1)) + np.tile(np.arange(0, n_frame * frameStep_sample, frameStep_sample),
                                                               (frameLength_sample, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
        #print(n_frame)
        self.frames = pad_signal[indices]  # 得到帧信号

    def addWindow(self,windowType="rectangle"):
        self.windowType = windowType
        if windowType == "rectangle":
            window_alpha = 0
        elif windowType =="hanning":
            window_alpha = 0.5
        elif windowType =="hamming":
            window_alpha = 0.46
        frameLength_sample = int(self.frameLength / 1000 * self.sampleRate)
        window = (1-window_alpha) - window_alpha * np.cos(2*np.pi*np.array(range(frameLength_sample))/(frameLength_sample-1))
        self.frames = self.frames * window

    def vad(self):
        E, zcr = self.getEnergyAndZCR(self.frames)
        startFrame = np.argmax(E)-200
        endFrame = startFrame + 499
        if startFrame<0:
            startFrame = 0
            endFrame = 499
        elif startFrame> len(self.frames)-300:
            startFrame = len(self.frames)-300
        self.frames = self.frames[startFrame:endFrame]

    def getEnergyAndZCR(self,frames):
        i = 0
        E = np.array([0.0])
        zcr = np.array([0.0])
        for f in frames:  # For each frame, calculate energy, zero-crossing rate
            if (i != 0):
                E = np.concatenate((E, [0.0]))
                zcr = np.concatenate((zcr, [0.0]))
            # Calculated energy
            for j in range(len(f)):
                E[i] += f[j] ** 2
            # zero crossing rate
            fa = f[0:len(f) - 1]
            fb = f[1:len(f)]
            zcr[i] = (sum(abs(np.sign(fa) - np.sign(fb)))) / 2 / len(fa)
            i += 1
        return E, zcr



