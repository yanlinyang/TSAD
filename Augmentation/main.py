from OAT import *
from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from prettytable import PrettyTable
import numpy as np

def preprocessing(file,originalSamplingRate=48000,originalChannels=2, sampleRate=48000, channels=2, alpha = 0.97,frameLength=20,windowType ="hanning"):
    prep = Preprocessing()
    prep.readSoundFile(file,sr=originalSamplingRate,n_channels=originalChannels)
    prep.channelConversion(channels)
    prep.resampling(sampleRate)
    prep.pre_emphasis(alpha)
    prep.framing(frameLength)
    prep.addWindow(windowType)
    prep.vad()
    return prep.frames


def getFeature(frames, sampleRate):
    fe = FeatureExtraction(frames, sampleRate)
    fe.mel()
    fe.getMFCC()
    return fe.mfccs

def getOrthogonalTable(case):
    oat = OAT()
    lable = ["실험번호"]
    for key ,value in case.items():
        lable.append(key)

    sets = oat.genSets(case)
    print("test:", len(sets))
    showTable(sets, lable)

    # 数据为None的部分均匀填充
    count = np.zeros(len(lable))
    for i in range(len(sets)):
        for j in range(1,len(lable),1):
            if sets[i][lable[j]] == None:
                sets[i][lable[j]] = case[lable[j]][int(count[j])]
                count[j] += 1
                if count[j] == len(case[lable[j]]): count[j] = 0


    showTable(sets, lable)

    # 检测是否有重复
    for i in range(len(sets)):

        for j in range(i + 1, len(sets), 1):
            count = 0
            for k in range(1,len(lable),1):
                if sets[i][lable[k]] == sets[j][lable[k]]:
                    count += 1
            if count == len(lable): print(i, j)
    return sets,lable

def showTable(sets,lable):
    x = PrettyTable(lable)
    for i in range(len(sets)):
        row = [i]
        for j in range(1,len(lable),1):
            row.append(sets[i][lable[j]])
        x.add_row(row)
    print(x)
def showTableAndResunlt(sets,lable,ss):
    x = PrettyTable(lable)
    for i in range(len(sets)):
        row = [i]
        for j in range(1,len(lable),1):
            row.append(sets[i][lable[j]])
        x.add_row(row)
    x.add_row(np.append(["제곱합"],np.mean(ss,axis=1)))
    print(x)
    print("SST=",np.sum(np.mean(ss,axis=1)))

if __name__ == '__main__':
    originalSamplingRate = 48000
    originalChannels = 2
    case = OrderedDict([ # ('n_channel', [1, 2]),
                        ('sample rate', [42000, 48000, 54000]),
                        ('alpha', [0.95, 0.97, 0.99]),
                        ('frame_length', [20, 40, 60]),
                        ('window type', ["rectangle", "hanning", "hamming"])
                        ])
    sets, lable = getOrthogonalTable(case)
    n_channel = 1
    SS = np.zeros((4,20))
    y = np.zeros((len(sets),20))
    file = "raw/1.raw"
    # for s in range(1):
    #     file = "raw/"+str(s)+".raw"

    for i in range(len(sets)):
        n_channel = 1  # n_channel = sets[i]['n_channel']
        sampleRate = sets[i]['sample rate']
        alpha = sets[i]['alpha']
        frame_length = sets[i]['frame_length']
        windowType = sets[i]['window type']
        print(n_channel,sampleRate,alpha,frame_length,windowType)
        frames = preprocessing(file, originalSamplingRate=originalSamplingRate, originalChannels=originalChannels, sampleRate=48000, channels=n_channel, alpha=0.97,
                      frameLength=frame_length, windowType=windowType)
        y[i] = getFeature(frames,sampleRate)
    print(y.shape)
    x = 1.0 / 6 * np.square(y + y[1] + y[2] - y[3] - y[4] - y[5])
    print(x)
    SS[0] = 1.0 / 6 * np.square(
        y[0] + y[1] + y[2] - y[3] - y[4] - y[5]) + 1.0 / 6 * np.square(
        y[0] + y[1] + y[2] - y[6] - y[7] - y[8]) + 1.0 / 6 * np.square(
        y[3] + y[4] + y[5] - y[6] - y[7] - y[8])
    SS[1] = 1.0 / 6 * np.square(y[0] + y[3] + y[6] - y[1] - y[4] - y[7]) + 1.0 / 6 * np.square(
        y[0] + y[3] + y[6] - y[2] - y[5] - y[8]) + 1.0 / 6 * np.square(
        y[1] + y[4] + y[7] - y[2] - y[5] - y[8])
    SS[2] = 1.0 / 6 * np.square(
        y[0] + y[5] + y[7] - y[2] - y[4] - y[6]) + 1.0 / 6 * np.square(
        y[0] + y[5] + y[7] - y[1] - y[3] - y[8]) + 1.0 / 6 * np.square(
        y[2] + y[4] + y[6] - y[1] - y[3] - y[8])
    SS[3] = 1.0 / 6 * np.square(
        y[0] + y[4] + y[8] - y[1] - y[5] - y[6]) + 1.0 / 6 * np.square(
        y[0] + y[4] + y[8] - y[2] - y[3] - y[7]) + 1.0 / 6 * np.square(
        y[1] + y[5] + y[6] - y[2] - y[3] - y[7])
    print(SS.shape)
    showTableAndResunlt(sets, lable, SS)





# 003_20181126163713
# 001_20181130081356
#
# 002_20190111175535
# 004_20190112145257
# 002_20190131152132
#
# 005_20190214163746
# 001_20190224142149