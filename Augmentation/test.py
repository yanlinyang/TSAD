# encoding: utf-8
from OAT import *
import json
from prettytable import PrettyTable
import numpy as np


def showTable(sets,lable):
    x = PrettyTable(lable)
    for i in range(len(sets)):
        row = []
        for j in range(len(lable)):
            row.append(sets[i][lable[j]])
        x.add_row(row)
    print(x)

if __name__ == "__main__":
    oat = OAT()
    case1 = OrderedDict([('K1', [0, 1]),
                         ('K2', [0, 1]),
                         ('K3', [0, 1])])

    case2 = OrderedDict([('A', ['A1', 'A2', 'A3']),
                         ('B', ['B1', 'B2', 'B3', 'B4']),
                         ('C', ['C1', 'C2', 'C3']),
                         ('D', ['D1', 'D2'])])

    case3 = OrderedDict([(u'对比度', [u'正常', u'极低', u'低', u'高', u'极高']),
                         (u'色彩效果', [u'无', u'黑白', u'棕褐色', u'负片', u'水绿色']),
                         (u'感光度', [u'自动', 100, 200, 400, 800]),
                         (u'白平衡', [u'自动', u'白炽光', u'日光', u'荧光', u'阴光']),
                         (u'照片大小', ['5M', '3M', '2M', '1M', 'VGA']),
                         (u'闪光模式', [u'开', u'关'])])

    case4 = OrderedDict([('A', ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']),
                         ('B', ['B1']),
                         ('C', ['C1,','C2'])])

    case5 = OrderedDict([('n_channel', [1, 2]),
                         ('sample rate', [32000, 44100, 48000]),
                         ('alpha', [0.95, 0.97, 0.99]),
                         ('frame_length', [20, 40, 60, 80]),
                         ('window type', ['rectangle', 'hanning', 'hamming'])
    ])
    case6 = OrderedDict([('A', [1, 2]),
                         ('B', [1, 2, 3]),
                         ('C', [1, 2, 3]),
                         ('D', [1, 2, 3, 4]),
                         ('E', [1, 2, 3])
                         ])
    sets = oat.genSets(case5)
    lable = ['n_channel', 'sample rate', 'alpha', 'frame_length', 'window type']
    print("test:",len(sets))

    showTable(sets,lable)

    # 数据为None的部分均匀填充
    count = np.zeros(len(lable))
    for i in range(len(sets)):
        for j in range(len(lable)):
            if sets[i][lable[j]]==None:
                sets[i][lable[j]] = case5[lable[j]][int(count[j])]
                count[j] += 1
                if count[j] == len(case5[lable[j]]): count[j] = 0

    showTable(sets, lable)

    # 检测是否有重复
    for i in range(len(sets)):

        for j in range(i+1, len(sets), 1):
            count = 0
            for k in range(len(lable)):
                if sets[i][lable[k]] == sets[j][lable[k]]:
                    count += 1
            if count == len(lable): print(i,j)


    #print(json.dumps(oat.genSets(case6)))

    # print(json.dumps(oat.genSets(case2)))
    # print(json.dumps(oat.genSets(case3)))
    # print(json.dumps(oat.genSets(case4)))
    # print(json.dumps(oat.genSets(case4, 1, 0)))
    # print(json.dumps(oat.genSets(case4, 1, 1)))
    # print(json.dumps(oat.genSets(case4, 1, 2)))
    # print(json.dumps(oat.genSets(case4, 1, 3)))