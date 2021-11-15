# -*- coding: utf-8 -*-
# @Time    : 2021年9月14日
# @Author  : Bole Ma
# @File    : atomic_reader.py   处理ATOMIC数据集脚本

import pandas as pd
import json
import random

event_pair = []
effect, cause = [], []
df = pd.read_csv("/data/CausalityReasoning.pytorch/dataset/atomic/v4_atomic_all_agg.csv")
df.iloc[:,1:9] = df.iloc[:,1:9].apply(lambda col: col.apply(json.loads))

for indexs in df.index:
    event = df.loc[indexs].values[0]
    if event.find('_') != -1:
        continue
    cause.append(event)

    oeffect = df.loc[indexs].values[1]
    for e in oeffect:
        if e != "none":
            if len(e.split(' ')) >=3:
                samp = {'event_1':event, 'event_2':e, 'label':1}
                effect.append(e)
                event_pair.append(samp)

    oreact = df.loc[indexs].values[2]
    for e in oreact:
        if e != "none":
            if len(e.split(' ')) >=3:
                samp = {'event_1':event, 'event_2':e, 'label':1}
                effect.append(e)
                event_pair.append(samp)

    xeffect = df.loc[indexs].values[5]
    for e in xeffect:
        if e != "none":
            if len(e.split(' ')) >=3:
                samp = {'event_1':event, 'event_2':e, 'label':1}
                effect.append(e)
                event_pair.append(samp)

    xreact = df.loc[indexs].values[8]
    for e in xreact:
        if e != "none":
            if len(e.split(' ')) >=3:
                samp = {'event_1':event, 'event_2':e, 'label':1}
                effect.append(e)
                event_pair.append(samp)

random.shuffle(effect)
effect = effect[:len(cause)]
assert len(cause) == len(effect)
for i in range(len(cause)):
    samp = {'event_1':cause[i], 'event_2':effect[i], 'label':0}
    event_pair.append(samp)

print(len(event_pair))
with open('/data/CausalityReasoning.pytorch/dataset/atomic/ATOMIC_event.json', 'w') as f:
    json.dump(event_pair, f)