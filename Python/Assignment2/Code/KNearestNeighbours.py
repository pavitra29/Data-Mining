import math
import sys
from collections import Counter
import arff
import numpy

infiletrain = sys.argv[1]
infiletest = sys.argv[2]
print("Name of train file:", infiletrain, "Name of the test file:", infiletest)

count = 0
i = 1

sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
for test in arff.load(infiletest):
    sepal_length.append(test.sepal_length)
    sepal_width.append(test.sepal_width)
    petal_length.append(test.petal_length)
    petal_width.append(test.petal_width)


def mean(cols):
    return numpy.mean(cols)

mu = []


def sigma(cols):
    return numpy.std(cols)

sig = []

mu_sl_tst = mean(sepal_length)
mu_sw_tst = mean(sepal_width)
mu_pl_tst = mean(petal_length)
mu_pw_tst = mean(petal_width)
sig__sl_tst = sigma(sepal_length)
sig_sw_tst = sigma(sepal_width)
sig_pl_tst = sigma(petal_length)
sig_pw_tst = sigma(petal_width)

sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
for train in arff.load(infiletrain):
    sepal_length.append(train.sepal_length)
    sepal_width.append(train.sepal_width)
    petal_length.append(train.petal_length)
    petal_width.append(train.petal_width)

mu_sl_tr = mean(sepal_length)
mu_sw_tr = mean(sepal_width)
mu_pl_tr = mean(petal_length)
mu_pw_tr = mean(petal_width)
sig__sl_tr = sigma(sepal_length)
sig_sw_tr = sigma(sepal_width)
sig_pl_tr = sigma(petal_length)
sig_pw_tr = sigma(petal_width)

for test in arff.load(infiletest):
    dict = {}
    for train in arff.load(infiletrain):
        dist = math.sqrt(math.pow((((test.sepal_length-mu_sl_tst)/sig__sl_tst)-((train.sepal_length-mu_sl_tr)/sig__sl_tr)), 2) +
                         math.pow((((test.sepal_width-mu_sw_tst)/sig_sw_tst)-((train.sepal_width-mu_sw_tr)/sig_sw_tr)), 2) +
                         math.pow((((test.petal_length-mu_pl_tst)/sig_pl_tst)-((train.petal_length-mu_pl_tr)/sig_pl_tr)), 2) +
                         math.pow((((test.petal_width-mu_pw_tst)/sig_pw_tst)-((train.petal_width-mu_pw_tr)/sig_pw_tr)), 2))
        dict[dist] = train.CLASS_LABEL
    sorted_dict = sorted(dict.items(), key=lambda x: x[0])
    val_list = []
    for key,val in sorted_dict:
        val_list.append(val)
    var = ""
    k_rows = {}
    dictlist = {}
    samefreqlist = {}
    for k in range(1, 11, 2):
        top = val_list[:k]
        k_rows = (sorted_dict[:k])
        most_common = Counter(top).most_common(k)
        for key,value in k_rows:
            for word,freq in most_common:
                if word == value:
                    dictlist[key,word] = freq
        sorted_freq = sorted(dictlist.items(), key=lambda x: x[1], reverse=True)
        first_row = sorted_freq[:1]
        for key,freq in first_row:
            max_freq = freq
            samefreqlist = {}
            for key, freq in sorted_freq:
                if freq == max_freq:
                    samefreqlist[key] = max_freq
        sorted_dist = sorted(samefreqlist.items(), key=lambda x: x[0])
        lowdisttuple = (sorted_dist[:1])
        for key,freq in lowdisttuple:
            for cls in key[1:]:
                var = var + cls + ","
    print test.sepal_length,",",test.sepal_width,",",test.petal_length,",",test.petal_width,",", var
    i += 1
