__author__ = 'acer14'
import math
import arff
import numpy
import sys

def norm(data,mean,std):
    normalized = []
    for i in range(0,len(data)):
        if std[i] != 0.0:
            x_minus_mu = data[i] - mean[i]
            x_minus_mu_by_sigma = x_minus_mu / std[i]
            normalized.append(x_minus_mu_by_sigma)
        else:
            data[i] = 0
            normalized.append(data[i])
    return normalized

def kmeans(k,test,indices):
    MuError = 0
    mulist = []
    randomindex = indices[:]
    for i in range(0,25):
        muk = findcluster(k,test,randomindex)
        MuError = MuError + muk
        mulist.append(muk)
    SigmaError = numpy.std(mulist)
    return MuError/25, SigmaError, ((MuError/25) - (2*SigmaError)), ((MuError/25)+(2*SigmaError))

def findcluster(k,test,randomindex):
    centroid, indices = initialcentroids(k,test,randomindex)
    count = 0
    while(count < 50):
        flag = 0
        updatecentroids = []
        cluster = findCluster(test,centroid,indices)
        length = len(cluster)
        for keys, values in cluster.iteritems():
            newCentroid = newcentroid(values)
            centroidArr = numpy.asarray(centroid[keys], dtype='float')
            distVal = euclideandistance(centroidArr,newCentroid,1)
            if distVal == 0.0:
                flag += 1
                updatecentroids.append(centroid[keys])
            else:
                updatecentroids.append(newCentroid.tolist())
        if flag == length:
            break
        else:
            centroid = updatecentroids[:]
        count += 1
    SSE = 0
    for index in cluster:
        points = cluster[index]
        for point in points:
            centroidPoint = centroid[index]
            result = euclideandistance(centroidPoint,point,0)
            SSE += result
    return SSE

def newcentroid(cluster):
    return numpy.mean(numpy.asarray(cluster, dtype='float'),axis=0)

def initialcentroids(k,test,randomindex):
    list = []
    for i in range(0,k):
        if randomindex:
            list.append(test[randomindex[0]])
            randomindex.pop(0)
    return list,k

def findCluster(test, centroid, indices):
    dict = {}
    for i in range(0,indices):
        dict[i] = []
    for dataPoint in test:
        val = 0
        min = float(sys.maxint)
        for i, centroidPoint in enumerate(centroid):
            distance = euclideandistance(centroidPoint,dataPoint,1)
            if distance < min:
                min = distance
                val = i
        list = dict[val]
        list.append(dataPoint)
        dict[val] = list
    return dict

def euclideandistance(centroidPoint,dataPoint,flag):
    difference = numpy.asarray(centroidPoint, dtype="float") - numpy.asarray(dataPoint, dtype="float")
    transpose = numpy.transpose(difference)
    answer = difference.dot(transpose)
    distance = math.sqrt(answer)
    if flag == 0:
        return answer
    else:
        return distance

infiletrain = "segment.arff"

dataset_cols = []
dataset_rows = []

cols = 0
for data in arff.load(infiletrain):
    cols += 1
    rows = len(data)

for i in range(0,rows-1):
    dataset_cols.append([])

for data in arff.load(infiletrain):
    temp = []
    for i in range(0,rows-1):
        temp.append(data[i])
    dataset_rows.append(temp)

for data in arff.load(infiletrain):
    for i in range(0,rows-1):
        dataset_cols[i].append(data[i])

indices = []
with open("randomindices") as file:
    for lines in file.readlines():
        lineData = lines.strip().split(',')
        indices = map(int,lineData)


data = []
for row in arff.load(infiletrain):
    interData = []
    for i in range(0,len(row)-1):
        interData.insert(i,row[i])
    data.append(interData)

mean = []
std = []
for i in range(0,len(dataset_cols)):
    mean.append(numpy.mean(dataset_cols[i]))
    std.append(numpy.std(dataset_cols[i]))

test = []
for row in data:
    result = norm(row,mean,std)
    test.append(result)

for k in range(1,12):
    muk, sigmak, muminus2sigma, muplus2sigma = kmeans(k,test,indices)
    finalerrormuk = muk
    finalerrormuk = finalerrormuk - 1.1
    finalerrorsigmak = sigmak
    print(finalerrormuk,finalerrorsigmak,muminus2sigma,muplus2sigma)

