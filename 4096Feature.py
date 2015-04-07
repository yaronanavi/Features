# TODO - write in database files that you read 
# Add catID, LInk, int descriptor 
# Change the floats to ints 
# When done add 100 vector with results of id

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import pickle
import shutil
import csv
import urllib
import sys
import cv2
from scipy.cluster import vq





# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def InitializeNet():
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    caffe.set_phase_test()
    caffe.set_mode_cpu()
    net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                           caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
    net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    return net

def RankImages(features, files, binaryFlag, copyFlag):
    rankedImages = []
    dictionaryRanks = {}
    for iList in range(len(features)):
        featuresDist = []
        featureTest  = features[iList]
        sumMin = 1000000000
        index = 0
        featureTestArr = np.asarray(featureTest)
        for iImage in range(len(features)):
            featureToCompare = features[iImage]
            featureToCompareArr = np.asarray(featureToCompare)
            dst = distance.euclidean(featureToCompareArr,featureTestArr)
            featuresDist.append(dst)
        featuresDistArr = np.asarray(featuresDist)
        sort_index = np.argsort(featuresDistArr)
        directory = 'Results/EucladianMetric/' + files[iList]
        print directory

        rankedImages = []
        for i100Images in range(100):
            rankedImages.append(files[sort_index[i100Images]])
            print files[sort_index[i100Images]]
            print "done"

        dictionaryRanks[files[iList]] = rankedImages
        if (copyFlag == 1):
            if not os.path.exists(directory):
                os.makedirs(directory)
        for iRank in range(20):
            print len(files)
            fileName = str(iRank)
            src = 'images/Images150000/' +  files[sort_index[iRank]]
            print src
            dst = directory + '/' + fileName + '.jpg'
            if (copyFlag == 1):
                shutil.copyfile(src, dst)

    print "ssss"
    print sort_index[0]
    print "ssss"
    print sort_index[1]
    #print "index is"
    #print index
    print "Image Test"
    print files[iList]
    print "Image simialr"
    print files[sort_index[1]]
    print "Image simialr"
    print files[sort_index[0]]

def RankDb(iterationCounter):
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    dataBase = client.descriptors
    iterations = 10;
    descriptorsDb1 = dataBase.descriptors.find({}).sort("_id").batch_size(5)
    dictTest = {}
    for recordToTest in descriptorsDb1:
        if ((iterationCounter % iterations) == 0):
            print iterationCounter
            featureToCompare = recordToTest['Feature']
            productID = recordToTest['productID']
            featureToCompareArr = np.asarray(featureToCompare)
            newFeature = []
            descriptorsDb1 = dataBase.descriptors.find({}).sort("_id").batch_size(5)
            scores = []
            fileNamesString = []
            for record in descriptorsDb1:
                feature = record['Feature']
                file2 = record['productID']
                link = record['Link']


                feature = np.asarray(feature)
                dst = distance.euclidean(featureToCompareArr,feature)        

                tupleFeature = (file2, dst, link)
                dictTest[productID] = tupleFeature
                scores.append(dst)
                fileNamesString.append(tupleFeature)

            featuresDistArr = np.asarray(scores)
            sort_index = np.argsort(featuresDistArr)
            filesOrdered = []
            for i in range(len(fileNamesString)):
                filesOrdered.append(fileNamesString[sort_index[i]])

            best100 = []
            for iBest100 in range(100):
                best100.append(filesOrdered[iBest100])
            
            dataBase.descriptors.update({'productID':productID}, {"$set":{"SimilarItems" : best100}}, upsert=True)
        iterationCounter = iterationCounter + 1

def GenerateFeatures(imagesPath,picklePath, net, generateFlag):
        from pymongo import MongoClient
        client = MongoClient('localhost:27017')
        dataBase = client.descriptors
        
        featuresListAll = []
        fileListAll = []

        featuresList = []
        fileList = []
        filesCounter = 0
        featuresListAll = []
        fileListAll = []

        f = open(imagesPath, 'rb')
        reader = csv.reader(f)

        for idx, row in enumerate(reader):
                        try:
                            print idx
                            fileLink = row[2]
                            catalogId = row[1]
                            identifier = row[0]
                            if (idx > 175000  and idx < 203000):
                                newFeature = []
                                scores = net.predict([caffe.io.load_image(fileLink)])
                                feat = net.blobs['fc6'].data[4]
                                for iFeature in range(4096):
                                    featureElement = feat[iFeature]
                                    featureElementIn = featureElement[0]
                                    newFeature.append(featureElementIn[0])

                                newFeature = np.int64(newFeature)
                                newFeature = [x.item() for x in newFeature]

                                dataBase.descriptors.insert({"productID" : identifier, "CatalogId" : catalogId, "Link" : fileLink, "Feature" : newFeature})
                                
                                
                                print catalogId
                                print fileLink
                                filesCounter = filesCounter + 1                           
                        except Exception, err:
                            print Exception, err

        return featuresListAll, fileListAll


def GenerateBinaryFeatures(features):
    featuresListBinary = []
    medianFeature = []
    for iBin in range(4096):
        print iBin
        iFeaturesList = []
        for iFeature in range(len(features)):
            featureI = features[iFeature]
            iFeaturesList.append(featureI[iBin])     
            iFeaturesList.append(featureI[iBin])
        medianFeature.append(np.median(np.asarray(iFeaturesList)))

    for iFeature in range(len(features)):
        binaryFeature = []
        featureI = features[iFeature]
        for iBin in range(4096):
            if featureI[iBin] > medianFeature[iBin]:
                binaryFeature.append(1)
            else:
                binaryFeature.append(0)
        featuresListBinary.append(binaryFeature)

    return featuresListBinary

def DownLoadImages():
    import shutil
    import requests
    from pymongo import MongoClient
    iterator = 0
    client = MongoClient('localhost:27017')
    dataBase = client.descriptors
    descriptorsDb1 = dataBase.descriptors.find({}).sort("_id").batch_size(1000)
    for recordToTest in descriptorsDb1:
        if ((iterator % 40) == 0):
            featureToCompare = recordToTest['Link']

            response = requests.get(featureToCompare, stream=True)
            iteratorStr = str(iterator)
            with open('ImagesToCluster/' + iteratorStr, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
        iterator = iterator + 1

   
def GenerateClusters(net):
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    dataBase = client.descriptors

    features = []
    folderPath = 'ImagesToCluster/'
    listing = os.listdir(folderPath)
    iterator = 0
    errorIterator = 0
    for infile in listing:
        try:
            print iterator
            newFeature = []
            scores = net.predict([caffe.io.load_image(folderPath + '/' + infile)])
            feat = net.blobs['fc6'].data[4]
            for iFeature in range(4096):
                featureElement = feat[iFeature]
                featureElementIn = featureElement[0]
                newFeature.append(featureElementIn[0])

            newFeature = np.int64(newFeature)
            newFeature = [x.item() for x in newFeature]
            features.append(newFeature)
            iterator = iterator + 1
        except:
            errorIterator = errorIterator + 1;
            print 'error'
            print errorIterator

    k = 100         # Number of clusters
    features = np.array(features)
    center,dist = vq.kmeans(features,k)
    code,distance = vq.vq(features,center)

    for cluster in center:
        cluster = cluster.tolist()
        myarray = np.asarray(cluster)
        dataBase.Clusters.insert({"Cluster": cluster})

def IndexImages():
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    dataBase = client.descriptors
    descriptorsClusters = dataBase.Clusters.find({}).sort("_id").batch_size(1000)
    descriptorsDresses = dataBase.Clusters.find({}).sort("_id").batch_size(1000)


    Clusters = []
    for record in descriptorsClusters:
        featureToCompare = record['Cluster']
        Clusters.append(featureToCompare)

    from sklearn.neighbors import NearestNeighbors
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # TODO make sure that the distances Ok
    # TODO make sure that
    for record in descriptorsDresses:

    # Go through each image and index it with nearest neighboor 
    # Then when i sort images i should sort only those with the same index

if __name__ == "__main__":
    #DownLoadImages()
    #net = InitializeNet()
    #GenerateClusters(net)
    IndexImages()
    #iterationCounter = int(sys.argv[-1])
    #print iterationCounter
    #imagesPath = '/home/donde/Work/caffe-master/examples/images/ImagesLinks/items.csv'
    #metric = 'Euclidian'
    #net = InitializeNet()
    #generateFlag = True
    #picklePath = '/home/donde/Work/caffe-master/examples/Results/PcklFiles/'
    #features, files = GenerateFeatures(imagesPath, picklePath, net, generateFlag)
    #RankDb(iterationCounter)
    #binaryFlag = 0
    #copyFlag = 1
    #RankImages(features, files, binaryFlag, copyFlag)







