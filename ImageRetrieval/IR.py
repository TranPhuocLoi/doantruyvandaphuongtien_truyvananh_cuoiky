import math
import operator

import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans,MiniBatchKMeans
import pickle
import os
from scipy.spatial import distance

def computeSIFT():
    if not os.path.isfile('features.bin'):
        print('Computing SIFT....')
        sift = cv2.xfeatures2d_SIFT.create()
        allImage = glob.glob('oxford\\images\\*.jpg')

        features = np.zeros((0,128))
        featuresPerImage = []
        print(np.shape(features))

        i = 1
        for fileName in allImage:
            print('Computing sift for %d/%d image'%(i,len(allImage)))
            i+=1
            img = cv2.imread(fileName,0)
            '''
                resize an image help us to implement SIFT faster
            '''
            height = img.shape[0]
            width = img.shape[1]
            img_resize = cv2.resize(img, (int(height*0.5), int(width*0.5)))

            kp, des = sift.detectAndCompute(img_resize, None)
            eps = 1*np.e - 7
            des /= (des.sum(axis=1, keepdims=True) + eps)
            des = np.sqrt(des)

            features = np.vstack((features,des))
            featuresPerImage.append(len(kp))

        with open('features.bin','wb') as fb:
            np.save(fb,features)
        with open('featuresPerImage.bin','wb') as fb:
            np.save(fb,featuresPerImage)
    else:
        with open('features.bin','rb') as fb:
            features = np.load(fb)
        print('da load file',len(features))
        print(features)
    return

def buildVocabulary():
    numberOfWord = 1000
    if not os.path.isfile('features.bin'):
        computeSIFT()
    else:
        with open('features.bin','rb') as fb:
            features = np.load(fb)
        with open('featuresPerImage.bin','rb') as fb:
            featuresPerImage = np.load(fb)

        print('number of features: ', len(features))
        print('Running kmeans clustering...')

        kmeans = MiniBatchKMeans(n_clusters=numberOfWord, random_state=0, max_iter=10).fit(features)
        print(kmeans.cluster_centers_)
        print('Finished kmeans clustering')
        with open('dictWord.bin','wb') as fb:
            np.save(fb,kmeans.cluster_centers_)

def square_euclidean_dist(a, b):
    return distance.euclidean(a,b)

def kmeans_classify(centers, feat):
    min_c = -1
    min_d = -1
    for c,center in enumerate(centers):
        d = square_euclidean_dist(feat,center)
        if min_c ==-1  or d < min_d:
            min_c,min_d = c,d
    return min_c

def ComputingWords():
    allImage = glob.glob('oxford\\images\\*.jpg')
    if not os.path.isfile('dictWord.bin'):
        buildVocabulary()
    else:
        with open('dictWord.bin','rb') as fb:
            dict = np.load(fb)
        with open('featuresPerImage.bin','rb') as fb:
            featuresPerImage = np.load(fb)
        with open('features.bin','rb') as fb:
            features = np.load(fb)
        print('number of bag of visual word: ',len(dict))
        numImage = len(allImage)
        frequencyVector= {}
        for i in range(numImage):
            print('Quantizing %d/%d images\n'%(i,numImage))
            if i==0:
                bindex = 0
            else:
                bindex = sum(featuresPerImage[:i])+1
            eindex = bindex + featuresPerImage[i]
            featuresOfImage = features[bindex:eindex, :]
            histogram = {}
            for j in range(len(featuresOfImage)):
                d = kmeans_classify(dict,featuresOfImage[j])
                histogram[d] = histogram.get(d,0) + 1
            frequencyVector[allImage[i]] = [histogram.get(d,0) for d in range(len(dict))]
        with open('frequencyVector.bin','wb') as fb:
            pickle.dump(frequencyVector,fb)

def createInvertedFile():
    if not os.path.isfile('frequencyVector.bin'):
        ComputingWords()
    else:
        print('creating inverted File...')
        with open('frequencyVector.bin','rb') as fb:
            vectors = pickle.load(fb)
        with open('dictWord.bin','rb') as fb:
            dict = np.load(fb)
        invFile = {}
        for i in range(len(dict)):
            invFile[i] = []
            for nameImg in vectors:
                if vectors.get(nameImg)[i] != 0:
                    invFile[i].append(nameImg)
        with open('invFile.bin','wb') as fb:
            pickle.dump(invFile,fb)

def distance2Vec(v1,v2):
    return distance.cosine(v1,v2)

def vectorImage():
    allImage = glob.glob('oxford\\images\\*.jpg')
    if not os.path.isfile('frequencyVector.bin'):
        ComputingWords()
    with open('frequencyVector.bin', 'rb') as fb:
        frequencyVector = pickle.load(fb)

    if not os.path.isfile('invFile.bin'):
        createInvertedFile()
    with open('invFile.bin', 'rb') as fb:
        invFile = pickle.load(fb)

    if not os.path.isfile('dictWord.bin'):
        buildVocabulary()
    with open('dictWord.bin', 'rb') as fb:
        dict = np.load(fb)


    idf = [1 + np.log(len(allImage) / (len(invFile.get(i)))) for i in invFile]
    idf = np.array(idf)
    if not os.path.isfile('weightVector.bin'):

        weight = {}
        for imgName in frequencyVector:
            tf = frequencyVector.get(imgName)
            tf = np.array(tf)
            weightCal= tf*idf
            weight[imgName] = weightCal
        with open('weightVector.bin','wb') as fb:
            pickle.dump(weight,fb)
    with open('weightVector.bin','rb') as fb:
        weight = pickle.load(fb)
    return


def query(pathQueryImg,gtFile = None,region = None):
    if not os.path.isfile('invFile.bin'):
        createInvertedFile()
    with open('invFile.bin','rb') as fb:
        invFile = pickle.load(fb)

    if not os.path.isfile('weightVector.bin'):
        vectorImage()
    with open('weightVector.bin','rb') as fb:
        weight = pickle.load(fb)

    if not os.path.isfile('dictWord.bin'):
        buildVocabulary()
    with open('dictWord.bin', 'rb') as fb:
        dict = np.load(fb)
    print(len(dict))
    allImage = glob.glob('oxford\\images\\*.jpg')
    idf = [1 + np.log(len(allImage) / (len(invFile.get(i)))) for i in invFile]
    idf = np.array(idf)

    #Load image query
    queryImg = cv2.imread(pathQueryImg)
    if region != None:
        x1 = int(float(region[0]))
        y1 = int(float(region[1]))
        x2 = int(float(region[2]))
        y2 = int(float(region[3]))

        queryImg = queryImg[y1:y2,x1:x2]

    sift = cv2.xfeatures2d_SIFT.create()
    height = queryImg.shape[0]
    width = queryImg.shape[1]
    queryImg = cv2.resize(queryImg, (int(height * 0.5), int(width * 0.5)))
    kp, des = sift.detectAndCompute(queryImg, None)
    eps = 1 * np.e - 7
    des /= (des.sum(axis=1, keepdims=True) + eps)
    des = np.sqrt(des)
    histogramQuery = {}
    print(des.shape)
    for j in range(len(des)):
        d = kmeans_classify(dict, des[j])
        histogramQuery[d] = histogramQuery.get(d, 0) + 1
    #print(idf)
    frequencyQueryVector = [histogramQuery.get(d, 0) for d in range(len(dict))]
    frequencyQueryVector = frequencyQueryVector*idf

    #Find Image Result
    resultImg = {}
    min = 1
    for img in weight:
        dis = distance2Vec(weight.get(img),frequencyQueryVector)

        resultImg[img] = dis
    #print(resultImg)
    sortedResult = sorted(resultImg.items(), key = operator.itemgetter(1))

    # reRanking = {}
    # for i in range(10):
    #     img2 = cv2.imread(sortedResult[i][0])
    #     match = drawMatchKeypoint(queryImg,img2,i)
    #     reRanking[sortedResult[i][0]] = match
    #
    # reRanking = sorted(reRanking.items(), key=operator.itemgetter(1),reverse=True)

    resultImageFile = []
    if gtFile!=None:
        nFinallyResult = len(gtFile)
    else:
        nFinallyResult = 10
    for i in range(nFinallyResult):
        # path, file = os.path.split(reRanking[i][0])
        # imgResult = cv2.imread(reRanking[i][0])
        path, file = os.path.split(sortedResult[i][0])
        imgResult = cv2.imread(sortedResult[i][0])
        resultImageFile.append(file)
        cv2.imwrite('result\\imageResult%d.jpg'%(i),imgResult)
    '''for i in resultImg:
        path, file = os.path.split(i)
        imgResult = cv2.imread(i)
        pathToSave = 'result\\'+file
        cv2.imwrite(pathToSave, imgResult)
        #cv2.imwrite('result\\imageResult%d.jpg'%(i),imgResult)'''
    # with open('resultImageFile.txt','w') as fb:
    #     for s in resultImageFile:
    #         s = s.replace('.jpg','')
    #         fb.writelines(s)
    return resultImageFile

def drawMatchKeypoint(img1, img2, ith):
    sift = cv2.xfeatures2d_SIFT.create()
    height1 = img1.shape[0]
    width1 = img1.shape[1]
    img1 = cv2.resize(img1, (int(height1 * 0.5), int(width1 * 0.5)))
    kp1, des1 = sift.detectAndCompute(img1, None)
    height2 = img2.shape[0]
    width2 = img2.shape[1]
    img2 = cv2.resize(img2, (int(height2 * 0.5), int(width2 * 0.5)))
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    MIN_MATCH_COUNT = 0
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    if matchesMask == None:
        return 0
    else:
        return matchesMask.count(1)

def evalAP(pathFileGroundTruth, pathFileQuery):
    #load gtFile
    with open(pathFileGroundTruth,'r') as fb:
        gtFile = fb.read()
    gtFile = gtFile.splitlines()
    with open(pathFileQuery,'r') as fb:
        queryImg = fb.read()
    queryImg = queryImg.split(' ')
    pathQueryImg = queryImg[0].replace('oxc1_','')
    pathQueryImg = 'oxford\\images\\'+pathQueryImg+'.jpg'
    region = queryImg[1:]
    resultImage = query(pathQueryImg,gtFile,region)
    resultImage = [s.replace('.jpg','') for s in resultImage]
    numberOfRelevantRsImage = 0
    ap = 0
    for i,rs in enumerate(resultImage):
        if rs in gtFile:
            #print(rs)
            numberOfRelevantRsImage +=1
            precision = numberOfRelevantRsImage/(i+1)
            ap += precision
    if numberOfRelevantRsImage == 0:
        ap = 0
    else:
        ap/=numberOfRelevantRsImage
    print('so luong dung: %d/%d' %(numberOfRelevantRsImage,len(gtFile) ))
    print('ap = ',ap)
    return ap

def evalMAP():
    allGTFile = glob.glob('oxford\\groundtruth\\*_good.txt')
    allQueryFile = glob.glob('oxford\\groundtruth\\*_query.txt')
    map = 0
    apFile = open('apFile.txt','w')

    for i in range(len(allGTFile)):
        print('computing query ', i)
        pathFileGroundTruth = allGTFile[i]
        pathFileQuery = allQueryFile[i]
        ap = evalAP(pathFileGroundTruth, pathFileQuery)
        map+=ap
        apFile.writelines(str(ap))
    map/=len(allGTFile)
    apFile.writelines(str(map))
    apFile.close()
    return map

def computeSIFTFor1(img):
    if not os.path.isfile('outfile.bin'):
        print('Computing SIFT....')
        #create Sift detector
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d_SIFT.create()
        kp, des = sift.detectAndCompute(imgGray,None)
        kmeans = KMeans(n_clusters = 2, random_state = 0).fit(des)
        with open('outfile.bin', 'wb') as fp:
            pickle.dump(des,fp)
    else:
        file= open('outfile.bin','rb')
        feat = pickle.load(file)
        print(feat)
    return
if __name__ == '__main__':

    pathQueryImg = 'oxford\\images\\all_souls_000001.jpg'
    listResult = query(pathQueryImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
