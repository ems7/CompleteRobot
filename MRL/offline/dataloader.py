'''
Created on 29 Apr 2019

@author: eli
'''
import os
import os.path as op
import cv2
import numpy as np
from datetime import datetime
import pytz
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, sampsPerObj=50, image_shape=(128,128), basePath="/home/eli/Documents/data/realShapes/images/cropped"):
        self.image_shape = image_shape
        self.basePath = basePath
        self.samps = sampsPerObj
        self.trainedAtrs = {}
        self.attrs = {
                # sizes
                "small":0, "medium":1, "big":2,
                 
                # colours
                "red":3,     "pink":4,  "orange":5,
                "yellow":6,  "green":7, "blue":8,
                "grey":9,    "black":10, "brown":11,
                "purple":12,
                 
                # objects
                "donut":13,  "duck":14, "brick":15,
                "cup":16,  "ball":17,
                "rectangle":18, "pepper":19, 
                
                }
        ''' 
        # position
        "topLeft":21, "topCentre":22, "topRight":23,
        "left":24,    "centre":25,    "right":26,
        "btmLeft":27, "btmCentre":28, "btmRight":29,
        '''
        # colour modifiers
        #"light":30, "dark":31
                
        
        self.names = dict((v,k) for k,v in self.attrs.iteritems())
        
        self.numAttrs = len(self.attrs.keys())
    
    def checkname(self, f, listOfAtrs, disallowed):
        name = op.basename(f)
        #print name
        
        for dis in disallowed:
            count = 0
            thresh = len(dis)
            for atr in name.split("_")[0:-1]:
                #print atr
                try:
                    int(atr)
                
                except:    
                    if atr not in listOfAtrs:
                        #print atr
                        return False
                    
                    
                    if atr in dis:
                        #print "count incremented"
                        count  += 1
            if count >= thresh:
                #print "thresh hit"
                return False

        return True
    
    
    def sharpen(self, image):
        #cv2.imshow("raw",image)
        blur = cv2.GaussianBlur(image, (3,3),0)
        #cv2.imshow("blur",blur)
        sharp = image - blur
        #cv2.imshow("sharp",sharp)
        sharpened = image + sharp 
        #cv2.imshow("sharpened",sharpened)
        #cv2.waitKey()
        return sharpened
        
    def MakeAverageImage(self, fnames, uniqueNames):
        for u in uniqueNames:
            print u
            FsWithName = [f for f in fnames if u in f]
            cv2.imshow("raw", cv2.resize(cv2.imread(FsWithName[0]), 
                               dsize=self.image_shape))
            samples = [cv2.resize(cv2.imread(f), 
                               dsize=self.image_shape) for f in FsWithName]
            imageAvg = np.mean(samples,
                              axis=0)
            #print np.shape(imageAvg)
            #print imageAvg
            
            cv2.imwrite("%s/avgs/%s.png"%(self.basePath,u), imageAvg)
            self.computeDistance(imageAvg, samples, u)
            
    def computeDistance(self, centre, samples, u):
        smallest = 99999999999;
        for sample in samples:
            dist = np.sum(np.abs(centre - sample))   
            if dist < smallest:
                smallest = dist
                toSave = sample
        
        cv2.imwrite("%s/avgs/Exemplar_%s.png"%(self.basePath,u), toSave)
        
    def loadExemplars(self, fnames):
        exemplarImages = []
        for f in fnames:
            exemplarExists = False
            for u in self.uniqueNames:            
                if u in f:
                    exemplar = cv2.imread("%s/exemplars/Exemplar_%s.png"%(self.basePath,u))
                    exemplarExists = True
            if not exemplarExists:
                exemplar = cv2.imread(f)
                
            exemplarImages.append(cv2.resize(exemplar, dsize=self.image_shape)/255.0)
            
        return exemplarImages
       
    def loadSubset(self, listOfAtrs, split, disallowed=[None]):
        fnames = self.getAllFiles("%s/%s"%(self.basePath,split))
        #checkedFiles = self.checkFileNames(fnames, listOfAtrs, disallowed)
        
        #a = raw_input("stop")
        checkedFiles = [f for f in fnames if self.checkname(f, listOfAtrs, disallowed)]
        sampledFiles = self.getSamples(checkedFiles)
        #self.checkColoursPerObject(sampledFiles)
        images = [cv2.resize(cv2.imread(f), dsize=self.image_shape)/255.0 for f in sampledFiles]
        
        attrs  = [self.convertName2OH(op.basename(f)) for f in sampledFiles]
        #print np.shape(images), np.shape(attrs) 
        return images, attrs, sampledFiles
    
              
    def getSamples(self, fnames):
        self.uniqueNames = self.getUniqueNew(fnames)
        #self.MakeAverageImage(fnames, self.uniqueNames)
        filesToReturn = []
        for u in self.uniqueNames:
            
            FsWithName = [f for f in fnames if u in f] # check if unique name is in file name and separate into a new list
            maxSamps = len(FsWithName)
            print u, maxSamps
            if self.samps > maxSamps:
                toAppend = FsWithName
            else:               
                toAppend = np.random.choice(FsWithName, self.samps, replace=False)
                
            for f in toAppend:
                filesToReturn.append(f)
        
        return filesToReturn   
         
    
    def checkColoursPerObject(self, fnames):
        
        colours =["red",  "yellow", 
                  "green",  "blue",
                  "black"]
        
        
        countsDict = {}
        
        for f in fnames:
            sz, col, obj = op.basename(f).split("_")[0:3]
            if col == "light":
                print f
                col, obj = op.basename(f).split("_")[2:4]
                                 
            if "%s_%s"%(obj, sz) not in countsDict.keys():
                countsDict["%s_%s"%(obj, sz)] = {"red":0,   "yellow":0, 
                                                "green":0,  "blue":0,
                                                "black":0}
                
                
                '''
                "red":0,   "orange":0, "yellow":0, 
                                                "green":0,  "blue":0,
                                                "grey":0,   "black":0,  "brown":0,
                                                "pink":0,   "purple":0
                '''
            countsDict["%s_%s"%(obj, sz)][col] +=1
                
            
        
        keys = [k for k in countsDict.keys()]   
        keys.sort()
        
        fig = plt.figure()
        ax = fig.gca()
        cols={"red":"r",   "orange":"m", "yellow":"y", 
              "green":"g",  "blue":"b",
              "grey":"w",   "black":"k",  "brown":"y",
              "pink":"m",   "purple":"b"}
        
        xpos = []
        for i, k in enumerate(keys):
            print k
            for j, col in enumerate(colours):
                print col, countsDict[k][col]
                ax.bar(i+(0.1*j), countsDict[k][col], 0.1, align="edge", color=col)
            xpos.append(i)
        
        plt.xticks(xpos, keys, rotation='vertical')        
        plt.show()          
       
    def getAllFiles(self, base_path, extension=""):
        """
        This method finds all files in a directory tree with the extension "extension"
        
        Args:
            @param base_path - string - the root directory to find files in
            @param extension - string/list of strings - the extension of files to be found
        """
        
        everything=[]
        for root, _, files in os.walk(base_path):
            for file_name in files:
                ext = file_name.split('.')[1]
                if ext in extension or extension == "":
                    # print root+"/"+file_name
                    everything.append(root+"/"+file_name)
            
        
        return everything
    
    def convertName2OH(self, fname):
        idxs = [self.attrs[atr] for atr in fname.split(".")[0].split("_") if atr in self.attrs.keys()]
        #print idxs
        #if len(idxs) < 3:
            #print fname
            #for atr in fname.split(".png")[0].split("_"):
                #print atr, self.attrs[atr]
                
        oh = np.zeros((self.numAttrs,))
        oh[idxs] = 1
        return oh
    
    def convertOH2Name(self, oh):
        atrs = [self.names[i] for i in range(self.numAttrs) if oh[i]==1]
        return "_".join(atrs)
         
    def loadData(self, split="train", showImages=False):
        fnames = self.getAllFiles("%s/%s"%(self.basePath,split))
        #self.createMetaTxt(fnames, split)
        images = [cv2.resize(cv2.imread(f), dsize=self.image_shape)/255.0 for f in fnames]
        attrs  = [self.convertName2OH(op.basename(f)) for f in fnames]
        
        if showImages:
            for a, f, i in zip(attrs, fnames, images):
                print a
                print op.basename(f)
                print self.convertOH2Name(a)
                cv2.imshow(op.basename(f), i)
                cv2.waitKey()
                cv2.destroyAllWindows()
            
        return images, attrs, fnames       
        
    def getUnique(self, fnames):
        unique = []
        pos = ["topLeft", "topCentre", "topRight",
               "left",    "centre",    "right",
               "btmLeft", "btmCentre", "btmRight"]
        
        for f in fnames:
            
            if op.basename(f).split("_")[0:3][-1] in pos:
                print op.basename(f)
            name = "_".join([x for x in op.basename(f).split("_") if x != "dark"][0:3])
            if name not in unique:
                unique.append(name)
        print "number of unique objects: ", len(unique) 
        for nm in unique:
            print nm 
        return unique  
    
    
    
    
    def getUniqueNew(self, fnames):
        unique = []
            
        colours =["red",   "orange", "yellow", 
                  "green",  "blue",
                  "grey",   "black",  "brown",
                  "pink",   "purple"]
        #ld = ["light", "dark"]
        
        
        for f in fnames:
            keep = True
            name = op.basename(f).split(".")[0]
            colour_count = 0
            for n in name.split():
                if n in colours:
                    colour_count += 1
            if colour_count >=1:
                keep = False
            
            if keep:
                name = "_".join([x for x in name.split("_")][0:3]) # if x not in ld
            
            if name not in unique:
                unique.append(name)
        print "number of unique objects: ", len(unique) 
        #for nm in unique:
            #print nm 
        return unique    
            
    def getMetaData(self, filename):
        ts = op.getctime(filename)
        return datetime.fromtimestamp(ts).replace(tzinfo=pytz.utc)
    
    def createMetaTxt(self, fnames, split):
        
        with open("%s/%s_meta.txt"%(self.basePath, split), "a") as meta:
            for f in fnames:
                session = f.split("/")[-2]
                ts = self.getMetaData(f)
                meta.write("%s/%s \t %s \n"%(session, op.basename(f), ts))
                
    
    def printStats(self, images, attrs, split):
        print "%s:"%split, np.shape(images), np.shape(attrs)

if __name__ == '__main__':
    dl = DataLoader() 
    atrs = ["big",  "medium", "small",
            
            "donut",  "brick",  "cup",
            "ball",   "pepper", "duck",
            
            "red",    "pink",   "orange",
            "yellow", "green",  "blue",
            "grey",   "black",  "brown",
            "purple", "chocolate",
            
            "topLeft", "topCentre", "topRight",
            "left",    "centre",    "right",
            "btmLeft", "btmCentre", "btmRight",
            
            "dark", "light"]
    
    
    disallowed = [[None]]#["medium", "brick"], ["chocolate"]]
    
    
    tr_im, tr_atrs, tr_nms = dl.loadSubset(atrs, "test", disallowed)
    for nm in tr_nms:
        
        for dis in disallowed:
            count = 0
            thresh = len(dis)
            for atr in op.basename(nm).split("_")[0:-1]:
                try:
                    int(atr)
                
                except:    
                    if atr in dis:
                        count +=1
            if count >= thresh:
                print nm
    
    dl.printStats(tr_im, tr_atrs, "test")
    
    ts_im, ts_atrs, ts_nms = dl.loadData("test")
    dl.printStats(ts_im, ts_atrs, "test")
    
    '''
    vl_im, vl_atrs, vl_nms = dl.loadData("val") 
    dl.printStats(vl_im, vl_atrs, "val")
    '''
     
     
     
