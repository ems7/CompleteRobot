'''
Created on 13 Nov 2019

@author: eli
'''
from net import NET
from base_module.BaseModule import BaseModule
import cv2
import numpy as np
import os.path as op
import os



class MAE(BaseModule):
    
    ENABLE_RPC = False
    
    PORTS = [
            ('input', 'imageCropped', 'image 64x64x3'),
            ('input', 'command', 'buffered'),
            ('input', 'entities', 'buffered'),
            ('output', 'MAEOutput', 'buffered'),
            ('output', 'confidences', 'buffered'),
            ('output', 'readyToContinue', 'buffered'),            
            ('output', 'imageGenerated', 'image 64x64x3'),            
            ]
    
    def __init__(self):
        BaseModule.__init__(self)
        
        
        self.checkPath = "/home/eli/Documents/data/CompleteRobot/CheckPoints"
        self.exemplarPath = "/home/eli/Documents/data/CompleteRobot/Images/exemplars"
        self.basePath = "/home/eli/Documents/data/CompleteRobot/Images/"

        self.imSize = (64, 64)
                
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
        self.numAttrs = len(self.attrs.keys())
        
        self.names = dict((v,k) for k,v in self.attrs.iteritems())
        
        self.net = NET(batch_size=1, image_shape=(self.imSize[0],self.imSize[1],3), text_shape=(self.numAttrs,))
        self.net.buildAutoEncSmall()
        self.net.compileAE()
        self.net.loadLatest(self.checkPath)
        print "MAE Compiled"
                
        self.exemplarList = self.checkExemplars()
        exFs = self.net.getAllFiles(self.exemplarPath, "png")
        self.exmImages = [cv2.resize(cv2.imread(_f),
                             dsize=(self.imSize[0],self.imSize[1]))/255.0 for _f in exFs]
        
        self.exmsOH = [self.convertText2OH(op.basename(_f).split(".")[0], "_") for _f in exFs]
        
    def configure(self, rf):
        print "configuring"
        result = BaseModule.configure(self, rf)               
        self.connect(self.outputPort["imageGenerated"].getName(), "//generated/Yarpview/image:i")      
        print "connected"   
        
        return result
    
    
    def updateModule(self):
        cmd = self.readPort("command")
        
        #self.clearOldImages()        
        if cmd == "learn":                       
            self.trainNet()
        elif cmd == "learnSingle":
            image = self.readImage("imageCropped")
            entities = self.readPort("entities") 
            self.trainSingle(image, entities)
        elif cmd == "predictFromImage":
            image = self.readImage("imageCropped")
            self.predictNet(image)         
               
        return True
 
    def clearOldImages(self):
        #for _ in range(10):
        _ =  self.readImage("imageCropped")
        
    
    def trainNet(self, image, text):        
        inputs, targets = self.makeInputs(image, text)        
        self.net.train(inputs, targets, self.checkPath)
        
    def trainSingle(self, image, text): 
        self.recalculateExemplars()       
        inputs, targets = self.loadTrainingData()        
        self.net.train(inputs, targets, self.checkPath)
        
            
    
    
    def loadTrainingData(self):
        exFs = self.net.getAllFiles(self.exemplarPath, "png")
        self.exmImages = [cv2.resize(cv2.imread(_f),
                             dsize=(self.imSize[0],self.imSize[1]))/255.0 for _f in exFs]
        
        self.exmsOH = [self.convertText2OH(op.basename(_f).split(".")[0], "_") for _f in exFs]
        
        exFs = self.net.getAllFiles(self.trainPath, "png")
        images = [cv2.resize(cv2.imread(_f),
                             dsize=(self.imSize[0],self.imSize[1]))/255.0 for _f in exFs]
        
        ohs = [self.convertText2OH(op.basename(_f).split(".")[0], "_") for _f in exFs]
        
        imagesIn = np.concatenate((images, images, np.zeros_like(images)),axis=0)
        textsIn  = np.concatenate((ohs, np.zeros_like(ohs), self.exmsOH), axis=0)
        
        imagesTr = np.concatenate((images, images, self.exmImages),axis=0)
        textsTr  = np.concatenate((ohs, ohs, self.exmsOH),axis=0)
        
        inputs = [imagesIn, textsIn]
        targets = [imagesTr, textsTr]
        
        return inputs, targets
        
    
    def makeInputs(self, image, text):
        
        text = text.strip('"')
          
        ohs = self.exmsOH
        images = self.exmImages
        
        image = image/255.0
        textOH = self.convertText2OH(text)
       
        imageEx = self.getExemplar(text, image)
        
        images.append(image)       
        ohs.append(textOH)
        

        imagesIn = np.concatenate((images, images, np.zeros_like(images)),axis=0)
        textsIn  = np.concatenate((ohs, np.zeros_like(ohs), ohs), axis=0)
        
        imagesEx = images[0:-1]
        imagesEx.append(imageEx)
        imagesTr = np.concatenate((images, images, imagesEx),axis=0)
        textsTr  = np.concatenate((ohs, ohs, ohs),axis=0)
        
        inputs = [imagesIn, textsIn]
        targets = [imagesTr, textsTr]
        
        return inputs, targets
        
    def checkExemplars(self):
        exFs = self.net.getAllFiles(self.exemplarPath, "png")
        exemplars = [op.basename(f).split(".")[0] for f in exFs]
        return exemplars
                
    def getExemplar(self, text, image):
        if text in self.exemplarList:
            exemplar = cv2.imread("%s/%s.png"%(self.exemplarPath, "_".join(text.split())))
            
        else:
            exemplar = image
            cv2.imwrite("%s/%s.png"%(self.exemplarPath, "_".join(text.split())), image*255)
            
        return cv2.resize(exemplar, dsize=self.imSize)
            
    def predictNet(self, image):
        newImage, pred, confs = self.net.predict(image/255.0, self.net.emptyText)
        newName = self.convertOH2Text(pred)
        self.sendPort("MAEOutput", newName)
        x = [str(_fl) for _fl in np.ndarray.tolist(confs)]        
        self.sendPort("confidences", "_".join(x))
        self.sendImage("imageGenerated", np.squeeze(newImage)*255)
 
    
    def convertText2OH(self, text, splitOn=" "):
        oh = np.zeros(self.net.text_shape)
        for ent in text.split(splitOn):
            try:
                idx = self.attrs[ent.strip('"')]
                oh[idx] = 1
            except:
                pass       
        
        return oh
    
    
    def recalculateExemplars(self):
        fnames = self.net.getAllFiles("%s/cropped"%self.basePath, "png")
        uniqueNames = self.getUniqueNew(fnames)
        self.MakeAverageImage(fnames, uniqueNames)
        
    
    def MakeAverageImage(self, fnames, uniqueNames):
        for u in uniqueNames:            
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
            
    def computeDistance(self, centre, samples, u):
        smallest = 99999999999;
        for sample in samples:
            dist = np.sum(np.abs(centre - sample))   
            if dist < smallest:
                smallest = dist
                toSave = sample
        
        cv2.imwrite("%s/exemplars/_%s.png"%(self.basePath,u), toSave)

    
    def convertOH2Text(self, oh):
        atrs = [self.names[i] for i in range(self.numAttrs) if oh[i]==1]
        return " ".join(atrs)


    def sendPort(self, port, string2Send):
        '''
        This method sends a string to the speech output port which is connected to MSpeak module
        '''
        Bottle =   self.outputPort[port].prepare()
        Bottle.clear()
        Bottle.addString(string2Send)
        self.outputPort[port].write()
        print "sent %s to %s"%(string2Send, port)
        
    def sendImage(self, port, image):
        '''
        This method sends an image <image> to the port <port>
        
        '''        
     
        self.outputPort[port].sendOpenCV(image)
        
    
    def readImage(self, port):
        '''
        This method reads an image from the port <port>
        
        @params
        returns contents - 3d Array
        '''
        
        port = self.inputPort[port]        

        if port.read(port.image):
            print "read image"
            # Make sure the image has not been re-allocated
            assert port.array.__array_interface__['data'][0] == port.image.getRawImage().__long__()

            # convert image to be usable in OpenCV
                        
            image = cv2.cvtColor(port.array, cv2.COLOR_BGR2RGB)
            
            return image
        
    def readPort(self, portID):
        '''
        This method reads a bottle containing a string from the "transcription" port
        
        @params
        returns contents - string
        '''
        
        port = self.inputPort[portID]      
        
        contents = port.read()
        
        return contents
    
if __name__ == '__main__':
    '''
    net = NET(1)
    base_path = "/home/eli/Documents/data/CompleteRobot/Images/cropped"
    dst = "/home/eli/Documents/data/CompleteRobot/Images/cropped/checked"
    fs = net.getAllFiles(base_path, "png") 
    for _f in fs:
        name = op.basename(_f)
        os.rename(_f, "%s/%s"%(dst,name))
    '''    
    
    MAE.main(MAE)
    
    
    