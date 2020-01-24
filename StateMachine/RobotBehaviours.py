'''
Created on 12 Nov 2019

@author: eli
'''
from scc_ip.base_module import BaseModule
import numpy as np
from Utterances import Utterances
import cv2
import logging
import datetime

LOG = logging.getLogger(__name__)


class RobotBehaviours(BaseModule, Utterances):
    
    ENABLE_RPC = False
    
    PORTS = [
            ('input', 'imageCropped', 'image 64x64x3'),
            ('input', 'imageRaw', 'image 640x480x3'),
            ('input', 'MAEOutput', 'buffered'),
            ('input', 'transcription', 'buffered'), 
            ('input', 'intent', 'buffered'),
            ('input', 'entities', 'buffered'),
            ('input', 'confidences', 'buffered'), 
            ('output', 'speech', 'buffered'),
            ('output', 'imageCropped', 'image 64x64x3'),
            ('output', 'entities', 'buffered'),
            ('output', 'command', 'buffered')
            
            ]
    
    def __init__(self):
        BaseModule.__init__(self)
        Utterances.__init__(self)
        self.repeats = 10  
        self.dataPath = "/home/eli/Documents/data/CompleteRobot/Images/"      
    
    def configure(self, rf):
        print "configuring"
        result = BaseModule.configure(self, rf)
        self.connect('/NLUManager/transcription:o', self.inputPort['transcription'].getName())
        self.connect('/NLUManager/intent:o', self.inputPort['intent'].getName())
        self.connect('/NLUManager/entities:o', self.inputPort['entities'].getName())
        self.connect('/ImageCropper/imageCropped:o', self.inputPort['imageCropped'].getName())
        self.connect('/MAE/MAEOutput:o', self.inputPort['MAEOutput'].getName())
        self.connect('/MAE/confidences:o', self.inputPort['confidences'].getName())
        self.connect('/image:o', self.inputPort['imageRaw'].getName()) 
        self.connect(self.outputPort['command'].getName(), '/MAE/command:i')
        self.connect(self.outputPort['entities'].getName(), '/MAE/entities:i')
        self.connect(self.outputPort['imageCropped'].getName(), '/MAE/imageCropped:i')
        self.connect(self.outputPort['speech'].getName(), '/read')              
        print "connected"   
        
        return result

    def greet(self):
        print self.greetings[np.random.randint(self.numGreets)]
        self.sendPort("speech", self.greetings[np.random.randint(self.numGreets)])
        
    def goodbye(self):
        self.sendPort("speech", self.goodbyes[np.random.randint(self.numByes)])
        self.sendPort("speech", "I will now go over everything you have taught me so that next time I do better.")
        self.sendPort("command", "learn")                
    
    def affirm(self):
        self.sendPort("speech", self.affirms[np.random.randint(self.numAffirms)])
    
    def deny(self):
        self.sendPort("speech", self.denials[np.random.randint(self.numDenials)])
    
    def mood_great(self):
        self.sendPort("speech", self.postiveMood[np.random.randint(self.numPostiveMood)])
    
    def mood_unhappy(self):
        self.sendPort("speech", self.negativeMood[np.random.randint(self.numNegativeMood)])
    
    def bot_challenge(self):
        self.sendPort("speech", self.botChallenges[np.random.randint(self.numBotChallenges)])
        
    
    def askColour(self):
        self.sendPort("command", "predictFromImage")
        self.image = self.readImage("imageCropped")
        self.sendImage("imageCropped", self.image)
        
        self.maeResponse = self.readPort("MAEOutput")
        self.confidences = [float(x) for x in self.readPort("confidences").strip('"').split("_")]
        
        answer = self.generateAnswer("colour", self.maeResponse, self.confidences)
        self.sendPort("speech", answer) 
        
    
    def askSize(self):
        self.sendPort("command", "predictFromImage")
        self.image = self.readImage("imageCropped")
        self.sendImage("imageCropped", self.image)
        
        self.maeResponse = self.readPort("MAEOutput")
        self.confidences = [float(x) for x in self.readPort("confidences").strip('"').split("_")]
        
        answer = self.generateAnswer("size", self.maeResponse, self.confidences)
        self.sendPort("speech", answer) 
        
    
    def askObject(self):
        self.sendPort("command", "predictFromImage")
        self.image = self.readImage("imageCropped")
        self.sendImage("imageCropped", self.image)
        
        self.maeResponse = self.readPort("MAEOutput")
        self.confidences = [float(x) for x in self.readPort("confidences").strip('"').split("_")]
        answer = self.generateAnswer("object", self.maeResponse, self.confidences)
        self.sendPort("speech", answer) 
    
    
    def parseEntities(self, entities):
        
        entSize= ""
        entColour = ""
        entObject = ""
        for ent in entities:
            
            eType, eValue = ent.split("_")
            if eType == "size":        
                entSize = eValue
                
            elif eType == "colour":                    
                entColour = eValue
                
            elif eType == "object":
                entObject = eValue
        
        return entSize, entColour, entObject
        
        
            
    
    def learnObject(self):
        size, colour, obj = self.parseEntities(self.entities)
        text = "_".join([size, colour, obj])
        for _ in range(self.repeats):
            image = self.readImage("imageCropped")
            rawImage = self.readImage("imageRaw")
            self.saveData(text, image, rawImage)
        self.sendPort("speech", "I have added that to my database, thanks for helping me learn")
        
    
    def getCorrection(self):
        size, colour, obj = self.parseEntities(self.entities)
        text = "_".join([size, colour, obj])
        for _ in range(self.repeats):
            rawImage = self.readImage("imageRaw")
            image = self.readImage("imageCropped")            
            self.saveData(text, image, rawImage)
        self.sendPort("speech", "I have added that to my database, thanks for correcting me")
        
    
    def pointToObject(self):
        LOG.warn("Not implemented: %s" %self.pointToObject__name__)
    
    def learnSelfSupervised(self):
        
        res = self.maeResponse.split()
        length = len(res)
        print len(res)
        print res
        if length == 3:
            size   = res[0]
            colour = res[1]
            obj    = res[2]
        
        elif length > 3:
            size, colour, obj = self.FindMostLIkely(self.confidences)
        for _ in range(self.repeats):
            rawImage = self.readImage("imageRaw")
            image = self.readImage("imageCropped")
            self.learn(size, colour, obj, image, rawImage)
        self.sendPort("speech", "I have updated my knowledge of this object as I was correct")
        
    
    def learn(self, size, colour, obj, crop, raw):        
        self.sendPort("command", "learnSingle")
        self.sendPort("entities", "%s %s %s"%(size, colour, obj))
        self.sendImage("imageCropped", crop)
        text = "_".join([size, colour, obj])
        self.saveData(text, crop, raw)        
    
    def saveData(self, text, crop, raw):
        time = self.getTimeStamp()       
        cv2.imwrite("%s/raw/%s_%s.png"%(self.dataPath, time, text), raw)
        cv2.imwrite("%s/cropped/%s_%s.png"%(self.dataPath, time, text), crop)
        cv2.waitKey(10)     
    
    def sendPort(self, port, string2Send):
        '''
        This method sends a string to the speech output port which is connected to MSpeak module
        '''
        Bottle =   self.outputPort[port].prepare()
        Bottle.clear()
        Bottle.addString(string2Send)
        self.outputPort[port].write()
        print "sent %s to %s"%(string2Send, port)
        
    def sendImage(self, portID, image):
        '''
        This method sends an image <image> to the port <port>
        
        '''        
        self.outputPort[portID].sendOpenCV(image)
        print "sent image to %s:o"%portID
    
    def readImage(self, portID):
        '''
        This method reads an image from the port <port>
        
        @params
        returns contents - 3d Array
        '''
        
        port = self.inputPort[portID]        

        if port.read(port.image):
            
            # Make sure the image has not been re-allocated
            assert port.array.__array_interface__['data'][0] == port.image.getRawImage().__long__()

            # convert image to be usable in OpenCV
            
            image = cv2.cvtColor(port.array, cv2.COLOR_BGR2RGB)            
            print "read image from %s:i" %portID
            return image
        
    def readPort(self, port):
        '''
        This method reads a bottle containing a string from the "transcription" port
        
        @params
        returns contents - string
        '''
        
        port = self.inputPort[port]        
        bottle = port.read()
        contents = bottle.toString()
        
        return contents
    
    def getTimeStamp(self):
        time = str(datetime.datetime.now())
        time = "_".join("_".join("_".join("_".join(time.split("-")).split()).split(":")).split("."))
        return time
        
    
     
    