'''
Created on 12 Nov 2019

@author: eli
'''
from scc_ip.base_module import BaseModule
import requests
import json


class NLUManager(BaseModule):
    
    ENABLE_RPC = False
        
    PORTS = [('input', 'transcription', 'buffered'), 
             ('output', 'intent', 'buffered'),
             ('output', 'entities', 'buffered'),
             ('output', 'transcription', 'buffered')
             
             ]
    
    def __init__(self):
        BaseModule.__init__(self)        
        
    def configure(self, rf):
        print "configuring"
        result = BaseModule.configure(self, rf)
        self.connect('/AudioTranscriber/transcription:o', self.inputPort['transcription'].getName())
        print "connected"
    
        self.ip = rf.find("ip").asString()
        if self.ip == "":
            self.ip = "137.195.27.55"
        print "rasa ip is: %s"%self.ip
        return result
    

    def updateModule(self):        
        transcription = self.readTranscription()
        try:
            r = requests.post('http://%s:5005/model/parse'%self.ip, json={"text": transcription})
            self.parseNLUResponse(r)
            self.sendPropogatedTranscription(transcription)
        except:
            print "failed to Post to <hhtp://%s:5005/model/parse>. Are you sure the rasa server is running?"%(self.ip)
        
        return True
    
    def parseNLUResponse(self, r):
        obj = json.loads(r.text)
        intent = obj["intent"]["name"]
        confidence = obj["intent"]["confidence"]
        
        entities = self.parseEntities(obj["entities"])
        self.sendIntent(intent, confidence)
        self.sendEntities(entities)
    
    def parseEntities(self, entities):
        toReturn = []
        for ent in entities:
            k = ent["entity"]
            v = ent["value"]
            toReturn.append("%s_%s"%(k,v))
        return toReturn
        
    def sendIntent(self, intent, confidence):
        bottle =   self.outputPort['intent'].prepare()
        bottle.clear()
        bottle.addString(str(intent)) 
        bottle.addDouble(float(confidence))       
        print "added intent to bottle: %s with confidence: %s"%(intent, confidence) 
        self.outputPort['intent'].write()
        
    
    def sendEntities(self, entities):
        bottle =   self.outputPort['entities'].prepare()
        bottle.clear()
        for ent in entities:
            bottle.addString(str(ent))        
            print "added entitiy to bottle: %s"%ent 
        self.outputPort['entities'].write()
    
    def sendPropogatedTranscription(self, transcription):
        bottle =   self.outputPort['transcription'].prepare()
        bottle.clear()
        bottle.addString(str(transcription))        
        print "added string to bottle: %s"%transcription 
        self.outputPort['transcription'].write()
        
    
        
        
    def readTranscription(self):
        '''
        This method reads a bottle containing a string from the "transcription" port
        
        @params
        returns transcription - string
        '''
        
        transcription_port = self.inputPort['transcription']        
        bottle = transcription_port.read()
        transcription = bottle.toString()
        
        return transcription
        
        
def main():
    NLUManager.main(NLUManager)
    
if __name__ == '__main__':
    main()

