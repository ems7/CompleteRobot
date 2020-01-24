'''
Created on 12 Nov 2019

@author: eli
'''

from RobotBehaviours import RobotBehaviours


class StateMachine(RobotBehaviours):

    def __init__(self):
        
        RobotBehaviours.__init__(self)
        
        self.prevState = None     
        self.knownActions = {
                             "greet":self.greet, 
                             "goodbye":self.goodbye,
                             "affirm":self.affirm, 
                             "deny":self.deny,
                             "mood_great":self.mood_great,
                             "mood_unhappy":self.mood_unhappy,
                             "bot_challenge":self.bot_challenge,
                             "AskColour":self.askColour,
                             "AskSize":self.askSize,
                             "AskObject":self.askObject,
                             "TeachObject":self.learnObject,
                             "PointToObject":self.pointToObject,
                             "GetCorrection":self.getCorrection,
                             "RobotIsRight":self.learnSelfSupervised
                             }
            
                    
    
    def updateModule(self):        
        self.transcription = self.readPort("transcription")
        print "got transcription %s"%self.transcription
        self.intent, confidence = self.readPort("intent").split()
        self.confidence = float(confidence)
        print "got intent %s"%self.intent
        self.entities = self.readPort("entities").split()
        print "got entities %s"%self.entities
        if self.confidence > 0.5:
            self.selectNextState()
        else:
            self.execute(self.unKnownState())
        
        return True
    
    
    def selectNextState(self):
        
        
        if self.intent in self.knownActions.keys():
            
            self.nextState = self.knownActions[self.intent]
            print "going to state: %s" %self.intent
            self.prevState = self.intent
            self.execute(self.nextState)
        
        else:
            self.execute(self.unknownState)
    
    def unKnownState(self):
        self.sendPort("Sorry I didn't understand that")        
    
    def execute(self, action):
        action()
    
        
      
if __name__ == '__main__':
    StateMachine.main(StateMachine)    
        
        
        