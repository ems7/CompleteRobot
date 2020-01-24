'''
Created on 12 Nov 2019

@author: eli
'''
import numpy as np

class Utterances():
    def __init__(self):
        
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
        
        self.names = dict((v,k) for k,v in self.attrs.iteritems())
        
        self.greetings = ["hello",
                          "hi"                          
                          ]
        self.numGreets = len(self.greetings)
        
        self.goodbyes = ["bye",
                         "goodbye",
                         "farewell",
                         "good day"
                         ]
        self.numByess = len(self.goodbyes) 
        
        
        self.affirms = ["affirmative",
                        "yes",
                        "yeah",
                        "yep"
                        ]
        self.numAffirms = len(self.affirms) 
        
        self.denials = ["no",
                        "nope"                        
                        ]
        self.numDenials = len(self.denials) 
         
        self.postiveMood = ["I am glad you are feeling good, I am feeling good too.",
                            "I'm happy you're happy!"]
        self.numPostiveMood = len(self.postiveMood) 
        
        
        self.negativeMood = ["Perhaps I can cheer you up.",
                             "That's a shame, I hope you feel better soon."] 
        self.numNegativeMood = len(self.negativeMood) 
         
         
        self.botChallenges = ["I am a robot made by the I I T",
                             "I am an eye cub robot"]
        self.numBotChallenges = len(self.botChallenges) 
        
    def FindMostLIkely(self, confidences):
        
        idxTop3 = (-np.asarray(confidences)).argsort()[:3]
        idxTop3.sort()
        print idxTop3
        size   = self.names[idxTop3[0]]
        colour = self.names[idxTop3[1]]
        obj    = self.names[idxTop3[2]]
        print "##########################"
        
        print size, colour, obj
        print "##########################"
        return size, colour, obj
        
    def generateAnswer(self, qType, maeResponse, confidences):
        
        res = maeResponse.split()
        length = len(res)
        
        if length == 3:
            size   = res[0].strip('"')
            colour = res[1].strip('"')
            obj    = res[2].strip('"')
        
        elif length > 3:
            size, colour, obj = self.FindMostLIkely(confidences)
        
        else:
            return "I am confused, I don't know what I am looking at. Can you tell me?"
        
        sizeMatch = False
        colourMatch = False
        objectMatch = False      
        
        entSize   = ""
        entColour = ""
        entObject = ""     
        
        if self.entities != "":
            for ent in self.entities:
                
                eType, eValue = ent.split("_")
                
                if eType == "size" and eValue == size:
                    sizeMatch = True
                    entSize = eValue
                
                elif eType == "size" and eValue != size:
                    sizeMatch = False
                    entSize = eValue
                    
                    
                elif eType == "colour" and eValue == colour:
                    colourMatch = True
                    entColour = eValue
                
                elif eType == "colour" and eValue != colour:
                    colourMatch = False
                    entColour = eValue
                    
                    
                elif eType == "object" and eValue == obj:
                    objectMatch = True
                    entObject = eValue
                    
                elif eType == "object" and eValue != obj:
                    objectMatch = False
                    entObject = eValue
                    
                    
        if qType == "colour" and colourMatch and entColour != "":
            answer = "yes that is %s. I think it is a %s %s %s" %(colour, size, colour, obj)
        elif qType == "colour" and not colourMatch and entColour != "":
            answer = "no that is not %s, that is %s. I think it is a %s %s %s" %(entColour, colour, size, colour, obj)
        elif qType == "colour" and entColour == "":
            answer = "I think that its colour is %s. I think it is a %s %s %s" %(colour, size, colour, obj)
        
            
        elif qType == "size" and sizeMatch and entSize != "":
            answer = "yes that is %s. I think it is a %s %s %s" %(size, size, colour, obj)
        elif qType == "size" and not sizeMatch and entSize != "":
            answer = "no that is not %s, I think its size is %s. I think it is a %s %s %s" %(entSize, size, size, colour, obj)
        elif qType == "size" and entSize == "":
            answer = "I think its size is %s. I think it is a %s %s %s" %(size, size, colour, obj)
        
        
        elif qType == "object" and objectMatch and entObject != "":
            answer = "yes that is a %s. I think it is a %s %s %s" %(obj, size, colour, obj)
        elif qType == "object" and not objectMatch and entObject != "":
            answer = "no that is not a %s, That is a %s. I think it is a %s %s %s" %(entObject, obj, size, colour, obj)
        elif qType == "object" and entObject == "":
            answer = "That is a %s. I think it is a %s %s %s" %(obj, size, colour, obj)
        
        
        return answer 
        
        