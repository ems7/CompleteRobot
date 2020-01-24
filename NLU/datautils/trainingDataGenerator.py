'''
Created on 12 Nov 2019

@author: eli
'''

objs = ["brick", "duck", "rectangle", "cup", "ball", "donut"]
cols = ["red", "orange", "yellow", "green", "blue",
        "indigo", "violet", "pink", "purple", "grey",
        "black", "brown", "white"
        ]
sizes = ["big", "medium", "small", "large", "average sized", "little", "tiny", "huge"]

synonyms = {"large": "big", "average sized":"medium", "little":"small",
            "tiny":"small", "huge": "big",
            "violet":"pink", "indigo":"purple"}

intents = ["AskColour"]#, "AskColour", "AskSize"]

grammars = {"AskColour": ["is the",                                                           
                            ]
            
            }


for intent in intents:
    print "\n \n"
    print "## intent:%s"%intent 
    for gr in grammars[intent]: 
         
        for ob in objs:
            
            #for col in cols:
                
                for sz in sizes:
                    if sz in synonyms.keys():
                        se =  synonyms[sz]
                    else:
                        se = sz
                    '''        
                    if col in synonyms.keys():
                        ce =  synonyms[col]
                    else:
                        ce = col
                    '''          
                        
                    line = "  - " + gr + " ["+ob+"]" + "(object:%s)"%ob + " ["+sz+"]" + "(size:%s)"%se  # " ["+col+"]" + "(colour:%s)"%ce+ \
                            #"  + \["+ob+"]" + "(object:%s)"%ob
                        
                
                    print line
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                