'''
Created on 11 Oct 2018

@author: eli
'''
import yarp
from scc_ip.base_module import BaseModule
import cv2
import sys, getopt

class Yarpview(BaseModule):
    
    ENABLE_RPC = False
        
    PORTS = [('input', 'image', 'image 640x480x3')]
  
    def __init__(self):
        
        rf = yarp.ResourceFinder()
        rf.setVerbose(True)
        rf.configure(sys.argv)
        
        try:
            width = rf.find('width').asInt()
        except:
            pass
        if width == 0:
            width = 640
            
        try:
            height = rf.find('height').asInt()
        except:
            pass
        if height == 0:
            height = 480
            
        try:
            self.channels = rf.find('channels').asInt()
        except:
            pass
        if self.channels == 0:
            self.channels = 3
            
        self.PORTS = [('input', 'image', 'image %sx%sx%s' %(width, height, self.channels))]
        print self.PORTS
        BaseModule.__init__(self)
        
               
    def configure(self, rf):
        
        
        result = BaseModule.configure(self, rf)
        print "yarpview created %s" %self.inputPort['image'].getName()
        
        
        return result
    
    
    def updateModule(self):
        port = self.inputPort['image']

        if port.read(port.image):

            # Make sure the image has not been re-allocated
            assert port.array.__array_interface__['data'][0] == port.image.getRawImage().__long__()

            # convert image to be usable in OpenCV
            if self.channels == 3:
                image = cv2.cvtColor(port.array, cv2.COLOR_BGR2RGB)
            else:
                image = port.array
            cv2.imshow("image", image)
            cv2.waitKey(10)
            if cv2.getWindowProperty('image',1) < 0:
                sys.exit(0)
            
            
            
            
        return True
    
    
    

if __name__ == "__main__":   
    Yarpview.main(Yarpview)

        