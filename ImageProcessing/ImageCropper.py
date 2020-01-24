'''
Created on 13 Nov 2019

@author: eli
'''

from scc_ip.base_module import BaseModule
from yarp import Bottle
import cv2
import numpy as np

class ImageCropper(BaseModule):
    
    ENABLE_RPC = False
    
    PORTS = [
            ('input', 'image', 'image 640x480x3'),
            ('output', 'imagePropogated', 'image 640x480x3'),
            ('output', 'imageCropped', 'image 64x64x3')
            ]
    
    def __init__(self):
        BaseModule.__init__(self)
        self.image_shape = (64,64)
        self.cropWidth = 200
        self.cropCoords = (128,128)
        
    def configure(self, rf):
        print "configuring"
        result = BaseModule.configure(self, rf)
        self.connect('/image:o', self.inputPort['image'].getName())
        
        return result
        
        
    def updateModule(self):
        
        self.rawImage = self.readImage("image")
        self.cropImage(self.cropCoords)
        self.sendImage("imagePropogated", self.rawImage)
        self.sendImage("imageCropped", self.crop_img)
        return True
    
    def cropImage(self, coords):
        
        
        crop_img = self.rawImage[coords[0]:coords[0]+self.cropWidth,
                                 coords[1]:coords[1]+self.cropWidth]
        
        
        
        
        self.crop_img = cv2.resize(crop_img, dsize=self.image_shape)
       
    def readImage(self, port):
        '''
        This method reads an image from the port <port>
        
        @params
        returns contents - 3d Array
        '''
        
        port = self.inputPort[port]        

        if port.read(port.image):            
            # Make sure the image has not been re-allocated
            assert port.array.__array_interface__['data'][0] == port.image.getRawImage().__long__()

            # convert image to be usable in OpenCV
            
            image = cv2.cvtColor(port.array, cv2.COLOR_BGR2RGB)            

            return image
        
    def sendImage(self, port, image):
        '''
        This method sends an image <image> to the port <port>
        
        '''        
     
        self.outputPort[port].sendOpenCV(image)
        
        
        
        
if __name__ == '__main__':
    ImageCropper.main(ImageCropper)        
        
        
        
        
        