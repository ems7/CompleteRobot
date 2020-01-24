'''
Created on 1 May 2019

@author: eli
'''
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Dense
from keras.layers import Input, Flatten, Reshape, Concatenate #, Lambda
import keras
from keras.losses import mse
#import keras.backend as K
import numpy as np
import datetime
import os
import os.path as op


class NET():
    def __init__(self, batch_size, image_shape=(64,64,3), text_shape=(20,)):
        self.image_shape = image_shape
        self.text_shape = text_shape
        self.emptyText = np.zeros(text_shape[0])        
        self.embSize = (4,4,16)
        self.embMagnitude = self.embSize[0]*self.embSize[1]*self.embSize[2]
        self.batch_size = batch_size
        self.batch_shape = [self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]]
    
    
       
    def buildImageEncoderSmall(self, trainable=[True]*100):
        self.imageInput = Input(self.image_shape, batch_shape=(self.batch_shape))
        x1 = Conv2D(64, kernel_size=(3, 3),
                         activation='relu', padding="same")(self.imageInput)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(16, (3, 3), strides=(2,2), activation='relu', padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(16, (3, 3), strides=(2,2), activation='relu', padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(16, (3, 3), strides=(2,2), activation='relu', padding="same")(x1)
        x1 = BatchNormalization()(x1)
        #x1 = MaxPooling2D(pool_size=(2, 2))(x1)
        self.iconv5 = Dropout(0.25)(x1)
                              
        
        self.imageEncoder = keras.models.Model(inputs=[self.imageInput], outputs=[self.iconv5])
        self.imageEncoder.name="imageEncoder" 
        #self.imageEncoder.summary()    
    
    def buildImageDecoderSmall(self, size, trainable=[True]*100):
        self.input3 = Input((2,2,size), name="im_dec_inp")
        y1 = Dropout(0.25)(self.input3)  
                                                
        y1 = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2,2), activation='tanh', padding="same")(y1)
        y1 = BatchNormalization()(y1)
        y1 = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2,2), activation='tanh', padding="same")(y1)
        y1 = BatchNormalization()(y1)
        y1 = Conv2DTranspose(16, kernel_size=(3, 3), strides=(2,2), activation='tanh', padding="same")(y1)
        y1 = BatchNormalization()(y1)
        y1 = Conv2DTranspose(32, kernel_size=(3, 3), strides=(2,2), activation='tanh', padding="same")(y1)
        y1 = BatchNormalization()(y1)           
        y1 = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2,2), activation='tanh', padding="same")(y1)
        y1 = BatchNormalization()(y1) 
        y1 = Conv2DTranspose(64, kernel_size=(3, 3), activation='tanh', padding="same")(y1)
        y1 = BatchNormalization()(y1) 
        self.y1 = Conv2DTranspose(self.image_shape[2], kernel_size=(3, 3), activation='sigmoid', padding="same")(y1)

        self.imageDecoder = keras.models.Model(inputs=[self.input3], outputs=[self.y1])
        self.imageDecoder.name="imageDecoder" 
        
        
    
    def buildTextEncoderSmall(self, trainable=[True]*100):
        
        self.tagInput = Input(self.text_shape[0], batch_shape = (self.batch_size, self.text_shape[0]))
        x2 = Dense((self.image_shape[0]*self.image_shape[1]/32), activation="tanh")(self.tagInput)
        x2 = BatchNormalization()(x2)
        x2 = Dense((self.image_shape[0]*self.image_shape[1]/16), activation="tanh")(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.5)(x2)
        x2 = Reshape(((self.image_shape[1]/4), (self.image_shape[0]/4), 1))(x2)
        x2 = Conv2D(16, (3,3), strides=(2,2), activation="relu", padding="same")(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(16, (3,3), strides=(2,2), activation="relu", padding="same")(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(16, (3,3), strides=(2,2), activation="relu", padding="same")(x2)
        x2 = BatchNormalization()(x2)
        self.txconv5 = Dropout(0.2, name="tx_enc_12")(x2)
        
        self.textEncoder = keras.models.Model(inputs=[self.tagInput], outputs=[self.txconv5])
        self.textEncoder.name="textEncoder" 
        
    
    def buildTextDecoderSmall(self, size, trainable=[True]*100):
        self.input4 = Input((2,2,size), name="tx_dec_inp")
        y2 = Conv2DTranspose(16, (3,3), strides=(2,2), activation="relu", padding="same")(self.input4)
        y2 = BatchNormalization()(y2)
        y2 = Conv2DTranspose(16, (3,3), strides=(2,2), activation="relu", padding="same")(y2)
        y2 = BatchNormalization()(y2)
        y2 = Conv2DTranspose(16, (3,3), strides=(2,2), activation="relu", padding="same")(y2)
        y2 = BatchNormalization()(y2) 
        y2 = Flatten()(y2)
        y2 = Dropout(0.5)(y2)
        y2 = Dense((self.image_shape[0]*self.image_shape[1]/16), activation="tanh")(y2)
        y2 = BatchNormalization()(y2)
        y2 = Dense((self.image_shape[0]*self.image_shape[1]/32), activation="tanh")(y2)
        y2 = BatchNormalization()(y2)
        self.y2 = Dense(self.text_shape[0], activation="sigmoid")(y2)
        
        self.textDecoder = keras.models.Model(inputs=[self.input4], outputs=[self.y2])
        self.textDecoder.name="textDecoder" 

    
    def buildAutoEncSmall(self, size=296, train=True):

        self.buildImageEncoderSmall()           

        #encIm = Reshape(self.embSize)(self.iconv5)

        self.buildTextEncoderSmall()

        #enctxRS = Reshape(self.embSize)(self.txconv5)

        merged = Concatenate()([self.iconv5, self.txconv5])       
        self.mergeLayer = Conv2D(size, (3,3), activation="relu", padding="same", name="merged")  
        merged = self.mergeLayer(merged)
        self.merged = BatchNormalization()(merged) 
        
        self.buildImageDecoderSmall(size)
        decIm = self.imageDecoder(self.merged)

        self.buildTextDecoderSmall(size)
        dectx = self.textDecoder(self.merged)
        
        self.model = keras.models.Model(inputs=[self.imageInput, self.tagInput], 
                                            outputs=[decIm, dectx])
        
        self.encoder = keras.models.Model(inputs=[self.imageInput, self.tagInput], 
                                            outputs=[self.merged])


        self.model.name   = "MAE"
        self.encoder.name = "encoder"
        

    def compileAE(self):
        """
        compile MAE model
        """
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.001),
                                  loss=mse) 
    
    
    def loadLatest(self, path):
        '''
        This method checks for the latest model in a directory and loads the weights if they exist
        '''
        wfs = self.getAllFiles(path)
        
        if len(wfs)>0:
            mostRecent = 0
            for w in wfs:
                timeModified = os.stat(w)[8]
                if timeModified > mostRecent:
                    mostRecent = timeModified
                    toLoad = w
            print("Loading weights from %s"%(op.basename(toLoad)))
            self.model.load_weights(toLoad)
        
        
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
    
    def getTimeStamp(self):
        time = str(datetime.datetime.now())
        time = "_".join("_".join("_".join(time.split("-")).split()).split(":")).split(".")[0]
        return time
    
    def train(self, inputs, targets, checkPath):
        time = self.getTimeStamp()             
        self.model.fit(inputs,
                    targets,
                    epochs = 1,
                    batch_size=self.batch_size,
                    verbose=1)
        
        self.model.save_weights("%s/%s.h5"%(checkPath, time))
    
    def predict(self, image, text):          
        image, pred = self.model.predict([np.reshape(image, (1,64,64,3)),
                                      np.reshape(text, (1,self.text_shape[0]))],
                                     batch_size=1)
                                      
                                      
        newAttr = np.squeeze(pred)        
        rnd_atr = np.zeros_like(newAttr)
        rnd_atr[newAttr>=0.5] = 1
        return image, rnd_atr, newAttr                                 
    