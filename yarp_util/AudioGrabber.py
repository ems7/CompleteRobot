'''
Created on 12 Oct 2018

@author: eli
'''


import numpy as np
from scc_ip.base_module import BaseModule
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset
#import cv2
from sklearn.preprocessing import MinMaxScaler
from yarp import Bottle
import sys

#from scipy.io.wavfile import write
import wave


class AudioGrabber(BaseModule):
    
    ENABLE_RPC = False
        
    PORTS = [('output', 'audioAsImage', 'image 320x160x1'), 
             ('output', 'trigger', 'buffered')
             ]
    
    def __init__(self):

        BaseModule.__init__(self)
        self.start = True
        #self.scaler = MinMaxScaler()
        self.chunk_count = 0
        self.audioShape = 320
        self.max_len = 160 
            

        

    def calibrate(self):
        '''
        This method calibrates the MinMaxScaler, self.scaler, by capturing 10 seconds
        of audio and applying MinMaxScaler fit method.
        See sklearn.preprocessing.MinMaxScaler for details.
        
        This is redundant, scaling is not necessary.
        
        '''
        a = raw_input("Calibrate normalisation, press return then make noises from your mouth hole.")
        if self.audioPath == None:
            asource = ADSFactory.ads(sampling_rate=self.sr, max_time=10)
        else: 
            asource = ADSFactory.ads(filename=self.audioPath, sampling_rate=self.sr, max_time=10)
        
        validator = AudioEnergyValidator(
                                        sample_width=asource.get_sample_width(),
                                        energy_threshold=self.energy)
        
        tokenizer = StreamTokenizer(validator=validator,
                                         min_length=self.min_len,
                                         max_length=self.max_len,
                                         max_continuous_silence=self.max_con_si)
        
        def calib_callback(data, start, end):
            audio = np.fromstring(data[0], dtype=np.int8)
            self.scaler.fit_transform(np.swapaxes(np.asarray([audio]), 0,1))
            print "Audio sample found {0}--{1}".format(start, end)
        
        asource.open()
        
        tokenizer.tokenize(asource, callback=calib_callback)
        print "Scaler paramaters found: min: {0} max: {1}".format(self.scaler.data_min_,
                                                                  self.scaler.data_max_)
        
            
        print "calibration done"
        self.mini = self.scaler.data_min_
        self.maxi = self.scaler.data_max_
        
        
        
            
    
    def configure(self, rf):
        result = BaseModule.configure(self, rf)
        #print "yarpview created %s" %self.inputPort['image'].getName()
        
        if rf.check("audioSource"):
            self.audioPath = rf.find("audioSource").asString()
            print "audio being recieved from %s"%self.audioPath
            
        else:
            self.audioPath = None
            print "no audio source specified using built in microphone"
                
        if rf.check("sampleRate"):
            self.sr = rf.find("sampleRate").asInt()
        else:
            self.sr = 16000
            print "using default sample rate %s"%self.sr
        
        if rf.check("energy"):
            self.energy = rf.find("energy").asInt()
        else:
            self.energy = 60
            print "using default energy %s"%self.energy
        
        if rf.check("min_len"):
            self.min_len = rf.find("min_len").asInt()
        else:
            self.min_len = 60
            print "using default minimum length %s"%self.min_len
        
        if rf.check("maxSilence"):
            self.max_con_si = rf.find("maxSilence").asInt()
        else:
            self.max_con_si = 60
            print "using default maximum continuous silence %s"%self.max_con_si
        
        
        
        self.record = rf.check("record")
        
        if not self.record:
            print "audio is not being recorded"
       
        wf = rf.check("writeFolder")
        if wf: 
            self.writeFolder = rf.find("writeFolder").asString()
        if self.record and not wf:
            self.record=False
            print "audio is not being recorded, a Folder to write to must be specified"
        elif self.record and wf:
            print "audio will be saved to %s"%self.writeFolder
        
            
        self.PLAYBACK = rf.check("playback")
        
        if not self.PLAYBACK :            
            print "audio is not being played back"
        
        
        #self.calibrate()           

        return result
    
    def runAuditok(self):
        
        '''
        This method captures sound from the audio source specified in self.audioPath
        if self.audioPath is None, the built in microphone is used.
        
        
        '''
              
        #a = raw_input("waiting for start")
        if self.audioPath == None:
            self.asource = ADSFactory.ads(sampling_rate=self.sr)
        else: 
            self.asource = ADSFactory.ads(filename=self.audioPath, sampling_rate=self.sr)
            
        self.validator = AudioEnergyValidator(
                                        sample_width=self.asource.get_sample_width(),
                                        energy_threshold=self.energy)
        
        self.tokenizer = StreamTokenizer(validator=self.validator,
                                         min_length=self.min_len,
                                         max_length=self.max_len,
                                         max_continuous_silence=self.max_con_si)
        
        
        self.player = player_for(self.asource)
        
        self.prev_data = np.zeros([1])
        
        
        def audio_callback(data, start, end):
            
            
            if not np.array_equal(data, self.prev_data):
                self.sendTrigger() # send notice that audio has been detected
                
                print("Acoustic activity at: {0}--{1}".format(start, end))
                
                stamp = (start, end, self.chunk_count)
                
                
                if self.record:
                    self.saveAudio(data)
                    
                                                    
                copied = []                 
                for x in data:
                    
                    np_data = np.frombuffer(x, dtype=np.uint8)
                    #print np_data
                    copied.append(np_data)
                    

                data_rs = self.reshapeAudio(np.asarray(copied))
                

                self.sendAudio(data_rs, stamp)
                
                self.prev_data=data
                if self.PLAYBACK:
                    print "playing audio"
                    self.playback(data_rs)
                
                self.chunk_count += 1
                

        self.asource.open()
        self.sendTrigger() # send notice that the audio has started to be processed
        self.tokenizer.tokenize(self.asource, callback=audio_callback)
        sys.exit(0)
        
    
    def updateModule(self):
        if self.start:
            self.start = False
            print "Auditok is running"
            self.runAuditok()
        return True
    
    def saveAudio(self, data):
        """
        This method saves audio data in a wav file.
        
        @params
        data             - a list of byte strings containing the audio to be saved
        self.writeFolder - the folder to write the audio to
        self.chunk_count - the number of saved audio chunks, used as a file name
        
        returns:
        .wav             - a wav file containing <data> saved to
                           self.writeFolder/self.chunk_count.wav 
        
        """
        wavef = wave.open("%s/%s.wav"%(self.writeFolder, self.chunk_count),
                                          'w')
        wavef.setnchannels(1) #set mono
        wavef.setsampwidth(self.asource.get_sample_width()) # set number of bytes per sample
                                                            # to match the audio source
        wavef.setframerate(self.sr)
        wavef.writeframesraw("".join(data))
        wavef.writeframesraw("")
        wavef.close()
    
    def playback(self, audioAsImage):
        '''
        Plays back audio from an image like array (3D array)
        @params
        audioAsImage - 3D numpy array containing the audio information
        
        '''
        # remove any padded rows from the end
        trimmed = []
        for row in reversed(audioAsImage[:,:,0]):
            if np.sum(row) == 0:
                pass
            else:
                trimmed.append(row)


        audio = []
        for row in reversed(trimmed):
            audio.append(row)
        
        audio = (np.asarray(audio)).astype(np.uint8)
        copied = []
        for x in audio:
            copied.append(x.tobytes()) # convert uint8 values to byte string
        
        
        self.player.play(''.join(copied)) # playback audio
      
            
    
    
    def reshapeAudio(self, audio):
        '''
        Reshape audio to be a standard sized image of 320x320x3
        Audio data is padded to the fixed size by appending rows of zeros
        The data is scaled to be between zero and one
        
        @params:
        audio       - 2D numpy array whos rows contain one frame of audio data each
        
        Returns:
        audioAsImage - 3D numpy array containing three copies of the scaled and padded audio data
                       Each "channel" contains identical data, the redundant data is provided
                       as Basemodule.sendopencv() only handles colour images
        '''
        
        num = self.max_len - len(audio)
                
        padded = np.zeros([self.max_len, self.audioShape], dtype=np.uint8)
        if num > 0: # copy audio data row by row into the correctly sized array
            for i, x in enumerate(audio):
                padded[i,:] = x
                
        # make the audio data look like a colour image
        audioAsImage = padded.reshape((self.max_len, self.audioShape, 1))/255.0       
        #audioAsImage = np.concatenate((audioAsImage, audioAsImage, audioAsImage), axis = 2)
        
    
        return audioAsImage
        
        


    
    def sendAudio(self, audio, stamp):
        '''
        This method sends an envelope containing a time stamp, chunk number and 
        image to the audioAsImage output port
        '''
        
        bottle =  Bottle()
        bottle.addInt(stamp[0]) #start
        bottle.addInt(stamp[1]) #end
        bottle.addInt(stamp[2]) #number
        
        self.outputPort['audioAsImage'].setEnvelope(bottle)
        self.outputPort['audioAsImage'].sendOpenCV(audio)
    
    def sendTrigger(self):
        bottle =   self.outputPort['trigger'].prepare()
        bottle.clear()
        bottle.addInt(1) 
        self.outputPort['trigger'].write()
        
    

        
    
        
        
def main():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    AudioGrabber.main(AudioGrabber)
    

if __name__ == '__main__':
    main()