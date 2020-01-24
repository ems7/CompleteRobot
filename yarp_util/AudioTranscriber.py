'''
Created on 12 Oct 2018

@author: eli
'''


from base_module.BaseModule import BaseModule
import speech_recognition as spr
import wave


class AudioTranscriber(BaseModule):
    
    ENABLE_RPC = False
        
    PORTS = [('output', 'transcription', 'buffered'), 
             ('output', 'trigger', 'buffered')
             ]
    
    def __init__(self):

        BaseModule.__init__(self)
        self.start = True
        #self.scaler = MinMaxScaler()
        self.chunk_count = 0
        self.audioShape = 320
        self.max_len = 160 
        self.recognizer = spr.Recognizer()
                             
    
    def configure(self, rf):
        result = BaseModule.configure(self, rf)
                
                
        if rf.check("maxSilence"):
            self.max_con_si = rf.find("maxSilence").asInt()
        else:
            self.max_con_si = 2
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
        

        self.calibrate()           

        return result
    
    def calibrate(self):
        '''
        This method calibrates the MinMaxScaler, self.scaler, by capturing 10 seconds
        of audio and applying MinMaxScaler fit method.
        See sklearn.preprocessing.MinMaxScaler for details.
        
        This is redundant, scaling is not necessary.
        
        '''
        _ = raw_input("Calibrate for background noise, press return then sit quitely whilst the recogniser adjusts")
        with spr.Microphone() as source:            
            self.recognizer.adjust_for_ambient_noise(source, 5)
        print "calibrated for background noise"
    
    def runTranscriber(self):
        #transcription = raw_input("say: ")
        #self.sendTranscription(transcription)
        
        transcription = None
        print "listening"
        with spr.Microphone() as source:
            audio = self.recognizer.listen(source,
                                            timeout=self.max_con_si)
            
            try:
                print "waiting for transcription..."
                transcription = self.recognizer.recognize_google(audio, language="en-GB").lower()
                print "transcription received: ", transcription
                self.sendTranscription(transcription)
                self.sendTrigger(1)
                self.chunk_count += 1
                if self.record:
                    self.saveAudio(audio, transcription)
                
            except:
                self.sendTrigger(0)
                
            
            
        
    
    def updateModule(self):       
        self.runTranscriber()
        return True
    
    def saveAudio(self, audioData, transcription):
        """
        This method saves audio data in a wav file.
        
        @params
        AudioSource      - an AudioSource object
        self.writeFolder - the folder to write the audio to
        self.chunk_count - the number of saved audio chunks, used as a file name
        
        returns:
        .wav             - a wav file containing <data> saved to
                           self.writeFolder/self.chunk_count.wav 
        
        """
        wavef = wave.open("%s/%s_%s.wav"%(self.writeFolder, transcription, self.chunk_count),
                                          'w')
        wavef.setnchannels(1) #set mono
        wavef.setsampwidth(audioData.sample_width) # set number of bytes per sample
                                                            # to match the audio source
        wavef.setframerate(audioData.sample_rate)
        wavef.writeframesraw("".join(audioData.get_raw_data()))
        wavef.writeframesraw("")
        wavef.close()
    
    
    
    def sendTrigger(self, x):
        bottle =   self.outputPort['trigger'].prepare()
        bottle.clear()
        bottle.addInt(x) 
        self.outputPort['trigger'].write()
        
        
    def sendTranscription(self, transcription):
        bottle =   self.outputPort['transcription'].prepare()
        bottle.clear()
        bottle.addString(str(transcription))        
        print "added string to bottle: %s"%transcription 
        self.outputPort['transcription'].write()
        
        
    

        
    
        
        
def main():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    AudioTranscriber.main(AudioTranscriber)
    

if __name__ == '__main__':
    main()