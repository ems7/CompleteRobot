'''
Created on 16 Oct 2018

@author: eli
'''

import numpy as np
from scc_ip.base_module import BaseModule
from auditok import ADSFactory, player_for
import cv2


class AudioPlayer(BaseModule):
    '''
    This class takes in 3D arrays and plays them back as audio. 
    See AudioGrabber.py for an example of generating 3D audio arrays
    '''
    PORTS = [('input', 'audioAsImage', 'image 320x320x1'), 
             ]
    

    def configure(self, rf):
        result = BaseModule.configure(self, rf)
        
        if rf.check("sampleRate"):
            sr = rf.find("sampleRate").asInt()
        else:
            sr = 16000
            
        asource = ADSFactory.ads(sampling_rate=sr) # create a default ADSFactory 16000Hz 2bytes
                                   # If non default values are needed 
                                   # e.g. if the audio to be played back 
                                   # was captured at a different rate
                                   # add parameters to the method creating the ads object     
        
        self.player = player_for(asource) # create a player for the audio source
        
        return result
        
    
    def playback(self, audioAsImage):
        
        trimmed = []
        for row in reversed(audioAsImage[:,:]):
            if np.sum(row) == 0:
                pass
            else:
                trimmed.append(row)


        audio = []
        for row in reversed(trimmed):
            audio.append(row)
        
        audio = (np.asarray(audio)*255).astype(np.uint8)
        copied = []
        for x in audio:
            copied.append(x.tobytes())
        
        self.player.play(''.join(copied))
        
    def updateModule(self):
        
        audio = self.readAudio()
        self.playback(audio)
        return True
    
    def readAudio(self):
        '''
        This method reads an envelope containing an 
        image from the audioAsImage input port
        
        @params
        returns audio - 2D numpy array
        '''
        transcription_port = self.inputPort['audioAsImage']
        if transcription_port.read(transcription_port.image):
    
            # Make sure the image has not been re-allocated
            assert transcription_port.array.__array_interface__['data'][0] == transcription_port.image.getRawImage().__long__()

            audio = transcription_port.array
            
            return audio
            
            
def main():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    AudioPlayer.main(AudioPlayer)
    
    
if __name__ == '__main__':
    main()
    
    
    