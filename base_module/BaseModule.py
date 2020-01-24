####################################################################################################
#    Copyright (C) 2016 by Ingo Keller                                                             #
#    <brutusthetschiepel@gmail.com>                                                                #
#                                                                                                  #
#    This file is part of SPY (A Sensor Module Package for Yarp).                                  #
#                                                                                                  #
#    SPY is free software: you can redistribute it and/or modify it under the terms of the         #
#    GNU Affero General Public License as published by the Free Software Foundation, either        #
#    version 3 of the License, or (at your option) any later version.                              #
#                                                                                                  #
#    SPY is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;              #
#    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.     #
#    See the GNU Affero General Public License for more details.                                   #
#                                                                                                  #
#    You should have received a copy of the GNU Affero General Public License                      #
#    along with SPY.  If not, see <http://www.gnu.org/licenses/>.                                  #
####################################################################################################
import time
import types

try:
    import cv2
except ImportError:
    print '[BaseModule] Can not import cv2. This module will raise a RuntimeException.'

import numpy as np

try:
    import yarp
except ImportError:
    print '[BaseModule] Can not import yarp. This module will raise a RuntimeException.'


EMSG_YARP_NOT_FOUND  = "Could not connect to the yarp server. Try running 'yarp detect'."
EMSG_WRONG_CLS       = "Given class %s does not inherit BaseModule"

PORT_TYPE = {   'unbuffered': yarp.Port,
                'buffered':   yarp.BufferedPortBottle,
                'rpcclient':  yarp.RpcClient,
                'rpcserver':  yarp.RpcServer,
             }


def sendOpenCV(self, image):
    """ Sends an OpenCV image through the port. This method gets bound to a port during its 
        creation. If the image is not in the correct size it will be resized using OpenCV.
        
    @param image - OpenCV image 
    """
    h, w, c = image.shape
    
    # resize input if necessary
    if h != self.imageHeight or w != self.imageWidth:
        image = cv2.resize(image, (self.imageWidth, self.imageHeight))
    
    if c == 1:
        self.array[:,:] = np.squeeze(image)
    
    else:
        # and convert image back to something the yarpview can understand
        self.array[:,:] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Send the result to the output port
    self.write(self.image)


class BaseModule(yarp.RFModule):

    # if enabled a RPC server port will be created
    ENABLE_RPC = False

    # defines the input ports [ ( <type>, <name>, <mode> )* ]
    PORTS = []


    def __init__(self):
        yarp.RFModule.__init__(self)
        self._ports        = []
        self._input_ports  = {}
        self._output_ports = {}

        if self.ENABLE_RPC:
            self.rpc_port = None

        self.prefix       = ''


    def configure(self, rf):

        # Module name handling
        if rf is not None:
            self.prefix = rf.find('name').asString()

        cls_name    = self.__class__.__name__

        self.setName(self.prefix + '/' + cls_name if self.prefix else cls_name)

        # RPC port
        if self.ENABLE_RPC:
            self.rpc_port = self.createRpcServerPort('')
            self.attach_rpc_server(self.rpc_port)

        # input port creation
        for _type, _name, _mode in self.PORTS:
            method       = self.createInputPort if _type == 'input' else self.createOutputPort
            store        = self._input_ports    if _type == 'input' else self._output_ports
            new_port     = method(_name, mode = _mode)
            store[_name] = new_port

        return True


    def interruptModule(self):
        for port in reversed(self._ports):
            port.interrupt()
        return True


    def close(self):
        for port in reversed(self._ports):
            port.close()
        return True


    def getPeriod(self):
        return 0.001


    def updateModule(self):
        # I do not know why we need that, but if method is empty the module gets stuck
        time.sleep(0.000001)
        return True


    @property
    def inputPort(self):
        return self._input_ports


    @property
    def outputPort(self):
        return self._output_ports


####################################################################################################
# Factory methods
####################################################################################################

    def _createPort(self, name, target = None, mode = 'unbuffered'):
        """ This method returns a port object.

        @param name     - yarp name for the port
        @param obj      - object for which the port is created
        @param buffered - if buffered is True a buffered port will be used otherwise not;
                          default is True.
        @result port
        """

        is_image_port = mode.startswith('image')

        # handle image port creation
        if is_image_port:
            width, height, channels = mode.split(' ')[1].split('x')
            width                   = int(width)
            height                  = int(height)
            channels                = int(channels)
            mode                    = 'unbuffered'

        # create port
        port             = PORT_TYPE[mode]()
        port.isImagePort = is_image_port

        # add image handling to port
        if port.isImagePort:

            # add image information to port
            port.imageWidth    = width
            port.imageHeight   = height
            port.imageChannels = channels

            # add image buffer to port
            port.image, port.array = BaseModule.createImageBuffer(width, height, channels)

            # add send method to port by monkey patching
            port.sendOpenCV = types.MethodType( sendOpenCV, port )

        # open port
        if not port.open('/%s/%s' % (self.getName(), name) ):
            raise RuntimeError, EMSG_YARP_NOT_FOUND

        # add output if given
        if target:
            port.addOutput(target)

        # add port to port list
        self._ports.append(port)

        return port


    def createInputPort(self, name, mode = 'unbuffered'):
        """ This method returns an input port.

        @param obj      - the object that the port is created for
        @param name     - if a name is provided it gets appended to the modules name
        @param buffered - if buffered is True a buffered port will be used otherwise not;
                          default is True.
        @result port
        """
        return self._createPort(name + ':i', None, mode)


    def createOutputPort(self, name, target = None, mode = 'unbuffered'):
        """ This method returns an output port.

        @param obj      - the object that the port is created for
        @param name     - if a name is provided it gets appended to the modules name
        @param buffered - if buffered is True a buffered port will be used otherwise not;
                          default is True.
        @result port
        """
        return self._createPort(name + ':o', target, mode)


    def createRpcClientPort(self, name, target = None):
        """ This method returns an output port.

        @param obj      - the object that the port is created for
        @param name     - if a name is provided it gets appended to the modules name
        @param buffered - if buffered is True a buffered port will be used otherwise not;
                          default is True.
        @result port
        """
        return self._createPort(':rpc' if name else 'rpc', target, 'rpcclient')


    def createRpcServerPort(self, name, obj = None, target = None):
        """ This method returns an output port.

        @param obj      - the object that the port is created for
        @param name     - if a name is provided it gets appended to the modules name
        @param buffered - if buffered is True a buffered port will be used otherwise not;
                          default is True.
        @result port
        """
        return self._createPort(':rpc' if name else 'rpc', target, 'rpcserver')


    @staticmethod
    def createImageBuffer(width = 320, height = 240, channels = 3):
        """ This method creates image buffers with the specified \a width, \a height and number of
            color channels \a channels.

        @param width    - integer specifying the width of the image   (default: 320)
        @param height   - integer specifying the height of the image  (default: 240)
        @param channels - integer specifying number of color channels (default: 3)
        @return image, buffer array
        """

        if channels == 1:
            buf_image = yarp.ImageFloat()
            buf_image.resize(width, height)

            buf_array = np.zeros((height, width), dtype = np.float32)

        else:
            buf_image = yarp.ImageRgb()
            buf_image.resize(width, height)

            buf_array = np.zeros((height, width, channels), dtype = np.uint8)

        buf_image.setExternal( buf_array,
                               buf_array.shape[1],
                               buf_array.shape[0] )

        return buf_image, buf_array


    @staticmethod
    def connect(source, target):
        """ This method connects two ports.

        If given a port it resolves the port names on its own.

        @param source   - can be either a string or an object with a getName method
        @param target   - can be either a string or an object with a getName method
        @result boolean - returns whether the port could be connected or not
        """
        return yarp.Network.connect( source.getName() if hasattr(source, 'getName') else source,
                                     target.getName() if hasattr(target, 'getName') else target )


    @staticmethod
    def main(module_cls):
        """ This is a main method to run a module from command line.

        @param module_cls - class inheriting BaseModule that can be started as a standalone module.
        """
        assert issubclass(module_cls, BaseModule), EMSG_WRONG_CLS % module_cls.__name__

        import sys

        yarp.Network.init()

        resource_finder = yarp.ResourceFinder()
        resource_finder.setVerbose(True)
        resource_finder.configure(sys.argv)

        module_cls().runModule(resource_finder)

        yarp.Network.fini()
