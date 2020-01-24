'''
Created on 2 May 2019

@author: eli
'''
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import sys
from MRL.net import NET
from dataloader import DataLoader
import keras
import keras.backend as K
import numpy as np
import os.path as op
import pickle as pkl
import cv2


def trainNet(net, inputs, targets, val_inputs, val_targets, run, checkPath, name):
    net.model.fit(inputs,
                targets,
                epochs = 50,
                batch_size=net.batch_size,
                verbose=1,
                validation_data=(val_inputs, val_targets), 
                                                     
                callbacks=[keras.callbacks.ModelCheckpoint(
                            "%s/%s_%s_epoch_{epoch:02d}_loss_{loss:.4f}_val_loss_{val_loss:.4f}.h5"
                            %(checkPath, name, run),
                            monitor='val_loss',
                            verbose=0,
                            save_best_only=False,
                            save_weights_only=False,
                            mode='auto',
                            period=1)])


def my_pred(images, text, text_true, net):
    
    _, pred = net.model.predict([np.reshape(images, (1,64,64,3)), np.reshape(text, (1,net.text_shape[0]))], batch_size=1)
    
    pred_labs = np.zeros_like(pred[0])
    pred_labs[pred[0]>0.5] = 1

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for tr, pr in zip(text_true, pred_labs):
        if tr == 1 and pr == 1:
            tp +=1 
        elif tr == 0 and pr == 1:
            fp +=1
        elif tr == 0 and pr == 0:
            tn +=1
        elif tr == 1 and pr == 0:
            fn +=1
    
           
    return tp, fp, tn, fn


def f1_score(tp,fp,tn,fn): 
    pr = (tp)/float(tp+fp)
    rc = (tp)/float(tp+fn)
    f1 = 2*((pr*rc)/float(pr+rc)) 
    
    return f1, pr, rc  

def addRes2Dict(resultsDict, res, key, test_type, epoch, tp, fp, tn, fn, f1, pr, rc):
    
    for i in range(len(res)):
        resultsDict[key][test_type][epoch][i] += res[i]
    
    resultsDict[key][test_type][epoch][i+1] += tp
    resultsDict[key][test_type][epoch][i+2] += fp
    resultsDict[key][test_type][epoch][i+3] += tn
    resultsDict[key][test_type][epoch][i+4] += fn
    resultsDict[key][test_type][epoch][i+5] += f1
    resultsDict[key][test_type][epoch][i+6] += pr
    resultsDict[key][test_type][epoch][i+7] += rc
    
    return resultsDict

def generateImages(ims, atrs, atr_ts, net, dl, resPath, epoch, cond):
    
    print "%s/imageOutput/%s/"%(resPath,epoch)
    if not op.exists("%s/imageOutput/%s"%(resPath,epoch)):
        print "making %s/imageOutput/%s"%(resPath,epoch)
        os.mkdir("%s/imageOutput/%s"%(resPath,epoch))
        
    if not op.exists("%s/imageOutput/%s/%s"%(resPath,epoch, cond)):
        print "making %s/imageOutput/%s/%s"%(resPath,epoch, cond)
        os.mkdir("%s/imageOutput/%s/%s"%(resPath,epoch, cond))
    
    
    for i, (im, at, atr_t) in enumerate(zip(ims, atrs, atr_ts)):
        newIm, _ = net.model.predict([np.reshape(im,(1,64,64,3)),np.asarray(at).reshape(1,net.text_shape[0])], batch_size=1)
    
    
        
        newName = dl.convertOH2Name(atr_t)
        #print newName
        #cv2.imshow("im", np.squeeze(newIm))
        cv2.imwrite("%s/imageOutput/%s/%s/%s_%s.png"%(resPath,epoch, cond,newName, i), np.squeeze(newIm)*255)
        #cv2.waitKey()

def run_tests(net, dl, checkPath, resPath, numRuns, atrs, disallowed):
    
    K.clear_session()
    imagesTest, textTest, _ = dl.loadSubset(atrs, "test", disallowed)
    dl.printStats(imagesTest, textTest, "test")
    net.batch_size = 1
    net.batch_shape = [1, net.image_shape[0], net.image_shape[1], net.image_shape[2]]
    
    wfs = dl.getAllFiles(checkPath)
    
    resultsDict = {}#pkl.load(open("/home/eli/Documents/data/realShapes/results/resDict.p", "r"))}
    #wfs = ["%s/MAE_dirtyquartPretrained_unlocked_64_1_04-0.21.h5"%path,
    #       "%s/MAE_dirtyquartPretrained_unlocked_64_0_04-0.21.h5"%path,
    #       "%s/MAE_dirtyquartPretrained_unlocked_64_2_04-0.21.h5"%path,
    #       "%s/MAE_dirtyquartPretrained_unlocked_64_3_04-0.21.h5"%path,]
    for wf in wfs:
        
        name    = op.basename(wf)
        print name
        epoch   = int(name.split("_")[3]) - 1
        modType = name.split("_")[0]
        
        
        net.buildAutoEncSmall()
        net.compileAE()
        net.model.load_weights(wf)
        
        
        key = "%s"%(modType)                  
        if key not in resultsDict.keys():
            resultsDict[key] = {"bimodal":[],
                                "image_only":[],
                                "text_only":[]}
            for _ in range(50):
                
                resultsDict[key]["bimodal"].append(   [0,0,0,0,0,0,0,0,0,0])
                resultsDict[key]["image_only"].append([0,0,0,0,0,0,0,0,0,0])
                resultsDict[key]["text_only"].append( [0,0,0,0,0,0,0,0,0,0]) 
     
        
        print "bimodal"
        length = len(imagesTest)
        
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        rows = 20
        for i in range(length):
            _tp, _fp, _tn, _fn = my_pred(imagesTest[i], textTest[i], textTest[i], net)
            tp += _tp
            fp += _fp
            tn += _tn
            fn += _fn
                    
        print tp, fp, tn, fn
        print tp /float(length*rows)
        f1, pr, rc = f1_score(tp, fp, tn, fn)
        print f1, pr, rc
        
        res = net.model.evaluate([imagesTest, textTest],
                                 [imagesTest, textTest],
                                 batch_size=1,
                                 verbose=0)
        print net.model.metrics_names
        print res
        
        resultsDict = addRes2Dict(resultsDict, res, key, "bimodal", epoch,
                                  (tp/float(length*rows))*100,
                                  (fp/float(length*rows))*100,
                                  (tn/float(length*rows))*100,
                                  (fn/float(length*rows))*100,
                                  f1,
                                  pr,
                                  rc)
       

        
        print "image only"
        
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        rows = 20
        for i in range(length):
            _tp, _fp, _tn, _fn = my_pred(imagesTest[i], textTest[i], textTest[i], net)
            tp += _tp
            fp += _fp
            tn += _tn
            fn += _fn
                    
        print tp, fp, tn, fn
        print tp /float(length*rows)
        f1, pr, rc = f1_score(tp, fp, tn, fn)
        print f1, pr, rc
              
        res = net.model.evaluate([imagesTest, np.zeros_like(textTest)],
                                 [imagesTest, textTest],
                                 batch_size=1,
                                 verbose=0)
        print res  
        resultsDict = addRes2Dict(resultsDict, res, key, "image_only", epoch,
                                  (tp/float(length*rows))*100,
                                  (fp/float(length*rows))*100,
                                  (tn/float(length*rows))*100,
                                  (fn/float(length*rows))*100,
                                  f1,
                                  pr,
                                  rc)
        
        print "text only"
        
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        rows = 20
        for i in range(length):
            _tp, _fp, _tn, _fn = my_pred(imagesTest[i], textTest[i], textTest[i], net)
            tp += _tp
            fp += _fp
            tn += _tn
            fn += _fn
                    
        print tp, fp, tn, fn
        print tp /float(length*rows)
        f1, pr, rc = f1_score(tp, fp, tn, fn)
        print f1, pr, rc
        
        res = net.model.evaluate([np.zeros_like(imagesTest), np.asarray(textTest)],
                                 [imagesTest, textTest],
                                 batch_size=1,
                                 verbose=0)
        print res 
        resultsDict = addRes2Dict(resultsDict, res, key, "text_only", epoch,
                                  (tp/float(length*rows))*100,
                                  (fp/float(length*rows))*100,
                                  (tn/float(length*rows))*100,
                                  (fn/float(length*rows))*100,
                                  f1,
                                  pr,
                                  rc)
                   
        K.clear_session()
        print 
        
    pkl.dump(resultsDict, open("%s/resDict.p"%resPath, "w"))     
    
    resultsDict = pkl.load(open("%s/resDict.p"%resPath, "r"))
    for fname in resultsDict.keys():
        print fname
        with open("%s/%s.txt"%(resPath, fname), "a") as txt:
            
            for i, _ in enumerate(resultsDict[fname]["bimodal"]):
                for cond in ["bimodal", "image_only", "text_only"]:
                    txt.write("%s"%cond)
                    avg = []
                    _list = resultsDict[fname][cond]
                    print _list
                    for j in _list:
                        print j
                        for k in j:
                            avg.append(k/numRuns)
                    
                    txt.write(
                    "%s \t loss \t %s \t"%(i,avg[0]) + \
                    "imageDecoder_loss \t %s \t"%avg[1] + \
                    "textDecoder_loss \t %s \n"%avg[2] + \
                    
                    #"multi_label_accuracy \t %s \t"%avg[4] + \
                    "true_positive_rate \t %s \t"%avg[3] + \
                    "false_positive_rate \t %s \t"%avg[4] + \
                    "true_negative_rate \t %s \t"%avg[5] + \
                    "false_negative_rate \t %s \t"%avg[6] + \
                    "f1_score \t %s \t"%avg[7] + \
                    "precsion \t %s \t"%avg[8] + \
                    "recall \t %s \n"%avg[9])   
                    

def representationAlgebra(net, templateImage, templateAttr, addImage, addAttr, subImage, subAttr):
        
        
        tRep = net.encoder.predict([np.asarray([templateImage]), np.zeros_like([templateAttr])])
        
        addRep = net.encoder.predict([np.asarray([addImage]), np.asarray([addAttr])])
        subRep = net.encoder.predict([np.asarray([subImage]), np.asarray([subAttr])])
        
        newRep = (tRep - subRep) + addRep
        
        
        newIm   = net.imageDecoder.predict(newRep)
        newAttr = net.textDecoder.predict(newRep)
        
        return newIm, newAttr 
    
def getNewAttrs(dl, originalAttr, attrName):
        newAttr = [0]*len(originalAttr)
        for atName in attrName.split():
            idx = dl.attrs[atName]
        #print idx
        #print originalAttr
            
            
        newAttr[idx] = 1 
                
        
        return np.asarray(newAttr)
                
    

    
def ask(net, dl, images, attrs, resPath):
        
        for i, (im, at) in enumerate(zip(images, attrs)):
            name = dl.convertOH2Name(at)
            print
            print name
            cv2.imshow("orig", im)
            
            print "Atrributes to select from: "
            print dl.attrs.keys()
            subName = raw_input("Enter an attribute to remove: ")
            subAttr = getNewAttrs(dl, at, subName)
            addName = raw_input("Enter an attribute to add: ")
            addAttr = getNewAttrs(dl, at, addName)
            
            print at
            print subAttr
            print addAttr
                  
            sumAttr = np.add(np.subtract(at, subAttr), addAttr).astype(np.int8)
            print np.ndarray.tolist(sumAttr)
            
            addImage = subImage =  np.zeros_like(im)
            
            #addRep = self.representationModel.predict([np.asarray([addImage]), np.asarray([addAttr])])
            #addImg, addAt = self.generatorModel.predict(addRep)
            
            newImg, newAttr = representationAlgebra(net, im, at, addImage, addAttr, subImage, subAttr)
            newAttr = np.squeeze(newAttr)
            rnd_atr = np.zeros_like(newAttr)
            rnd_atr[newAttr>=0.5] = 1
            newName = dl.convertOH2Name(rnd_atr)
            cv2.imshow("new", np.squeeze(newImg))
            cv2.waitKey()
            subSpace = np.concatenate((subImage, np.squeeze(newImg)))          
            origAdd = np.concatenate((im, addImage))
            toSave = np.concatenate((origAdd, subSpace), axis=1)
            print np.shape(toSave)

            cv2.imwrite("%s/generated/%s_%s_%s_%s.png"%(resPath, name, subName, addName, newName), (np.squeeze(toSave)*255).astype(np.uint8))
    

def argmaxGenerate(net, dl, respath, epoch):
    
    if not op.exists("%s/argmax/%s"%(respath,epoch)):
        os.mkdir("%s/argmax/%s"%(respath,epoch))
        
    atrs = ["medium", "small", "big",
            
            "brick", "duck", "donut",
            "cup", #"ball",
            "rectangle", #"pepper",
            
            "red",   
            "yellow", 
            "green",  "blue",
            "black"#,  "orange", 
            #"pink", "grey", "brown",
            #"purple"
            ] #"chocolate", 
            
            #"topLeft", "topCentre", "topRight",
            #"left",    "centre",    "right",
            #"btmLeft", "btmCentre", "btmRight",
            
            #"dark", "light"]
    for at1 in atrs:
        for at2 in atrs:
            print at1, at2
            
            oh = getNewAttrs(dl, np.ones(dl.numAttrs), at1)
            oh2 = getNewAttrs(dl, np.ones(dl.numAttrs), at2)
            if at1 != at2:
                oh = oh + oh2
                
            im, newAttr = net.model.predict([np.zeros((1, 64,64,3)), np.asarray([oh])])
            newAttr = np.squeeze(newAttr)
            rnd_atr = np.zeros_like(newAttr)
            rnd_atr[newAttr>=0.5] = 1
            newName = dl.convertOH2Name(rnd_atr)
            #cv2.imshow("%s_%s"%(at,newName), (np.squeeze(im)*255).astype(np.uint8))
            #cv2.waitKey()
            cv2.imwrite("%s/argmax/%s/%s_%s.png"%(respath, epoch, at1, at2), (np.squeeze(im)*255).astype(np.uint8))
            
        
def main():
    train = True
    useExemplars = True
    extraInputs = False
    name = "mostExemplar"
    numRuns = 4
    checkPath = "/home/eli/Documents/data/realShapes/checkpoints/mostExemplar"
    resPath =   "/home/eli/Documents/data/realShapes/results/mostExemplar"
    
    
    net = NET(batch_size=5, image_shape=(64,64,3), text_shape=(20,))
    dl = DataLoader(image_shape=(64,64), sampsPerObj=450)
    
    
    if extraInputs:
        net.buildAutoEncSmallExtraInputs()
        net.compileAEEmbTarCost()
    else:
        net.buildAutoEncSmall()
        net.compileAE()
        
    net.model.summary()

    
    atrs = ["medium", "small", "big",
            
            "brick", "duck", "donut",
            "cup", #"ball",
            "rectangle", #"pepper",
            
            "red",   
            "yellow", 
            "green",  "blue",
            "black",  #"orange", 
            #"pink", "grey", "brown",
            #"purple", #"chocolate", 
            
            "topLeft", "topCentre", "topRight",
            "left",    "centre",    "right",
            "btmLeft", "btmCentre", "btmRight",
            
            "dark", "light"]
    
    dl.trainedAtrs = atrs
    
    disallowed = [["chocolate", "donut"], ["huge", "duck"], ["pink", "donut"], ["yellow", "pink", "duck"]]
    
    
    if train:
        tr_im, tr_atrs, tr_nms = dl.loadSubset(atrs, "train", disallowed)
        dl.printStats(tr_im, tr_atrs, "train")
        
        tr_im_conc = np.concatenate((tr_im, np.zeros_like(tr_im), tr_im), axis=0)
        tr_atr_conc = np.concatenate((tr_atrs, tr_atrs, np.zeros_like(tr_atrs)), axis=0)
        
        if not useExemplars:
            tr_im_tar = np.concatenate((tr_im, tr_im, tr_im), axis=0)
            tr_atr_tar = np.concatenate((tr_atrs, tr_atrs, tr_atrs), axis=0)
                
        else:
            exemplars = dl.loadExemplars(tr_nms)
            tr_im_tar = np.concatenate((tr_im, exemplars, tr_im), axis=0)
            tr_atr_tar = np.concatenate((tr_atrs, tr_atrs, tr_atrs), axis=0)
            
        if extraInputs:
            inputs = [tr_im_conc, tr_atr_conc, tr_im_tar, tr_atr_tar]
            
        else:
            inputs = [tr_im_conc, tr_atr_conc]
        
        targets = [tr_im_tar, tr_atr_tar]
        
        vl_im, vl_atrs, vl_nms = dl.loadSubset(atrs, "val", disallowed)
        dl.printStats(vl_im, vl_atrs, "val")
        
        vl_im_conc = np.concatenate((vl_im, np.zeros_like(vl_im), vl_im), axis=0)
        vl_atr_conc = np.concatenate((vl_atrs, vl_atrs, np.zeros_like(vl_atrs)), axis=0)
        exemplars_vl = dl.loadExemplars(vl_nms)
        
        if not useExemplars:
            vl_im_tar = np.concatenate((vl_im, vl_im, vl_im), axis=0)
            vl_atr_tar = np.concatenate((vl_atrs, vl_atrs, vl_atrs), axis=0)
        
        else:
            vl_im_tar = np.concatenate((vl_im, exemplars_vl, vl_im), axis=0)
            vl_atr_tar = np.concatenate((vl_atrs, vl_atrs, vl_atrs), axis=0)
            
        if extraInputs:
            vl_inputs = [vl_im_conc, vl_atr_conc, vl_im_tar, vl_atr_tar]
        
        else:
            vl_inputs = [vl_im_conc, vl_atr_conc]
        vl_targets = [vl_im_tar, vl_atr_tar]
    
    
        for run in range(numRuns):
            trainNet(net, inputs, targets, vl_inputs, vl_targets, run, checkPath, name)
    
    run_tests(net, dl, checkPath, resPath, numRuns, atrs, disallowed)
    
    
def makeAllCombos(dl, sizes=["medium", "small", "big"],\
                   
                   objects=["brick", "duck", "donut",
                            "cup", #"ball",
                            "rectangle", #"pepper"
                            ],\
                  
                  colours=["red",
                           "yellow", "green",             
                           "blue",
                           "black",  #"orange", 
                           #"pink", "grey", "brown",
                           #"purple"
                           ]):
    
    attrs = []
    for size in sizes:
        for obj in objects:
            for col in colours:
                
                fname = "%s_%s_%s.png"%(size,obj,col)
                print fname
                x = dl.convertName2OH(fname)
                print dl.convertOH2Name(x)
                attrs.append(x)
    return attrs
    
if __name__ == '__main__':
    
    atrs = ["medium", "small", "big",
            
            "brick", "duck", "donut",
            "cup", #"ball",
            "rectangle", #"pepper",
            
            "red",   
            "yellow", 
            "green",  "blue",
            "black",  #"orange", 
            #"pink", "grey", "brown",
            #"purple", #"chocolate", 
            
            "topLeft", "topCentre", "topRight",
            "left",    "centre",    "right",
            "btmLeft", "btmCentre", "btmRight",
            
            "dark", "light"]
    
    dl = DataLoader(image_shape=(64,64), sampsPerObj=450)
    dl.trainedAtrs = atrs
    
    disallowed = [["chocolate", "donut"], ["huge", "duck"], ["pink", "donut"], ["yellow", "pink", "duck"]]
    
        
    checkPath = "/home/eli/Documents/data/realShapes/checkpoints/mostExemplar/"
    resPath =   "/home/eli/Documents/data/realShapes/results/mostExemplar"
       
    net = NET(batch_size=1, image_shape=(64,64,3), text_shape=(20,))
    
    net.buildAutoEncSmall()
    net.compileAE()
    
    
    wfs = ["%s/mostExemplar_0_epoch_17_loss_0.0069_val_loss_0.0032.h5"%checkPath]#dl.getAllFiles(checkPath)
    vl_im, vl_atrs, vl_nms = dl.loadSubset(atrs, "test", disallowed)
    combo_atrs = makeAllCombos(dl)
    for wf in wfs:
        print "loading weights: %s"%wf
        net.model.load_weights("%s"%wf)
        print "loaded weights: %s"%wf
        epoch = int(wf.split("_")[3])
        print "epoch: %s"%epoch
        
        argmaxGenerate(net, dl, resPath, epoch)
        
        
        
        
        #generateImages(vl_im,vl_atrs, vl_atrs, net, dl, resPath, epoch, "bimodal")
        #generateImages(vl_im,np.zeros_like(vl_atrs), vl_atrs, net, dl, resPath, epoch, "image_only")
        generateImages(np.zeros_like(vl_im)[0:len(combo_atrs)],combo_atrs, combo_atrs, net, dl, resPath, epoch, "text_only")
       
    
        #vl_im = [cv2.resize(cv2.imread("/home/eli/Documents/data/realShapes/images/val/session1/medium_blue_rectangle_centre_4_50.png"),dsize=(128,128))/255.0]
        #vl_atrs = [dl.convertName2OH("big_black_brick_centre")]
    
    '''
    n = ""
    while n != "y":
        ask(net, dl, vl_im, vl_atrs, resPath)
        n = raw_input("exit? y/N")
    '''
    
    #main()
    
    
