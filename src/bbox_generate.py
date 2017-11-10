import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import pickle as pk

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pk.load(f)
    
class rectangle(object):
    def __init__(self,x=0,y=0,w=0,h=0):
        if isinstance(x,(tuple,list)):
            x,y,w,h=x
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x+w
        self.y2 = y+h
        
    def area(self):
        return self.w*self.h if self.w>0 and self.h>0 else 0
    
    def perimeter(self):
        return 2*(self.w+self.h) if self.w>0 and self.h>0 else 0
    
    def overlap(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        x = max(self.x, rect.x)
        y = max(self.y, rect.y)
        w = min(self.x2, rect.x2) - x
        h = min(self.y2, rect.y2) - y
        return rectangle(x,y,w,h)
    
    def boundrect(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        x = min(self.x, rect.x)
        y = min(self.y, rect.y)
        w = max(self.x2, rect.x2) - x
        h = max(self.y2, rect.y2) - y
        return rectangle(x,y,w,h)
    
    def overlap_IoBB(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        return self.overlap(rect).area()/self.area()
    
    def overlap_IoU(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        return self.overlap(rect).area()/(self.area()+rect.area()-self.overlap(rect).area())
        
    def __repr__(self):
        return "rectangle({},{},{},{})".format(self.x,self.y,self.w,self.h)
    

path='/home/shared_data/chest_xray8/'

files=os.listdir(join(path,'cams_rn'))
for f in files:
    print(f)
    file=join(path,'cams_rn',f)
    cam=load_obj(file)
    ret,mask=cv2.threshold(cam, cam.max()*0.5, 255, cv2.THRESH_BINARY)
    x,contour,hie=cv2.findContours(mask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE )
    img=cv2.imread(join(path,'bbox',f[:-4]))
    rect=[]
    for cnt in contour:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h >= 100:
            rect.append((x,y,w,h))
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
            cv2.imwrite(join(path,'bbox_rn',f[:-4]),img)
        
    save_obj(rect,join(path,'rect_rn',f[:-4]))
    

classes = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2, 'Infiltrate':3, \
                        'Mass':4, 'Nodule':5, 'Pneumonia':6, 'Pneumothorax':7, \
                        'Consolidation':8, 'Edema':9, 'Emphysema':10, 'Fibrosis':11, \
                        'Pleural_Thickening':12, 'Hernia':13   }
bbox = pd.read_csv(join(path,'BBox_list_2017.csv'))
bbox['rect'] = bbox.iloc[:,[2,3,4,5]].apply(lambda x: tuple(x),axis=1)
bbox = bbox[['Image Index','Finding Label','rect']]

def perform(df):
    cnt=load_obj(join(path,'rect_rn','label{}_{}.pkl'.format(classes[df['Finding Label']],df['Image Index'])))
    df['detect_num']=len(cnt)
    df['iou']=[rectangle(df['rect']).overlap_IoU(c) for c in cnt]
    df['iobb']=[rectangle(df['rect']).overlap_IoBB(c) for c in cnt]

    df['best_match']=cnt[max(range(len(cnt)),key=lambda x: df['iou'][x])]
    return df

bdresult = bbox.apply(perform,axis=1)

thres = np.linspace(0,0.9,10)

temp=bdresult['iobb'].apply(lambda x: np.array(x)<thres[0])
bdresult['match']=temp.apply(lambda x: not x.all())
bdresult['fp']=temp.apply(lambda x: x.sum())

res = bdresult.groupby('Finding Label')['match','fp'].mean()



    