import csv 
import os
import sys
from scipy import misc
from scipy.ndimage import rotate
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm 
sys.path.append('RAFT/core')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import OrderedDict
import cv2
import numpy as np
import torch
import imutils
from raft import RAFT
from utils import flow_viz
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from os import listdir
from os.path import isfile, join
import time
from tensorflow.keras.preprocessing import image


 ###################Inputs ##########################################
MIM_size=11
####################Loading EfficientNet-B0-based classifier##########
img_rows, img_cols = 150,150
CNN_model= load_model('./models/EfficientNetB0-medium-patches-12-RAFT.h5')
CNN_model.load_weights('./models/EfficientNetB0-medium-patches-12-RAFT.h5')
####################Classification##############################################

def classify(im):
    
    im = cv2.resize(im, (img_rows, img_cols ), interpolation = cv2.INTER_AREA)
    im = np.array([im]).reshape((1, 150, 150, 3))
    group=CNN_model.predict(im)
    if(group>=0.5):
        return 1
    else:
        return 0 

 ###########################Identifying patches borders################################
def patchBorder(rows, cols, left_top_x,left_top_y,right_bottom_x, right_bottom_y ):
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        
        width=right_bottom_x-left_top_x
        height=right_bottom_y-left_top_y
        for i in range(rows):
            for j in range(cols):
                patch_width=int(width/cols)
                patch_height=int(height/rows)
            
                left_x=left_top_x+patch_width*j
                right_x=left_top_x+patch_width*(j+1)
                left_y=left_top_y+patch_height*i
                right_y=left_top_y+patch_height*(i+1)
                
                x1.append(left_x)
                x2.append(right_x)
                y1.append(left_y)
                y2.append(right_y)
        if(x1[0]>x2[0]):
            return(x2,y1,x1,y2)
        else:
            return(x1,y1,x2,y2)
                         
###########################################Labeling patches##########################
def label(mim,name):
    rec=[name]
    for patch in range (0,len(x1)):
         mimPatch=mim[int(y1[patch]):int(y2[patch]),int(x1[patch]):int(x2[patch])]
          
         if classify(mimPatch)==1:
            cate=1
         else:
            cate=0
         rec.append(cate)
    return (rec)
##############################################################################################
def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame

######################################False Reduction Approach ##############################
def falseRed(a):
    for i in range(len(a)-2):
    
        if  a[i]!=a[i+1] and  np.count_nonzero(a[i+2:i+5]==a[i])>1:
                if a[i+1]==1:
                    a[i+1]=0    
                else:
                    a[i+1]=1

    #First 2 Clips
    if  np.count_nonzero(a[0:3]==a[0])!=3:

        if np.count_nonzero(a[2:5]==a[1])==0:
                if a[1]==1:
                        a[1]=0
                else:
                        a[1]=1

        if np.count_nonzero(a[1:4]==a[0])==0:
                if a[0]==1:
                        a[0]=0
                else:
                        a[0]=1                         

    #Last two clips
    l=len(a)

    if  a[l-1]!=a[l-2] and a[l-1]!=a[l-3]:
            if a[l-1]==1:
                a[l-1]=0
            else:
                a[l-1]=1
    elif   a[l-1]!=a[l-2] and a[l-1]==a[l-3]:
            if a[l-2]==1:
                a[l-2]=0
            else:
                a[l-2]=1
    elif a[l-1]==a[l-2] and  a[l-1]not in a[l-5:l-2]:
            if a[l-1]==1:
                a[l-1]=0
                a[l-2]=0
            else:
                a[l-1]=1 
                a[l-2]=1 

    return a

#######################Visualization##########################################################
def vizualize_flow(img, flo, counter,roi):
    
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    counter=counter- MIM_size
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)
    ###########################################
    rec=label(flo,counter)
    
    return rec

######################################naming images#########################
def img_name(name):
        name=str(name)
        if len(name)==1:
             name="00000"+name
        elif len(name)==2:
             name="0000"+name
        elif len(name)==3:
             name="000"+name
        elif len(name)==4:
             name="00"+name
        elif len(name)==5:
             name="0"+name
                
        return name
############################################# Annotation Visualization ########################

def visualize_label(file_name,roi):

    fileL = open(file_name,encoding='utf-8-sig')
    arr1 = np.loadtxt(fileL, delimiter=",")
    name= ( file_name.split("/")[-1]).split(".")[0]

    new_arr=np.zeros((len(arr1[:,0]), (len(arr1[0,:])*2)-1))
    new_arr[:,0]=arr1[:,0]
    to=len(x1)+1
    new_arr[:,1:to]=arr1[:,1:to]
    shift=len(x1)
    for col in range (1,len(x1)+1):
       new_arr[:,(col+shift)]=falseRed(arr1[:,(col)])
     
    np.savetxt("./Outputs/"+name+"-result.csv", new_arr  , delimiter=',',fmt="%d")

                 
    for row in new_arr:
        frame_name=str(int(row[0]))+".png"
        first_frame="./Outputs/frames/"+frame_name
        first_frame=cv2.imread(first_frame)
       
        for patch in range (0,len(x1)):
            if int(row[(patch+to)])==1: 
                #cv2.putText(first_frame, str(patch+1), (int(x1[patch])+5,int(y1[patch])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                cv2.rectangle(first_frame, (int(x1[patch]),int(y1[patch])), (int(x2[patch]),int(y2[patch])), (0,0,255),2)
        cv2.putText(first_frame,  "Frame: "+str(int(row[0])), (int(x1[0]),int(y1[0]+20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.imwrite("./Outputs/annotatedFrames/"+img_name(int(row[0]))+".png",first_frame[int(roi[1]-5):int(roi[3]+5),int(roi[0]-5):int(roi[2]+5)])
    fileL.close()

######################################################################################################
def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model

###################################Scaling and resizing the frames and ROI######################################
def new_dim(w,h,ratio, roi):
    new_w=int(w*ratio)
    new_w=new_w -(new_w%24)
    new_h=int(h*ratio)
    new_h=new_h -(new_h%24)
    
    roi[0]=int(roi[0]*ratio)
    roi[1]=int(roi[1]*ratio)
    roi[2]=int(roi[2]*ratio)
    roi[3]=int(roi[3]*ratio)

    return new_w,new_h,roi

#######################################Rotating point######################
def point_rotate(image, xy, angle):
    im_rot = rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return  new+rot_center


#######################################Generation video#####################
def generate_video(video_name ,frame_dir,fps):
    images = [img for img in sorted(os.listdir(frame_dir))
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]
    dur=fps/MIM_size 
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'),  dur, (width, height)) 

    for image in images: 
            video.write(cv2.imread(os.path.join(frame_dir, image))) 
            
    video.release()

######################################Empty folder###################################
def empty(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
 
def inference(args):

    
    #################pretrained RAFT weights#######################################
    model = RAFT(args)
    pretrained_weights = torch.load("./models/raft-sintel.pth",map_location ='cpu')
    ################################################################################
     
    device = "cpu"
    pretrained_weights = get_cpu_model(pretrained_weights)
    model.load_state_dict(pretrained_weights)
    model.eval()
    video_path = args.video
   
    # capture the video and get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame_1 = cap.read()
    #################################Information about input video###############
    h = frame_1.shape[0]
    w = frame_1.shape[1]
    fps = cap.get(cv2.CAP_PROP_FPS)
    ##########################################
    ##################New dim, roi, patches borders#################
    
    new_w,new_h,new_roi=new_dim(w,h,args.ratio, args.roi)
    
    dim=(new_w,new_h)
    frame_1= cv2.resize(frame_1, dim, interpolation = cv2.INTER_AREA)
    left_top_roi_corner=np.array([new_roi[0],new_roi[1]])
    right_bottom_roi_corner=np.array([new_roi[2],new_roi[3]])
    rotated_left_left=point_rotate(frame_1, left_top_roi_corner, args.angle)
    rotated_right_bottom=point_rotate(frame_1, right_bottom_roi_corner, args.angle)

    left_top_x=int(rotated_left_left[0])
    left_top_y=int(rotated_left_left[1])
    right_bottom_x=int(rotated_right_bottom[0])
    right_bottom_y=int(rotated_right_bottom[1])
    roi=[left_top_x,left_top_y,right_bottom_x, right_bottom_y]
    if roi[0]>roi[2]:
        swap=roi[0]
        roi[0]=roi[2]
        roi[2]=swap

    if roi[1]>roi[3]:
        swap=roi[1]
        roi[1]=roi[3]
        roi[3]=swap
    
    rows=args.patch[0]
    cols=args.patch[1]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    global x1,x2,y1,y2
    x1,y1,x2,y2=patchBorder(rows, cols, roi[0],roi[1],roi[2], roi[3] )
    ####################################################################
    print(5*"\n")
    print("**************************************************************************")
    print("*****Automatic Deep learning framework for detecting pushing behavior*****")
    print("**************************************************************************")
    print("\n")
    
    #########################Basic Information################################
    print ("-------------------------------------------------------------------------")
    print ("Basic Information")
    print ("-------------------------------------------------------------------------")
    print("Video name: ", args.video)
    print("Duration (S) :" + str(frame_count/fps))
    print("Number of frames: ",frame_count )
    print("Frame rate per second:", fps)
    print("Original width: ", w)
    print("Original height: ", h)
    print("Bounding box of original ROI: ",args.roi)
    print("Ratio of resizing: ",args.ratio)
    print("Angle of rotation: ",args.angle)
    print("New width: ", new_w)
    print("New height: ", new_h)
    print("Bounding box of new ROI: ",roi)
    print("Number of patches: ",str(rows*cols))
    print ("-------------------------------------------------------------------------")
    
    #############################################################################
    video_name= (args.video.split("/")[-1]).split(".")[0]
    file_name=video_name+"-labels.csv"
    filePath="./Outputs/"+ file_name
    f = open( filePath, 'w',newline="")
    writer = csv.writer(f )
    generated_video_name= video_name+".mp4"
    frame_dir="./Outputs/frames/"
    empty(frame_dir)
    annotatedFrame_dir="./Outputs/annotatedFrames/"    
    start = time.time()
    #frames extraction and preprocessing
    
    frame_1 = rotate(frame_1,args.angle) 
    cv2.imwrite("./Outputs/frames/0.png", frame_1) 
    frame_1 = frame_preprocess(frame_1, device)
    
    counter = 1
    
    with torch.no_grad():
      
        for fr in tqdm(range(0,frame_count)):
             

           # read the next frame
            ret, frame_2 = cap.read()
            if not ret:
                f.close() 
                empty(annotatedFrame_dir )
                visualize_label(filePath,roi)
                generate_video(generated_video_name,annotatedFrame_dir,fps)
                end = time.time()
                print(f"Runtime of the program is {end - start}")   
               
                break
            # read the next frame
            
            
            if (counter)% MIM_size==0: 
                
                # preprocessing
                frame_2= cv2.resize(frame_2, dim, interpolation = cv2.INTER_AREA)
                frame_2 = rotate(frame_2,args.angle) 
                cv2.imwrite("./Outputs/frames/"+str(counter)+".png", frame_2)
                frame_2 = frame_preprocess(frame_2, device)
                #predict the flow
               
                flow_low, flow_up = model(frame_1, frame_2, iters=20, test_mode=True)
                
                # transpose the flow output and convert it into numpy array
                row= vizualize_flow(frame_1, flow_up,  counter,roi)
                writer.writerow(row)

                #if not ret:
                    #break
               
                frame_1 = frame_2   
                
            counter += 1


def main():
    
    parser = ArgumentParser()
    parser.add_argument("--model", help="CNN-based classifier")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--video", type=str, default="./videos/crowd.mp4")
    parser.add_argument("--save", action="store_true", help="save demo frames")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--angle", type=float, default=1, help="Baseline direction from left to right")

    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument('--roi','--list', nargs='+',type=int, help='<Required> Set flag', required=True)
    parser.add_argument('--patch','--list1', nargs='+',type=int, help='<Required> Set flag', required=True)


    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
