# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 07:10:13 2023

@author: fmizt
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
import shelve
#import time

def get_frame_index(filepath,index:int,save=False,show=0):
    movie = cv2.VideoCapture(filepath)
    movie.set(cv2.CAP_PROP_POS_FRAMES, index) 
    ret, frame=movie.read()
    if type(show)==int:
        cv2.imshow('first', frame)
        if cv2.waitKey(show) & 0xFF == ord('q'):
            cv2.destroyWindow('first')

    else:
        pass
        
    if save:
        cv2.imwrite(save, frame)
    return frame

def HSVonMouse(event, x, y, flags, params):   
    img=params["img"]
    if event == cv2.EVENT_LBUTTONDOWN:#左クリックがされたら
        h = img[[y], [x]].T[0].flatten().mean()#
        s = img[[y], [x]].T[1].flatten().mean()
        v = img[[y], [x]].T[2].flatten().mean()             
        print("H: {}, S: {}, V: {}".format(h, s, v))

def get_HSVonMouse(img):
    param={"img":img}
    cv2.imshow('first', img)
    cv2.setMouseCallback('first', HSVonMouse,param)
    cv2.waitKey(0)

def make_framemask(img,lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if len(lower)!=len(upper):
        raise Exception("length of lower and upper for mask should be the same")
    else:
        frame_mask =sum([cv2.inRange(hsv, lower[i], upper[i]) for i in range(len(lower))])
    return frame_mask

def check_color_param(img,lower, upper,save=False):
    frame_mask=make_framemask(img,lower, upper)
    dst = cv2.bitwise_and(img, img, mask=frame_mask)
    resized_dst= cv2.resize(dst, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow("img", resized_dst)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(save, dst)

def get_moment(contour):
    M = cv2.moments(contour)
    return int(M["m10"]/M["m00"]) , int(M["m01"]/M["m00"])

def get_saturation(img,contour):
    black=np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(black, [contour], -1, 255, -1)
    xys=np.where(black!=0)
    sat=[img[[xys[0][i]], [xys[1][i]]].T[1].flatten().mean() for i in range(len(xys[0]))]
    return np.mean([s for s in sat if not np.isnan(s)])

def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

def get_onMouse(event, x, y, flags, params):
    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            params["points"].append((x,y))
    except KeyError as e:
        print("list type variable \"points\" must be defined")
        raise e

def check(img,contours,save=False):
    frame=img.copy()
    if type(contours[0])==list:
        def hsv_to_rgb(h, s, v):
            bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
            return (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        delta=int(np.floor(180/len(contours)))
        for k in range(len(contours)):
            for i in range(len(contours[k])):
                cv2.polylines(frame, contours[k][i], True, hsv_to_rgb(0+delta*k,255,255), 5)
    else:
        for c in contours:
            cv2.polylines(frame, c, True, (0, 255, 0), 5)
    
    cv2.imshow("img", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(save, frame)
        
def circularity(contour):
    '''
    円形度を求める

    Parameters
    ----------
    contour : ndarray
        輪郭の(x,y)座標の配列

    Returns
    -------
        円形度

    '''
    # 面積
    area = cv2.contourArea(contour)
    # 周囲長
    length = cv2.arcLength(contour, True)
    if length==0:
        check(img,contour)

    # 円形度を返す
    return 4*np.pi*area/length/length

def circularity_plot_standard(contour):
    x = np.linspace( 0, 1, 100)
    y = ((math.pi)**2)*x/(((math.pi)*x+1)**2)
    plt.plot(x, y)
    plt.set_title("circularity of x=1/q circle")
    plt.show()


def subtracted_ratio(fgm,contour):
    cnt_img = np.zeros(fgm.shape[:2], dtype=np.uint8)
    cv2.drawContours(cnt_img, [contour], -1, 255, -1)
    #fgm_=cnt_img*fgm*255
    #cv2.imshow("img", fgm_)
    #if cv2.waitKey(0) & 0xFF == ord('q'):
        #cv2.destroyAllWindows()
    if np.sum(cnt_img/255)==0:
        return 0
    else:
        return  np.sum(cnt_img*fgm)/np.sum(cnt_img/255)

def get_marker_centroid(img,fgmk,lower,upper,ref_point,thresh=None,scope=False,cir=0.1,area=5,rad=15,sat_thresh=100,animal="finch"):
    
    """
    """"""
    function to get marker centers combining MoG and HSV
    """"""
    img : raw color image from captured video
    
    fgmk : binary image representing bird shillhouette (0/255)
    
    lower : lower limits for hsv color feature extraction on color marker. 
            In np.array([[H0,S0,V0],[H1,S1,V1]...]) format for lower limit0,1,... 0<=H<=180,0<=S<=255,0<=V<=255
    
    upper : upper limits for hsv color feature extraction on color marker. 
            In np.array([[H0,S0,V0],[H1,S1,V1]...]) format for upper limit0,1,... 0<=H<=180,0<=S<=255,0<=V<=255
    
    ref_point : the center of the most recent marker positions 
                to be used in the restriction of searching area and selection of marker contours
    
    thresh : half lenth (pixcel) of sides of square centered on the 'ref_point' to restrict the searching area
                to be used in the restriction of searching area and selection of marker contours
    
    scope : Bool. If True, searching area is restricted to the square area centered on 'ref_point' of length 'thresh'*2.1
    
    cir : 0-1. minimum threshold for filtering marker contours by circularity
    
    area : pixcel. minimum threshold for filtering marker contours by area
    
    rad : pixcel. maximum threshold for filtering marker contours by radius
    
    sat_thresh : 0-255. minimum threshold for filtering marker contours by saturation value
    """
    
    if thresh==None:
        thresh=max(img.shape[:2])
    if scope:
        #print(scope)
        if type(scope)==bool:
            scope=(sorted([math.floor(thresh*2.1),math.ceil(thresh*2.1)],key=lambda x:x%2)[0],sorted([math.floor(thresh*2.1),math.ceil(thresh*2.1)],key=lambda x:x%2)[0])
        else:
            pass
            #if not all([s%2==0 for s in scope]):
                #raise ValueError("scope shold be composed of odd numbers")
        x0=math.floor(max([0,ref_point[0]-scope[0]//2]))
        x1=math.ceil(min([img.shape[:2][1],ref_point[0]+scope[0]//2]))
        y0=math.floor(max([0,ref_point[1]-scope[1]//2]))
        y1=math.ceil(min([img.shape[:2][0],ref_point[1]+scope[1]//2]))
        #print([v0,v1,w0,w1])
        frame=img[y0:y1+1,x0:x1+1]
        fgm=fgmk[y0:y1+1,x0:x1+1]
        #ref_point_=([ref_point[0],ref_point[0]-scope[0]//2][max(1,v0)//v0],[ref_point[1],ref_point[1]-scope[1]//2][max(1,w0)//w0])
        #ref_point=(v1-scope[0]//2,w1-scope[1]//2)
        ref_point=(ref_point[0]-x0,ref_point[1]-y0)
    else:
        frame=img.copy()
        fgm=fgmk.copy()
        
    frame_b = cv2.blur(frame, (5, 5))
    hsv = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
    if len(lower)!=len(upper):
        raise Exception("length of lower and upper for mask should be the same")
    else:
        frame_mask =sum([cv2.inRange(hsv, lower[i], upper[i]) for i in range(len(lower))])
    #dst = cv2.bitwise_and(img, img, mask=frame_mask)
    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours1 = list(filter(lambda x: cv2.contourArea(x) >= area, contours))
    contours2=list(filter(lambda x: circularity(x) >= cir, contours1))#理論的に金網によって球が最大4分割される。しかし輪郭の不正確さや羽による日隠も考慮して0.4にする 低めでもいいかも
    #contours3 = list(filter(lambda x: cv2.contourArea(x) >= 50, contours2))
    #contours4=list(filter(lambda x: cv2.minEnclosingCircle(x)[1] <=rad, contours3))
    contours4=list(filter(lambda x: cv2.minEnclosingCircle(x)[1] <=rad, contours2))
    #contours5=sorted(contours4,key=lambda x:subtracted_ratio(fgm,x))#sorted, 値が同じときの対処
    #contours4=list(filter(lambda x: get_saturation(frame,x[0])<sat_thresh, contours4))
    #paper fig
    #with open(r'I:\motion_paper\figures\230815\{}.txt'.format("-".join(list(map(str,list(lower[0]))))), mode='a') as f:
        #f.write("{}\n".format(str(len(contours4))))
    
    contours5=sorted([(s,subtracted_ratio(fgm,s)) for s in contours4],key=lambda x:(x[1],-sum([(ref_point[i]-s)**2 for i, s in enumerate(get_moment(x[0]))])),reverse=True)
    center,radius=(None,None),None
    if len(contours5)>0:
        if contours5[0][1]<0.1:
            black = np.zeros(frame_mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(black, contours1, -1, 255, -1)
            contours_, _ = cv2.findContours((fgm*black)*255, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours1_ = list(filter(lambda x: cv2.contourArea(x) >= area, contours_))
            contours2_=list(filter(lambda x: circularity(x) >= cir, contours1_))#理論的に金網によって球が最大4分割される。しかし輪郭の不正確さや羽による日隠も考慮して0.4にする 低めでもいいかも
            #contours3_ = list(filter(lambda x: cv2.contourArea(x) >= 50, contours2_))
            #contours4_=list(filter(lambda x: cv2.minEnclosingCircle(x)[1] <=rad, contours3_))
            contours4_=list(filter(lambda x: cv2.minEnclosingCircle(x)[1] <=rad, contours2_))
            #contours4_=list(filter(lambda x: get_saturation(frame,x[0])<sat_thresh, contours4_))
            contours5=sorted([(s,subtracted_ratio(fgm,s)) for s in contours4+contours4_],key=lambda x:(x[1],-sum([(ref_point[i]-s)**2 for i, s in enumerate(get_moment(x[0]))])),reverse=True)

        if len(contours5)>1 and contours5[0][1]-contours5[1][1]<0.1:
            #global C5
            #C5=contours5.copy()
            contours5=sorted(contours5[:2],key=lambda x:get_saturation(frame,x[0]),reverse=True)
            #print("a")
            if get_saturation(frame,contours5[0][0])>sat_thresh and get_saturation(frame,contours5[1][0])>sat_thresh:
                contours5=sorted(contours5[:2],key=lambda x:sum([(ref_point[i]-s)**2 for i, s in enumerate(get_moment(x[0]))]))
        if contours5[0][1]<0.1 or get_saturation(frame,contours5[0][0])<sat_thresh*0.6:
            #print("b")
            """
            cv2.imshow("img", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            """
            if animal=="mice":
                center, radius = cv2.minEnclosingCircle(contours5[0][0])
                if scope:  
                    center=(center[0]+x0,center[1]+y0)
                else:
                    #print(ref_point)
                    pass
            return center, radius
        center, radius = cv2.minEnclosingCircle(contours5[0][0])
        if scope:  
            center=(center[0]+x0,center[1]+y0)
        else:
            #print(ref_point)
            pass
            #raise Exception("thresh might not be appropriate. bird moved faster?? or color was not detected")
    """
    fgc1, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    fgc2=sorted(fgc1,key=lambda x:(get_moment(x)[0]-center[0])**2+(get_moment(x)[1]-center[1])**2)
    ellipse = cv2.fitEllipse(fgc2[0])
    cv2.ellipse(frame,ellipse,(255,0,0),2)
    cv2.imshow("img", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    """
    return center, radius

def track(filepath,colors,thresh=None,scope=False,first_ind=0,first_points="manual",bird_area=600,cir=0.1,area=5,rad=15,sat_thresh=100,mog_thresh=300,weight=1,animal="finch"):
    
    """
    """"""
    function to track markers on songle subject
    """"""
    
    filepath : video file path on which marker tracking applied
    
    colors : sets of lower and upper limits for HSV color feature extraction for each marker.
            key name is used to represent the results ex) {marler1:{"lower":np.array([[H,S,V],[],...),"upper":np.array([[[H,S,V],[],...])}}
            secondly keys should include "lower" and "upper". see 'get_marker_centroid'
    thresh : see 'get_marker_centroid'
    
    scope : see 'get_marker_centroid'
    
    first_ind : first index to start the tracking
    
    first_points : sets of first positions for each marker.  ex) {marler0:(x0,y0), marler1:(x1,y1))
                   If "manual", you would click.
    
    bird_area : pixcel. minumum threshold for filtering MoG (bird shilhoutte) contours by area
    
    cir : see 'get_marker_centroid'
    
    area : see 'get_marker_centroid'
    
    rad : see 'get_marker_centroid'
    
    sat_thresh : see 'get_marker_centroid'
    
    mog_thresh : index to start using MoG backgrounf subtraction. 
    
    weight : weights on caluculating reference point(see 'get_marker_centroid') for each target marker
    
    """
    
    
    #results=pd.DataFrame(columns=["video_path","frame"]+sum([[s+"_x",s+"_y",s+"_r"] for s in colors.keys()],[])+["bird_x","bird_y"]+["prepare_time"]+[key+"_time" for key in colors.keys()])
    results=pd.DataFrame(columns=["video_path","frame"]+sum([[s+"_x",s+"_y",s+"_r"] for s in colors.keys()],[])+["bird_x","bird_y"])
    #movie = cv2.VideoCapture(r"F:\2022-11-17-08-48-08.mp4")"F:\2022-11-17-08-49-18.mp4"
    movie = cv2.VideoCapture(filepath)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #fgbg = cv2.createBackgroundSubtractorMOG2(history=60,detectShadows=False)
    #video = cv2.VideoWriter(r"F:\test_2022-11-17-08-48-08.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    #video = cv2.VideoWriter("/".join(re.split(r'[.\\]',filepath)[:3])+"/analysis/"+re.split(r'[.\\]',filepath)[3]+"_bird.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while True:
        try:
            ind=movie.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = movie.read()
            #st=time.perf_counter()
            if not ret: #or ind==200:
                break
            I={}
            if ind<first_ind:
                for ky in colors.keys():
                    results.loc[int(ind),[ky+"_x",ky+"_y",ky+"_r"]]=[None,None,None]
                results.iloc[int(ind),:2]=[filepath,ind]
                continue
            elif ind==first_ind:
                if first_points=="manual":
                    first_points={}
                    cv2.imshow('first', frame)
                    cv2.setMouseCallback('first', onMouse)
                    cv2.waitKey(0)
                    for ky in colors.keys():
                        first_points[ky]=tuple([int(s) for s in input("{}:: ex)1,2: ".format(ky)).split()])
                else:
                    first_points=first_points
                for ky in colors.keys():
                    results.loc[int(ind),[ky+"_x",ky+"_y",ky+"_r"]]=list(first_points[ky])+[None]
                results.iloc[int(ind),:2]=[filepath,ind]
                #ref_point=(np.mean([val[0] for val in first_points.values() if len(val)>1]),np.mean([val[1] for val in first_points.values() if len(val)>1]))
                xs=[val[0] for val in first_points.values() if len(val)>1]
                ys=[val[1] for val in first_points.values() if len(val)>1]
                xyi=0
            else:
                xyi=0
                while True:
                    #ref_point=(results.iloc[-1][key+"_x"],results.iloc[-1][key+"_y"])
                    xs=[t for t in list(results.iloc[-1-xyi][[s for s in list(results.columns) if s[-1]=="x"]]) if t!=None]
                    ys=[t for t in list(results.iloc[-1-xyi][[s for s in list(results.columns) if s[-1]=="y"]]) if t!=None]
                    if len(xs)==0 and len(ys)==0:#片方はありえない
                        xyi+=1
                    else:
                        break
                #ref_point=(np.mean(xs),np.mean(ys))
                #print(ref_point)
            #ref_point=(min([ref_point[0],frame.shape[1]]),min([ref_point[1],frame.shape[0]]))
            
            fgmask = fgbg.apply(frame)
            fgc, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            #fg=frame.copy()
            fg=np.stack([fgmask,fgmask,fgmask], 2)
            for i in range(len(fgc)):
                cv2.polylines(fg, fgc[i], True, (255,255,255), 3) 
            fgm_b = cv2.blur(fg, (5, 5))
            fgm_mask =cv2.threshold(fgm_b[:,:,0], 0, 255, cv2.THRESH_BINARY)[1]
            fgc1, _ = cv2.findContours(fgm_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area, fgc1))
            #fgc2=sorted(list(filter(lambda x: cv2.contourArea(x) >= 600 and cv2.contourArea(x) < 3000, fgc1)),key=lambda y:cv2.contourArea(y),reverse=True)
            #ff=frame.copy()
            if len(fgc2)>0 and ind>mog_thresh:
                fgmk=np.zeros(fgmask.shape[:2], dtype=np.uint8)
                #cv2.drawContours(fgmk, fgc2, -1, 255, -1)
                fgc3=sorted(fgc2,key=lambda x:sum([(ref_point[i]-s)**2 for i, s in enumerate(get_moment(x))]))
                bpos=get_moment(fgc3[0])
                cv2.drawContours(fgmk, fgc3[:3], -1, 255, -1)
                #cv2.polylines(ff, fgc3[0], True, (0,255,0), 3)
 
            else:
                fgmk=np.full(fgmask.shape[:2], 255, dtype=np.uint8)
                bpos=(None,None)
            #I["prepare_time"]=time.perf_counter()-st
            #st=time.perf_counter()
            for key in colors.keys():
                ref_point=(np.mean([t for t in xs+[results.iloc[-1-xyi][key+"_x"] for wi in range((weight-1))] if t!=None]),np.mean([t for t in ys+[results.iloc[-1-xyi][key+"_y"] for wi in range((weight-1))] if t!=None]))
                ref_point=(min([ref_point[0],frame.shape[1]]),min([ref_point[1],frame.shape[0]]))
                #ref_point_c=tuple(results.iloc[-1][[s for s in list(results.columns) if s in [key+"_x",key+"_y"]]])
                #center, radius= get_marker_centroid(frame,colors[key]["lower"],colors[key]["upper"],ref_point,ref_point_c=ref_point_c,thresh=thresh*(1+xyi),scope=scope)
                #paper fig
                #if ind==460 and key=="top":
                    #return frame,fgmk,colors[key]["lower"],colors[key]["upper"],ref_point,thresh*(1+xyi),fgmask
                center, radius= get_marker_centroid(frame,fgmk,colors[key]["lower"],colors[key]["upper"],ref_point,thresh*(1+xyi),scope=scope,cir=cir,area=area,rad=rad,sat_thresh=sat_thresh,animal=animal)
                #I[key+"_time"]=time.perf_counter()-st
                #st=time.perf_counter()
                I[key+"_x"]=center[0]
                I[key+"_y"]=center[1]
                I[key+"_r"]=radius
                I["bird_x"]=bpos[0]
                I["bird_y"]=bpos[1]

                
            results.loc[len(results),:]=[filepath,ind]+[I[s] for s in list(results.columns)[2:]]
            #video.write(ff)
        except Exception as e:
            #return C5
            print(e)
            break
    #video.release()
    movie.release()
    return results

def track_mult(filepath,colors,thresh=None,scope=False,first_ind=0,first_points={"s0":{"tip0":(),"tip0":()},"s1":{"tip1":(),"tip1":()}},bird_area=600,cir=0.1,area=5,rad=15,sat_thresh=100,mog_thresh=300,weight=1,cnt_img=None,animal="finch"):
    """
    """"""
    function to track markers of multiple subject
    """"""
    see 'get_marker_centroid' and see 'track'
    
    first_points : sets of first positions for each marker for each subject.  
                    ex) {"s0":{"maeker0":(),"maeker1":()},"s1":{"maeker3":(),"maeker4":()}} where s0,s1 represents subject
                   If locations have been not specified, you would click but you shold put the structure.
    
    """
    
    results=pd.DataFrame(columns=["video_path","frame"]+sum([[s+"_x",s+"_y",s+"_r"] for s in colors.keys()],[])+sum([[s+"_x",s+"_y"] for s in first_points.keys()],[]))
    #movie = cv2.VideoCapture(r"F:\2022-11-17-08-48-08.mp4")"F:\2022-11-17-08-49-18.mp4"
    movie = cv2.VideoCapture(filepath)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #fgbg = cv2.createBackgroundSubtractorMOG2(history=60,detectShadows=False)
    #video = cv2.VideoWriter(r"F:\test_2022-11-17-08-48-08.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    #video = cv2.VideoWriter("/".join(re.split(r'[.\\]',filepath)[:3])+"/analysis/"+re.split(r'[.\\]',filepath)[3]+"_bird.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    sbj_dic={v:key0 for key0,val0 in first_points.items()  for v in val0}
    while True:
        try:
            ind=movie.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = movie.read()
            if not ret: #or ind==200:
                break
            I={}
            if ind<first_ind:
                for ky in colors.keys():
                    results.loc[int(ind),[ky+"_x",ky+"_y",ky+"_r"]]=[None,None,None]
                results.iloc[int(ind),:2]=[filepath,ind]
                continue
            elif ind==first_ind:
                if all(list(map(lambda x:not x,list(first_points.values())[0].values()))):#直してもいい
                    fp={}
                    cv2.imshow('first', frame)
                    cv2.setMouseCallback('first', onMouse)
                    cv2.waitKey(0)
                    for ky in colors.keys():
                        fp[ky]=tuple([int(s) for s in input("{}:: ex)1,2: ".format(ky)).split()])
                    first_points={key0:{key1:fp[key1] for key1 in val0.keys() } for key0,val0 in first_points.items()}
                else:
                    first_points=first_points
                for ky in colors.keys():
                    results.loc[int(ind),[ky+"_x",ky+"_y",ky+"_r"]]=list(first_points[sbj_dic[ky]][ky])+[None]
                results.iloc[int(ind),:2]=[filepath,ind]
                #ref_point=(np.mean([val[0] for val in first_points.values() if len(val)>1]),np.mean([val[1] for val in first_points.values() if len(val)>1]))
                xs={key0:[val[0] for val in first_points[key0].values() if len(val)>1] for key0 in first_points.keys()}
                ys={key0:[val[1] for val in first_points[key0].values() if len(val)>1] for key0 in first_points.keys()}
                xyi=0
            else:
                xyi=0
                while True:
                    #ref_point=(results.iloc[-1][key+"_x"],results.iloc[-1][key+"_y"])
                    if animal=="mice":
                        xs={key0:[t for t in list(results.iloc[-1-xyi][[s for s in list(results.columns) if s.split("_")[0] in list(first_points[key0].keys()) and s[-1]=="x"]]) if t!=None] for key0 in first_points.keys()}
                        ys={key0:[t for t in list(results.iloc[-1-xyi][[s for s in list(results.columns) if s.split("_")[0] in list(first_points[key0].keys()) and s[-1]=="y"]]) if t!=None] for key0 in first_points.keys()}
                    else:
                        xs={key0:[t for t in list(results.iloc[-1-xyi][[s for s in list(results.columns) if s.split("_")[0] in list(first_points[key0].keys())+[key0] and s[-1]=="x"]]) if t!=None] for key0 in first_points.keys()}
                        ys={key0:[t for t in list(results.iloc[-1-xyi][[s for s in list(results.columns) if s.split("_")[0] in list(first_points[key0].keys())+[key0] and s[-1]=="y"]]) if t!=None] for key0 in first_points.keys()}
                    if len(xs)==0 and len(ys)==0:#片方はありえない
                        xyi+=1
                    else:
                        break
                #ref_point=(np.mean(xs),np.mean(ys))
                #print(ref_point)
            #ref_point=(min([ref_point[0],frame.shape[1]]),min([ref_point[1],frame.shape[0]]))
            
            fgmask = fgbg.apply(frame)
            fgc, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            #fg=frame.copy()
            fg=np.stack([fgmask,fgmask,fgmask], 2)
            for i in range(len(fgc)):
                cv2.polylines(fg, fgc[i], True, (255,255,255), 3) 
            fgm_b = cv2.blur(fg, (5, 5))
            fgm_mask =cv2.threshold(fgm_b[:,:,0], 0, 255, cv2.THRESH_BINARY)[1]
            fgc1, _ = cv2.findContours(fgm_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area, fgc1))
            #fgmk=np.zeros(fgmask.shape[:2], dtype=np.uint8)
            #cv2.drawContours(fgmk, fgc2, -1, 255, -1)
            #cv2.imshow("img0",fgmk)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            
            #fgc2=sorted(list(filter(lambda x: cv2.contourArea(x) >= 600 and cv2.contourArea(x) < 3000, fgc1)),key=lambda y:cv2.contourArea(y),reverse=True)
            #ff=frame.copy()
            if len(fgc2)>0 and ind>mog_thresh:
                fgmk=np.zeros(fgmask.shape[:2], dtype=np.uint8)
                #cv2.drawContours(fgmk, fgc2, -1, 255, -1)
                fgc3={key0:sorted(fgc2,key=lambda x:sum([(ref_point[key0][i]-s)**2 for i, s in enumerate(get_moment(x))])) for key0 in first_points.keys()}
                bpos={key0:get_moment(fgc3[key0][0]) for key0 in fgc3.keys()}
                fgmk={key0:cv2.drawContours(np.zeros(fgmask.shape[:2], dtype=np.uint8), fgc3[key0][:3], -1, 255, -1) for key0 in fgc3.keys()}
                #cv2.polylines(ff, fgc3[0], True, (0,255,0), 3)
 
            else:
                fgmk={key0:np.full(fgmask.shape[:2], 255, dtype=np.uint8) for key0 in first_points.keys()}
                bpos={key0:(None,None) for key0 in first_points.keys()}
            #for kt,vt in fgmk.items():
                #cv2.imshow(kt,vt)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #cv2.destroyAllWindows()
            ref_point={key0:(np.mean(xs[key0]),np.mean(ys[key0])) for key0 in xs.keys()}
            ref_point={key0:(min([ref_point[key0][0],frame.shape[1]]),min([ref_point[key0][1],frame.shape[0]])) for key0 in xs.keys()}
            for key in colors.keys():
                rfp=(np.mean([t for t in xs[sbj_dic[key]]+[results.iloc[-1-xyi][key+"_x"] for wi in range((weight-1))] if t!=None]),np.mean([t for t in ys[sbj_dic[key]]+[results.iloc[-1-xyi][key+"_y"] for wi in range((weight-1))] if t!=None]))
                rfp=(min([rfp[0],frame.shape[1]]),min([rfp[1],frame.shape[0]]))
                #ref_point_c=tuple(results.iloc[-1][[s for s in list(results.columns) if s in [key+"_x",key+"_y"]]])
                #center, radius= get_marker_centroid(frame,colors[key]["lower"],colors[key]["upper"],ref_point,ref_point_c=ref_point_c,thresh=thresh*(1+xyi),scope=scope)
                #paper fig
                #if ind==2135 and key=="tip1":
                    #return frame,fgmk,colors[key]["lower"],colors[key]["upper"],ref_point,thresh*(1+xyi),fgmask
                if len(cnt_img)>0:
                    center, radius= get_marker_centroid(cv2.bitwise_and(np.stack([cnt_img,cnt_img,cnt_img],2),frame),cv2.bitwise_and(cnt_img,fgmk[sbj_dic[key]]),colors[key]["lower"],colors[key]["upper"],rfp,thresh*(1+xyi),scope=scope,cir=cir,area=area,rad=rad,sat_thresh=sat_thresh,animal=animal)
                else:
                    center, radius= get_marker_centroid(frame,fgmk[sbj_dic[key]],colors[key]["lower"],colors[key]["upper"],rfp,thresh*(1+xyi),scope=scope,cir=cir,area=area,rad=rad,sat_thresh=sat_thresh,animal=animal)
                I[key+"_x"]=center[0]
                I[key+"_y"]=center[1]
                I[key+"_r"]=radius
                I[sbj_dic[key]+"_x"]=bpos[sbj_dic[key]][0]
                I[sbj_dic[key]+"_y"]=bpos[sbj_dic[key]][1]

                
            results.loc[len(results),:]=[filepath,ind]+[I[s] for s in list(results.columns)[2:]]
            #video.write(ff)
            #for key0 in fgmk.keys():
                #cv2.imshow(key0,fgmk[key0])
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
        except Exception as e:
            #return C5
            #print(e)
            print(ind)
            raise(e)
            break
    #video.release()
    movie.release()
    return results

def track_out_movie(results,fileout,colors,filepath=None):
    if filepath==None:
        filepath=results.iloc[0]["video_path"]
    #movie = cv2.VideoCapture(r"F:\2022-11-17-08-48-08.mp4")"F:\2022-11-17-08-49-18.mp4"
    movie = cv2.VideoCapture(filepath)
    #video = cv2.VideoWriter(r"F:\test_2022-11-17-08-48-08.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    video = cv2.VideoWriter(fileout, cv2.VideoWriter_fourcc(*'mp4v'), movie.get(cv2.CAP_PROP_FPS), (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for i in range(1,len(results)):
        try:
            movie.set(cv2.CAP_PROP_POS_FRAMES, results.loc[i,"frame"]) 
            ret, frame=movie.read()
            if not ret:
                break
            for key in colors.keys():
                if key+"_r" in results.columns and results.loc[i,key+"_r"]!=None and not np.isnan(results.loc[i,key+"_r"]):
                    x_= [int(np.floor(results.loc[i,key+"_x"])),int(np.ceil(results.loc[i,key+"_x"]))][min(enumerate([results.loc[i,key+"_x"]-np.floor(results.loc[i,key+"_x"]),np.ceil(results.loc[i,key+"_x"])-results.loc[i,key+"_x"]]), key = lambda x:x[1])[0]]
                    y_= [int(np.floor(results.loc[i,key+"_y"])),int(np.ceil(results.loc[i,key+"_y"]))][min(enumerate([results.loc[i,key+"_y"]-np.floor(results.loc[i,key+"_y"]),np.ceil(results.loc[i,key+"_y"])-results.loc[i,key+"_y"]]), key = lambda x:x[1])[0]]
                    if key+"_r" in results.columns:
                        r_=[int(np.floor(results.loc[i,key+"_r"])),int(np.ceil(results.loc[i,key+"_r"]))][min(enumerate([results.loc[i,key+"_r"]-np.floor(results.loc[i,key+"_r"]),np.ceil(results.loc[i,key+"_r"])-results.loc[i,key+"_r"]]), key = lambda x:x[1])[0]]
                        cv2.circle(frame, (x_,y_), r_, colors[key]["color"], 3)
                    cv2.circle(frame, (x_,y_), 1, colors[key]["color"], -1)
            video.write(frame)
        except Exception as e:
            print(e)
            break
    video.release()
    movie.release()
    

#analysis

def nothing(x):
    pass

def draw(event,x,y,flags,param):
    b,g,r=255,255,255
    s = cv2.getTrackbarPos('Size','image')
    i = cv2.getTrackbarPos('0 : draw \n1 : erase','image')

    if event == cv2.EVENT_LBUTTONDOWN:
        param["drawing"] = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if i == 1:
            b,g,r = 0,0,0
        if param["drawing"] == True:
            cv2.circle(param["mask"],(x,y),s,(b,g,r),-1)
            if i==1:
                mask_re=param["mask"].copy()
                mask_re[:,:,0]=np.where(mask_re[:,:,0] == 0, 0, param["color"][0])
                mask_re[:,:,1]=np.where(mask_re[:,:,1] == 0, 0, param["color"][1])
                mask_re[:,:,2]=np.where(mask_re[:,:,2] == 0, 0, param["color"][2])
                ft=cv2.bitwise_and(param["frame"], param["frame"], mask=cv2.bitwise_not(param["mask"][:,:,0]))
                param["image"]=cv2.bitwise_or(ft, mask_re)
            else:
                cv2.circle(param["image"],(x,y),s,param["color"],-1)
    elif event == cv2.EVENT_LBUTTONUP:
        if i == 1:
            b,g,r = 0,0,0
        param["drawing"] = False
        cv2.circle(param["mask"],(x,y),s,(b,g,r),-1)
        if i==1:
            mask_re=param["mask"].copy()
            mask_re[:,:,0]=np.where(mask_re[:,:,0] == 0, 0, param["color"][0])
            mask_re[:,:,1]=np.where(mask_re[:,:,1] == 0, 0, param["color"][1])
            mask_re[:,:,2]=np.where(mask_re[:,:,2] == 0, 0, param["color"][2])
            ft=cv2.bitwise_and(param["frame"], param["frame"], mask=cv2.bitwise_not(param["mask"][:,:,0]))
            param["image"]=cv2.bitwise_or(ft, mask_re)
        else:
            cv2.circle(param["image"],(x,y),s,param["color"],-1)


def get_object_contours(path,i=0,mask=False):
    frame=get_frame_index(path,i,show=10)
    img = frame.copy()
    if type(mask)==bool and not mask:
        mask = np.zeros(frame.shape,np.uint8)

    param={"drawing":False,"color":(0,255,0),"frame":frame,"image":img,"mask":mask}

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask')
    #cv2.moveWindow("mask", 80,60)
    cv2.createTrackbar('Size', 'image',1,30,nothing)
    cv2.setMouseCallback('image',draw,param)
    #switch = '0 : draw \n1 : erase'
    cv2.createTrackbar('0 : draw \n1 : erase', 'image',0,1,nothing)
    while True:  
        cv2.imshow('image',param['image'])
        cv2.imshow('mask',param['mask'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rt,pm=cv2.threshold(param['mask'][:,:,0], 254,255, cv2.THRESH_TOZERO)
            param['mask']=np.stack([pm,pm,pm],axis=2)
            #return param['mask']
            contours, _ = cv2.findContours(param['mask'][:,:,0], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            print(len(contours))
            for i in range(len(contours)):
                c=contours[i]
                marked=param['image'].copy()
                cv2.polylines(marked, c, True, (0, 0, 255), 1)
                cv2.imshow("contours{}".format(str(i)), marked)
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    pass
                cv2.destroyWindow("contours{}".format(str(i)))
            x=input("continue:c,finish:index ::")
            if x=="c":
                continue
            else:
                param['contour']=[contours[int(x)]]
                break
    cv2.destroyAllWindows()
    if param["drawing"]:
        param["drawing"]=False
    return param


def linear(n1, n2):
    if n2[0]==n1[0]:
        return None,None
    a = (n2[1]-n1[1])/(n2[0]-n1[0])
    b = (n2[0]*n1[1]-n1[0]*n2[1])/(n2[0]-n1[0])

    return a, b
def cross_point(n1, n2, m1, m2):
    a1, b1 = linear(n1, n2)
    a2, b2 = linear(m1, m2)

    if a1 == a2:
        return None, None
    if a1==None:
        return n1[0],a2*n1[0]+b2
    if a2==None:
        return m1[0],a1*m1[0]+b1
    return (b2-b1)/(a1-a2), (a1*b2-b1*a2)/(a1-a2)
def calc_straight(img, n1, n2):
    if n1[0]==n2[0]:
        tp = (n1[0],0)
        bp = (n1[0],img.shape[0])
    else:
        top = [[0, 0], [img.shape[1], 0]]
        btm = [[0, img.shape[0]], [img.shape[1], img.shape[0]]]
    
        tp = cross_point(n1, n2, top[0], top[1])
        bp = cross_point(n1, n2, btm[0], btm[1])
        if tp[0]==None and bp[0]==None:
            tp = (img.shape[1],n1[1])
            bp = (0,n1[1])
    tp = list(map(int, tp))
    bp = list(map(int, bp))
    #print(tp[0], tp[1])
    return tp, bp

def calc_half_straight(img, n1, n2):#n1が端点
    if n1[0]==n2[0]:
        tp = (n1[0],0)
        bp = (n1[0],img.shape[0])
        tp=sorted([(n1[0],0),(n1[0],img.shape[0])],key=lambda x:(n1[1]-n2[1])*(n1[1]-x[1]),reverse=True)[0]
        bp=n1
    elif n1[1]==n2[1]:
        tp = sorted([(0,n1[1]),(img.shape[1],n1[1])],key=lambda x:(n1[0]-n2[0])*(n1[0]-x[0]),reverse=True)[0]
        bp = n1
    else:
        top = [[0, 0], [img.shape[1], 0]]
        btm = [[0, img.shape[0]], [img.shape[1], img.shape[0]]]

        tp = sorted([cross_point(n1, n2, top[0], top[1]),cross_point(n1, n2, btm[0], btm[1])],key=lambda x:(n1[0]-n2[0])*(n1[0]-x[0]),reverse=True)[0]
        bp = n1
                
    tp = list(map(int, tp))
    bp = list(map(int, bp))
    #print(tp[0], tp[1])
    return tp, bp

def shape_line_cross(cnt_img,n1,n2,half=False):
    img= np.zeros(cnt_img.shape[:2], dtype=np.uint8)
    if not half:
        tp, bp = calc_straight(img, n1, n2)
    else:
        tp, bp = calc_half_straight(img, n1, n2)
    cv2.line(img, (tp[0], tp[1]), (bp[0], bp[1]), 255, 1)

    #cv2.imshow('first', img)
    #if cv2.waitKey(100) & 0xFF == ord('q'):
        #cv2.destroyAllWindows()

    return (img*cnt_img).max()    #np.uint8なので255*255=1

def perpendicular_straight_cross(dt,cnt_img,t_marks=[],colname=None):
    cross=[]
    for i in range(len(dt)):
        if np.isnan(dt.loc[i,t_marks[0]+"_x"]) or np.isnan(dt.loc[i,t_marks[1]+"_x"]):
            cross.append(np.nan)
            continue
        n1=(int(sorted([np.ceil((dt.loc[i,t_marks[0]+"_x"]+dt.loc[i,t_marks[1]+"_x"])/2),np.floor((dt.loc[i,t_marks[0]+"_x"]+dt.loc[i,t_marks[1]+"_x"])/2)],key=lambda x:x%2)[0]),int(sorted([np.ceil((dt.loc[i,t_marks[0]+"_y"]+dt.loc[i,t_marks[1]+"_y"])/2),np.floor((dt.loc[i,t_marks[0]+"_y"]+dt.loc[i,t_marks[1]+"_y"])/2)],key=lambda x:x%2)[0]))
        n2=(n1[0]+int(1000*(dt.loc[i,t_marks[0]+"_y"]-dt.loc[i,t_marks[1]+"_y"])),n1[1]-int(1000*(dt.loc[i,t_marks[0]+"_x"]-dt.loc[i,t_marks[1]+"_x"])))
        cross.append(shape_line_cross(cnt_img,n1,n2))
    if colname==None:
        colname="".join(t_marks)+"_p"
    dt[colname]=cross

    return dt 

def half_straight_cross(dt,cnt_img,t_marks=[],colname=None):#t_marks[0]が端点
    cross=[]
    for i in range(len(dt)):
        if np.isnan(dt.loc[i,t_marks[0]+"_x"]) or np.isnan(dt.loc[i,t_marks[1]+"_x"]):
            cross.append(np.nan)
            continue
        n1=(int(sorted([np.ceil(dt.loc[i,t_marks[0]+"_x"]),np.floor(dt.loc[i,t_marks[0]+"_x"])],key=lambda x:x-dt.loc[i,t_marks[0]+"_x"])[0]),int(sorted([np.ceil(dt.loc[i,t_marks[0]+"_y"]),np.floor(dt.loc[i,t_marks[0]+"_y"])],key=lambda x:x-dt.loc[i,t_marks[0]+"_y"])[0]))
        n2=(int(sorted([np.ceil(dt.loc[i,t_marks[1]+"_x"]),np.floor(dt.loc[i,t_marks[1]+"_x"])],key=lambda x:x-dt.loc[i,t_marks[1]+"_x"])[0]),int(sorted([np.ceil(dt.loc[i,t_marks[1]+"_y"]),np.floor(dt.loc[i,t_marks[1]+"_y"])],key=lambda x:x-dt.loc[i,t_marks[1]+"_y"])[0]))
        cross.append(shape_line_cross(cnt_img,n1,n2,half=True))
    if colname==None:
        colname="".join(t_marks)+"_p"
    dt[colname]=cross
    return dt 

def sight_out_movie(results,dt,cnt_img,colors,fileout,filepath=None):
    if filepath==None:
        filepath=results.iloc[0]["video_path"]
    #movie = cv2.VideoCapture(r"F:\2022-11-17-08-48-08.mp4")"F:\2022-11-17-08-49-18.mp4"
    movie = cv2.VideoCapture(filepath)
    #video = cv2.VideoWriter(r"F:\test_2022-11-17-08-48-08.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    video = cv2.VideoWriter(fileout, cv2.VideoWriter_fourcc(*'mp4v'), movie.get(cv2.CAP_PROP_FPS), (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for i in range(len(results)):
        try:
            movie.set(cv2.CAP_PROP_POS_FRAMES, results.loc[i,"frame"]) 
            ret, frame=movie.read()
            if not ret:
                break
            n1=(int(sorted([np.ceil((dt.loc[i,"tip_x"]+dt.loc[i,"top_x"])/2),np.floor((dt.loc[i,"tip_x"]+dt.loc[i,"top_x"])/2)],key=lambda x:x%2)[0]),int(sorted([np.ceil((dt.loc[i,"tip_y"]+dt.loc[i,"top_y"])/2),np.floor((dt.loc[i,"tip_y"]+dt.loc[i,"top_y"])/2)],key=lambda x:x%2)[0]))
            n2=(n1[0]+int(1000*(dt.loc[i,"tip_y"]-dt.loc[i,"top_y"])),n1[1]-int(1000*(dt.loc[i,"tip_x"]-dt.loc[i,"top_x"])))
            img= np.zeros(cnt_img.shape[:2], dtype=np.uint8)
            tp, bp = calc_straight(img, n1, n2)
            cv2.line(frame, (tp[0], tp[1]), (bp[0], bp[1]), (0,0,255), 1)
            if shape_line_cross(cnt_img,n1,n2):
                cv2.putText(frame,text='in sight',org=(100, 100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0, 255, 0),thickness=2)
            #cv2.imshow("img",  frame)
            #if cv2.waitKey(0) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            for key in colors.keys():
                if results.loc[i,key+"_x"]!=None:
                    x_= [int(np.floor(results.loc[i,key+"_x"])),int(np.ceil(results.loc[i,key+"_x"]))][min(enumerate([results.loc[i,key+"_x"]-np.floor(results.loc[i,key+"_x"]),np.ceil(results.loc[i,key+"_x"])-results.loc[i,key+"_x"]]), key = lambda x:x[1])[0]]
                    y_= [int(np.floor(results.loc[i,key+"_y"])),int(np.ceil(results.loc[i,key+"_y"]))][min(enumerate([results.loc[i,key+"_y"]-np.floor(results.loc[i,key+"_y"]),np.ceil(results.loc[i,key+"_y"])-results.loc[i,key+"_y"]]), key = lambda x:x[1])[0]]
                    r_=[int(np.floor(results.loc[i,key+"_r"])),int(np.ceil(results.loc[i,key+"_r"]))][min(enumerate([results.loc[i,key+"_r"]-np.floor(results.loc[i,key+"_r"]),np.ceil(results.loc[i,key+"_r"])-results.loc[i,key+"_r"]]), key = lambda x:x[1])[0]]
                    cv2.circle(frame, (x_,y_), r_, colors[key]["color"], 3)
                    cv2.circle(frame, (x_,y_), 1, colors[key]["color"], -1)
            video.write(frame)
        except Exception as e:
            raise e
            break
    video.release()
    movie.release()
    
    
def moving_dist_frame(dt,t_marks,colname=None):#t_marks[0]が端点
    dist=[]
    for i in range(len(dt)):
        if i==0:
            dist.append(np.sqrt((dt.loc[i,t_marks+"_x"]-dt.loc[i+1,t_marks+"_x"])**2+(dt.loc[i,t_marks+"_y"]-dt.loc[i+1,t_marks+"_y"])**2))
        elif i==len(dt)-1:
            dist.append(np.sqrt((dt.loc[i,t_marks+"_x"]-dt.loc[i-1,t_marks+"_x"])**2+(dt.loc[i,t_marks+"_y"]-dt.loc[i-1,t_marks+"_y"])**2))
        else:
            dist.append((np.sqrt((dt.loc[i,t_marks+"_x"]-dt.loc[i+1,t_marks+"_x"])**2+(dt.loc[i,t_marks+"_y"]-dt.loc[i+1,t_marks+"_y"])**2)+np.sqrt((dt.loc[i,t_marks+"_x"]-dt.loc[i-1,t_marks+"_x"])**2+(dt.loc[i,t_marks+"_y"]-dt.loc[i-1,t_marks+"_y"])**2))/2)
    if colname==None:
        colname="".join(t_marks)+"_velo"
    dt[colname]=dist
    return dt 

def arg_vector(vec1,vec2):
    absvec1=np.linalg.norm(vec1)
    absvec2=np.linalg.norm(vec2)
    inner=np.inner(vec1,vec2)
    cos_theta=inner/(absvec1*absvec2)
    if abs(cos_theta)>1:
        cos_theta=cos_theta.round(2)
    theta=math.degrees(math.acos(cos_theta))
    if cos_theta==0 and any(np.array(vec1)*np.array(vec2)<0) and theta==0:
        theta=180
           
    return theta

def arg_vector_sets(dt,t_marks,colname=None):
    arg=[]
    for i in range(len(dt)):
        vec1=[dt.loc[i,t_marks[0][1]+"_x"]-dt.loc[i,t_marks[0][0]+"_x"],dt.loc[i,t_marks[0][1]+"_y"]-dt.loc[i,t_marks[0][0]+"_y"]]
        vec2=[dt.loc[i,t_marks[1][1]+"_x"]-dt.loc[i,t_marks[1][0]+"_x"],dt.loc[i,t_marks[1][1]+"_y"]-dt.loc[i,t_marks[1][0]+"_y"]]
        arg.append(arg_vector(vec1,vec2))
    if colname==None:
        colname="arg_"+t_marks[0][1]+"-"+t_marks[0][0]+"_"+t_marks[1][1]+"-"+t_marks[1][0]
    dt[colname]=arg
    return dt 

def delta_vec_arg_frame(dt,t_marks,colname=None):#t_marks[0]から#t_marks[1]へのベクトル
    delta_args=[]
    for i in range(len(dt)):
        if i==0:
            vec1=[dt.loc[i,t_marks[1]+"_x"]-dt.loc[i,t_marks[0]+"_x"],dt.loc[i,t_marks[1]+"_y"]-dt.loc[i,t_marks[0]+"_y"]]
            vec2=[dt.loc[i+1,t_marks[1]+"_x"]-dt.loc[i+1,t_marks[0]+"_x"],dt.loc[i+1,t_marks[1]+"_y"]-dt.loc[i+1,t_marks[0]+"_y"]]
            delta_args.append(arg_vector(vec1,vec2))
        elif i==len(dt)-1:
            vec1=[dt.loc[i,t_marks[1]+"_x"]-dt.loc[i,t_marks[0]+"_x"],dt.loc[i,t_marks[1]+"_y"]-dt.loc[i,t_marks[0]+"_y"]]
            vec2=[dt.loc[i-1,t_marks[1]+"_x"]-dt.loc[i-1,t_marks[0]+"_x"],dt.loc[i-1,t_marks[1]+"_y"]-dt.loc[i-1,t_marks[0]+"_y"]]
            delta_args.append(arg_vector(vec1,vec2))
        else:
            vec1=[dt.loc[i,t_marks[1]+"_x"]-dt.loc[i,t_marks[0]+"_x"],dt.loc[i,t_marks[1]+"_y"]-dt.loc[i,t_marks[0]+"_y"]]
            vec2=[dt.loc[i-1,t_marks[1]+"_x"]-dt.loc[i-1,t_marks[0]+"_x"],dt.loc[i-1,t_marks[1]+"_y"]-dt.loc[i-1,t_marks[0]+"_y"]]
            vec3=[dt.loc[i+1,t_marks[1]+"_x"]-dt.loc[i+1,t_marks[0]+"_x"],dt.loc[i+1,t_marks[1]+"_y"]-dt.loc[i+1,t_marks[0]+"_y"]]
            delta_args.append((arg_vector(vec1,vec2)+arg_vector(vec1,vec3))/2)

    if colname==None:
        colname="".join(t_marks)+"_delta_arg"
    dt[colname]=delta_args
    return dt 

def sight_arg_frame(dt,t_marks,cnt_img,step=1):#t_marks[0]から#t_marks[1]へのベクトルが0°
    #delta_args=[]
    #cross={j:0 for j in range(0,360,step)}
    #calcf=0
    for j in range(0,360,step):
        dt[j]=np.nan
    for i in range(len(dt)):
        if np.isnan(dt.loc[i,t_marks[0]+"_x"]) or np.isnan(dt.loc[i,t_marks[1]+"_x"]):
            continue
        #calcf+=1
        vec0=[1,0]
        vec1=[dt.loc[i,t_marks[1]+"_x"]-dt.loc[i,t_marks[0]+"_x"],dt.loc[i,t_marks[1]+"_y"]-dt.loc[i,t_marks[0]+"_y"]]
        if vec1[0]**2+vec1[1]**2==0:
            continue
        fai=arg_vector(vec0,vec1)
        if vec1[1]>0:
            fai=360-fai
        
        n1=(int(sorted([np.ceil((dt.loc[i,t_marks[0]+"_x"]+dt.loc[i,t_marks[1]+"_x"])/2),np.floor((dt.loc[i,t_marks[0]+"_x"]+dt.loc[i,t_marks[1]+"_x"])/2)],key=lambda x:x%2)[0]),int(sorted([np.ceil((dt.loc[i,t_marks[0]+"_y"]+dt.loc[i,t_marks[1]+"_y"])/2),np.floor((dt.loc[i,t_marks[0]+"_y"]+dt.loc[i,t_marks[1]+"_y"])/2)],key=lambda x:x%2)[0]))
        for x in range(0,360,step):
            n2=(n1[0]+(lambda th:int(1000*np.round(math.cos(math.radians(th)),3)) if not (th%90==0 and int(abs(th//90))%2==1) else 0)(-(fai+x)),n1[1]+(lambda th:int(1000*np.round(math.sin(math.radians(th)),3)) if not (th%90==0 and int(abs(th//90))%2==0) else 0)(-(fai+x)))
            dt.loc[i,x]=shape_line_cross(cnt_img,n1,n2,half=True)
            #cross[x]+=shape_line_cross(cnt_img,n1,n2,half=True)
    #cross={k:cross[k]/calcf for k in cross.keys()}
    return dt 

def trace_marker(dt,t_mark,savepath,frame_id=0,filepath=None,back_frame=None):

    if filepath==None:
        filepath=dt.iloc[0]["video_path"]
    if type(back_frame)!=type(None):
        frame=back_frame
    else:
        frame=get_frame_index(filepath,frame_id,show=1)
    inds=list(dt[t_mark+"_x"])
    inds=list(np.where(~np.isnan(inds))[0])
    for i in range(len(inds)-1):
        if dt.loc[inds[i]+1,"group"]=="before":
            color=(255,0,0)
            #color=(255,144,30)
        elif dt.loc[inds[i+1],"group"]=="present":
            color=(0,0,255)
            #color=(60,20,220)
        pt1_x=int(sorted([np.ceil(dt.loc[inds[i],t_mark+"_x"]),np.floor(dt.loc[inds[i],t_mark+"_x"])],key=lambda x:x-dt.loc[inds[i],t_mark+"_x"])[0])
        pt1_y=int(sorted([np.ceil(dt.loc[inds[i],t_mark+"_y"]),np.floor(dt.loc[inds[i],t_mark+"_y"])],key=lambda x:x-dt.loc[inds[i],t_mark+"_y"])[0])
        pt2_x=int(sorted([np.ceil(dt.loc[inds[i+1],t_mark+"_x"]),np.floor(dt.loc[inds[i+1],t_mark+"_x"])],key=lambda x:x-dt.loc[inds[i+1],t_mark+"_x"])[0])
        pt2_y=int(sorted([np.ceil(dt.loc[inds[i+1],t_mark+"_y"]),np.floor(dt.loc[inds[i+1],t_mark+"_y"])],key=lambda x:x-dt.loc[inds[i+1],t_mark+"_y"])[0])
        cv2.line(frame,(pt1_x,pt1_y),(pt2_x,pt2_y),color=color,thickness=1,lineType=cv2.LINE_4,shift=0)
        
    cv2.imwrite(savepath, frame)

def trace_marker2(dt,t_marks,savepath,frame_id=0,filepath=None,back_frame=None):

    if filepath==None:
        filepath=dt.iloc[0]["video_path"]
    if type(back_frame)!=type(None):
        frame=back_frame
    else:
        frame=get_frame_index(filepath,frame_id,show=1)
    for t_mark,color in t_marks.items():
        inds=list(dt[t_mark+"_x"])
        inds=list(np.where(~np.isnan(inds))[0])
        for i in range(len(inds)-1):
            pt1_x=int(sorted([np.ceil(dt.loc[inds[i],t_mark+"_x"]),np.floor(dt.loc[inds[i],t_mark+"_x"])],key=lambda x:x-dt.loc[inds[i],t_mark+"_x"])[0])
            pt1_y=int(sorted([np.ceil(dt.loc[inds[i],t_mark+"_y"]),np.floor(dt.loc[inds[i],t_mark+"_y"])],key=lambda x:x-dt.loc[inds[i],t_mark+"_y"])[0])
            pt2_x=int(sorted([np.ceil(dt.loc[inds[i+1],t_mark+"_x"]),np.floor(dt.loc[inds[i+1],t_mark+"_x"])],key=lambda x:x-dt.loc[inds[i+1],t_mark+"_x"])[0])
            pt2_y=int(sorted([np.ceil(dt.loc[inds[i+1],t_mark+"_y"]),np.floor(dt.loc[inds[i+1],t_mark+"_y"])],key=lambda x:x-dt.loc[inds[i+1],t_mark+"_y"])[0])
            cv2.line(frame,(pt1_x,pt1_y),(pt2_x,pt2_y),color=color,thickness=1,lineType=cv2.LINE_4,shift=0)
        
    cv2.imwrite(savepath, frame)

def approx_body_ellipse(filepath,results,t_marks,max_bird_size,min_bird_size,noise_judge=(),moment_d=None,thresh=None,scope=False,first_ind=0,bird_area=600,area=5,mog_thresh=300,judge_range=100,weight=1,sti=None,marker_out=0.25,display=False):
    
    """
    """"""
    function to estimate body center and direction from shilhouette using MoG backgroung sbtruction 
    only for single subject estimation
    """"""
    
    see 'get_marker_centroid' and see 'track'
    
    filepath : video file path on which body location and estimation applied
    
    results : file including the x,y coodinates of head markers. out put from 'trck' or 'track_mult'
    
    t_marks : name of markers on head
    
    max_bird_size : pixcel. hald of maximum size of bird.
    
    min_bird_size : pixcel. hald of minumum size of bird.
    
    noise_judge :(p0,0<=p1<=1,0<=p2<=1) parameters to decide to blur(p1) or to go on to the futher noise filtering after watershed(p2) based on the occupation of circle(radius: p0).
    
    moment_d : maximum threshold to adapt contour by the distance between of its center and reference point
    
    mog_thresh, judge_range : For the first 'mog_thresh' frames unless the posterior marker on head is outside of circle centered on the ellipse center with radius*'marker_out' of semi-major axis, 
                            two sets of body directions started with each vector stemming from the center to apexes of major axis were traced until body direction was estimated for 'judge_range' frame without loosing successive two frames
    
    marker_out : when the posterior marker on head is outside of circle centered on the ellipse center with radius*'marker_out' of semi-major axis, the semi-major axis on ellipse quarter comprising it  was picked as candidates
    """
    
    
    if len(noise_judge)==0:
        noise_judge=(int((max_bird_size+min_bird_size)/2),0.4,0.6)
    if moment_d==None:
        moment_d=int((max_bird_size+min_bird_size)/2)*0.8
    movie = cv2.VideoCapture(filepath)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    t_=np.nan
    xc_=np.nan
    yc_=np.nan
    fai_=np.nan
    fai__=np.nan
    rminor_=np.nan
    ixs=-1
    
    sk=0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) #blurの二倍程度
    
    mh=0
    t1_=np.nan
    t2_=np.nan
    while True:
        try:
            ind=movie.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = movie.read()
            if not ret:
                break
            
            ix=max(list(results[results["frame"]==ind].index))
            if ixs==-1:
                ixs=ix
            xs=[t for t in list(results.iloc[ix][[s for s in list(results.columns) if s in [sm+"_x" for sm in t_marks]]]) if t!=None and not np.isnan(t)]
            ys=[t for t in list(results.iloc[ix][[s for s in list(results.columns) if s in [sm+"_y" for sm in t_marks]]]) if t!=None and not np.isnan(t)]
            if len(xs)>0 and len(ys)>0:
                ref_n=(np.mean(xs),np.mean(ys))
                ref_n=(min([ref_n[0],frame.shape[1]]),min([ref_n[1],frame.shape[0]]))
            else:
                sk+=1
                #print("e")
                continue
                       
            fgmask = fgbg.apply(frame)
            
            scope=(sorted([math.floor(thresh*2.1),math.ceil(thresh*2.1)],key=lambda x:x%2)[0],sorted([math.floor(thresh*2.1),math.ceil(thresh*2.1)],key=lambda x:x%2)[0])
            #ref_point_=(ref_point[0]-math.floor(max([0,ref_point[0]-scope[0]//2])),ref_point[1]-math.floor(max([0,ref_point[1]-scope[1]//2])))
            #t_area = fgmask[math.floor(max([0,ref_point[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_point[1]+scope[1]//2]))+1,math.floor(max([0,ref_point[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_point[0]+scope[0]//2]))+1]
            #fgmask = np.zeros(fgmask.shape[:2], dtype=np.uint8)
            #fgmask[math.floor(max([0,ref_point[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_point[1]+scope[1]//2]))+1,math.floor(max([0,ref_point[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_point[0]+scope[0]//2]))+1]=t_area
            
            ref_n_=tuple(map(int,[ref_n[0]-math.floor(max([0,ref_n[0]-scope[0]//2])),ref_n[1]-math.floor(max([0,ref_n[1]-scope[1]//2]))]))
            xs_=[t-math.floor(max([0,ref_n[0]-scope[0]//2])) for t in xs]
            ys_=[t-math.floor(max([0,ref_n[1]-scope[1]//2])) for t in ys]
            
            t_area = fgmask[math.floor(max([0,ref_n[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_n[1]+scope[1]//2]))+1,math.floor(max([0,ref_n[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_n[0]+scope[0]//2]))+1]
            fgc, hir = cv2.findContours(t_area, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            fgc=[s[1] for s in enumerate(fgc) if hir[0][s[0]][-1]==-1]
            fg = np.zeros(t_area.shape[:2], dtype=np.uint8)
            cv2.drawContours(fg, fgc,-1, 255, -1)
            fg_tc=np.zeros(t_area.shape[:2], dtype=np.uint8)
            cv2.circle(fg_tc,ref_n_,noise_judge[0], 255, -1)
            fg_t=cv2.bitwise_and(fg,fg_tc)

            if (fg_t/255).sum()>math.pi*noise_judge[0]*noise_judge[0]*noise_judge[1]:
                fgc_t = list(filter(lambda x: cv2.contourArea(x) >= area, fgc))
                fg = np.zeros(t_area.shape[:2], dtype=np.uint8)              
                cv2.drawContours(fg, fgc_t,-1, 255, -1)
                        
            fgm_b = cv2.blur(fg, (5, 5))
            fgm_mask =cv2.threshold(fgm_b, 0, 255, cv2.THRESH_BINARY)[1]
            
            fgc1, hir = cv2.findContours(fgm_mask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            fgc1=[s[1] for s in enumerate(fgc1) if hir[0][s[0]][-1]==-1]
            fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area, fgc1))
            
            #cv2.imshow("img1",t_area)
            #cv2.imshow("img2",fgm_mask)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #cv2.destroyAllWindows()
            #continue
            
            #fgc2=sorted(list(filter(lambda x: cv2.contourArea(x) >= 600 and cv2.contourArea(x) < 3000, fgc1)),key=lambda y:cv2.contourArea(y),reverse=True)
            #ff=frame.copy()
            
            #print(sk)
            if len(fgc2)>0:# and ind>mog_thresh:
                
                #ref_n=(None,None)
                #fgc4=sorted(fgc2,key=lambda x:sum([(ref_n[i]-s)**2 for i, s in enumerate(get_moment(x))]))
                fgc4=sorted(fgc2,key=lambda x:(-sum([int(cv2.pointPolygonTest(x, (int(xs_[i]),int(ys_[i])), measureDist=False)>=0) for i in range(len(xs_))]),sum([(ref_n_[i]-s)**2 for i, s in enumerate(get_moment(x))])))
                
                #fgmask_ = fgm_b[:,:,0]
                fgmask_ = np.zeros(fg.shape[:2], dtype=np.uint8)
                cv2.drawContours(fgmask_, fgc4[:1],-1, 255, -1)
                if (sum([int(cv2.pointPolygonTest(fgc4[0], (int(xs_[i]),int(ys_[i])), measureDist=False)>=0) for i in range(len(xs_))])==0 and np.sqrt(sum([(ref_n_[i]-s)**2 for i, s in enumerate(get_moment(fgc4[0]))]))>moment_d) or cv2.minEnclosingCircle(fgc4[0])[1] >max_bird_size:

                    #fgmask = np.zeros(fgmask.shape[:2], dtype=np.uint8)
                    #cv2.drawContours(fgmask, fgc4[:1],-1, 255, -1)
                    c_img=np.zeros(fgmask_.shape[:2], dtype=np.uint8)
                    cv2.circle(c_img,tuple(map(int,list(ref_n_))),min_bird_size,255,-1)
                    f_img=cv2.bitwise_and(fgmask_,c_img)
                    c_img=np.zeros(fgmask_.shape[:2], dtype=np.uint8)
                    cv2.circle(c_img,tuple(map(int,list(ref_n_))),max_bird_size,255,-1)
                    fgmask_=cv2.bitwise_and(fgmask_,c_img)
                    #c_img= cv2.bitwise_not(c_img)
                    #b_img=cv2.bitwise_and(fgmask_,c_img)
                    """if rich in noise"""
                    #fgmask_=fgmask[math.floor(max([0,ref_point[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_point[1]+scope[1]//2]))+1,math.floor(max([0,ref_point[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_point[0]+scope[0]//2]))+1]
                    #sure_bg = cv2.dilate(fgmask_, kernel, iterations=1)
                    sure_bg = fgmask_
                    #sure_bg = b_img
                    dist = cv2.distanceTransform(f_img, cv2.DIST_L2, 5)
                    ret_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
                    sure_fg = sure_fg.astype(np.uint8)
                    unknown = cv2.subtract(sure_bg, sure_fg)
                    ret_, markers = cv2.connectedComponents(sure_fg)
                    markers += 1
                    markers[unknown == 255] = 0
                    markers = cv2.watershed(frame[math.floor(max([0,ref_n[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_n[1]+scope[1]//2]))+1,math.floor(max([0,ref_n[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_n[0]+scope[0]//2]))+1], markers)
                    labels = np.unique(markers)
                    fgc1 = []
                    for label in labels[2:]:  # 0:背景ラベル １：境界ラベル は無視する。
                    
                        # ラベル label の領域のみ前景、それ以外は背景となる2値画像を作成する。
                        target = np.where(markers == label, 255, 0).astype(np.uint8)
                    
                        # 作成した2値画像に対して、輪郭抽出を行う。
                        contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        fgc1.append(contours[0])
                    fgc2=list(filter(lambda x: cv2.contourArea(x) >= area, fgc1))
                    #fgc4=sorted(fgc2,key=lambda x:sum([(ref_n[i]-s)**2 for i, s in enumerate(get_moment(x))])) 
                    fgc4=sorted(fgc2,key=lambda x:(-sum([int(cv2.pointPolygonTest(x, (int(xs_[i]),int(ys_[i])), measureDist=False)>=0) for i in range(len(xs_))]),sum([(ref_n_[i]-s)**2 for i, s in enumerate(get_moment(x))])))

                    if len(fgc4)==0 or (sum([int(cv2.pointPolygonTest(fgc4[0], (int(xs_[i]),int(ys_[i])), measureDist=False)>=0) for i in range(len(xs_))])==0 and np.sqrt(sum([(ref_n_[i]-s)**2 for i, s in enumerate(get_moment(fgc4[0]))]))>moment_d) or cv2.minEnclosingCircle(fgc4[0])[1] >max_bird_size:
                        sk+=1
                        fgmask = np.zeros(fgmask.shape[:2], dtype=np.uint8)
                        fgmask[math.floor(max([0,ref_n[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_n[1]+scope[1]//2]))+1,math.floor(max([0,ref_n[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_n[0]+scope[0]//2]))+1]=fgmask_

                        #cv2.imshow("img", frame)
                        #cv2.imshow("img2", fgmask)
                        #cv2.imshow("img3", cnt_img)
                        #if cv2.waitKey(1) & 0xFF == ord('q'):
                            #cv2.destroyAllWindows()
                        #print("d")
                        #if len(xs)>1:
                            #break
                        continue
                    fgmask_ = np.zeros(fgmask_.shape[:2], dtype=np.uint8)
                    cv2.drawContours(fgmask_, fgc4[:1],-1, 255, -1)
                #fgmask = np.zeros(fgmask.shape[:2], dtype=np.uint8)
                #cv2.drawContours(fgmask, fgc4[:1],-1, 255, -1)
                #fgmask=cv2.bitwise_and(fgmask,fgmask_)
                fgc4=list(filter(lambda x: len(x) >= 5, fgc4))
                if len(fgc4)<1:
                    sk+=1
                    #print("a")
                    continue
                ellipse = cv2.fitEllipse(fgc4[0])
                ellipse =(ellipse[0],tuple([0.5 if np.isnan(s) else s for s in ellipse[1]]),ellipse[2])
                fg_tc=np.zeros(t_area.shape[:2], dtype=np.uint8)
                cv2.ellipse(fg_tc, ellipse, 255, -1)
                fg_t=cv2.bitwise_and(fgmask_,fg_tc)
    
                if ((fg_t/255).sum())/((fg_tc/255).sum())<noise_judge[2]:
                    eroded_image = cv2.erode(fgmask_, kernel, iterations=1)
                    fgc1, hir = cv2.findContours(eroded_image, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                    fgc1=[s[1] for s in enumerate(fgc1) if hir[0][s[0]][-1]==-1]
                    fgc2=list(filter(lambda x: cv2.contourArea(x) >= area, fgc1))
                    #fgc4=sorted(fgc2,key=lambda x:sum([(ref_n[i]-s)**2 for i, s in enumerate(get_moment(x))])) #小さいときは探すみたいな処理いれるか
                    fgc4=sorted(fgc2,key=lambda x:(-sum([int(cv2.pointPolygonTest(x, (int(xs_[i]),int(ys_[i])), measureDist=False)>=0) for i in range(len(xs_))]),sum([(ref_n_[i]-s)**2 for i, s in enumerate(get_moment(x))])))
                    eroded_image = np.zeros(eroded_image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(eroded_image, fgc4[:1],-1, 255, -1)
                    fgmask_ = cv2.dilate(eroded_image, kernel, iterations=1)
                fgmask = np.zeros(fgmask.shape[:2], dtype=np.uint8)
                fgmask[math.floor(max([0,ref_n[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_n[1]+scope[1]//2]))+1,math.floor(max([0,ref_n[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_n[0]+scope[0]//2]))+1]=fgmask_

                fgc4,_=cv2.findContours(fgmask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                fgc4=list(filter(lambda x: len(x) >= 5, fgc4))
                if len(fgc4)<1:
                    sk+=1
                    #print("a")
                    continue

                #print(len(fgc4[0]))
                ellipse = cv2.fitEllipse(fgc4[0])
                if  any([np.isnan(s) for s in list(ellipse[0])+list(ellipse[1])]):
                    continue

                xc,yc=ellipse[0]
                rmajor = max(ellipse[1])/2
                rminor = min(ellipse[1])/2
                angle=ellipse[2]
                if angle > 90:
                    angle = angle - 90
                else:
                    angle = angle + 90
                x1 = xc + math.cos(math.radians(angle))*rmajor
                y1 = yc + math.sin(math.radians(angle))*rmajor
                x2 = xc + math.cos(math.radians(angle+180))*rmajor
                y2 = yc + math.sin(math.radians(angle+180))*rmajor
                x3 = xc + math.cos(math.radians(angle+90))*rminor
                y3 = yc + math.sin(math.radians(angle+90))*rminor
                x4 = xc + math.cos(math.radians(angle+270))*rminor
                y4 = yc + math.sin(math.radians(angle+270))*rminor
                results.loc[ix,"el1_x"]=x1
                results.loc[ix,"el1_y"]=y1
                results.loc[ix,"el2_x"]=x2
                results.loc[ix,"el2_y"]=y2
                results.loc[ix,"el3_x"]=x3
                results.loc[ix,"el3_y"]=y3
                results.loc[ix,"el4_x"]=x4
                results.loc[ix,"el4_y"]=y4
                results.loc[ix,"bird_elp_center_x"]=xc
                results.loc[ix,"bird_elp_center_y"]=yc
                if not all([s in results.columns for s in sum([[sm+"_x",sm+"_y"] for sm in t_marks],[])]):
                    continue
                
                if (np.isnan(results.loc[ix,t_marks[0]+"_x"]) or np.isnan(results.loc[ix,t_marks[1]+"_x"])):
                    sk+=1
                    #print("b")
                    continue
                #calcf+=1
                vec0=[1,0]
                vec1=[results.loc[ix,t_marks[1]+"_x"]-results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[1]+"_y"]-results.loc[ix,t_marks[0]+"_y"]]
                if vec1[0]**2+vec1[1]**2==0:
                    sk+=1
                    #print("c")
                    continue
                fai=arg_vector(vec0,vec1)
                if vec1[1]<0:
                    fai=360-fai
                ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                if ind<mog_thresh and mh<judge_range and len(ms)==0:
                    mh+=1
                    w=1
                    if (type(t1_)!=tuple and np.isnan(t1_)) and (type(t2_)!=tuple and np.isnan(t2_)) or sk>2:
                        mh=0
                        #results.loc["bird_elp_center_x"]=np.nan    
                        #results.loc["bird_elp_center_y"]=np.nan    
                        results.loc[ix,"bird_elp_headside_x1"]=np.nan
                        results.loc[ix,"bird_elp_headside_y1"]=np.nan
                        results.loc[ix,"bird_elp_headside_x2"]=np.nan
                        results.loc[ix,"bird_elp_headside_y2"]=np.nan    
                        """
                        dr=results.loc[ixs:,[t_marks[0]+"_x",t_marks[0]+"_y",t_marks[1]+"_x",t_marks[1]+"_y","bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"]].dropna()
                        tm1=np.mean((np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[1]+"_y"])**2))/2)
                        tm2=np.mean((np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[1]+"_y"])**2))/2)
                        if tm1==tm2:
                            dr=results.loc[int((ix+ixs)/2):ix,[t_marks[0]+"_x",t_marks[0]+"_y",t_marks[1]+"_x",t_marks[1]+"_y","bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"]].dropna()
                            tm1=np.mean((np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[1]+"_y"])**2))/2)
                            tm2=np.mean((np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[1]+"_y"])**2))/2)
        
                        if tm1<tm2:
                            results["bird_elp_headside_x"]=results["bird_elp_headside_x1"]
                            results["bird_elp_headside_y"]=results["bird_elp_headside_y1"]
                            t_=t1
                        else:
                            results["bird_elp_headside_x"]=results["bird_elp_headside_x2"]
                            results["bird_elp_headside_y"]=results["bird_elp_headside_y2"]
                            t_=t2
                        """
                        ixs=ix

                        t1=tuple(map(int,[x1,y1]))
                        t2=tuple(map(int,[x2,y2]))
                    elif np.isnan(fai_):   #下とうまく組み合わせたい
                        #print("c")
                        print(ix)
                        cand_p1=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[t1_[0]-xc_,t1_[1]-yc_]))[:2]#
                        cand_p2=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[t2_[0]-xc_,t2_[1]-yc_]))[:2]#np.sqrt(((x[0]-xc)-(t_[0]-xc_))**2+((x[1]-yc)-(t_[1]-yc_))**2))[:2]
                        t1=tuple(map(int,sorted(cand_p1,key=lambda x:x[2]*w,reverse=True)[0][:2]))

                        t2=tuple(map(int,sorted(cand_p2,key=lambda x:x[2]*w,reverse=True)[0][:2]))
                    else:
                        #print("d")
                        th=math.radians(np.mean(list(filter(lambda x: ~np.isnan(x),[sorted([fai-fai_,-(360-abs(fai-fai_))],key=lambda n:abs(n))[0],sorted([fai_-fai__,-(360-abs(fai_-fai__))],key=lambda n1:abs(n1))[0]]))))
                        rvec=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
                        cand_v1=np.dot(rvec, np.array([t1_[0]-xc_,t1_[1]-yc_]))#+np.array([xc-xc_,yc-yc_])
                        cand_v2=np.dot(rvec, np.array([t2_[0]-xc_,t2_[1]-yc_]))
                        cva1=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,list(cand_v1)),list(cand_v1))
                        cva2=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,list(cand_v2)),list(cand_v2))
                        t_va1=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,[t1_[0]-xc_,t1_[1]-yc_]),[t1_[0]-xc_,t1_[1]-yc_])#[t_[0]+xc-xc_,t_[1]+yc-yc_]
                        t_va2=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,[t2_[0]-xc_,t2_[1]-yc_]),[t2_[0]-xc_,t2_[1]-yc_])#

                        #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(mocap.arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(mocap.arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                        ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                        #wms=list(map(lambda x:min([np.sqrt((x[0]-ms[i][0])**2+(x[1]-ms[i][1])**2) for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]]))
                        wms=list(map(lambda x:min([(lambda n2:n2 if abs(n2)>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor else 0)(-np.sqrt((xc-ms[i][0])**2+(yc-ms[i][1])**2)*math.cos(math.radians(angle+x[3]-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[ms[i][0]-xc,ms[i][1]-yc]),[ms[i][0]-xc,ms[i][1]-yc]))))+x[2] for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor,0],[x2,y2,rmajor,180],[x3,y3,rminor,90],[x4,y4,rminor,270]]))
                        wts=[np.mean([1/wms[0],1/wms[2]]),np.mean([1/wms[1],1/wms[2]]),np.mean([1/wms[1],1/wms[3]]),np.mean([1/wms[0],1/wms[3]])]
                        cand_p=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]],reverse=True)
                        if wts[cand_p[0][0]]==wts[cand_p[1][0]] and wms[int(cand_p[0][0]%3==0)]==1:
                            cand_p1=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]]*(lambda y,z:sum([0 if y[2*iy]>=z[2*(iz+1)-1] or z[2*iz]>=y[2*(iy+1)-1] else (lambda k:k[2]-k[1])(sorted(y[2*iy:2*(iy+1)]+z[2*iz:2*(iz+1)])) for iz in range(len(z)//2) for iy in range(len(y)//2)]))((lambda n1:[n1[0],n1[1]] if n1[1]-n1[0]<=90 else [n1[1],360,0,n1[0]])(sorted([[angle%360,(angle+90)%360],[(angle+90)%360,(angle+180)%360],[(angle+180)%360,(angle+270)%360],[(angle+270)%360,(angle)%360]][x[0]])),(lambda n2:[n2[0],n2[1]] if n2[1]-n2[0]<=180 else [n2[1],360,0,n2[0]])(sorted([t_va1,cva1]))),reverse=True)[0][1]
                            cand_p2=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]]*(lambda y,z:sum([0 if y[2*iy]>=z[2*(iz+1)-1] or z[2*iz]>=y[2*(iy+1)-1] else (lambda k:k[2]-k[1])(sorted(y[2*iy:2*(iy+1)]+z[2*iz:2*(iz+1)])) for iz in range(len(z)//2) for iy in range(len(y)//2)]))((lambda n1:[n1[0],n1[1]] if n1[1]-n1[0]<=90 else [n1[1],360,0,n1[0]])(sorted([[angle%360,(angle+90)%360],[(angle+90)%360,(angle+180)%360],[(angle+180)%360,(angle+270)%360],[(angle+270)%360,(angle)%360]][x[0]])),(lambda n2:[n2[0],n2[1]] if n2[1]-n2[0]<=180 else [n2[1],360,0,n2[0]])(sorted([t_va2,cva2]))),reverse=True)[0][1]
                        else:
                            cand_p1=cand_p[0][1]
                            cand_p2=cand_p[0][1]
    
                        wa=np.mean([min([abs(cand_p1[1][3]+angle-cva1),abs(360-abs(cand_p1[1][3]+angle-cva1))])/max([0.0000000001,min([abs(cand_p1[0][3]+angle-cva1),abs(360-abs(cand_p1[0][3]+angle-cva1))])]),min([abs(cand_p1[1][3]+angle-t_va1),abs(360-abs(cand_p1[1][3]+angle-t_va1))])/max([0.0000000001,min([abs(cand_p1[0][3]+angle-t_va1),abs(360-abs(cand_p1[0][3]+angle-t_va1))])])])
                        wi=np.mean([min([abs(cand_p1[0][3]+angle-cva1),abs(360-abs(cand_p1[0][3]+angle-cva1))])/max([0.0000000001,min([abs(cand_p1[1][3]+angle-cva1),abs(360-abs(cand_p1[1][3]+angle-cva1))])]),min([abs(cand_p1[0][3]+angle-t_va1),abs(360-abs(cand_p1[0][3]+angle-t_va1))])/max([0.0000000001,min([abs(cand_p1[1][3]+angle-t_va1),abs(360-abs(cand_p1[1][3]+angle-t_va1))])])])
                        if rmajor/rminor<1.2 and sk<1:
                            #w=1
                            w=wa-wi
                        #w=abs(rmajor-rminor_)*wa/rminor_-abs(rminor-rminor_)*wi/rminor_
                        cand_p1=[p[:3] for p in cand_p1]
                        t1=tuple(map(int,sorted(cand_p1,key=lambda x:x[2]*w,reverse=True)[0][:2]))
                        w=1
                        wa=np.mean([min([abs(cand_p2[1][3]+angle-cva2),abs(360-abs(cand_p2[1][3]+angle-cva2))])/max([0.0000000001,min([abs(cand_p2[0][3]+angle-cva2),abs(360-abs(cand_p2[0][3]+angle-cva2))])]),min([abs(cand_p2[1][3]+angle-t_va2),abs(360-abs(cand_p2[1][3]+angle-t_va2))])/max([0.0000000001,min([abs(cand_p2[0][3]+angle-t_va2),abs(360-abs(cand_p2[0][3]+angle-t_va2))])])])
                        wi=np.mean([min([abs(cand_p2[0][3]+angle-cva2),abs(360-abs(cand_p2[0][3]+angle-cva2))])/max([0.0000000001,min([abs(cand_p2[1][3]+angle-cva2),abs(360-abs(cand_p2[1][3]+angle-cva2))])]),min([abs(cand_p2[0][3]+angle-t_va2),abs(360-abs(cand_p2[0][3]+angle-t_va2))])/max([0.0000000001,min([abs(cand_p2[1][3]+angle-t_va2),abs(360-abs(cand_p2[1][3]+angle-t_va2))])])])
                        if rmajor/rminor<1.2 and sk<1:
                            #w=1
                            w=wa-wi
                        #w=abs(rmajor-rminor_)*wa/rminor_-abs(rminor-rminor_)*wi/rminor_
                        cand_p2=[p[:3] for p in cand_p2]
                        t2=tuple(map(int,sorted(cand_p2,key=lambda x:x[2]*w,reverse=True)[0][:2]))
                    t1_=t1
                    t2_=t2
                    xc_=xc
                    yc_=yc
                    rminor_=rminor
                    if sk>2:
                        fai__=np.nan
                    else:
                        fai__=fai_
                    fai_=fai
                    #ellipse_=ellipse
                    sk=0
                    #results.loc[ix,"bird_elp_center_x"]=xc
                    #results.loc[ix,"bird_elp_center_y"]=yc
                    results.loc[ix,"bird_elp_headside_x1"]=t1[0]
                    results.loc[ix,"bird_elp_headside_y1"]=t1[1]
                    results.loc[ix,"bird_elp_headside_x2"]=t2[0]
                    results.loc[ix,"bird_elp_headside_y2"]=t2[1]
                    #print("1")

                    if mh==judge_range:
                        dr=results.loc[ixs:,[t_marks[0]+"_x",t_marks[0]+"_y",t_marks[1]+"_x",t_marks[1]+"_y","bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"]].dropna()
                        tm1=np.mean((np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[1]+"_y"])**2))/2)
                        tm2=np.mean((np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[1]+"_y"])**2))/2)
                        if tm1==tm2:
                            dr=results.loc[int((ix+ixs)/2):ix,[t_marks[0]+"_x",t_marks[0]+"_y",t_marks[1]+"_x",t_marks[1]+"_y","bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"]].dropna()
                            tm1=np.mean((np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[1]+"_y"])**2))/2)
                            tm2=np.mean((np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[1]+"_y"])**2))/2)
        
                        if tm1<tm2:
                            results["bird_elp_headside_x"]=results["bird_elp_headside_x1"]
                            results["bird_elp_headside_y"]=results["bird_elp_headside_y1"]
                            t_=t1
                        else:
                            results["bird_elp_headside_x"]=results["bird_elp_headside_x2"]
                            results["bird_elp_headside_y"]=results["bird_elp_headside_y2"]
                            t_=t2
                        ixs=ix+1
                        results.drop(["bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"],axis=1,inplace=True)
                    continue
                if "bird_elp_headside_x1" in results.columns:
                    results.drop(["bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"],axis=1,inplace=True)
                    
                #w=abs(rmajor-rminor_)/rminor_-abs(rminor-rminor_)/rminor_
                w=1
                if (type(t_)!=tuple and np.isnan(t_)) or sk>2:
                    ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]

                    #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if max([abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle-fai))),abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle+90-fai)))])>=0.25*rmajor]
                    if len(ms)>0:
                        #cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:min([np.sqrt((x[0]-ms[i][0])**2+(x[1]-ms[i][1])**2) for i in range(len(ms))]))[:2]

                        cand_p=sorted(list(map(lambda n:(n[0],(lambda x:min([(lambda x2:x2 if abs(x2)>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor else 0)(-np.sqrt((xc-ms[i][0])**2+(yc-ms[i][1])**2)*math.cos(math.radians(angle+x[3]-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[ms[i][0]-xc,ms[i][1]-yc]),[ms[i][0]-xc,ms[i][1]-yc]))))+x[2] for i in range(len(ms))])/x[2])(n[1])),enumerate([[x1,y1,rmajor,0],[x2,y2,rmajor,180],[x3,y3,rminor,90],[x4,y4,rminor,270]]))),key=lambda n_:n_[1])
                        if cand_p[0][1]==cand_p[1][1] and [[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[0][0]][2]==[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[1][0]][2]:
                            sk+=1
                            continue   
                        cand_p=[[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[0][0]],[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[1][0]]]
                        #rmajor/rminor<1.2 and 
                        if rmajor/rminor<1.2 and sorted([(-1,0)]+[(i,(-np.sqrt((xc-results.loc[ix,t_marks[i]+"_x"])**2+(yc-results.loc[ix,t_marks[i]+"_y"])**2)*math.cos(math.radians(angle+0-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[results.loc[ix,t_marks[i]+"_x"]-xc,results.loc[ix,t_marks[i]+"_y"]-yc]),[results.loc[ix,t_marks[i]+"_x"]-xc,results.loc[ix,t_marks[i]+"_y"]-yc]))))) for i in range(len(t_marks))],key=lambda xa:xa[1])[1][0]==1:
                            cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[results.loc[ix,t_marks[1]+"_x"]-results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[1]+"_y"]-results.loc[ix,t_marks[0]+"_y"]]))[:2]

                        #print("a")
                        #print(ix)
                    else:
                        #print("b")
                        sk+=1
                        continue
                    ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                    #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if max([abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle-fai))),abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle+90-fai)))])>=0.25*rmajor]

                elif np.isnan(fai_):   #下とうまく組み合わせたい
                    #print("c")
                    cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[t_[0]-xc_,t_[1]-yc_]))[:2]#np.sqrt(((x[0]-xc)-(t_[0]-xc_))**2+((x[1]-yc)-(t_[1]-yc_))**2))[:2]
                else:
                    #print("d")
                    th=math.radians(np.mean(list(filter(lambda x: ~np.isnan(x),[sorted([fai-fai_,-(360-abs(fai-fai_))],key=lambda n:abs(n))[0],sorted([fai_-fai__,-(360-abs(fai_-fai__))],key=lambda n1:abs(n1))[0]]))))
                    rvec=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
                    cand_v=np.dot(rvec, np.array([t_[0]-xc_,t_[1]-yc_]))#+np.array([xc-xc_,yc-yc_])
                    cva=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,list(cand_v)),list(cand_v))
                    t_va=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,[t_[0]-xc_,t_[1]-yc_]),[t_[0]-xc_,t_[1]-yc_])#[t_[0]+xc-xc_,t_[1]+yc-yc_]
                    #**これじゃだめか
                    #cand_p=sorted(enumerate([[[x1,y1,rmajor],[x3,y3,rminor]],[[x3,y3,rminor],[x2,y2,rmajor]],[[x2,y2,rmajor],[x4,y4,rminor]],[[x4,y4,rminor],[x1,y1,rmajor]]]),key=lambda x:(lambda y,z:0 if y%360>=z[1] or z[0]>=y%360+90 else (lambda k:k[2]-k[1])(sorted([y%360,y%360+90]+z)))([angle,angle+90,angle+180,angle+270][x[0]],(lambda x1:sorted([x1[0]+360*min([1,(x1[1]-x1[0])//180]),x1[1]]))(sorted([t_va,cva]))),reverse=True)[0][1]
                    #wts=list(map(lambda x:mocap.arg_vector([x[0]-xc,x[1]-yc],[t_[0]-xc_,t_[1]-yc_]),[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]]))
                    #wts=[min(wts)/s for s in wts]
                    #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                    ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                    #wms=list(map(lambda x:min([np.sqrt((x[0]-ms[i][0])**2+(x[1]-ms[i][1])**2) for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]]))
                    wms=list(map(lambda x:min([(lambda n2:n2 if abs(n2)>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor else 0)(-np.sqrt((xc-ms[i][0])**2+(yc-ms[i][1])**2)*math.cos(math.radians(angle+x[3]-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[ms[i][0]-xc,ms[i][1]-yc]),[ms[i][0]-xc,ms[i][1]-yc]))))+x[2] for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor,0],[x2,y2,rmajor,180],[x3,y3,rminor,90],[x4,y4,rminor,270]]))
                    wts=[np.mean([1/wms[0],1/wms[2]]),np.mean([1/wms[1],1/wms[2]]),np.mean([1/wms[1],1/wms[3]]),np.mean([1/wms[0],1/wms[3]])]
                    cand_p=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]],reverse=True)
                    if wts[cand_p[0][0]]==wts[cand_p[1][0]] and wms[int(cand_p[0][0]%3==0)]==1:
                        cand_p=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]]*(lambda y,z:sum([0 if y[2*iy]>=z[2*(iz+1)-1] or z[2*iz]>=y[2*(iy+1)-1] else (lambda k:k[2]-k[1])(sorted(y[2*iy:2*(iy+1)]+z[2*iz:2*(iz+1)])) for iz in range(len(z)//2) for iy in range(len(y)//2)]))((lambda n1:[n1[0],n1[1]] if n1[1]-n1[0]<=90 else [n1[1],360,0,n1[0]])(sorted([[angle%360,(angle+90)%360],[(angle+90)%360,(angle+180)%360],[(angle+180)%360,(angle+270)%360],[(angle+270)%360,(angle)%360]][x[0]])),(lambda n2:[n2[0],n2[1]] if n2[1]-n2[0]<=180 else [n2[1],360,0,n2[0]])(sorted([t_va,cva]))),reverse=True)[0][1]
                    else:
                        cand_p=cand_p[0][1]

                    wa=np.mean([min([abs(cand_p[1][3]+angle-cva),abs(360-abs(cand_p[1][3]+angle-cva))])/max([0.0000000001,min([abs(cand_p[0][3]+angle-cva),abs(360-abs(cand_p[0][3]+angle-cva))])]),min([abs(cand_p[1][3]+angle-t_va),abs(360-abs(cand_p[1][3]+angle-t_va))])/max([0.0000000001,min([abs(cand_p[0][3]+angle-t_va),abs(360-abs(cand_p[0][3]+angle-t_va))])])])
                    wi=np.mean([min([abs(cand_p[0][3]+angle-cva),abs(360-abs(cand_p[0][3]+angle-cva))])/max([0.0000000001,min([abs(cand_p[1][3]+angle-cva),abs(360-abs(cand_p[1][3]+angle-cva))])]),min([abs(cand_p[0][3]+angle-t_va),abs(360-abs(cand_p[0][3]+angle-t_va))])/max([0.0000000001,min([abs(cand_p[1][3]+angle-t_va),abs(360-abs(cand_p[1][3]+angle-t_va))])])])
                    if rmajor/rminor<1.2 and sk<1:
                        #w=1
                        w=wa-wi
                    #w=abs(rmajor-rminor_)*wa/rminor_-abs(rminor-rminor_)*wi/rminor_
                    cand_p=[p[:3] for p in cand_p]
                #体軸を選ぶ
                ##単軸にとっての変化量の差分で重みづける
                #w=abs(rmajor-rminor_)/rminor_-abs(rminor-rminor_)/rminor_#プラスマイナスでいいか
                #if rminor_>35 and w>-0.2 and w<0:
                    #w=1
                #w=1
                #print(w)
                #cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda:mocap.arg_vector(cand_v,[x[0],x[1]]))[:2]
                t=tuple(map(int,sorted(cand_p,key=lambda x:x[2]*w,reverse=True)[0][:2]))

                #print(t)
                if display or (sti!=None and ix==sti):
                    cv2.line(frame, (int(xc),int(yc)), t, (0, 0, 255), 3)
                    cv2.circle(frame, (int(xc),int(yc)), 2, (0, 255, 0), -1)
                    cv2.ellipse(frame,ellipse,(255,0,0),2)
                    cv2.line(frame, (int(results.loc[ix,t_marks[0]+"_x"]),int(results.loc[ix,t_marks[0]+"_y"])), (int(results.loc[ix,t_marks[1]+"_x"]),int(results.loc[ix,t_marks[1]+"_y"])), (255, 0, 255), 3)
                    
                    #paper fig
                    #cv2.line(frame, (int(xc),int(yc)), t, (200, 0, 100), 3)
                    #cv2.circle(frame, (int(xc),int(yc)), 2, (200, 0, 100), -1)
                    #cv2.ellipse(frame,ellipse,(200, 0, 100),2)
                    
                    cv2.imshow("img", frame)
                    cv2.imshow("img2", fgmask)
                   
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                
                #if type(t_)==tuple and type(t)==tuple and mocap.arg_vector(list(map(int,[t[0]-xc,t[1]-yc])),list(map(int,[t_[0]-xc_,t_[1]-yc_])))>105:
                if sti!=None and ix==sti:
                #if ix!=374 and (cand_p[0]==[x2,y2,rmajor] or cand_p[1]==[x2,y2,rmajor]):
                #if type(t_)==tuple and type(t)==tuple and ix>100 and (cand_p[0]==[x2,y2,rmajor] or cand_p[1]==[x2,y2,rmajor]):
                    print(t)
                    print(t_)
                    print(fai)
                    print(fai_)
                    print(fai__)
                    print(rminor)
                    print(rminor_)
                    print((xc,yc))
                    print((xc_,yc_))
                    print(ix)
                    print(arg_vector(list(map(int,[t[0]-xc,t[1]-yc])),list(map(int,[t_[0]-xc_,t_[1]-yc_]))))
                    movie.release()
                    cv2.imshow("img", frame)
                    cv2.imshow("img2", fgmask)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                    #cv2.imwrite(r"I:\motion_paper\pictures\elp_bd_tracked.png", frame)
                    #cv2.imwrite(r"I:\motion_paper\pictures\elp_fgmask.png", fgmask)
                    #cv2.imwrite(r"I:\motion_paper\figures\230815\fig1\fig1_d_0.png", frame)
                    #cv2.imwrite(r"I:\motion_paper\figures\230815\fig1\fig1_d_1.png", fgmask)
                    #231214
                    #cv2.imwrite(r"F:\motion_paper\figures\231109\program_validation\approx_body_elp_direct_result.png", frame)
                    #cv2.imwrite(r"F:\motion_paper\figures\231109\program_validation\body_blur_for_elp.png", fgmask)
                    #cv2.imwrite(r"F:\motion_paper\figures\231109\program_validation\approx_body_elp_direct_result_scoped.png", frame[math.floor(max([0,ref_n[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_n[1]+scope[1]//2]))+1,math.floor(max([0,ref_n[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_n[0]+scope[0]//2]))+1])
                    #cv2.imwrite(r"F:\motion_paper\figures\231109\program_validation\body_blur_for_elp_scoped.png", fgmask[math.floor(max([0,ref_n[1]-scope[1]//2])):math.ceil(min([fgmask.shape[:2][0],ref_n[1]+scope[1]//2]))+1,math.floor(max([0,ref_n[0]-scope[0]//2])):math.ceil(min([fgmask.shape[:2][1],ref_n[0]+scope[0]//2]))+1])
                    
                    break
                    #return ellipse
                t_=t
                xc_=xc
                yc_=yc
                #rmajor_=rmajor
                #if (not np.isnan(rminor) or not np.isnan(rminor_)) and not np.isnan(w) and abs(w)>0.4:
                    #print("e")
                    #rminor_=np.mean([s for s in [rminor_,rminor] if not np.isnan(s)])
                #elif not np.isnan(rminor) and np.isnan(w):
                    #rminor_=rminor
                rminor_=rminor
                if sk>2:
                    fai__=np.nan
                else:
                    fai__=fai_
                fai_=fai
                #ellipse_=ellipse
                sk=0
                
                results.loc[ix,"bird_elp_headside_x"]=t[0]
                results.loc[ix,"bird_elp_headside_y"]=t[1]
            else:
                sk+=1
                #xc_=np.nan
                #yc_=np.nan
                #cv2.imshow("img", frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #cv2.destroyAllWindows()
            #results.loc[len(results),:]=[filepath,ind]+[I[s] for s in list(results.columns)[2:]]
            #video.write(ff)
    
        #except Exception as e:
        except KeyboardInterrupt:
            #return C5
            #print(e)
            cv2.imshow("img", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            #el=ellipse
            print(t)
            print(t_)
            print(fai_)
            print(ix)
            movie.release()
            #return el
            break
    #video.release()
    cv2.destroyAllWindows()
    movie.release()
    return results

def direct_out_movie(results,fileout,t_marks,colors,color=(255,0,0),filepath=None):
    if filepath==None:
        filepath=results.iloc[0]["video_path"]
    #movie = cv2.VideoCapture(r"F:\2022-11-17-08-48-08.mp4")"F:\2022-11-17-08-49-18.mp4"
    movie = cv2.VideoCapture(filepath)
    #video = cv2.VideoWriter(r"F:\test_2022-11-17-08-48-08.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    video = cv2.VideoWriter(fileout, cv2.VideoWriter_fourcc(*'mp4v'), movie.get(cv2.CAP_PROP_FPS), (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for i in range(len(results)):
        try:
            movie.set(cv2.CAP_PROP_POS_FRAMES, results.loc[i,"frame"]) 
            ret, frame=movie.read()
            if not ret:
                break
            if not np.isnan(results.loc[i,t_marks[0]+"_x"]) and not np.isnan(results.loc[i,t_marks[1]+"_x"]):
                n1=(int(sorted([np.ceil(results.loc[i,t_marks[0]+"_x"]),np.floor(results.loc[i,t_marks[0]+"_x"])],key=lambda x:abs(results.loc[i,t_marks[0]+"_x"]-x))[0]),int(sorted([np.ceil(results.loc[i,t_marks[0]+"_y"]),np.floor(results.loc[i,t_marks[0]+"_y"])],key=lambda x:abs(results.loc[i,t_marks[0]+"_y"]-x))[0]))
                n2=(int(sorted([np.ceil(results.loc[i,t_marks[1]+"_x"]),np.floor(results.loc[i,t_marks[1]+"_x"])],key=lambda x:abs(results.loc[i,t_marks[1]+"_x"]-x))[0]),int(sorted([np.ceil(results.loc[i,t_marks[1]+"_y"]),np.floor(results.loc[i,t_marks[1]+"_y"])],key=lambda x:abs(results.loc[i,t_marks[1]+"_y"]-x))[0]))
                cv2.line(frame, n1, n2, color, 3)
            #cv2.imshow("img",  frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            for key in colors.keys():
                if results.loc[i,key+"_x"]!=None and not np.isnan(results.loc[i,key+"_x"]) and i>0 and not np.isnan(results.loc[i,key+"_r"]):
                    x_= [int(np.floor(results.loc[i,key+"_x"])),int(np.ceil(results.loc[i,key+"_x"]))][min(enumerate([results.loc[i,key+"_x"]-np.floor(results.loc[i,key+"_x"]),np.ceil(results.loc[i,key+"_x"])-results.loc[i,key+"_x"]]), key = lambda x:x[1])[0]]
                    y_= [int(np.floor(results.loc[i,key+"_y"])),int(np.ceil(results.loc[i,key+"_y"]))][min(enumerate([results.loc[i,key+"_y"]-np.floor(results.loc[i,key+"_y"]),np.ceil(results.loc[i,key+"_y"])-results.loc[i,key+"_y"]]), key = lambda x:x[1])[0]]
                    r_=[int(np.floor(results.loc[i,key+"_r"])),int(np.ceil(results.loc[i,key+"_r"]))][min(enumerate([results.loc[i,key+"_r"]-np.floor(results.loc[i,key+"_r"]),np.ceil(results.loc[i,key+"_r"])-results.loc[i,key+"_r"]]), key = lambda x:x[1])[0]]
                    cv2.circle(frame, (x_,y_), r_, colors[key]["color"], 3)
                    cv2.circle(frame, (x_,y_), 1, colors[key]["color"], -1)
            video.write(frame)
        except Exception as e:
            raise e
            break
    video.release()
    movie.release()

def get_silhouette_mult(filepath,results,sbj,scope=sorted([math.floor(70*2.1),math.ceil(70*2.1)],key=lambda x:x%2)[0],bird_area=600,display=False,savepath=None):
    """
    """"""
    function to get shilhouettes of subjects using OpticalFlow
    """"""
    see 'get_marker_centroid' and see 'track'
    
    filepath : video file path on which body location and estimation applied
    
    results : file including the x,y coodinates of head markers. out put from 'trck' or 'track_mult'
    
    sbj : dictionary to specifi the marker sets on each subject {"s0":["marker0","marker1"],"s1":["marker2","marker3"]} where s0,s1 represents subject
    """
    slab={key:i+1 for i,key in enumerate(list(sbj.keys()))}
    
    def innerfnc_to_get_id_cntr(fgmask,fglabel,fgc2,xs,ys,ref_n):
        fgc4_uniq,fgc4_not_uniq={},{}
        fglabel_id={key:cv2.inRange(fglabel,slab[key],slab[key]) for key in slab.keys()}                   
        if len(fgc2)>0:# and ind>mog_thresh:
            fgc4={key0:sorted(enumerate(fgc2),key=lambda x:(-np.mean([(lambda x1:x1 if x1<0 else 0)(cv2.pointPolygonTest(x[1], (int(xs[key0][i]),int(ys[key0][i])), measureDist=True)) for i in range(len(xs[key0]))]),sum([(ref_n[key0][i]-s)**2 for i, s in enumerate(get_moment(x[1]))])))[0] for key0 in ref_n.keys()}
            fgc4_=[]
            [s for s in (lambda x1:[x1_ for x1_ in x1 if x1.count(x1_)==1])([s_[0] for s_ in list(fgc4.values())]) if s not in fgc4_ and fgc4_.append(s)]
            fgc4_uniq={key:[val[1]] for key,val in fgc4.items() if val[0] in fgc4_ and all([subtracted_ratio(fglabel_id[key_],val[1])==0 for key_ in xs.keys() if len(xs[key_])==0])}
            
            for key,val in fgc4_uniq.items():
                cnt_img=np.zeros(fglabel.shape[:2], dtype=np.uint8)
                cv2.drawContours(cnt_img, val, -1, 255, -1)
                if sum([subtracted_ratio(cnt_img,s) for s in (lambda fh:[s[1] for s in enumerate(fh[0]) if fh[1][0][s[0]][-1]==-1])(cv2.findContours(fglabel_id[key], cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE))])>0.5:      
                    fgc4_uniq[key]=(lambda fh:(lambda fh_:fh_ if sum([cv2.contourArea(fs) for fs in fh_])>=1500 else val)([s[1] for s in enumerate(fh[0]) if fh[1][0][s[0]][-1]==-1]))(cv2.findContours(cv2.bitwise_and(fglabel_id[key],cnt_img), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE))
            
            fgc4_not_uniq={key:[val[1]] for key,val in fgc4.items() if val[0] not in fgc4_ or any([subtracted_ratio(fglabel_id[key_],val[1])>0 for key_ in xs.keys() if len(xs[key_])==0])}

            if len(fgc4_not_uniq)>0:#くっついているとき
                #多分残骸として複雑 rly_optflowが現状セレクションしてないのでfgmask=cv2.threshold(fglabel, 0, 255, cv2.THRESH_BINARY)[1]と同じな気がする
                fgmask=cv2.bitwise_and(fgmask, cv2.threshold(fglabel, 0, 255, cv2.THRESH_BINARY)[1])

                rly_optflow={ky:fglabel_id[{v:k for k,v in slab.items()}[ky]] for ky in list(set(list(np.ravel(fglabel)))) if ky>0}
                for ky in rly_optflow.keys():
                    fgmask=cv2.bitwise_or(fgmask,cv2.bitwise_and(rly_optflow[ky],cv2.bitwise_xor(rly_optflow[ky],fgmask)))
                    #fglabel_[np.where(fglabel==ky)]=0
                   
                fgc=sum([(lambda fh:[s[1] for s in enumerate(fh[0]) if fh[1][0][s[0]][-1]==-1])( cv2.findContours(cv2.inRange(cv2.bitwise_and(fglabel,fgmask),int(ky),int(ky)), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)) for ky in list(set(list(np.ravel(fglabel)))) if ky>0],[])
                fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area, fgc))
                if len(fgc2)>0:
                    fgc4={key0:sorted(enumerate(fgc2),key=lambda x:((lambda x0:-subtracted_ratio(fglabel_id[x0],x[1]) if slab[x0] in rly_optflow.keys() else 0)(key0),-np.mean([(lambda x1:x1 if x1<0 else 0)(cv2.pointPolygonTest(x[1], (int(xs[key0][i]),int(ys[key0][i])), measureDist=True)) for i in range(len(xs[key0]))]),sum([(ref_n[key0][i]-s)**2 for i, s in enumerate(get_moment(x[1]))])))[0] for key0 in ref_n.keys()}
                    fgc4_=[]
                    [s for s in (lambda x1:[x1_ for x1_ in x1 if x1.count(x1_)==1])([s_[0] for s_ in list(fgc4.values())]) if s not in fgc4_ and fgc4_.append(s)]
                    fgc4_uniq={key:[val[1]] for key,val in fgc4.items() if val[0] in fgc4_}
                    fgc4_not_uniq={key:[val[1]] for key,val in fgc4.items() if val[0] not in fgc4_ }

        return fgc4_uniq,fgc4_not_uniq
    
    movie = cv2.VideoCapture(filepath)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    #optical flow
    #area=15
    winsize=15
    polyN=(5,1.1)#(7,1.5)
    #thresh=70
    #scope=sorted([math.floor(thresh*2.1),math.ceil(thresh*2.1)],key=lambda x:x%2)[0]
    BS_Get=False
    mog_optflow={}
    while True:
        try:
            ind=movie.get(cv2.CAP_PROP_POS_FRAMES)
            
            ret, frame = movie.read()
            
            if not ret:
                break
            
            if ind==0:
                prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                fglabel=np.zeros(frame.shape[:2], dtype=np.uint8)
                flow=np.zeros((frame.shape[0],frame.shape[1],2), dtype=np.uint8)
                
                #if thresh==None:
                    #thresh=max([frame.shape[0],frame.shape[1]])
                #scope=sorted([math.floor(thresh*2.1),math.ceil(thresh*2.1)],key=lambda x:x%2)[0]
                
                continue
            
            elif ind>0:
                #if ind==66:
                    #break
                next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, winsize, 3, polyN[0], polyN[1], 0)
                prvs = next
                if BS_Get:
                    #fgflow=np.zeros(fglabel.shape[:2], dtype=np.uint8)
                    m=fglabel.max()
                    sl=[s for s in list(np.unique(fglabel)) if s>0]
                    r_dic={}
                    for si in sl:
                        fgflow_inds=np.array(list(map(int,np.ravel(np.array(np.where(fglabel==si)).T+flow[np.where(fglabel==si)][:,[1,0]])))).reshape(-1,2).T
                        r_dic[si]=fgflow_inds
                    for si,fgflow_inds in r_dic.items():
                        fglabel[fgflow_inds[0],fgflow_inds[1]]=si+m
                    fglabel[(np.where((fglabel>0)&(fglabel<=m)))]=0
                    fglabel[np.where(fglabel>0)]=fglabel[np.where(fglabel>0)]-m
                    #cv2.imshow("img4", np.stack([fglabel*125,fglabel*125,fglabel*125],axis=2))                  
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #cv2.destroyAllWindows()
            ix=max(list(results[results["frame"]==ind].index))
            xs={key0:[t for t in list(results.iloc[ix][[s for s in list(results.columns) if s in [sm+"_x" for sm in t_marks]]]) if t!=None and not np.isnan(t)] for key0,t_marks in sbj.items()}
            ys={key0:[t for t in list(results.iloc[ix][[s for s in list(results.columns) if s in [sm+"_y" for sm in t_marks]]]) if t!=None and not np.isnan(t)] for key0,t_marks in sbj.items()}
            ref_n={key0:(np.mean(xs[key0]),np.mean(ys[key0])) for key0 in xs.keys() if len(xs[key0])>0 and len(ys[key0])>0}
            ref_n={key0:(min([ref_n[key0][0],frame.shape[1]]),min([ref_n[key0][1],frame.shape[0]])) for key0 in ref_n.keys()}
            
            fgmask = fgbg.apply(frame)
            #if ind==69:
                #break
            mask=np.zeros(frame.shape[:2], dtype=np.uint8)
            for key in ref_n.keys():
                cv2.circle(mask,tuple(map(lambda x:int(sorted([math.floor(x),math.ceil(x)],key=lambda x_:abs(x-x_))[0]),ref_n[key])),scope, 255, -1)
            fgmask=cv2.bitwise_and(fgmask,mask)
            fglabel=cv2.bitwise_and(fglabel,mask)
            mog_optflow[ind]={"mog":fgmask,"optflow":{"flow":flow,"foward":fglabel},"flag":[1,0],"result":{"unique":{},"ununique":{}}}
            
            #cv2.imshow("img2", fgmask)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            if (fgmask==255).all():
                continue
            
            fglabel=cv2.morphologyEx(fglabel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
            #fgmask=cv2.morphologyEx(fglabel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            fgmask = cv2.blur(fgmask, (5, 5))
            fgmask =cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)[1]
            fgc, hir = cv2.findContours(fgmask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            fgc=[s[1] for s in enumerate(fgc) if hir[0][s[0]][-1]==-1]
            fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area, fgc))
            #if ind==24:
                #break
            if len(fgc2)==0:
                mog_optflow[ind]["flag"][0]=0
                fgmask=cv2.threshold(fglabel, 0, 255, cv2.THRESH_BINARY)[1]
                fgc, hir = cv2.findContours(fgmask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                fgc=[s[1] for s in enumerate(fgc) if hir[0][s[0]][-1]==-1]
                #240404修正　エラーが出てなかったら関係ない
                #fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area and sum([int(cv2.pointPolygonTest(x, (int(xs[key][i]),int(ys[key][i])), measureDist=False)>=0) for i in range(len(xs[key])) for key in xs.keys()])>0, fgc))
                fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area and sum([int(cv2.pointPolygonTest(x, (int(xs[key][i]),int(ys[key][i])), measureDist=False)>=0) if len(xs[key])>0 else 0 for i in range(len(xs[key])) for key in xs.keys()])>0, fgc))

            fgc4_uniq,fgc4_not_uniq=innerfnc_to_get_id_cntr(fgmask,fglabel,fgc2,xs,ys,ref_n)
        
            #fgmask_ = np.zeros(fgmask.shape[:2], dtype=np.uint8)
            #cv2.drawContours(fgmask_, sum(list(fgc4_uniq.values()),[]),-1, 255, -1)
    
            if ~BS_Get and len(fgc4_uniq)>0:
                BS_Get=True
            #cv2.imshow("img2", fgmask_)                  
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            if len(fgc4_uniq)>0:
                for key in fgc4_uniq.keys():
                    fglabel[np.where(fglabel==slab[key])]=0
                #fglabel_test=np.stack([fglabel,fglabel,fglabel],axis=2)
                for key in fgc4_uniq.keys():
                    cv2.drawContours(fglabel, fgc4_uniq[key],-1, slab[key], -1)
                    #cv2.drawContours(fglabel_test, fgc4_uniq[key],-1, [0,(255,0,0),(0,255,0)][slab[key]], -1)
                #cv2.imshow("img3", fglabel_test)                  
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #cv2.destroyAllWindows()
            
            mog_optflow[ind]["result"]["unique"]=fgc4_uniq
            mog_optflow[ind]["result"]["ununique"]=fgc4_not_uniq
            mog_optflow[ind]["optflow"]["foward"]=fglabel
            if len(fgc4_uniq)>0 and len(fgc4_not_uniq)==0:#1つ目の条件は2つ目が満たされるとき必ず満たされる
                mog_optflow[ind]["flag"][1]=1
            
            if display:
                fgmask_ = np.zeros(fgmask.shape[:2], dtype=np.uint8)
                fgmask_=np.stack([fgmask_,fgmask_,fgmask_],axis=2)
                for k,v in fgc4_uniq.items():
                    color=tuple(map(lambda cx:max([int(cx),255]),list(np.array(sns.blend_palette(['blue', 'red', 'green'], max(list(slab.values())))[slab[k]-1])*255)))
                    cv2.drawContours(fgmask_, v,-1, color, -1)
                cv2.imshow("img1", fgmask_)                  
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
        except KeyboardInterrupt as e:
            print(e)
            print("foward")
            print(ind)
            movie.release()
            return mog_optflow

        
    movie.release()
    
    s=0        
    for i,ind in enumerate(sorted(list(mog_optflow.keys()),reverse=True)):
        try:
            fgmask=mog_optflow[ind]["mog"]
            flow=mog_optflow[ind]["optflow"]["flow"]
            if (s==0 and len(mog_optflow[ind]["result"]["unique"])>0) or mog_optflow[ind]["flag"][1]==1:
                fglabel=np.zeros(fgmask.shape[:2], dtype=np.uint8)
                for key in fgc4_uniq.keys():
                    cv2.drawContours(fglabel, fgc4_uniq[key],-1, slab[key], -1)
                s=1
                continue
            
            if s==0 or (len(mog_optflow[ind]["result"]["unique"])>0 and len(mog_optflow[ind]["result"]["ununique"])==0):
                continue
            
            m=fglabel.max()
            sl=[s for s in list(np.unique(fglabel)) if s>0]
            r_dic={}
            for si in sl:
                fgflow_inds=np.array(list(map(int,np.ravel(np.array(np.where(fglabel==si)).T+flow[np.where(fglabel==si)][:,[1,0]])))).reshape(-1,2).T
                r_dic[si]=fgflow_inds
            for si,fgflow_inds in r_dic.items():
                fglabel[fgflow_inds[0],fgflow_inds[1]]=si+m
            fglabel[(np.where((fglabel>0)&(fglabel<=m)))]=0
            fglabel[np.where(fglabel>0)]=fglabel[np.where(fglabel>0)]-m
            
            ix=max(list(results[results["frame"]==ind].index))
            xs={key0:[t for t in list(results.iloc[ix][[s for s in list(results.columns) if s in [sm+"_x" for sm in t_marks]]]) if t!=None and not np.isnan(t)] for key0,t_marks in sbj.items()}
            ys={key0:[t for t in list(results.iloc[ix][[s for s in list(results.columns) if s in [sm+"_y" for sm in t_marks]]]) if t!=None and not np.isnan(t)] for key0,t_marks in sbj.items()}
            ref_n={key0:(np.mean(xs[key0]),np.mean(ys[key0])) for key0 in xs.keys() if len(xs[key0])>0 and len(ys[key0])>0}
            ref_n={key0:(min([ref_n[key0][0],fgmask.shape[1]]),min([ref_n[key0][1],fgmask.shape[0]])) for key0 in ref_n.keys()}
            mask=np.zeros(fgmask.shape[:2], dtype=np.uint8)
            for key in ref_n.keys():
                cv2.circle(mask,tuple(map(lambda x:int(sorted([math.floor(x),math.ceil(x)],key=lambda x_:abs(x-x_))[0]),ref_n[key])),scope, 255, -1)
            fgmask=cv2.bitwise_and(fgmask,mask)
            fglabel=cv2.bitwise_and(fglabel,mask)
        
            mog_optflow[ind]["optflow"]["backward"]=fglabel
            fglabel=cv2.morphologyEx(fglabel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
            
            if mog_optflow[ind]["flag"][0]>0:
                fgmask = cv2.blur(fgmask, (5, 5))
                fgmask =cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)[1]
                fgc, hir = cv2.findContours(fgmask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                fgc=[s[1] for s in enumerate(fgc) if hir[0][s[0]][-1]==-1]
                fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area, fgc))
            #if ind==520:
                #break
            if mog_optflow[ind]["flag"][0]==0 or len(fgc2)==0:      
                #fgmask=cv2.threshold(cv2.blur(fglabel, (5, 5)), 0, 255, cv2.THRESH_BINARY)[1]
                fgmask=cv2.threshold(fglabel, 0, 255, cv2.THRESH_BINARY)[1]
                fgc, hir = cv2.findContours(fgmask, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                fgc=[s[1] for s in enumerate(fgc) if hir[0][s[0]][-1]==-1]
                
                #240404修正　エラーが出てなかったら関係ない
                #fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area and sum([int(cv2.pointPolygonTest(x, (int(xs[key][i]),int(ys[key][i])), measureDist=False)>=0) for i in range(len(xs[key])) for key in xs.keys()])>0, fgc))
                fgc2=list(filter(lambda x: cv2.contourArea(x) >= bird_area and sum([int(cv2.pointPolygonTest(x, (int(xs[key][i]),int(ys[key][i])), measureDist=False)>=0) if len(xs[key])>0 else 0 for i in range(len(xs[key])) for key in xs.keys()])>0, fgc))

            fgc4_uniq,fgc4_not_uniq=innerfnc_to_get_id_cntr(fgmask,fglabel,fgc2,xs,ys,ref_n)
                
            #fgmask_ = np.zeros(fgmask.shape[:2], dtype=np.uint8)
            #cv2.drawContours(fgmask_, sum(list(fgc4_uniq.values()),[]),-1, 255, -1)
            #cv2.imshow("img2", fgmask_)                  
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            if len(fgc4_uniq)>len(mog_optflow[ind]["result"]["unique"]) or len(fgc4_not_uniq)<len(mog_optflow[ind]["result"]["ununique"]):#betterならば
                for key in fgc4_uniq.keys():
                    fglabel[np.where(fglabel==slab[key])]=0
                #fglabel_test=np.stack([fglabel,fglabel,fglabel],axis=2)
                for key in fgc4_uniq.keys():
                    cv2.drawContours(fglabel, fgc4_uniq[key],-1, slab[key], -1)
                    #cv2.drawContours(fglabel_test, fgc4_uniq[key],-1, [0,(255,0,0),(0,255,0)][slab[key]], -1)
                #cv2.imshow("img3", fglabel_test)                  
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #cv2.destroyAllWindows()
                mog_optflow[ind]["result"]["unique"]=fgc4_uniq
                mog_optflow[ind]["result"]["ununique"]=fgc4_not_uniq
                mog_optflow[ind]["optflow"]["backward"]=fglabel
            if display:
                fgmask_ = np.zeros(fgmask.shape[:2], dtype=np.uint8)
                fgmask_=np.stack([fgmask_,fgmask_,fgmask_],axis=2)
                for k,v in fgc4_uniq.items():
                    color=tuple(map(lambda cx:max([int(cx),255]),list(np.array(sns.blend_palette(['blue', 'red', 'green'], max(list(slab.values())))[slab[k]-1])*255)))
                    cv2.drawContours(fgmask_, v,-1, color, -1)
                cv2.imshow("img2", fgmask_)                  
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
        except KeyboardInterrupt as e:
            print(e)
            print("backward")
            print(ind)
            movie.release()
            return mog_optflow
        
    if savepath:
        if not os.path.exists(savepath[0]):
            os.mkdir(savepath[0])
        with shelve.open("{}/{}".format(savepath[0],savepath[1]),"c") as f:
            f["memo"]="mog_optflow"
            f["mog_optflow"]=mog_optflow
    res={ind:mog_optflow[ind]["result"] for ind in sorted(list(mog_optflow.keys()))}
    return res

#ntime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#savepath=("{}/{}".format(basepath,ntime+"_silhouette"),",filepath.split("\\")[-1])
def approx_body_ellipse_mult(filepath,results,sbj,scope=sorted([math.floor(70*2.1),math.ceil(70*2.1)],key=lambda x:x%2)[0],bird_area=600,sti=None,marker_out=0.25,mog_thresh=300,judge_range=100,display=False,savepath=None,est_body_direct=True):
    """
    
    """"""
    functio to estimate body center and direction from shilhouettes from get_silhouette_mult
    """"""
    
    see 'get_marker_centroid', track', 'approx_body_ellipse' and 'get_silhouette_mult'
    
        
    sbj : dictionary to specifi the marker sets on each subject {"s0":["marker0","marker1"],"s1":["marker2","marker3"]} where s0,s1 represents subject
    """
    
    res=get_silhouette_mult(filepath,results,sbj,scope=scope,bird_area=bird_area,display=display,savepath=savepath)
    #res=get_silhouette_mult(filepath,results,sbj,display=True)

    for key,t_marks in sbj.items():
        try:
            t_=np.nan
            xc_=np.nan
            yc_=np.nan
            fai_=np.nan
            fai__=np.nan
            rminor_=np.nan
            sk=0
            mh=0
            ixs=-1
            t1_=np.nan
            t2_=np.nan
            for ind in sorted(list(res.keys())):
                ix=max(list(results[results["frame"]==ind].index))
                if ixs==-1:
                    ixs=ix
                if key in res[ind]["unique"].keys() and len(res[ind]["unique"][key][0]) >= 5:
                    ellipse = cv2.fitEllipse(res[ind]["unique"][key][0])
                    if  any([np.isnan(s) for s in list(ellipse[0])+list(ellipse[1])]):
                        continue
        
                    xc,yc=ellipse[0]
                    rmajor = max(ellipse[1])/2
                    rminor = min(ellipse[1])/2
                    if rminor==0:
                        sk+=1
                        continue
                    angle=ellipse[2]
                    if angle > 90:
                        angle = angle - 90
                    else:
                        angle = angle + 90
                    x1 = xc + math.cos(math.radians(angle))*rmajor
                    y1 = yc + math.sin(math.radians(angle))*rmajor
                    x2 = xc + math.cos(math.radians(angle+180))*rmajor
                    y2 = yc + math.sin(math.radians(angle+180))*rmajor
                    x3 = xc + math.cos(math.radians(angle+90))*rminor
                    y3 = yc + math.sin(math.radians(angle+90))*rminor
                    x4 = xc + math.cos(math.radians(angle+270))*rminor
                    y4 = yc + math.sin(math.radians(angle+270))*rminor
                    results.loc[ix,key+"_el1_x"]=x1
                    results.loc[ix,key+"_el1_y"]=y1
                    results.loc[ix,key+"_el2_x"]=x2
                    results.loc[ix,key+"_el2_y"]=y2
                    results.loc[ix,key+"_el3_x"]=x3
                    results.loc[ix,key+"_el3_y"]=y3
                    results.loc[ix,key+"_el4_x"]=x4
                    results.loc[ix,key+"_el4_y"]=y4
                    results.loc[ix,key+"_elp_center_x"]=xc
                    results.loc[ix,key+"_elp_center_y"]=yc
                    if (not all([s in results.columns for s in sum([[sm+"_x",sm+"_y"] for sm in t_marks],[])])) or (not est_body_direct):
                        continue
                    
                    if (np.isnan(results.loc[ix,t_marks[0]+"_x"]) or np.isnan(results.loc[ix,t_marks[1]+"_x"])):
                        sk+=1
                        #print("b")
                        continue
                    #calcf+=1
                    vec0=[1,0]
                    vec1=[results.loc[ix,t_marks[1]+"_x"]-results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[1]+"_y"]-results.loc[ix,t_marks[0]+"_y"]]
                    if vec1[0]**2+vec1[1]**2==0:
                        sk+=1
                        #print("c")
                        continue
                    fai=arg_vector(vec0,vec1)
                    if vec1[1]<0:
                        fai=360-fai
                    ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                    if ind<mog_thresh and mh<judge_range and len(ms)==0:
                        mh+=1
                        w=1
                        if (type(t1_)!=tuple and np.isnan(t1_)) and (type(t2_)!=tuple and np.isnan(t2_)) or sk>2:
                            mh=0
                            #results.loc["bird_elp_center_x"]=np.nan    
                            #results.loc["bird_elp_center_y"]=np.nan    
                            results.loc[ix,key+"_elp_headside_x1"]=np.nan
                            results.loc[ix,key+"_elp_headside_y1"]=np.nan
                            results.loc[ix,key+"_elp_headside_x2"]=np.nan
                            results.loc[ix,key+"_elp_headside_y2"]=np.nan 
                            ixs=ix
                            t1=tuple(map(int,[x1,y1]))
                            t2=tuple(map(int,[x2,y2]))
                        elif np.isnan(fai_):   #下とうまく組み合わせたい
                            #print("c")
                            print(ix)
                            cand_p1=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[t1_[0]-xc_,t1_[1]-yc_]))[:2]#
                            cand_p2=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[t2_[0]-xc_,t2_[1]-yc_]))[:2]#np.sqrt(((x[0]-xc)-(t_[0]-xc_))**2+((x[1]-yc)-(t_[1]-yc_))**2))[:2]
                            t1=tuple(map(int,sorted(cand_p1,key=lambda x:x[2]*w,reverse=True)[0][:2]))
                            t2=tuple(map(int,sorted(cand_p2,key=lambda x:x[2]*w,reverse=True)[0][:2]))
                        else:
                            #print("d")
                            th=math.radians(np.mean(list(filter(lambda x: ~np.isnan(x),[sorted([fai-fai_,-(360-abs(fai-fai_))],key=lambda n:abs(n))[0],sorted([fai_-fai__,-(360-abs(fai_-fai__))],key=lambda n1:abs(n1))[0]]))))
                            rvec=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
                            cand_v1=np.dot(rvec, np.array([t1_[0]-xc_,t1_[1]-yc_]))#+np.array([xc-xc_,yc-yc_])
                            cand_v2=np.dot(rvec, np.array([t2_[0]-xc_,t2_[1]-yc_]))
                            cva1=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,list(cand_v1)),list(cand_v1))
                            cva2=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,list(cand_v2)),list(cand_v2))
                            t_va1=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,[t1_[0]-xc_,t1_[1]-yc_]),[t1_[0]-xc_,t1_[1]-yc_])#[t_[0]+xc-xc_,t_[1]+yc-yc_]
                            t_va2=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,[t2_[0]-xc_,t2_[1]-yc_]),[t2_[0]-xc_,t2_[1]-yc_])#
        
                            #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(mocap.arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(mocap.arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                            ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                            #wms=list(map(lambda x:min([np.sqrt((x[0]-ms[i][0])**2+(x[1]-ms[i][1])**2) for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]]))
                            wms=list(map(lambda x:min([(lambda n2:n2 if abs(n2)>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor else 0)(-np.sqrt((xc-ms[i][0])**2+(yc-ms[i][1])**2)*math.cos(math.radians(angle+x[3]-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[ms[i][0]-xc,ms[i][1]-yc]),[ms[i][0]-xc,ms[i][1]-yc]))))+x[2] for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor,0],[x2,y2,rmajor,180],[x3,y3,rminor,90],[x4,y4,rminor,270]]))
                            wts=[np.mean([1/wms[0],1/wms[2]]),np.mean([1/wms[1],1/wms[2]]),np.mean([1/wms[1],1/wms[3]]),np.mean([1/wms[0],1/wms[3]])]
                            cand_p=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]],reverse=True)
                            if wts[cand_p[0][0]]==wts[cand_p[1][0]] and wms[int(cand_p[0][0]%3==0)]==1:
                                cand_p1=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]]*(lambda y,z:sum([0 if y[2*iy]>=z[2*(iz+1)-1] or z[2*iz]>=y[2*(iy+1)-1] else (lambda k:k[2]-k[1])(sorted(y[2*iy:2*(iy+1)]+z[2*iz:2*(iz+1)])) for iz in range(len(z)//2) for iy in range(len(y)//2)]))((lambda n1:[n1[0],n1[1]] if n1[1]-n1[0]<=90 else [n1[1],360,0,n1[0]])(sorted([[angle%360,(angle+90)%360],[(angle+90)%360,(angle+180)%360],[(angle+180)%360,(angle+270)%360],[(angle+270)%360,(angle)%360]][x[0]])),(lambda n2:[n2[0],n2[1]] if n2[1]-n2[0]<=180 else [n2[1],360,0,n2[0]])(sorted([t_va1,cva1]))),reverse=True)[0][1]
                                cand_p2=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]]*(lambda y,z:sum([0 if y[2*iy]>=z[2*(iz+1)-1] or z[2*iz]>=y[2*(iy+1)-1] else (lambda k:k[2]-k[1])(sorted(y[2*iy:2*(iy+1)]+z[2*iz:2*(iz+1)])) for iz in range(len(z)//2) for iy in range(len(y)//2)]))((lambda n1:[n1[0],n1[1]] if n1[1]-n1[0]<=90 else [n1[1],360,0,n1[0]])(sorted([[angle%360,(angle+90)%360],[(angle+90)%360,(angle+180)%360],[(angle+180)%360,(angle+270)%360],[(angle+270)%360,(angle)%360]][x[0]])),(lambda n2:[n2[0],n2[1]] if n2[1]-n2[0]<=180 else [n2[1],360,0,n2[0]])(sorted([t_va2,cva2]))),reverse=True)[0][1]
                            else:
                                cand_p1=cand_p[0][1]
                                cand_p2=cand_p[0][1]
        
                            wa=np.mean([min([abs(cand_p1[1][3]+angle-cva1),abs(360-abs(cand_p1[1][3]+angle-cva1))])/max([0.0000000001,min([abs(cand_p1[0][3]+angle-cva1),abs(360-abs(cand_p1[0][3]+angle-cva1))])]),min([abs(cand_p1[1][3]+angle-t_va1),abs(360-abs(cand_p1[1][3]+angle-t_va1))])/max([0.0000000001,min([abs(cand_p1[0][3]+angle-t_va1),abs(360-abs(cand_p1[0][3]+angle-t_va1))])])])
                            wi=np.mean([min([abs(cand_p1[0][3]+angle-cva1),abs(360-abs(cand_p1[0][3]+angle-cva1))])/max([0.0000000001,min([abs(cand_p1[1][3]+angle-cva1),abs(360-abs(cand_p1[1][3]+angle-cva1))])]),min([abs(cand_p1[0][3]+angle-t_va1),abs(360-abs(cand_p1[0][3]+angle-t_va1))])/max([0.0000000001,min([abs(cand_p1[1][3]+angle-t_va1),abs(360-abs(cand_p1[1][3]+angle-t_va1))])])])
                            if rmajor/rminor<1.2 and sk<1:
                                #w=1
                                w=wa-wi
                            #w=abs(rmajor-rminor_)*wa/rminor_-abs(rminor-rminor_)*wi/rminor_
                            cand_p1=[p[:3] for p in cand_p1]
                            t1=tuple(map(int,sorted(cand_p1,key=lambda x:x[2]*w,reverse=True)[0][:2]))
                            w=1
                            wa=np.mean([min([abs(cand_p2[1][3]+angle-cva2),abs(360-abs(cand_p2[1][3]+angle-cva2))])/max([0.0000000001,min([abs(cand_p2[0][3]+angle-cva2),abs(360-abs(cand_p2[0][3]+angle-cva2))])]),min([abs(cand_p2[1][3]+angle-t_va2),abs(360-abs(cand_p2[1][3]+angle-t_va2))])/max([0.0000000001,min([abs(cand_p2[0][3]+angle-t_va2),abs(360-abs(cand_p2[0][3]+angle-t_va2))])])])
                            wi=np.mean([min([abs(cand_p2[0][3]+angle-cva2),abs(360-abs(cand_p2[0][3]+angle-cva2))])/max([0.0000000001,min([abs(cand_p2[1][3]+angle-cva2),abs(360-abs(cand_p2[1][3]+angle-cva2))])]),min([abs(cand_p2[0][3]+angle-t_va2),abs(360-abs(cand_p2[0][3]+angle-t_va2))])/max([0.0000000001,min([abs(cand_p2[1][3]+angle-t_va2),abs(360-abs(cand_p2[1][3]+angle-t_va2))])])])
                            if rmajor/rminor<1.2 and sk<1:
                                #w=1
                                w=wa-wi
                            #w=abs(rmajor-rminor_)*wa/rminor_-abs(rminor-rminor_)*wi/rminor_
                            cand_p2=[p[:3] for p in cand_p2]
                            t2=tuple(map(int,sorted(cand_p2,key=lambda x:x[2]*w,reverse=True)[0][:2]))
                        t1_=t1
                        t2_=t2
                        xc_=xc
                        yc_=yc
                        rminor_=rminor
                        if sk>2:
                            fai__=np.nan
                        else:
                            fai__=fai_
                        fai_=fai
                        #ellipse_=ellipse
                        sk=0
                        #results.loc[ix,"bird_elp_center_x"]=xc
                        #results.loc[ix,"bird_elp_center_y"]=yc
                        results.loc[ix,key+"_elp_headside_x1"]=t1[0]
                        results.loc[ix,key+"_elp_headside_y1"]=t1[1]
                        results.loc[ix,key+"_elp_headside_x2"]=t2[0]
                        results.loc[ix,key+"_elp_headside_y2"]=t2[1]
                        #print("1")
                        if mh==judge_range:
                            dr=results.loc[ixs:,[t_marks[0]+"_x",t_marks[0]+"_y",t_marks[1]+"_x",t_marks[1]+"_y","bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"]].dropna()
                            tm1=np.mean((np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[1]+"_y"])**2))/2)
                            tm2=np.mean((np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[1]+"_y"])**2))/2)
                            if tm1==tm2:
                                dr=results.loc[int((ix+ixs)/2):ix,[t_marks[0]+"_x",t_marks[0]+"_y",t_marks[1]+"_x",t_marks[1]+"_y","bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"]].dropna()
                                tm1=np.mean((np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x1"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y1"]-dr[t_marks[1]+"_y"])**2))/2)
                                tm2=np.mean((np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[0]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[0]+"_y"])**2)+np.sqrt((dr["bird_elp_headside_x2"]-dr[t_marks[1]+"_x"])**2+(dr["bird_elp_headside_y2"]-dr[t_marks[1]+"_y"])**2))/2)
            
                            if tm1<tm2:
                                results["bird_elp_headside_x"]=results["bird_elp_headside_x1"]
                                results["bird_elp_headside_y"]=results["bird_elp_headside_y1"]
                                t_=t1
                            else:
                                results["bird_elp_headside_x"]=results["bird_elp_headside_x2"]
                                results["bird_elp_headside_y"]=results["bird_elp_headside_y2"]
                                t_=t2
                            ixs=ix+1
                            results.drop(["bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"],axis=1,inplace=True)

                            continue
                    if "bird_elp_headside_x1" in results.columns:
                        results.drop(["bird_elp_headside_x1","bird_elp_headside_y1","bird_elp_headside_x2","bird_elp_headside_y2"],axis=1,inplace=True)
 
                    
                    #w=abs(rmajor-rminor_)/rminor_-abs(rminor-rminor_)/rminor_
                    w=1
                    if (type(t_)!=tuple and np.isnan(t_)) or sk>2:
                        ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
        
                        #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if max([abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle-fai))),abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle+90-fai)))])>=0.25*rmajor]
                        if len(ms)>0:
                            #cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:min([np.sqrt((x[0]-ms[i][0])**2+(x[1]-ms[i][1])**2) for i in range(len(ms))]))[:2]
        
                            cand_p=sorted(list(map(lambda n:(n[0],(lambda x:min([(lambda x2:x2 if abs(x2)>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor else 0)(-np.sqrt((xc-ms[i][0])**2+(yc-ms[i][1])**2)*math.cos(math.radians(angle+x[3]-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[ms[i][0]-xc,ms[i][1]-yc]),[ms[i][0]-xc,ms[i][1]-yc]))))+x[2] for i in range(len(ms))])/x[2])(n[1])),enumerate([[x1,y1,rmajor,0],[x2,y2,rmajor,180],[x3,y3,rminor,90],[x4,y4,rminor,270]]))),key=lambda n_:n_[1])
                            if cand_p[0][1]==cand_p[1][1] and [[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[0][0]][2]==[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[1][0]][2]:
                                sk+=1#180度では判別できない
                                continue   
                            cand_p=[[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[0][0]],[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]][cand_p[1][0]]]
                            #rmajor/rminor<1.2 and 
                            if rmajor/rminor<1.2 and sorted([(-1,0)]+[(i,(-np.sqrt((xc-results.loc[ix,t_marks[i]+"_x"])**2+(yc-results.loc[ix,t_marks[i]+"_y"])**2)*math.cos(math.radians(angle+0-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[results.loc[ix,t_marks[i]+"_x"]-xc,results.loc[ix,t_marks[i]+"_y"]-yc]),[results.loc[ix,t_marks[i]+"_x"]-xc,results.loc[ix,t_marks[i]+"_y"]-yc]))))) for i in range(len(t_marks))],key=lambda xa:xa[1])[1][0]==1:
                                #楕円らしくないか、tipのほうがcenterに近いか
                                cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[results.loc[ix,t_marks[1]+"_x"]-results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[1]+"_y"]-results.loc[ix,t_marks[0]+"_y"]]))[:2]
        
                            #print("a")
                            #print(ix)
                        else:
                            #print("b")
                            sk+=1
                            continue
                        ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                        #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if max([abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle-fai))),abs(np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*math.cos(math.radians(angle+90-fai)))])>=0.25*rmajor]
        
                    elif np.isnan(fai_):   #下とうまく組み合わせたい
                        #print("c")
                        cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda x:arg_vector([x[0]-xc,x[1]-yc],[t_[0]-xc_,t_[1]-yc_]))[:2]#np.sqrt(((x[0]-xc)-(t_[0]-xc_))**2+((x[1]-yc)-(t_[1]-yc_))**2))[:2]
                    else:
                        #print("d")
                        th=math.radians(np.mean(list(filter(lambda x: ~np.isnan(x),[sorted([fai-fai_,-(360-abs(fai-fai_))],key=lambda n:abs(n))[0],sorted([fai_-fai__,-(360-abs(fai_-fai__))],key=lambda n1:abs(n1))[0]]))))
                        rvec=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]])
                        cand_v=np.dot(rvec, np.array([t_[0]-xc_,t_[1]-yc_]))#+np.array([xc-xc_,yc-yc_])
                        cva=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,list(cand_v)),list(cand_v))
                        t_va=(lambda x,y:360-x if y[1]<0 else x)(arg_vector(vec0,[t_[0]-xc_,t_[1]-yc_]),[t_[0]-xc_,t_[1]-yc_])#[t_[0]+xc-xc_,t_[1]+yc-yc_]
                        #**これじゃだめか
                        #cand_p=sorted(enumerate([[[x1,y1,rmajor],[x3,y3,rminor]],[[x3,y3,rminor],[x2,y2,rmajor]],[[x2,y2,rmajor],[x4,y4,rminor]],[[x4,y4,rminor],[x1,y1,rmajor]]]),key=lambda x:(lambda y,z:0 if y%360>=z[1] or z[0]>=y%360+90 else (lambda k:k[2]-k[1])(sorted([y%360,y%360+90]+z)))([angle,angle+90,angle+180,angle+270][x[0]],(lambda x1:sorted([x1[0]+360*min([1,(x1[1]-x1[0])//180]),x1[1]]))(sorted([t_va,cva]))),reverse=True)[0][1]
                        #wts=list(map(lambda x:mocap.arg_vector([x[0]-xc,x[1]-yc],[t_[0]-xc_,t_[1]-yc_]),[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]]))
                        #wts=[min(wts)/s for s in wts]
                        #ms=[s for s in [[results.loc[ix,t_marks[0]+"_x"],results.loc[ix,t_marks[0]+"_y"]],[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                        ms=[s for s in [[results.loc[ix,t_marks[1]+"_x"],results.loc[ix,t_marks[1]+"_y"]]] if np.sqrt((s[0]-xc)**2+(s[1]-yc)**2)*max([abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc])))),abs(math.cos(math.radians(angle-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[s[0]-xc,s[1]-yc]),[s[0]-xc,s[1]-yc]))))])>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor]
                        #wms=list(map(lambda x:min([np.sqrt((x[0]-ms[i][0])**2+(x[1]-ms[i][1])**2) for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]]))
                        wms=list(map(lambda x:min([(lambda n2:n2 if abs(n2)>=marker_out*(1+int(sk>0)-0.5*int(sk>0))*rmajor else 0)(-np.sqrt((xc-ms[i][0])**2+(yc-ms[i][1])**2)*math.cos(math.radians(angle+x[3]-(lambda x_,y:360-x_ if y[1]<0 else x_)(arg_vector(vec0,[ms[i][0]-xc,ms[i][1]-yc]),[ms[i][0]-xc,ms[i][1]-yc]))))+x[2] for i in range(len(ms))])/x[2] if len(ms)>0 else 1,[[x1,y1,rmajor,0],[x2,y2,rmajor,180],[x3,y3,rminor,90],[x4,y4,rminor,270]]))
                        wts=[np.mean([1/wms[0],1/wms[2]]),np.mean([1/wms[1],1/wms[2]]),np.mean([1/wms[1],1/wms[3]]),np.mean([1/wms[0],1/wms[3]])]
                        cand_p=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]],reverse=True)
                        if wts[cand_p[0][0]]==wts[cand_p[1][0]] and wms[int(cand_p[0][0]%3==0)]==1:#msが0か、偏りなし
                            #扇形overwrap    
                            cand_p=sorted(enumerate([[[x1,y1,rmajor,0],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x3,y3,rminor,90]],[[x2,y2,rmajor,180],[x4,y4,rminor,270]],[[x1,y1,rmajor,0],[x4,y4,rminor,270]]]),key=lambda x:wts[x[0]]*(lambda y,z:sum([0 if y[2*iy]>=z[2*(iz+1)-1] or z[2*iz]>=y[2*(iy+1)-1] else (lambda k:k[2]-k[1])(sorted(y[2*iy:2*(iy+1)]+z[2*iz:2*(iz+1)])) for iz in range(len(z)//2) for iy in range(len(y)//2)]))((lambda n1:[n1[0],n1[1]] if n1[1]-n1[0]<=90 else [n1[1],360,0,n1[0]])(sorted([[angle%360,(angle+90)%360],[(angle+90)%360,(angle+180)%360],[(angle+180)%360,(angle+270)%360],[(angle+270)%360,(angle)%360]][x[0]])),(lambda n2:[n2[0],n2[1]] if n2[1]-n2[0]<=180 else [n2[1],360,0,n2[0]])(sorted([t_va,cva]))),reverse=True)[0][1]
                        else:
                            cand_p=cand_p[0][1]
        
                        wa=np.mean([min([abs(cand_p[1][3]+angle-cva),abs(360-abs(cand_p[1][3]+angle-cva))])/max([0.0000000001,min([abs(cand_p[0][3]+angle-cva),abs(360-abs(cand_p[0][3]+angle-cva))])]),min([abs(cand_p[1][3]+angle-t_va),abs(360-abs(cand_p[1][3]+angle-t_va))])/max([0.0000000001,min([abs(cand_p[0][3]+angle-t_va),abs(360-abs(cand_p[0][3]+angle-t_va))])])])#0.0000000001は0の代わりのダミー
                        wi=np.mean([min([abs(cand_p[0][3]+angle-cva),abs(360-abs(cand_p[0][3]+angle-cva))])/max([0.0000000001,min([abs(cand_p[1][3]+angle-cva),abs(360-abs(cand_p[1][3]+angle-cva))])]),min([abs(cand_p[0][3]+angle-t_va),abs(360-abs(cand_p[0][3]+angle-t_va))])/max([0.0000000001,min([abs(cand_p[1][3]+angle-t_va),abs(360-abs(cand_p[1][3]+angle-t_va))])])])
                        if rmajor/rminor<1.2 and sk<1:
                            #w=1
                            w=wa-wi
                        #w=abs(rmajor-rminor_)*wa/rminor_-abs(rminor-rminor_)*wi/rminor_
                        cand_p=[p[:3] for p in cand_p]
                    #体軸を選ぶ
                    ##単軸にとっての変化量の差分で重みづける
                    #w=abs(rmajor-rminor_)/rminor_-abs(rminor-rminor_)/rminor_#プラスマイナスでいいか
                    #if rminor_>35 and w>-0.2 and w<0:
                        #w=1
                    #w=1
                    #print(w)
                    #cand_p=sorted([[x1,y1,rmajor],[x2,y2,rmajor],[x3,y3,rminor],[x4,y4,rminor]],key=lambda:mocap.arg_vector(cand_v,[x[0],x[1]]))[:2]
                    t=tuple(map(int,sorted(cand_p,key=lambda x:x[2]*w,reverse=True)[0][:2]))
        
                    #print(t)
                    if display or (sti!=None and ix==sti):
                        frame=get_frame_index(filepath,ix,show=False)
                        fgmask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.drawContours(fgmask, res[ind]["unique"][key], -1, 255, -1)
                        cv2.line(frame, (int(xc),int(yc)), t, (0, 0, 255), 3)
                        cv2.circle(frame, (int(xc),int(yc)), 2, (0, 255, 0), -1)
                        cv2.ellipse(frame,ellipse,(255,0,0),2)
                        cv2.line(frame, (int(results.loc[ix,t_marks[0]+"_x"]),int(results.loc[ix,t_marks[0]+"_y"])), (int(results.loc[ix,t_marks[1]+"_x"]),int(results.loc[ix,t_marks[1]+"_y"])), (255, 0, 255), 3)
                        
                        #paper fig
                        #cv2.line(frame, (int(xc),int(yc)), t, (200, 0, 100), 3)
                        #cv2.circle(frame, (int(xc),int(yc)), 2, (200, 0, 100), -1)
                        #cv2.ellipse(frame,ellipse,(200, 0, 100),2)
                        
                        cv2.imshow("img", frame)
                        cv2.imshow("img2", fgmask)
                       
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                    
                    #if type(t_)==tuple and type(t)==tuple and mocap.arg_vector(list(map(int,[t[0]-xc,t[1]-yc])),list(map(int,[t_[0]-xc_,t_[1]-yc_])))>105:
                    if sti!=None and ix==sti:
                    #if ix!=374 and (cand_p[0]==[x2,y2,rmajor] or cand_p[1]==[x2,y2,rmajor]):
                    #if type(t_)==tuple and type(t)==tuple and ix>100 and (cand_p[0]==[x2,y2,rmajor] or cand_p[1]==[x2,y2,rmajor]):
                        print(t)
                        print(t_)
                        print(fai)
                        print(fai_)
                        print(fai__)
                        print(rminor)
                        print(rminor_)
                        print((xc,yc))
                        print((xc_,yc_))
                        print(ix)
                        print(arg_vector(list(map(int,[t[0]-xc,t[1]-yc])),list(map(int,[t_[0]-xc_,t_[1]-yc_]))))
                        #movie.release()
                        cv2.imshow("img", frame)
                        cv2.imshow("img2", fgmask)
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                        #cv2.imwrite(r"I:\motion_paper\pictures\elp_bd_tracked.png", frame)
                        #cv2.imwrite(r"I:\motion_paper\pictures\elp_fgmask.png", fgmask)
                        #cv2.imwrite(r"I:\motion_paper\figures\230815\fig1\fig1_d_0.png", frame)
                        #cv2.imwrite(r"I:\motion_paper\figures\230815\fig1\fig1_d_1.png", fgmask)
                        break
                        #return ellipse
                    t_=t
                    xc_=xc
                    yc_=yc
                    #rmajor_=rmajor
                    #if (not np.isnan(rminor) or not np.isnan(rminor_)) and not np.isnan(w) and abs(w)>0.4:
                        #print("e")
                        #rminor_=np.mean([s for s in [rminor_,rminor] if not np.isnan(s)])
                    #elif not np.isnan(rminor) and np.isnan(w):
                        #rminor_=rminor
                    rminor_=rminor
                    if sk>2:
                        fai__=np.nan
                    else:
                        fai__=fai_
                    fai_=fai
                    #ellipse_=ellipse
                    sk=0
                    
                    results.loc[ix,key+"_elp_headside_x"]=t[0]
                    results.loc[ix,key+"_elp_headside_y"]=t[1]
                else:
                    sk+=1
        except KeyboardInterrupt:
            print(key)
            print(ix)
            #movie.release()
            #return el
            break
    cv2.destroyAllWindows()
    #movie.release()
    return results

def direct_out_movie_mult(results,fileout,t_marks,colors,color=(255,0,0),filepath=None):
    if filepath==None:
        filepath=results.iloc[0]["video_path"]
    #movie = cv2.VideoCapture(r"F:\2022-11-17-08-48-08.mp4")"F:\2022-11-17-08-49-18.mp4"
    movie = cv2.VideoCapture(filepath)
    #video = cv2.VideoWriter(r"F:\test_2022-11-17-08-48-08.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    video = cv2.VideoWriter(fileout, cv2.VideoWriter_fourcc(*'mp4v'), movie.get(cv2.CAP_PROP_FPS), (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for i in range(len(results)):
        try:
            movie.set(cv2.CAP_PROP_POS_FRAMES, results.loc[i,"frame"]) 
            ret, frame=movie.read()
            if not ret:
                break
            for key,t_mark in t_marks.items(): 
                if t_mark[0]+"_x" in results.columns and not np.isnan(results.loc[i,t_mark[0]+"_x"]):
                    n1=(int(sorted([np.ceil(results.loc[i,t_mark[0]+"_x"]),np.floor(results.loc[i,t_mark[0]+"_x"])],key=lambda x:abs(results.loc[i,t_mark[0]+"_x"]-x))[0]),int(sorted([np.ceil(results.loc[i,t_mark[0]+"_y"]),np.floor(results.loc[i,t_mark[0]+"_y"])],key=lambda x:abs(results.loc[i,t_mark[0]+"_y"]-x))[0]))
                    cv2.circle(frame, n1, 3, color[key], -1)
                
                if all([t_mark[0]+"_x" in results.columns,t_mark[1]+"_x" in results.columns]) and not np.isnan(results.loc[i,t_mark[0]+"_x"]) and not np.isnan(results.loc[i,t_mark[1]+"_x"]):
                    n1=(int(sorted([np.ceil(results.loc[i,t_mark[0]+"_x"]),np.floor(results.loc[i,t_mark[0]+"_x"])],key=lambda x:abs(results.loc[i,t_mark[0]+"_x"]-x))[0]),int(sorted([np.ceil(results.loc[i,t_mark[0]+"_y"]),np.floor(results.loc[i,t_mark[0]+"_y"])],key=lambda x:abs(results.loc[i,t_mark[0]+"_y"]-x))[0]))
                    n2=(int(sorted([np.ceil(results.loc[i,t_mark[1]+"_x"]),np.floor(results.loc[i,t_mark[1]+"_x"])],key=lambda x:abs(results.loc[i,t_mark[1]+"_x"]-x))[0]),int(sorted([np.ceil(results.loc[i,t_mark[1]+"_y"]),np.floor(results.loc[i,t_mark[1]+"_y"])],key=lambda x:abs(results.loc[i,t_mark[1]+"_y"]-x))[0]))
                    cv2.line(frame, n1, n2, color[key], 3)
                
            #cv2.imshow("img",  frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.destroyAllWindows()
            for key in colors.keys():
                if key+"_x" in results.columns and results.loc[i,key+"_x"]!=None and not np.isnan(results.loc[i,key+"_x"]) and i>0:
                    x_= [int(np.floor(results.loc[i,key+"_x"])),int(np.ceil(results.loc[i,key+"_x"]))][min(enumerate([results.loc[i,key+"_x"]-np.floor(results.loc[i,key+"_x"]),np.ceil(results.loc[i,key+"_x"])-results.loc[i,key+"_x"]]), key = lambda x:x[1])[0]]
                    y_= [int(np.floor(results.loc[i,key+"_y"])),int(np.ceil(results.loc[i,key+"_y"]))][min(enumerate([results.loc[i,key+"_y"]-np.floor(results.loc[i,key+"_y"]),np.ceil(results.loc[i,key+"_y"])-results.loc[i,key+"_y"]]), key = lambda x:x[1])[0]]
                    r_=[int(np.floor(results.loc[i,key+"_r"])),int(np.ceil(results.loc[i,key+"_r"]))][min(enumerate([results.loc[i,key+"_r"]-np.floor(results.loc[i,key+"_r"]),np.ceil(results.loc[i,key+"_r"])-results.loc[i,key+"_r"]]), key = lambda x:x[1])[0]]
                    cv2.circle(frame, (x_,y_), r_, colors[key]["color"], 3)
                    cv2.circle(frame, (x_,y_), 1, colors[key]["color"], -1)
            video.write(frame)
        except Exception as e:
            raise e
            break
    video.release()
    movie.release()

def get_rotated_vec(ADBD,rotate_angle):
    theta = np.deg2rad(rotate_angle)
    rotateMat = np.matrix([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    cxy = (rotateMat * ADBD)
    return cxy

def get_local_coordinate(ex,ey,origin,coordinate):
    dim=ex.shape[0]
    #print(dim)
    R_GtoL = np.identity(dim+1)
    R_GtoL[:dim, :dim] = np.array([ex, ey])
    R_GtoL[:dim, dim] = - np.dot(np.array([ex, ey]), origin)
    return np.dot(R_GtoL,np.array(list(coordinate)+[1]))[:dim]
#おもろい
def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]