import glob
import cv2
import numpy as np
import _pickle as pc
import os
from sklearn import cluster
import src.compute_map as compute_map
import src.convert_for_eval as convert_for_eval
import argparse
from skimage.util import view_as_windows
import scipy

np.random.seed(7) #lol random


"""
def generate_random(a,b,percentage=0,default=-1,intmi=False):
        #returns default with a change of percentage
        if( np.random.ranf() > percentage ):
                retval= (b-a) * np.random.ranf() + a
        else:
                retval = default

        if(intmi==True):
                return int(retval)
        return retval
"""

def log_results(args):

        name = get_result_filename(args)
        convert_for_eval.mainn(["filler",name])
        converted_filename = "converted_"+name
        map_score = compute_map.mainn(["filler",converted_filename,"./src/validation_gt.dat"])
        print("map score: ",map_score)


        if args.log==False:
                return
        with open("log_"+name,"a") as filem:
                filem.write("Best Score : %f \n" % map_score )
                #TODO
                pass




def get_image_names():
    return glob.glob("./dataset/*")

def eucledian(v1,v2):
    return scipy.spatial.distance.cosine(v1,v2)
    return np.linalg.norm(v1-v2)

def gridden(img,grid):
    if grid!=1:
        img = view_as_windows(img,grid,step=grid)
        img = img.reshape(-1,grid*grid)
        img = np.average(img,axis=1)
    else:
        img = img.flatten()
    return img
    
def grayscale_histogram(img,k,grid):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = gridden(img,grid)
    hist = np.histogram(img, bins=k,range=(0,255))[0]
    return hist

def color_histogram(img,k,grid):
    c=[0]*3
    c[0] = gridden(img[:,:,0],grid)
    c[1] = gridden(img[:,:,1],grid)
    c[2] = gridden(img[:,:,2],grid)
    return np.histogramdd(c,bins=(k,k,k))[0].flatten()

def edge_histogram(img,k):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(img,100,200)
    grad_map = np.gradient(img)[0]
    grad = grad_map[edge_map>0]
    return np.histogram(grad,bins=k)[0]

def extract_features(img,typef,k,grid):
            
    if typef==1:
        feat = grayscale_histogram(img,k,grid)
    elif typef==2:
        feat = color_histogram(img,k,grid)
    elif typef==3:
        feat = edge_histogram(img,k)
        
    feat = feat/np.sum(feat)
    return feat

def update_feature_vectors(fname,feat_type,k,grid):
        pic_names_array = get_image_names()
        pic_count = len(pic_names_array)
        feat_filename = fname
        features = np.zeros((pic_count,k))
        print("extracting features")
        for index,pic in enumerate(pic_names_array):
                if(index%100==0):
                       print("Feature extracted %d images" % index)
                curr_image = cv2.imread(pic)
                features[index] = extract_features(curr_image,feat_type,k,grid)
        to_save =  { "features" : features, "names":pic_names_array } 
        print("done extracting")
        with open(feat_filename,"wb") as feat_file:
                 pc.dump(to_save,feat_file)
                
        return features, pic_names_array



def get_similiar_images(img_name,features,names,args):


        curr_image_gray = cv2.imread("./dataset/"+img_name)
        current_features = extract_features(curr_image_gray,args.t,args.bin,args.grid)
        distance_array = np.zeros(shape=(len(names),))
        image_names_array = [0]*(len(names))
        for index,feat in enumerate(features):

                distance_array[index] = eucledian(current_features,feat)
                image_names_array[index] = names[index]


        return list(zip(distance_array,image_names_array))



def get_feat_filename(args):
    if args.t==1:
        a = "gray"
    elif args.t==2:
        a = "color"
    elif args.t==3:
        a = "edge"
        
    retval = a+"_"+str(args.grid)+"_"+str(args.bin)+".pc"
    return retval
    
def get_result_filename(args):
    if args.t==1:
        a = "gray"
    elif args.t==2:
        a = "color"
    elif args.t==3:
        a = "edge"
        
    retval = a+"_"+str(args.grid)+"_"+str(args.bin)+"_result.txt"
    return retval

def get_feature_vectors(fname):
    
    with open(fname,"rb") as bof_file:
        dic = pc.load(bof_file)
    features = dic["features"]
    names = dic["names"]
        
    return features,names
    
def main(args):
        
        feat_filename = get_feat_filename(args)
        print(feat_filename)
        if args.reuse:
            features,names = get_feature_vectors(feat_filename)
        else:
            features,names = update_feature_vectors(feat_filename,feat_type=args.t, k=args.bin, grid=args.grid)
        
        if args.test:
            validation_file = "./src/test_queries.dat"
        else:
            validation_file = "./src/validation_queries.dat"
        result_filename = get_result_filename(args)

        with open(result_filename,"w") as res_file:
                for line in open(validation_file,"r"):
                        curr_image_name = line.rstrip('\n')
                        query_result = get_similiar_images(curr_image_name,features,names,args)
                        query_result = list( map(lambda x:str(x[0])+" "+str(x[1]),query_result) )
                        query_string =" ".join(query_result)
                        res_file.write(curr_image_name+": "+query_string+'\n')

        log_results(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ZSL")
  
    parser.add_argument('-t', type=int, default=1)
    parser.add_argument('--test',  dest='test', action='store_true',
                        help='if not given, validation is done')
    parser.add_argument('--reuse',  dest='reuse', action='store_true',
                        help='Reuse old features from file')
    parser.add_argument('--log',  dest='log', action='store_true',
                        help='Create log files')
    parser.add_argument('--grid',  dest='grid', type=int,default=1)
    parser.add_argument('--bin',  dest='bin', type=int,default=10)


    main(parser.parse_args())
