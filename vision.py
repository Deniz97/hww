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
    #return np.linalg.norm(v1-v2)

def gridden(img,grid):
    if grid!=1:
        img = view_as_windows(img,grid,step=grid)
        img = img.reshape(-1,grid*grid)
        img = np.average(img,axis=1)
    else:
        img = img.flatten()
    return img
    
def grayscale_histogram(img,k,grid):
    img = gridden(img,grid)
    hist = np.histogram(img, bins=k,range=(0,255))[0]
    return hist

def color_histogram(img,k,grid):
    c=[0]*3
    c[0] = gridden(img[:,:,0],grid)
    c[1] = gridden(img[:,:,1],grid)
    c[2] = gridden(img[:,:,2],grid)
    retval = np.histogramdd(c,bins=(k,k,k))[0].flatten()
    return retval

def edge_histogram(img,k):
    fx = np.asarray( [ [1,0,-1], [2,0,-2], [1,0,-1] ] )
    fy = np.asarray( [ [1,2,1],[0,0,0],[-1,-2,-1] ] )
    sobelx = scipy.ndimage.convolve(img,fx)
    sobely = scipy.ndimage.convolve(img,fy)
    magn = np.hypot(sobelx,sobely)

    direc = np.arctan2(sobely,sobelx)
    
    edges = np.histogram(direc,bins=k)[1]
    print(edges.shape)
    bin_indices = np.digitize(direc,edges)
    
    hist = np.zeros(k)
    magnf = magn.flatten()
    for index,x in enumerate( np.nditer(bin_indices) ):
        hist[ x-2 ] += magnf[index]

    return hist
    
    

def edge_histogram2(img,k):
    """
    edge_map = cv2.Canny(img,100,200)
    grad_map = np.gradient(img)[0]
    grad = grad_map[edge_map>0]
    retval = np.histogram(grad,bins=k)[0]
    
    return retval
    """
    fx = np.asarray( [ [1,0,-1], [2,0,-2], [1,0,-1] ] )
    fy = np.asarray( [ [1,2,1],[0,0,0],[-1,-2,-1] ] )
    sobelx = scipy.ndimage.convolve(img,fx)
    sobely = scipy.ndimage.convolve(img,fy)
    magn = np.square(sobelx) + np.square(sobely)
    magn = np.sqrt(magn)

    direc = np.arctan2(sobely,sobelx)

    hist1 = np.histogram(magn,bins=k)[0]
    hist2 = np.histogram(direc,bins=k)[0]
    return np.concatenate((hist1,hist2))



def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """

    assert len(arr.shape)==2, "Grid feature not yet implemented for color histogram ("+str(arr.shape)+")"
    if arr.shape[0]%2!=0:
        arr = arr[:-1,:]
    if arr.shape[1]%2!=0:
        arr = arr[:,:-1]
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def extract_features(img,typef,k,grid,divide,norm=True):
    
    
    
    #average the picture over grid*grid regions
    if grid>1:
        org_h = img.shape[0]
        org_w = img.shape[1]
        imgs = blockshaped(img,grid,grid)
        img = np.asarray( [ np.average(x) for x in imgs] )[org_h//grid* org_w//grid:].reshape( org_h//grid, org_w//grid  )
        return extract_features(x,typef,k,1,divide)
        
    #divide the picture in to divide*divide regions
    if divide > 1:
        org_h = img.shape[0]
        org_w = img.shape[1]
        imgs = blockshaped(img,org_h//divide,org_w//divide)
        hists = [extract_features(x,typef,k,grid,1,norm=False) for x in imgs]
        hists = np.concatenate(hists)
        hists = hists/np.sum(hists)
        return hists
    
    if typef==1:
        feat = grayscale_histogram(img,k,grid)
    elif typef==2:
        feat = color_histogram(img,k,grid)
    elif typef==3:
        feat = edge_histogram(img,k)
    
    if norm:
        #print("feat: ",feat)
        #print("s: ",np.sum(feat))
        #print("f/s: ",feat/np.sum(feat))
        retval = feat/np.sum(feat)
        return retval
    
    return feat

def update_feature_vectors(fname,feat_type,k,grid,divide):
        pic_names_array = get_image_names()
        pic_count = len(pic_names_array)
        feat_filename = fname
        sec_dim = k * divide**2
        if feat_type==3:
                sec_dim = sec_dim
        elif feat_type==2:
                sec_dim = sec_dim*k*k
        features = np.zeros((pic_count,sec_dim))
        print("extracting features")
        for index,pic in enumerate(pic_names_array):
                if(index%1==0):
                       print("Feature extracted %d images" % index)
                curr_image = cv2.imread(pic) if feat_type == 2 else cv2.imread(pic,0)
                #print("a: ",features[index].shape)
                temp = extract_features(curr_image,feat_type,k,grid,divide)
                #print("b: ",temp.shape)
                features[index] = temp
        to_save =  { "features" : features, "names":pic_names_array } 
        print("done extracting")
        with open(feat_filename,"wb") as feat_file:
                 pc.dump(to_save,feat_file)
                
        return features, pic_names_array



def get_similiar_images(img_name,features,names,args):

        
        curr_image = cv2.imread("./dataset/"+img_name) if args.t == 2 else cv2.imread("./dataset/"+img_name,0)
        current_features = extract_features(curr_image,args.t,args.bin,args.grid,args.divide)
        distance_array = np.zeros(shape=(len(names),))
        image_names_array = [0]*(len(names))
        for index,feat in enumerate(features):
                
                distance_array[index] = eucledian(current_features,feat)
                image_names_array[index] = names[index].replace("./dataset/","")


        return list(zip(distance_array,image_names_array))



def get_feat_filename(args):
    if args.t==1:
        a = "gray"
    elif args.t==2:
        a = "color"
    elif args.t==3:
        a = "edge"
        
    retval = a+"_grid"+str(args.grid)+"_bin"+str(args.bin)+"_divide"+str(args.divide)+".pc"
    return retval
    
def get_result_filename(args):
    if args.t==1:
        a = "gray"
    elif args.t==2:
        a = "color"
    elif args.t==3:
        a = "edge"
        
    retval = a+"_grid"+str(args.grid)+"_bin"+str(args.bin)+"_divide"+str(args.divide)+"_result.txt"
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
            features,names = update_feature_vectors(feat_filename,feat_type=args.t, k=args.bin, grid=args.grid,divide=args.divide)
        
        if args.test:
            validation_file = "./src/test_queries.dat"
        else:
            validation_file = "./src/validation_queries.dat"
        result_filename = get_result_filename(args)

        with open(result_filename,"w") as res_file:
                for index,line in enumerate(open(validation_file,"r")):
                        if(index%50==0):
                            print("Done %d query images" % index )
                        curr_image_name = line.rstrip('\n')
                        query_result = get_similiar_images(curr_image_name,features,names,args)
                        query_result = list( map(lambda x:str(x[0])+" "+str(x[1]),query_result) )
                        query_string =" ".join(query_result)
                        res_file.write(curr_image_name+": "+query_string+'\n')

        log_results(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ZSL")
  
    parser.add_argument('-t', type=int, default=1,help="1: gray hist, 2:color hist, 3:edge hist")
    parser.add_argument('--test',  dest='test', action='store_true',
                        help='if not given, validation is done')
    parser.add_argument('--reuse',  dest='reuse', action='store_true',
                        help='Reuse old features from file')
    parser.add_argument('--log',  dest='log', action='store_true',
                        help='Create log files')
    parser.add_argument('--grid',  dest='grid', type=int,default=1,help="Sampling size of pixels (not in actual hw)")
    parser.add_argument('--divide',  dest='divide', type=int,default=1,
                       help="Parition to picture and histogram the sub pictures (in actual hw)")
    parser.add_argument('--bin',  dest='bin', type=int,default=10,
                       help="Number of bins for histograms")


    main(parser.parse_args())
