import openslide
import numpy as np
from openslide.deepzoom import DeepZoomGenerator
from shapely.geometry import Polygon
from bs4 import BeautifulSoup
from multiprocessing.pool import Pool
import os
import logging
from shapely import affinity
logging.basicConfig(filename='app.log', format='%(asctime)s - %(message)s')
import matplotlib.pyplot as plt
import cv2
undersample={
}

dictionary={
    "fiber":0.2,
    "normal":0.1,
    "necrosis":0.3,
    "default":0.4,
    "ccRCC":1,
    "pRCC":2,
    "CHROMO":3,
    "ONCOCYTOMA":4,
}

def is_informative(im, size):
    img_array = np.array(im)
    
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100: 
        return False

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    if np.mean(saturation > 30) < 0.10: 
        return False

    
    white_fraction = np.sum(np.mean(img_array, axis=2) > 235) / (size * size)
    if white_fraction > 0.60: 
        return False

    return True

def extract_Annotation_xml(path):
  with open(path, 'r') as f:
    data = f.read()
  bs_data = BeautifulSoup(data, 'xml')
  all_classes={}
  for i in bs_data.find_all('Group'):
    all_classes[i.get('Name')]=[]

  for annotation in bs_data.find_all('Annotation'):
    ls=[]
    for coordinate in annotation.find_all('Coordinate'):
      ls.append((int(coordinate.get('Order')),float(coordinate.get('X')),float(coordinate.get('Y'))))

    all_classes[annotation.get('PartOfGroup')].append(ls)
  return all_classes

def extract_ROI(annotation):
  xs=[i[1] for i in annotation]
  ys=[i[2] for i in annotation]
  x_max=int(max(xs))
  x_min=int(min(xs))

  y_max=int(max(ys))
  y_min=int(min(ys))
  rectangle=[(x_min,y_min),(x_max,y_max)]
  return rectangle


def read_img(tile,coordinate,level,size):
        x=coordinate[0]//(2**(tile.level_count-1 - level))
        y=coordinate[1]//(2**(tile.level_count-1 - level))

        x1=x//size
        x2=x%size
        y1=y//size
        y2=y%size
        xlimit,ylimit=tile.level_tiles[level]

        im1 = np.array(tile.get_tile(level , (x1 , y1)))
        if(x1+1<xlimit):
            im2 = np.array(tile.get_tile(level , (x1+1 , y1)))
            im = np.hstack((im1, im2))
        if(y1+1<ylimit):
            im3 = np.array(tile.get_tile(level , (x1 , y1+1)))
            im=np.vstack((im1, im3))

        if(x1+1<xlimit and y1+1<ylimit):
            im4 = np.array(tile.get_tile(level , (x1 +1, y1+1)))

            if (im2.shape[1] > im4.shape[1]):
                im2 = im2[:,:im4.shape[1]]
            elif (im4.shape[1] > im2.shape[1]):
                im4 = im4[:,:im2.shape[1]]

            if (im3.shape[0] > im4.shape[0]):
                im3 = im3[:im4.shape[0],:]
            elif (im4.shape[0] > im3.shape[0]):
                im4 = im4[:im3.shape[0],:]


            xi=np.vstack((im1, im3))
            xj=np.vstack((im2, im4))


            im = np.hstack((xi,xj))

        return im[y2:y2+size,x2:x2+size]


def extract_patches_from_ROI(img, annotation, size=256, level=0):
    rectangle = extract_ROI(annotation)
    x = rectangle[0][0]
    patches = []
    tiles=DeepZoomGenerator(img,size,level)
    level_inverse=tiles.level_count -1-level
    zoom_factor = 2**level


    relative_annotation = [(i[0], (i[1] - rectangle[0][0]) / zoom_factor, (i[2] - rectangle[0][1]) / zoom_factor) for i
                           in annotation]

    annotation_poly = Polygon([(i[1], i[2]) for i in relative_annotation])
    if(annotation_poly.is_valid==False):
        logging.warning('annotation poly false')
        annotation_poly = annotation_poly.buffer(0)
        logging.warning(str(annotation_poly.is_valid))
        if(annotation_poly.is_valid==False):
            return []

    while x < rectangle[1][0]:

        y = rectangle[0][1]
        while y < rectangle[1][1]:

            square = Polygon([
                ((x - rectangle[0][0]) / zoom_factor, (y - rectangle[0][1]) / zoom_factor),
                (min((x + size * zoom_factor - rectangle[0][0]),rectangle[1][0]) / zoom_factor, (y - rectangle[0][1]) / zoom_factor),
                (min((x + size * zoom_factor - rectangle[0][0]),rectangle[1][0]) / zoom_factor,
                 min((y + size * zoom_factor - rectangle[0][1]), rectangle[1][1])/ zoom_factor),
                ((x - rectangle[0][0]) / zoom_factor, (y + size * zoom_factor - rectangle[0][1]) / zoom_factor)
            ])


            if (annotation_poly.intersects(square) == False):
                y += size * zoom_factor
                continue

            ratio = annotation_poly.intersection(square).area / square.area
            if (ratio < 0.1):
                y += size * zoom_factor
                continue
            im=read_img(tiles,(x,y),level_inverse,size)


            std = np.array(im)[:, :, 0].std(), np.array(im)[:, :, 1].std(), np.array(im)[:, :, 2].std()
            m = max(abs(std[0] - std[1]), abs(std[2] - std[1]), abs(std[0] - std[2]))
            k = np.sum(np.array(im).mean(axis=2) > 230) / (size * size)

            if (np.array(im).min(axis=2).mean() > 240 or m < 2 or k>0.5):
                y += size * zoom_factor
                continue

            c = affinity.translate(square.intersection(annotation_poly), xoff=-square.bounds[0], yoff=-square.bounds[1])
            if c.geom_type == 'Polygon':
                xx, yy = c.exterior.coords.xy
                coords = [[xx.tolist(), yy.tolist()]]
            elif c.geom_type == 'MultiPolygon':
                coords=[ [p.exterior.coords.xy[0].tolist(),p.exterior.coords.xy[1].tolist()]  for p in c.geoms]
            else:
                coords=[]
            patches.append((im,coords,ratio))

            y += size * zoom_factor
        x += size * zoom_factor
    return patches

#this contains the important functionality
##############################################################################
def extract_patches_chromo_ocno(path, size=256, level=0):
    img = openslide.open_slide(path)
    tiles = DeepZoomGenerator(img, size, 0)

    level_inverse = tiles.level_count - level - 1


    x, y = tiles.level_tiles[level_inverse]
    patches = []
    for i in range(x):
        for j in range(y):
            im = tiles.get_tile(level_inverse, (i, j))

            std = np.array(im)[:, :, 0].std(), np.array(im)[:, :, 1].std(), np.array(im)[:, :, 2].std()
            m = max(abs(std[0] - std[1]), abs(std[2] - std[1]), abs(std[0] - std[2]))

            k = np.sum(np.array(im).mean(axis=2) > 230) / (size * size)
            if not is_informative(im, size):
                continue
            else:
                x1 = (tiles.level_dimensions[level_inverse][0]-size)*(2**level)
                x2 = (tiles.level_dimensions[level_inverse][1]-size)*(2**level)
                if (im.width < size and im.height < size):
                    im = read_img(tiles, (x1,x2), level_inverse, size)
                elif (im.width < size):
                    im = read_img(tiles,(x1,(j*size)*(2**level)),level_inverse,size)
                elif (im.height < size):
                    im = read_img(tiles,((i*size)*(2**level),x2),level_inverse,size)
                patches.append(im)


    return patches
##############################################################################

def onco_chromo(input):
    file = input[0]
    label=input[1]
    if (os.path.isdir(file)):
        return  'done '+file
    if (os.path.splitext(file)[-1] not in ['.svs', '.tif', '.scn']):
        return  'done '+file

    k=extract_patches_chromo_ocno(file, input[2], input[3])
    print('saving '+file)
    np.savez_compressed(input[4],np.array([(np.array(i),dictionary[label]) for i in k],dtype=object))
    print('saved ' + file)
    return  'done ' +file

def loop_ocno_chromo(path,out,size,level,cores):
    label=os.path.basename(path)
    if(label not in dictionary.keys()):
        if("CHROMO" in label.upper()):
            label="CHROMO"
        elif "ONCO" in label.upper():
            label="ONCOCYTOMA"
        else:
            label="default"

    files=[(os.path.join(path,i),label,size,level,os.path.join(out,i[:-4])) for i in os.listdir(path)]
    with Pool(processes=cores) as pool:
        for results in pool.map(func=onco_chromo, iterable=files):
            logging.warning(results)


def downsample(ls,annotations,tumor):
    new_ls=[]
    length_anotation={}
    for key in annotations:
        k=key.strip()
        if(k=='tumor'):
            k=tumor
        length_anotation[k]=len(annotations[key])
    for key in annotations:

        k=key.strip()
        if(k=='tumor'):
            k=tumor
        if (k not in dictionary.keys()):
            print(k)
            continue
        print(k)

        sample_from=[i for i in ls if i[3]==dictionary[k]]
        weights=[i[2] for i in ls if i[3]==dictionary[k]]
        weights=np.array(weights)
        weights=weights/sum(weights)
        quantity=undersample[k][2]
        if(length_anotation[k]> undersample[k][0]):
            quantity += (length_anotation[k] / undersample[k][0]) * undersample[k][1]
        quantity=int(quantity)
        if(quantity>len(sample_from)):
            quantity=len(sample_from)-2
        if(quantity<3):
            quantity = len(sample_from)

        temp=np.random.choice(range(len(sample_from)), size=quantity, replace=False, p=weights)
        new_ls +=[sample_from[i] for i in temp]
    return new_ls
def cc_p_RCC(info):

    ls = []
    if('pre' not in info[0]):
        annotationfolder = os.path.join(info[0], os.path.basename(info[0]) + '_xml')
    else:
        annotationfolder = os.path.join(info[0], 'pre_'+os.path.basename(info[0]) + '_xml')
    file = info[1]
    tumor=''
    if (os.path.isdir(file)):
        return  'done ' + file
    if (os.path.splitext(file)[-1] not in ['.svs', '.tif', '.scn']):
        return  'done ' + file
    if(os.path.exists(os.path.join(annotationfolder,os.path.basename( info[1])[:-3] + 'xml'))==False):
        return  'done ' + file
    annotations = extract_Annotation_xml(os.path.join(annotationfolder, os.path.basename( info[1])[:-3] + 'xml'))
    img = openslide.open_slide(file)

    for key in annotations.keys():

        logging.warning(key)
        if (key.strip() == 'tumor'):
            label = os.path.basename(info[0]).strip()
            tumor=label
        elif key.strip() in dictionary.keys():
            label = key.strip()
        else:
            label = 'default'
        count = 0
        for annotation in annotations[key]:
            logging.warning(f"annotation {count}/{len(annotations[key])}")
            count += 1
            k = extract_patches_from_ROI(img, annotation, info[2], info[3])
            k = [(np.array(i[0], dtype=np.ubyte),np.array(i[1],dtype=object),np.array(i[2]), dictionary[label]) for i in k]
            ls += k
    result=downsample(ls,annotations,tumor)
    logging.warning('saving ' + file)
    np.savez_compressed(info[4], np.array(result, dtype=object))
    logging.warning('saved ' + file)
    return  'done ' + file

def loop_cc_p_RCC(path,out,size,level,cores):

    inputs=[(path,os.path.join(path,i),size,level,os.path.join(out,i[:-4])) for i in os.listdir(path)]
    with Pool(processes=cores) as pool:
        for results in pool.map(func=cc_p_RCC,iterable=inputs):
            logging.warning(results)

def handle_extraction(path,mode,size=256,level=0,skip='',cores=5):

    if(os.path.exists(path + '_npy')==False):
        os.mkdir(path + '_npy')
    if(os.path.exists(os.path.join(path + '_npy', mode))==False):
        os.mkdir(os.path.join(path + '_npy', mode))

    for i in os.listdir(path):

        folder=os.path.join(path,i)
        if(i.lower() in skip.lower()):
            continue
        if(os.path.isfile(folder)):
            continue
        print(f" starting file {i}")
        if(mode=='training'):
            if(i=='CHROMO' or i=='ONCOCYTOMA'):
                continue
            elif('Annotations' in i):
                if(os.path.exists(os.path.join(os.path.join(path + '_npy', mode),i))==False):
                    os.mkdir(os.path.join(os.path.join(path + '_npy', mode),i))
                loop_ocno_chromo(folder,os.path.join(os.path.join(path + '_npy', mode),i),size,level,cores)


            elif(i=='pre'):
                prepRCC=os.path.join(folder,'pRCC')
                preccRCC=os.path.join(folder,'ccRCC')
                if (os.path.exists(os.path.join(os.path.join(path + '_npy', mode), i+'_pRCC'))==False):
                    os.mkdir(os.path.join(os.path.join(path + '_npy', mode), i+'_pRCC'))
                if (os.path.exists(os.path.join(os.path.join(path + '_npy', mode), i + '_ccRCC'))==False):
                    os.mkdir(os.path.join(os.path.join(path + '_npy', mode), i + '_ccRCC'))
                loop_cc_p_RCC(prepRCC,os.path.join(os.path.join(path + '_npy', mode), i+'_pRCC'),size,level,cores)
                loop_cc_p_RCC(preccRCC,os.path.join(os.path.join(path + '_npy', mode), i + '_ccRCC'),size,level,cores)


            else:
                if (os.path.exists(os.path.join(os.path.join(path + '_npy', mode),i))==False):
                    os.mkdir(os.path.join(os.path.join(path + '_npy', mode),i))
                loop_cc_p_RCC(folder,os.path.join(os.path.join(path + '_npy', mode),i),size,level,cores)
        elif(mode=='testing'):
            if ('Annotations' in i):
                continue
            elif(i=='pre'):
                prepRCC=os.path.join(folder,'pRCC')
                preccRCC=os.path.join(folder,'ccRCC')
                if (os.path.exists(os.path.join(os.path.join(path + '_npy', mode), i + '_pRCC'))==False):
                    os.mkdir(os.path.join(os.path.join(path + '_npy', mode), i + '_pRCC'))
                if (os.path.exists(os.path.join(os.path.join(path + '_npy', mode), i + '_ccRCC'))==False):
                    os.mkdir(os.path.join(os.path.join(path + '_npy', mode), i + '_ccRCC'))
                loop_ocno_chromo(prepRCC,os.path.join(os.path.join(path + '_npy', mode), i + '_pRCC'),size,level,cores)
                loop_ocno_chromo(preccRCC,os.path.join(os.path.join(path + '_npy', mode), i + '_ccRCC'),size,level,cores)
            else:
                if (os.path.exists(os.path.join(os.path.join(path + '_npy', mode),i))==False):
                    os.mkdir(os.path.join(os.path.join(path + '_npy', mode),i))
                loop_ocno_chromo(folder,os.path.join(os.path.join(path + '_npy', mode),i),size,level,cores)


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='creating the tumor classification datasets')
    parser.add_argument('--path', dest='path', type=str, help='path to data')
    parser.add_argument('--mode', dest='mode', type=str, help='mode training/testing')
    parser.add_argument('--size', dest='size',default=256, type=int, help='H/W')
    parser.add_argument('--nb_cores', dest='nb_cores', default=5, type=int, help='nb_cores')
    parser.add_argument('--level', dest='level',default=0, type=int, help='at how level the patches should be extracted')
    parser.add_argument('--skip', dest='skip', default='', type=str,help='which folders to skip ?')

    args = parser.parse_args()
    handle_extraction(args.path,args.mode,args.size,args.level,args.skip,args.nb_cores)

    '''
    os.mkdir('imgs')
    img=openslide.open_slide('RCC_WSIs_TRAIN/ccRCC/HP19.754.A4.ccRCC.scn')
    ann=extract_Annotation_xml('RCC_WSIs_TRAIN/ccRCC/ccRCC_xml/HP19.754.A4.ccRCC.xml')

    save=extract_patches_from_ROI(img,ann['necrosis'][0],args.size,args.level)
    k=0
    for i in save:
       plt.imshow(np.array(i))
       plt.savefig('imgs/'+str(k))
       k+=1
    '''
