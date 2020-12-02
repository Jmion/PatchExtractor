#!/home/mion/.conda/envs/patchExtract/bin/python

import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import cv2
from page_xml import xmlPAGE
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import logging


__author__ = "Jeremy Mion"
__version__ = "1.0.0"
__email__ = "jeremy.mion@epfl.ch"
__status__ = "Production"


"""


"""
class patchExtractor:
    
    def __init__(self, image_directory: str, xml_directory: str,
                 save_directory: str, dataframe_save: str,
                offest_above: int, offset_below: int, disable_progress_bar: bool):
        """
        Args:
            image_directory (string): Path to the image_directory
            xml_direcotry (string): Path to xml_direcotry
            save_directory (string): Path to save location
            dataframe_save (string): save location for dataframes
            offset_above (int): number of pixels above the baseline that must be part of a patch
            offset_below (int): number of pixels below the baseline that must be part of a patch
            disable_progress_bar (nool): disables the tqdm progress bar
        """
        
        self.image_directory = image_directory
        self.xml_directory = xml_directory
        self.save_directory = save_directory
        self.dataframe_save = dataframe_save
        self.disable_progress_bar = disable_progress_bar
        
        log_file_name = f'{image_directory}log.log'
        logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    
    def _extract_images_from_page(self, page_name):
        """
        Args:
            page_name (string): The name of the page to extract all the patches
            
        Returns:
        
            pandas dataframe: with the coordinates of the patches as well as the
                filename used to store the patch
        """
        logging.debug(f'page : {page_name}')
        
        x_arr_min = []
        x_arr_max = []
        y_arr_min = []
        y_arr_max = []
        filename_arr = []
        page_name_arr = []
        collumn_number = []
        patch_number = []
        MIN_LENGTH_OF_PATH = 50

        xml_path = os.path.join(self.xml_directory, page_name+".xml")
        image_path = os.path.join(self.image_directory, page_name+".jpg")



        #line detection

        def load_image(path: str):
            return cv2.imread(path)


        def show_image(img, title="") -> None:
            return
            #plt.figure(figsize=(10,15))
            #plt.imshow(img, cmap='Greys_r')
            #plt.title(title)
            #plt.show()

        def remove_connected_components(img, min_size=50):
            number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
            sizes = stats[1:, -1]  # first index represents the background and the last column the total area in pixels
            number_of_labels = number_of_labels - 1

            img_cleaned = np.full(img.shape, 0)
            for i in range(0, number_of_labels):
                if sizes[i] >= min_size:
                    img_cleaned[labels == i + 1] = 255
                else:
                    number_of_labels = number_of_labels - 1

            return img_cleaned
        
        def binarize_image(img, otsu=True):
            img_binarized = None
            if otsu:
                threshold, img_binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            else:
                threshold, img_binarized = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            return img_binarized


        def _get_kernel(theta) -> float:
            ksize = 31
            return cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)


        def filter_image(img, theta=np.pi):
            kernel = _get_kernel(theta)
            return cv2.filter2D(img, -1, kernel)


        def invert(img):
            return cv2.bitwise_not(img)

        #LOADING IMG
        connected_components_threshold = 600
        img_og = load_image(image_path)
        image = mpimg.imread(image_path)
        #LOADING PAGE
        page = xmlPAGE.pageData(xml_path)
        page.parse()
        imgBoundingBox = page.build_baseline_coordinates()

        img = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
        show_image(img, "Original")
        #Finding Region of Intreset ROI:
        x_min_global = 6000
        x_max_global = -1
        y_min_global = 6000
        y_max_global = -1
        for i in imgBoundingBox:
            for j in i[0]:
                x_max_global = max(j[0][0], x_max_global)
                x_min_global = min(j[0][0], x_min_global)
                y_max_global = max(j[0][1], y_max_global)
                y_min_global = min(j[0][1], y_min_global)

        x_min_global = max(300, x_min_global)
        x_max_global = min(4000, x_max_global)

        def ROILineMask(img):
            connected_components_threshold = 500
            img_filtered = filter_image(img, theta=np.pi)
            img_vertical_binarized = binarize_image(img_filtered.copy())
            img_vertical_binarized = invert(img_vertical_binarized)
            img_cleaned_cc = remove_connected_components(img_vertical_binarized.copy(), connected_components_threshold)
            #show_image(img_cleaned_cc, "adapt threshold cc")
            img_gg_blur = cv2.blur(img_cleaned_cc, (20, 20))
            _,img_bin = cv2.threshold(np.float32(img_gg_blur), 20,255, cv2.THRESH_BINARY)
            rows,cols = img_bin.shape
            M = np.float32([[1,0,0],[0,1,-150]])
            dst = cv2.warpAffine(img_bin,M,(cols,rows))
            res = cv2.bitwise_or(img_bin, dst)
            #show_image(res, "translationOR")
            connected_components_threshold = 50000
            img_col = remove_connected_components(np.uint8(res).copy(), connected_components_threshold)
            show_image(img_col, "cleanup")
            return img_col

        y_min_global_margin = -100
        y_max_global_margin = 150

        logging.debug(f'ROILineMask input coordinates {y_min_global+y_min_global_margin}:{y_max_global+y_max_global_margin}, {x_min_global}:{x_max_global}')
        
        img_col_ROI = ROILineMask(img[y_min_global+y_min_global_margin:y_max_global+y_max_global_margin, x_min_global:x_max_global])
        #img = img[y_min_global-100:y_max_global+150, x_min_global:x_max_global]
        #show_image(img, "ROI")

        def x_ogToROICoordinates(og_x):
            return max(0,og_x-x_min_global)



        def y_ogToROICoordinates(og_y):
            return og_y - (y_min_global+y_min_global_margin)



        #img_filtered = filter_image(img, theta=np.pi)
        #img_vertical_binarized = binarize_image(img_filtered.copy())
        #img_vertical_binarized = invert(img_vertical_binarized)
        #img_cleaned_cc = remove_connected_components(img_vertical_binarized.copy(), connected_components_threshold)
        #show_image(img_cleaned_cc, "cleaned")


        if not(os.path.isdir(self.save_directory)):
            os.mkdir(self.save_directory)


        patch_height_above = -100
        patch_height_below = 20

        for i in imgBoundingBox:
            x_min = 6000
            x_max = -1
            y_min = 6000
            y_max = -1
            for j in i[0]:
                x_max = max(j[0][0], x_max)
                x_min = min(j[0][0], x_min)
                y_max = max(j[0][1], y_max)
                y_min = min(j[0][1], y_min)
            og_patch_img = image[y_min+patch_height_above:y_max+patch_height_below, x_min:x_max]


            #show_image(og_patch_img, f'patch_{x_min}_{x_max}_{y_min}_{y_max}')
            #show_image(img_cleaned_cc[y_min+patch_height_above:y_max+patch_height_below, x_min:x_max], "vert lines")


            show_image(np.uint8(img_col_ROI)[y_ogToROICoordinates(y_min+patch_height_above):y_ogToROICoordinates(y_max+patch_height_below),:],"Line")
            number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                np.uint8(img_col_ROI)[y_ogToROICoordinates(y_min+patch_height_above):y_ogToROICoordinates(y_max+patch_height_below), :], 8, cv2.CV_32S)
            centroid_x_locations = centroids[1:,0]
            centroid_x_locations.sort()
            #print(f"centroid x location : {centroid_x_locations}")

            #Locations of lines x-coord
            number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                np.uint8(img_col_ROI)[y_ogToROICoordinates(y_min+patch_height_above):y_ogToROICoordinates(y_max+patch_height_below),
                                      x_ogToROICoordinates(x_min):x_ogToROICoordinates(x_max)],
                8, cv2.CV_32S)
            #print(f"centroidis raw location  = {centroids[:,0].astype(int)}")
            cut_locations = np.append(centroids[1:,0].astype(int) , [0, x_max-x_min],axis=0) + x_min- x_min_global
            cut_locations.sort()

            show_image(img_col_ROI[y_ogToROICoordinates(y_min+patch_height_above):y_ogToROICoordinates(y_max+patch_height_below),
                                   x_ogToROICoordinates(x_min):x_ogToROICoordinates(x_max)], "SIMPLE_PATH")

            #print(f"{x_min_global=}, {x_min=}")
            #print(f"{cut_locations=}")
            for c in range(1,len(cut_locations[1:])+1):
                #print("patches")
                if(cut_locations[c]- cut_locations[c-1] > MIN_LENGTH_OF_PATH):
                    locationOfPatch= cut_locations[c-1]+(cut_locations[c]-cut_locations[c-1])/2
                    #print(f"location of patch={locationOfPatch}")
                    col_location = np.searchsorted(
                        centroid_x_locations,
                        locationOfPatch)
 
                    x_min_temp = cut_locations[c-1]+ x_min_global
                    x_max_temp = cut_locations[c] + x_min_global
                    y_min_temp = y_min+patch_height_above
                    y_max_temp = y_max+patch_height_below
                
                    output_img = image[y_min_temp:y_max_temp, x_min_temp:x_max_temp]
    
                    x_arr_max.append(x_max_temp)
                    x_arr_min.append(x_min_temp)
                    y_arr_min.append(y_min_temp)
                    y_arr_max.append(y_max_temp)
                    collumn_number.append(col_location)
                    patch_number.append(c)
                    filename = f'{page_name}_{x_min_temp}_{x_max_temp}_{y_min_temp}_{y_max_temp}_{c}_{col_location}.jpg'
                    filename_arr.append(filename)
                    page_name_arr.append(page_name)
                    
                    plt.imsave(os.path.join(self.save_directory,filename), output_img)
                    

        return pd.DataFrame({"x_min" : x_arr_min, "x_max": x_arr_max, "y_min" : y_arr_min, "y_max": y_arr_max, "collumn_number": collumn_number, "patch_number": patch_number, "filename" : filename_arr, "page_name": page_name_arr})
    
    def _extraxt_page_and_save_df(self, page_name):
        csv_name = os.path.join(self.dataframe_save,page_name.split(".")[0]+".csv")
        self._extract_images_from_page(page_name.split(".")[0]).to_csv(csv_name)

    """
    Get's all the patches from the image directory and processes them.
    Only processes files with extensions jpg.
    
    """
    def extract_all_pages(self, page=None):
        if not(os.path.isdir(self.dataframe_save)):
            os.mkdir(self.dataframe_save)
        img_in_dir = [f for f in os.listdir(self.image_directory) if (len(f.split(".")) > 1 and f.split(".")[1] == "jpg")]
        for f in tqdm(os.listdir(self.image_directory), disable=self.disable_progress_bar, desc="pictures processed"):
            if len(f.split(".")) > 1 and f.split(".")[1] == "jpg" and f.split("_")[-1].split(".")[0].isnumeric() and (page==None or page==f):
                csv_name = os.path.join(self.dataframe_save,f.split(".")[0]+".csv")
                self._extract_images_from_page(f.split(".")[0]).to_csv(csv_name)
                

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    This script is to be ruin after running p2pala to extract the baselines out of 
    handwritten images. It uses will analyse the coordinates produced to create a patch around the baseline.
    These patches will be saved to a directory to allow for processing with htr methods.
    ''',epilog='''
    Here is an example of how to run it on the default path that is created by p2pala:
    
    ./extractPatches.py /dhlabdata4/mion/P2PaLA/work/results/prod/ /dhlabdata4/mion/P2PaLA/work/results/prod/page /dhlabdata4/mion/P2PaLA/work/results/prod/patches/ /dhlabdata4/mion/P2PaLA/work/results/prod/dataframe/''')
    parser.add_argument('image_directory', metavar='imgDir', type=str, 
                        help='directory containing the source images.')
    parser.add_argument('xml_directory', metavar='xmlDir', type=str, 
                        help='directory containing the xml.')
    parser.add_argument('patch_save_dir', metavar='patchDir', type=str, 
                        help='directory where the patches will be saved. If non existant will be created')
    parser.add_argument('dataframe_save_dir', metavar='dfDir', type=str, 
                        help='Locations where the dataframes contains x_min, y_min, x_max, y_max, filename, pagename; will be saved as a csv. If the directory does not exist it will be created.')
    parser.add_argument("--offset_above" ,"-oa", metavar="offsetAbove", type=int, default=100,
                        help="Offset above baseline to take. Larger values will increase the room above the patch.")
    parser.add_argument("--offset_below", "-ob", metavar="offsetBelow", type=int, default=20,
                        help="Offset below baseline to take. Larger values will increase the room below the patch.")
    
    parser.add_argument("--disable_progress_bar", type=bool, default=False,
                        help="dissables progress bar")

    parser.add_argument("--page",metavar="page", type=str, help="page name, jpg filename of the page to run the extraction on. Usefull for debuging if there is a crash on a specific page.")
    
    
    args = parser.parse_args()
        
        
    p = patchExtractor(args.image_directory, args.xml_directory,
                       args.patch_save_dir, args.dataframe_save_dir,
                      args.offset_above, args.offset_below, args.disable_progress_bar)
    if args.page:
        p.extract_all_pages(page=args.page)
    else:
        p.extract_all_pages()
