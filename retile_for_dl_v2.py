import os
import sys
import time
import datetime
import argparse
import multiprocessing
from loguru import logger

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box


def verify_input_kernel(input_kernel):
    k = int(input_kernel)
    if k % 2 != 0:
        parser.error("The user-specified chip size parameters ('-s' or '--chip_size') must be even integers.")
    else:
        return k

def verify_dir_path(string):
    '''
    Ensures a valid directory using argparse. 
    https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
    '''
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def verify_file_choices(choices,fname):
    '''
    Ensure a valid geospatial file using argparse.
    https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
    '''
    ext = os.path.splitext(fname)[1][1:]
    if ext not in choices:
       parser.error(f"Detections file doesn't end with one of the following valid formats: {choices}")
    return fname

@logger.catch
def rio_bounds_to_gpkg(rio_img, rio_img_filename, filename_attribute_header='srcimg'):
    '''
    A simple function that takes a rasterio dataset (already opened with rasterio.open()) and dumps the image's bounding box to
    a geopandas GeoDataFrame. The data set's CRS is properly set and the original filename is added to the attribute table under
    the <filename_attribute_header> column.

    '''
    bbox = box(*rio_img.bounds)
    row = {filename_attribute_header:[rio_img_filename], 
           'geometry':[bbox]}

    out = gpd.GeoDataFrame(row, crs=rio_img.crs)

    return out

@logger.catch
def create_chip_from_centroid(img, centroid, chip_size, output_dir, chip_basename):
    '''
    Given an image and a point; this function will create a small image subset (a "chip") that is centered on the
    point at (<centroid_x>, <centroid_y>). The chip will be of shape (<x_size>, <y_size>, depth) and it will be written 
    to the <output_dir> with filename of the format: "<chip_basename>_row#_col#.jpg".

    The function will additionally return the chip's geometry (formatted as a single row in a geopandas GeoDataFrame), and 
    the chip's affine transformation matrix.

    NOTE: this function is reads and writes "boundless" windows (rasterio.open(boundless=True)). This means that areas of 
            the output image chips that fall outside of the original image's extent will be written with black "no data" 
            pixel values. 

    Parameters:
    ------------
    - img: a rasterio image dataset with georeferencing information. This parameter should come directly from rasterio.open().
    - centroid: a tuple containing the X/Y coordinate of the desired image chip's centroid (the coordinate system should match img)
    - chip_size: a tuple containing the desired width/height of the image chip in pixel values
    - output_dir: the location of the folder in which to store the output chip image
    - chip_basename: a string value that will be used as the prefix of the output chip name. The row, col, and <chip_ext> will all 
                        be appended to create a unique chip filename.
    - chip_ext: a string value that contains the extension of the output image with '.' prefixed. Examples are '.jpg' or '.tif'.
    '''

    x_size = chip_size[0]
    y_size = chip_size[1]

    # determine the chip bbox's pixel coordinates based on kernel size and centroid
    x_stride = int(x_size / 2)
    y_stride = int(y_size / 2)

    target_lbl_centroid = rasterio.transform.rowcol(img.transform, centroid[0], centroid[1])
    
    min_y = target_lbl_centroid[0] - y_stride
    max_y = min_y + y_size
    min_x = target_lbl_centroid[1] - x_stride
    max_x = min_x + x_size
    
    
    try:
        # create a window into image based on chip bbox, pull affine transform info
        chip_window = rasterio.windows.Window.from_slices((min_y, max_y), (min_x, max_x), boundless=True)
        chip_transform = img.window_transform(chip_window)
    except:
        logger.error(f"ERROR! There is an issue slicing a window from {chip_basename} at location {(min_y, max_y), (min_x, max_x)}. Input centroid was {target_lbl_centroid}.")
    
    # write the cropped chip to disk using the affine transform info
    profile = img.profile
    profile.update({
        'driver': 'JPEG',
        'height': y_size,
        'width': x_size,
        'transform': chip_transform}) 

    chip_output_dir = os.path.join(output_dir, 'v2-chips')
    if not os.path.exists(chip_output_dir):
        os.makedirs(chip_output_dir, exist_ok=True)

    chip_filename = str(chip_basename) + f'_{min_x}_{min_y}.jpg'
    chip_filepath = os.path.join(chip_output_dir, chip_filename)

    logger.debug(f"{chip_filename} sliced at {(min_y, max_y), (min_x, max_x)} to produce {chip_window}.")

    with rasterio.open(chip_filepath, 'w', **profile) as dst:
      # Read the data from the window and write it to the output raster
      dst.write(img.read(window=chip_window, boundless=True))

      chip_gdf = rio_bounds_to_gpkg(dst, chip_filename, filename_attribute_header='filename')

    logger.debug(f"{chip_filename} written to: {chip_filepath}")

    return chip_gdf, chip_transform

@logger.catch
def add_px_coords_to_gdf(gdf, chip_transform, ):
    '''
    This function converts the coordinates of polygons (stored in a geopandas dataframe) into an image's pixel coordinates using
    an affine transformation matrix (typically this comes from the rasterio.transform module). This function assumes the polygons
    are contained within the image's extent. This function also assumes the polygons and image are of the same real-world coordinate
    system. 
    NOTE: This function is designed to run in a pd.apply() statement with the result_type='expand' flag. This will broadcast the
    output new_columns dictionary across the input gdf's table.
    
    Parameters:
    ------------
    - gdf: a geopandas GeoDataFrame that contains polygons. The GeoDataFrame should have 'geometry' column, which is used to calculate
           the output pixel coordinates. 
    - chip_tranform: an affine tranformation matrix (typically from the rasterio.transform module) that is used to compute pixel
           coordinates from real-world coordinates. 
    TODO:
    ------------
    ''' 
    # pull the real-world coordinates from the current gdf row
    coords = list(gdf.geometry.exterior.coords)
    
    # reformat the real-world coordinates into lists for rasterio
    xs = []
    ys = []
    for coord in coords:
      xs.append(coord[0])
      ys.append(coord[1])

    # transform the real-world coords into row/col pixel values
    pixels = rasterio.transform.rowcol(chip_transform, xs, ys)
    
    # dictionary keys become column names; dictionary values become row values
    new_columns = {'xmin': min(pixels[1]),
                   'xmax': max(pixels[1]),
                   'ymin': min(pixels[0]),
                   'ymax': max(pixels[0])}  
    return new_columns

@logger.catch
def retile_for_dl_v2(image_path, in_lbls, kernel_size, output_dir, INTERESECTION_THRESHOLD = 0.5):
    '''
    This function will convert geospatial polygons into a format suitable for deep learning-based object detection. A geo-referenced 
    image and a set of georeferenced polygon object labels (bounding boxes) are cross-referenced, and the geo-referenced image is "chipped"
    into smaller images, each of which will have all overlapping bounding boxes associated with it. Extraneous sections of imagery that do 
    not contain bounding boxes will be dropped. These smaller "image chips" can be of a small size, such as 640x640 or 1024x1024 pixels, which
    are common of deep learning but considered extremely small for geospatial images. 

    Image chips are written directly to disk. A geospatial "chip index" that shows each chip image's location and a geospatial "detections file"
    that contains all of the input object labels will be returned by the function. The attribute table of the detections file will contain
    associated chip names and chip coordinates. The detections file is suitable to be exported directly to a GIS format or directly to CSV 
    format for use in a variety of deep learning workflows.

    Parameters:
    ------------
    - image_path: the absolute path to a georeferenced image on disk.
    - in_lbls: a geopandas dataframe containing georeferenced polygons. These polygons should represent object detection bounding boxes. At 
                least some of these should overlap the target image (but it's okay if none do, the script will just pass).
    - kernel_size: the desired (width, height) of the output image chips in pixels.
    - output_dir: the location on disk to store output files: final detections (.gpkg and .csv), final chip index (.gpkg), 
                    image chips (.jpgs), logfile (.log)
    - INTERSECTION_THRESHOLD: a float value ranged 0 to 1.0 that specifies the threshold at which to keep to detections appearing in a chip. This 
                                prevents partial object labels that appear at chip edges from creeping into training data. 50% by default (0.5).

    TODO (v3):
    ------------
    - [ ] Allow user to specify INTERSECT_THRESH, output formats, etc.
    - [ ] Automatically mark partial annotations with attribute in table. This would be 
            done by flagging detections > INTERSECTION_THRESHOLD. (could use 'difficult' 
            flag that is common in DL annotation programs.)
    - [ ] Negative image chips?
    - [ ] The ability to use "n - 1" cores for multiprocessing. Also enable this auto-multicore mode by default.
    - [ ] The ability to specify a single chip dimension for equal-size images (i.e. '-s 512' would be the equivalent of '-s 512 512')
    - [ ] Brainstorm on if/how to allow odd-sized image. I don't see any problem conceptually, but must think more throughly.
    - [ ] Group loguru messages for each image and drop them into the logfile/terminal in an orderly fashion to enhance human readability.
    '''
    
    chip_basename = os.path.splitext(os.path.basename(image_path))[0]

    with rasterio.open(image_path, 'r') as img:
        if img.crs != in_lbls.crs:
            logger.warning(f'Input image {chip_basename} and the input annotation file do not have matching coordinate reference systems ({img.crs} and {in_lbls.crs}, respectively). These files cannot be compared.')
        else:
            # turn the chip bbox vector polygon into a single-row GeoDataFrame, filter lbls outside image 
            bbox_gdf = rio_bounds_to_gpkg(img, image_path)

            lbls = gpd.overlay(in_lbls, bbox_gdf)

            if len(lbls) == 0:
                logger.debug(f"{chip_basename} contains 0 labels!!")
            else:
                logger.debug(f"{chip_basename} contains {len(lbls)} labels.")
                # We use a hardcoded index because the gpd.overlay() operation does not preserve the original left index (our detections). 
                # TODO: Is this proper geopandas/pandas practice? I feel like this is kind of hacky...
                idx = [x for x in range(0, len(lbls))]
                lbls['idx'] = idx

                lbls['area'] = lbls.geometry.area
                lbls['captured'] = 0

                final_chips = []
                final_detections_by_chip = []
                round=0
                while len(lbls[lbls['captured']==0]) > 0:
                    # of the uncaptured labels, sort by largest, and take the top entry
                    sorted_lbls = lbls[lbls['captured']==0].sort_values(by='area', ascending=False)
                    target_lbl = sorted_lbls.iloc[0]

                    # pull the centroid, write the chip image to disk, get the chip's bbox 
                    centroid_xy = (target_lbl.geometry.centroid.x, target_lbl.geometry.centroid.y)
                    chip_bounds, chip_transform = create_chip_from_centroid(img, centroid_xy, kernel_size, output_dir, chip_basename) 

                    # put our chip in our collection of chips
                    final_chips.append(chip_bounds) 

                    # find all labels within chip, figure out which ones are 70% or more contained by our chip (default values)
                    lbls_within = gpd.overlay(lbls, chip_bounds, how='intersection')
                    lbls_within['intersect_area'] = lbls_within.geometry.area
                    lbls_within['intersect_percent'] =  lbls_within['intersect_area'] / lbls_within['area'] 
                    lbls_above_threshold = lbls_within[lbls_within['intersect_percent'] >= INTERESECTION_THRESHOLD] 

                    # convert to chip coordinates, get the chip name, format an attribute table
                    detection_px_coords = lbls_above_threshold.apply(add_px_coords_to_gdf, chip_transform=chip_transform, axis='columns', result_type='expand') 
                    to_concat = [lbls_above_threshold, detection_px_coords]
                    final_detections = gpd.GeoDataFrame(pd.concat(to_concat, axis='columns'),  
                                                            crs=to_concat[0].crs)

                    # update the "captured" tally
                    idxs_above_threshold = final_detections.idx.tolist()
                    lbls.loc[lbls['idx'].isin(idxs_above_threshold), 'captured'] = 1    

                    # get final_detections into final form (this will be the resulting attribute table entry). Then
                    # put it in our collection of detections.
                    final_detections = final_detections.drop(['area', 'intersect_area', 'intersect_percent', 'captured', 'idx', 'srcimg'], axis=1)

                    final_detections_by_chip.append(final_detections)   

                    round = round + 1

                final_chips_gdf = gpd.GeoDataFrame(pd.concat(final_chips, axis='rows'),  
                                                  crs=final_chips[0].crs)

                final_detections_gdf = gpd.GeoDataFrame(pd.concat(final_detections_by_chip, axis='rows'),  
                                                        crs=final_detections_by_chip[0].crs)

                logger.debug(f"{chip_basename} was chipped {len(final_chips_gdf)} times.")
                out = [final_detections_gdf, final_chips_gdf]
                
                return out

if __name__ == "__main__":
    accepted_raster_types = ('tif', 'TIF', 'tiff', 'TIFF', 'jpg', 'JPG', 'jpeg', 'JPEG')
    accepted_vector_types = ('shp','gpkg')

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--image_dir',
                    dest='image_dir', type=verify_dir_path,
                    help="A directory of geotiff (tif) or jpeg (jpg) images.")
    
    parser.add_argument('-a', '--annotations',
                    dest='annotations', type=lambda d:verify_file_choices(accepted_vector_types, d),
                    help="A geopackage (gpkg) or shapefile (shp) that contains polygon bounding boxes of object annotations to be used for deep learning-based object detection. If these files' attributes contain the values: xmin, ymin, xmax, or ymax then those fields will be overwritten in the output labels geopackage.")

    parser.add_argument('-s', '--chip_size',
                    dest='kernel_size', type=lambda s:verify_input_kernel(s),
                    nargs=2, 
                    help="The desired height and width of the output image chips (in pixels).")

    parser.add_argument('-o', '--output_dir',
                    dest='output_dir', type=verify_dir_path,
                    help="A directory to store image chips, a chip index geopackage, and the associated detections geopackage.")
    
    parser.add_argument('-n', '--num_cores',
                    dest='num_cores', type=int,
                    help="An integer that specifies the number of computer cores to use for computation. This number should be set based on your machine's CPU. Defaults to 1 core (very slow!!).")

    parser.add_argument('-v', '--verbose', 
                    dest='verbosity', action='store_true',
                    help='Including this switch will print DEBUG messages to the terminal. The debug messages are garbled, chatty, and really only useful when debugging the program or trapping errors. Note that this switch only controls terminal output; DEBUG messages are always written to the log file.')

    args = parser.parse_args()

    annotations = gpd.read_file(args.annotations)
    # NOTE: we quietly drop these columns from the original attribute table... maybe not best practice. We do put that in the help though.
    annotations = annotations.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1, errors='ignore')

    kernel_size = tuple(args.kernel_size)
    output_dir = args.output_dir
    # a one-liner to scan the input_dir for images w/ accepted raster extension and compile a list of lists with args for retile_for_dl_v2
    args_list = [[os.path.join(args.image_dir, f), annotations, kernel_size, output_dir] for f in os.listdir(args.image_dir) if f.endswith(accepted_raster_types)]

    logger.remove()
    if args.verbosity == True:
        logger.add(sys.stderr, colorize=True, format=" <cyan>{time:hh:mm}</cyan>| <level>{message}</level>", enqueue=True, level="DEBUG")
    else:
        logger.add(sys.stderr, colorize=True, format=" <cyan>{time:hh:mm}</cyan>| <level>{message}</level>", enqueue=True, level="INFO")
        
    timestamp = datetime.datetime.now()
    timestamp_formatted = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
    logger.add(os.path.join(output_dir, f"v2-log-{timestamp_formatted}.log"))
    
    logger.info("RETILE_FOR_DL_V2 START")
    logger.info(" -----------------------")
    logger.info("INITIAL PROJECT SUMMARY:")
    logger.info(f"{len(args_list)} images found in directory {args.image_dir}.")
    logger.info(f"Writing image chips of size {kernel_size} to {output_dir}.")
    logger.info(" -----------------------")
    
    pool=multiprocessing.Pool(processes=args.num_cores)
    map_results = pool.starmap_async(retile_for_dl_v2, args_list, chunksize=1) #chunksize=1 is to make the while loop below display the correct info.

    while not map_results.ready():
        logger.info(f"retile_for_deeplearning_V2.py | {map_results._number_left} of {len(args_list)} files remain. \n") #_number_left is prob wrong way to do this. https://stackoverflow.com/questions/49807345/multiprocessing-pool-mapresult-number-left-not-giving-result-i-would-expect
        time.sleep(5)

    pool.close()
    pool.join()

    # package up the "final final" results and return to user.
    results = map_results.get()
    clean = [x for x in results if x is not None]
    
    if len(clean) == 0:
       logger.error(f"RETILE_FOR_DL_V2 FAILED! No detections were found to overlap your input imagery. Exiting without final chip indicies or a detections record. Check that your detections do overlap your image and that the imagery and detections CRS match.")
    else:
        final_final_chips = []
        final_final_detections = []
        for res in clean:
            final_final_detections.append(res[0])
            final_final_chips.append(res[1])

        final_final_chips_gdf = gpd.GeoDataFrame(pd.concat(final_final_chips, axis='rows'),  
                                                    crs=final_final_chips[0].crs)
        final_final_detections_gdf = gpd.GeoDataFrame(pd.concat(final_final_detections, axis='rows'),  
                                                        crs=final_final_detections[0].crs)

        # set the final GPKG/CSV output column order
        preferred_column_order = ['filename', 'label', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'geometry']
        final_final_detections_gdf = final_final_detections_gdf.reindex(columns=preferred_column_order)

        # write GPKGs useful for geospatial applications or to visually review outputs
        final_final_chips_path = os.path.join(output_dir, 'v2-chip-index.gpkg')
        final_final_detections_path = os.path.join(output_dir, 'v2-labels.gpkg')

        final_final_chips_gdf.to_file(final_final_chips_path, layer='chips', driver="GPKG")
        final_final_detections_gdf.to_file(final_final_detections_path, layer='detections', driver="GPKG")

        # write detections CSV useful for deep learning
        final_final_detections_csv_path = os.path.join(output_dir, 'v2-labels.csv')

        final_final_detections_gdf.to_csv(final_final_detections_csv_path, header=True, index=False)

        # Close out the project!
        logger.info("RETILE_FOR_DL_V2 SUCCESSFUL!")
        logger.info(" -----------------------")
        logger.info("FINAL PROJECT SUMMARY:")
        logger.info(f"{len(final_final_chips_gdf)} chips were written to {output_dir}. The chip index was written to {final_final_chips_path}")
        logger.info(f"{len(final_final_detections_gdf)} associated detections were written to {final_final_detections_path}.")
        logger.info(" -----------------------")