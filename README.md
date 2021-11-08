# tfodapi-tools
Python scripts and Colab notebooks to supplement the Tensorflow Object Detection API v2 (TFODAPI)

## Installation

### `Dockerfiles/` for local installation of the Tensorflow Object Detection API

## Preprocessing

### `retile_for_dl_v2.py`

Retiling geospatial imagery and reformatting geospatial data is a common first step prior to training a deep learning (DL) model. `retile_for_dl_v2.py` is a preprocessing tool for converting GIS imagery and object polygons into formats suitable for training a deep learning-based object detection model in PyTorch or Tensorflow, or other programs.

![A visual depiction of retile_for_dl_v2.py. It shows input georeferenced imagery with shoreline marine debris individually annotated with GIS bounding boxes. retile_for_dl_v2 has retiled this large image into many smaller images, moreover, each bounding box contains new attributes specific to training object detection models (such as associated image pixel coordinates).](imgs/retile_for_dl_v2_readme.png)

There are several key differences between geospatial and DL data formats:

1. Remotely sensed geospatial imagery (typically from satellite, aircraft, or UAS) are typically mosaiced into single images larger than 5,000 x 5,000 pixels. Meanwhile DL often require images as small as 512 x 512 pixels.__Resampling would squash information about small objects contained in satellite images. And let's face it, almost all objects are small when viewed from space!__
2. Geospatial object annotations are located with real-world coordinates independent of the image from which they were derived. Meanwhile, deep learning annotations are bound directly to their image in pixel coordinates.__Affine matrix tranformations can be tricky.__
3. Deep learning often uses a top-left origin. Meanwhile, geospatial programs often utilize a bottom-left origin. __Always mind your y-axis__.

Taking these three challenges into account, `retile_for_dl_v2` is a simple, lightweight, and universal python routine for converting large-format geospatial imagery into a series of smaller images appropriate for deep learning-based object detection algorithms. The retiling procedure is guided by a set of user-specified GIS polygons, each of which denotes an object bounding box annotations within the geospatial imagery. See the example below for a visual abstract of `retile_for_dl_v2.py` in action preparing labeled GIS data about marine debris objects for use in the Tensorflow Object Detection API:

__Flags:__

- `--image_dir` ( `-i` ): a directory containing geospatial imagery data to be retiled for deep learning.
- `--annotations` ( `-a` ): a geospatial vector data file (gpkg or shp) that contains object annotation polygons spread across the images from `--image_dir`.
- `--chip_size` ( `-s` ): the output height and width of the retiled images in pixels.
- `--output_dir` ( `-o` ): a directory to store the outputs of this script. The following will be put there: a directory of re-tiled image chips, a geopackage that shows the real-world locations of image chips in a GIS program, a geopackage of the annotations with chip information written to it's attribute table, and a CSV file of the annotation's attribute table, and a log file.
- `--num_cores` ( `-n` ): an integer value that specifies the number of CPU cores to utilize for processing. For example, _-n 6_ will process 6 images simultaneously. This number should be set as high as possible for fast processing.
- `--verbose` ( `-v` ): print the entire chatty, garbled output of the script to the terminal. This is really only needed when debugging. All this info is written to the log file anyways, so you should probably just omit this flag and go read it there.

__Example Usage:__

- Create 640x640 pixel output image chips utilizing 11 CPU cores:

```python
python3 retile_for_dl_v2.py \
    --input_dir GIS_images/ \
    --annotations GIS_annotations.gpkg \
    --output_dir DL_data/ \ 
    --chip_size 640 640 \
    --num_cores 11
```

- Create 1024x1024 pixel output image chips utilizing 11 CPU cores with verbose output:

```python
python3 retile_for_dl_v2.py \
    --input_dir GIS_images/ \
    --annotations GIS_annotations.gpkg \
    --output_dir DL_data/ \ 
    --chip_size 640 640 \
    --num_cores 11 \
    --verbose
```

- Get additional help:

```python
python3 retile_for_dl_v2.py --help
```
