# `Imagetemplate.py`

This is a Python script that allows converting PNG and JPEG image files to bitmap format. It also provides methods for image encoding, such as radial hashing and mosaic encoding, and for template matching against encoded images.

The script requires the Python Imaging Library Pillow and the numpy library.

## Usage

The script takes command-line arguments with the following format:
```python
python Imagetemplate.py [option] [image list]
```
## Options

The available options are:

`--png_bmp` : convert the PNG image to BMP format and clamp it to a square.  
`--downgrade:n` : downgrade the PNG image by clamping it to a square and then downgrading it to an nxn grid using a mosaic encoding technique. The parameter n defines the size of the grid.
Methods

The script provides the following methods for image encoding:

`EncodeMosaic(n)` : encodes the image into an nxn matrix by cutting the image into a mosaic and counting the number of dark pixels in each sub-grid.  
`EncodeCircle(r_count, t_count)` : encodes the image into a radial map by calculating the fraction of dark pixels in a given set of concentric rings and angles.  
`EncodeDistance(n, rel_centre, spacing)` : encodes the image as a radial distribution function by calculating the average fraction of dark pixels in a set of concentric circles.

## Classes

The script uses three classes:

### `Template`
A class that represents an image and provides methods for encoding and saving the image.

### `Model`
An abstract class to define template matching models.

### `MosaicModel`
A class that inherits from Model and provides a method for comparing mosaic encoded images.

### `CircleModel`
A class that inherits from Model and provides a method for comparing radial hashing encoded images.

### `DistanceModel`
A class that inherits from Model and provides a method for comparing radial distribution encoded images.

### Example

To convert a PNG file my_image.png to BMP format and clamp it to a square:

```Shell
$ python main.py --png_bmp my_image.png
```

To downgrade the BMP image to a 4x4 grid mosaic encoding:

```Shell
$ python main.py --downgrade:4 my_image_clamped.bmp
```

To encode the image using radial encoding and save it:

```Python
t = Template("my_image.png")
t.EncodeCircle(20, 20)
Template.SaveBitmap(t.circleEncoding*255, "my_image_circleEncoded.bmp")
```


To train a model:

```Python
mosaicModel = MosaicModel(6)
mosaicModel.Train("path/to/training/data/")
mosaicModel.Save("mosaicModel.npy")
```

To load a model and test an image:

```Python
mosaicModel = MosaicModel(6)
mosaicModel.Load("mosaicModel.npy")
mosaicModel.Test("my_image_downgraded.bmp")
```