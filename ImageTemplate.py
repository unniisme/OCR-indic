from PIL import Image
import numpy as np
import os
import sys

def getRadialPixel(r,t, centre):
    return (int(r*np.cos(t) + centre[0]), int(r*np.sin(t) + centre[1]))


def ImgToBitmap(img):
    ary = np.array(img)

    # Split the three channels
    r,g,b,a = np.split(ary,4,axis=2)
    r=r.reshape(-1)
    g=g.reshape(-1)
    b=b.reshape(-1)

    # Standard RGB to grayscale 
    bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2], zip(r,g,b)))
    bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float),255)
    return bitmap

class Template:

    mosaic_N = 6

    def __init__(self, file):
        img = Image.open(file).convert("RGBA")
        bitmap = ImgToBitmap(img)
        bitmap = Template.ClampBitmap(bitmap)
        self.size = len(bitmap)
        self.bitmap = bitmap

    def SaveBitmap(bitmap, filename):
        im = Image.fromarray(bitmap.astype(np.uint8))
        im.save(filename)

    def ShowBitmap(bitmap):
        im = Image.fromarray(bitmap.astype(np.uint8))
        im.show()

    def Save(self, filename):
        Template.SaveBitmap(self.bitmap, filename)

    def EncodeMosaic(self, n):
        """
        Encodes the data about the image into an nxn matrix my cutting the image into a mosaic and counting the number of dark pixels in each subgrid
        """
        def subBitmapPolicy(subBitmap):
            return sum([1 if val == 0 else 0 for val in np.nditer(subBitmap)])/(subBitmap.shape[0] * subBitmap.shape[1])


        gridPoints = [int(x) for x in np.linspace(0, self.size, n+1)]
        self.mosaicEncoding = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                subBitmap = self.bitmap[gridPoints[i]:gridPoints[i+1], gridPoints[j]:gridPoints[j+1]]
                self.mosaicEncoding[i,j] = subBitmapPolicy(subBitmap)

        return self.mosaicEncoding

        ## Depriciate
        gridSize = int(self.size/n)
        self.mosaicEncoding = np.zeros((n+1,n+1))

        for i in range(n):
            for j in range(n):
                subBitmap = self.bitmap[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize]
                self.mosaicEncoding[i,j] = sum([1 if val == 0 else 0 for val in np.nditer(subBitmap)])/(gridSize**2)

        return self.mosaicEncoding

    def EncodeCircle(self, r_count, t_count):
        """
        Encodes the data into a radial map
        """        
        def tryInBitmap(bitmap, pos):
            try:
                return bitmap[pos]
            except:
                return 255

        r_unit = self.size/(2*r_count)
        t_unit = 2*np.pi/t_count

        circleEncoding = np.zeros((r_count-1,t_count-1))

        for i in range(r_count-1):
            for j in range(t_count-1):
                pixels = [tryInBitmap(self.bitmap,getRadialPixel(r, t, (self.size/2,self.size/2)))/255 for r in np.arange(i*r_unit, (i+1)*i*r_unit, 1) for t in np.arange(j*t_unit, (j+1)*t_unit, 0.2)]
                circleEncoding[i,j] = sum(pixels)/len(pixels) if not len(pixels) == 0 else 0

        self.circleEncoding = circleEncoding

        return circleEncoding
    
    def EncodeDistance(self, n=20):
        """
        Encodes the data as a radial distribution function
        """

        def tryInBitmap(bitmap, pos):
            try:
                return bitmap[pos]
            except:
                return 255
        
        centre = (self.size/2,self.size/2)
        
        R = np.linspace(0, self.size*int(np.sqrt(2)), n)

        self.distanceEncoding = np.zeros(n)

        for i,r in enumerate(R):
                radPixels = [tryInBitmap(self.bitmap, getRadialPixel(r, t, centre))/255 for t in np.linspace(0, 2*np.pi, int(2*np.pi*r))]
                self.distanceEncoding[i] = sum(radPixels)/len(radPixels) if len(radPixels) !=0 else 0

        return self.distanceEncoding

    def ClampBitmap(bitmap):
        """
        Removes whitespaces at each edge, return square image centered on the axis not clamped
        """
        negative = 255-bitmap
        i = 0
        while sum(negative[i]) == 0:
            negative = np.delete(negative, i, 0)

        i = -1
        while sum(negative[i]) == 0:
            negative = np.delete(negative, i, 0)       
            i = negative.shape[0]-1

        i = 0
        while sum(negative[:,i]) == 0:
            negative = np.delete(negative, i, 1)

        i = -1
        while sum(negative[:,i]) == 0:
            negative = np.delete(negative, i, 1)       
            i = negative.shape[1]-1

        diff = negative.shape[0] - negative.shape[1]
        if diff != 0:
            if diff > 0:
                if diff%2 == 0:
                    padding1 = int(diff/2)
                    padding2 = int(diff/2)
                else:
                    padding1 = int(diff/2)
                    padding2 = int(diff/2) + 1
                negative = np.c_[np.zeros((negative.shape[0], padding1)), negative, np.zeros((negative.shape[0], padding2))]
            else:
                if diff%2 == 0:
                    padding1 = int(-diff/2)
                    padding2 = int(-diff/2)
                else:
                    padding1 = int(-diff/2)
                    padding2 = int(-diff/2) + 1
                negative = np.r_[np.zeros((padding1, negative.shape[1])), negative, np.zeros((padding2, negative.shape[1]))]

        return 255-negative

    def DownGrade(self, filename, size):
        ar = self.EncodeMosaic(size)
        ar = 255 - 255*ar
        Template.SaveBitmap(ar, filename)


class Model:
    """Abstract class to define Template matching models"""

    def __init__(self, encoding, compareEncodings, *args):
        self.encoding = encoding
        self.templateEncodings = {}
        self.compareEncodings = compareEncodings
        self.encodingArgs = args

    def Train(self, directory, verbose = False):
        if os.path.isfile(directory):
            targetName = directory.split('/')[-1].split(".")[0]
            if verbose:
                print("Training Target: " + targetName)
            self.templateEncodings[targetName] = self.encoding(Template(directory), *self.encodingArgs)
        
        else:
            for filename in os.listdir(directory):
                self.Train(os.path.join(directory, filename), verbose)

    def Test(self, filename):
        testEncoding = self.encoding(Template(filename), *self.encodingArgs)
        comparisons = []
        for template_key in self.templateEncodings:
            comparisons.append((self.compareEncodings(testEncoding, self.templateEncodings[template_key]), template_key))

        return sorted(comparisons)[0][1]

    def Save(self, filename):
        np.save(filename, self.templateEncodings, allow_pickle=True)

    def Load(self, filename):
        self.templateEncodings = np.load(filename, allow_pickle=True)

class MosaicModel(Model):

    def __comparison(encoding_a, encoding_b):
        similarityMatrix = encoding_a - encoding_b
        return np.sum(np.abs(similarityMatrix))/len(similarityMatrix)

    def __init__(self, n):
        super().__init__(Template.EncodeMosaic, MosaicModel.__comparison, n)

class CircleModel(Model):

    def __comparison(encoding_a, encoding_b):
        similarityMatrix = encoding_a - encoding_b
        return np.sum(np.abs(similarityMatrix))/len(similarityMatrix)

    def __init__(self, r, t):
        super().__init__(Template.EncodeCircle, CircleModel.__comparison, r, t)

class DistanceModel(Model):

    def __comparison(encoding_a, encoding_b):
        similarityMatrix = encoding_a - encoding_b
        return np.sum(np.abs(similarityMatrix))/len(similarityMatrix)

    def __init__(self, n):
        super().__init__(Template.EncodeDistance, DistanceModel.__comparison, n)



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("A simple image processing script. Converts png or jgp to bitmap.")
        print("use --png_bmp [image] to convert png image to bmp and clamp it to a square.")
        print("use --downgrade:n to downgrade clamp png to a square, and downgrade it to an nxn grid.")
        print("Result files are saved beside the input files.")

    elif sys.argv[1] == "--png_bmp":
        for i in range(2, len(sys.argv)):
            print("Converting " + sys.argv[i])
            Template(sys.argv[i]).Save(sys.argv[i].replace(".png", "_clamped.bmp"))

    elif sys.argv[1].split(":")[0] == "--downgrade":
        for i in range(2, len(sys.argv)):
            print("Downgrading " + sys.argv[i])
            Template(sys.argv[i]).DownGrade(sys.argv[i].replace(".png", "_downgraded.bmp"), int(sys.argv[1].split(":")[1]))

    else:
        for i in range(1, len(sys.argv)):
            print("encoding " + sys.argv[i])
            t = Template(sys.argv[i])
            t.EncodeCircle(20, 20)
            Template.SaveBitmap(t.circleEncoding*255, sys.argv[i].replace(".png", "_circleEncoded.bmp"))
            





    