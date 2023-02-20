from PIL import Image
import numpy as np
import os
import sys


def ImgToBitmap(img):
    ary = np.array(img)

    # Split the three channels
    r,g,b,a = np.split(ary,4,axis=2)
    r=r.reshape(-1)
    g=g.reshape(-1)
    b=b.reshape(-1)

    # Standard RGB to grayscale 
    bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2], 
    zip(r,g,b)))
    bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float),255)
    return bitmap

class Template:

    mosaic_N = 6

    def __init__(self, file):
        img = Image.open(file)
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
        gridSize = int(self.size/n)
        self.mosaicEncoding = np.zeros((n+1,n+1))

        for i in range(n):
            for j in range(n):
                subBitmap = self.bitmap[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize]
                self.mosaicEncoding[i,j] = sum([1 if val == 0 else 0 for val in np.nditer(subBitmap)])/(gridSize**2)

        return self.mosaicEncoding

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



if __name__ == '__main__':
    if sys.argv[1] == "--png_bmp":
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
            print(Template(sys.argv[i]).EncodeMosaic(6))



    