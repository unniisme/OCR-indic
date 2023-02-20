from ImageTemplate import Template
import os
import sys
import numpy as np
from PIL import Image

np.set_printoptions(precision=3)

templates = {}
directory = 'Letters'

def visualizationMatrix(x):
    return 255*(x+1)/2
    
    
def visualizationMatrix_clamped(x):
    cutoff = 0.5
    if -0.5 < x < 0.5:
        return 255/2
    elif x <= -0.5:
        return 0
    else:
        return 255 


# Save each encoding
for file in os.listdir(directory):
    template = Template(os.path.join(directory, file))
    template.EncodeMosaic(8)
    templates[file.replace(".png", "")] = template

# Test similarity with A
def testSimilarity(target_key):
    print("Testing similarity with ", target_key)

    target = templates[target_key]
    for template_key in templates:
        print(template_key)
        similarityMatrix = target.mosaicEncoding - templates[template_key].mosaicEncoding
        # print(similarityMatrix)
        print("Crude difference: ", np.sum(np.abs(similarityMatrix)))
        # print()
        similarityDir = f'similarity_with_{target_key}'
        if not os.path.exists(similarityDir):
            os.mkdir(similarityDir)
        Template.SaveBitmap(np.vectorize(visualizationMatrix_clamped)(similarityMatrix), similarityDir + "/" + template_key + ".bmp")

testSimilarity('A')
print()
testSimilarity('H')

