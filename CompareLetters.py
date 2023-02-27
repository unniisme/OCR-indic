from ImageTemplate import Template
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    print("Encoding " + file)
    template = Template(os.path.join(directory, file))
    template.EncodeMosaic(16)
    # template.EncodeCircle(20, 10)
    template.EncodeDistance(10)
    templates[file.replace(".png", "")] = template

# Test similarity with A
def testSimilarity(target_key, saveImage = True, showGraph = True):
    print("Testing similarity with ", target_key)

    target = templates[target_key]
    for template_key in templates:
        print(template_key)
        similarityMatrix = target.mosaicEncoding - templates[template_key].mosaicEncoding
        # similarityMatrix = target.circleEncoding - templates[template_key].circleEncoding
        # similarityMatrix = target.distanceEncoding - templates[template_key].distanceEncoding 
        # print(similarityMatrix)
        print("Crude difference: ", np.sum(np.abs(similarityMatrix))/len(similarityMatrix))
        # print()
        if saveImage:
            similarityDir = f'similarity_with_{target_key}'
            if not os.path.exists(similarityDir):
                os.mkdir(similarityDir)
            Template.SaveBitmap(np.vectorize(visualizationMatrix)(similarityMatrix.reshape(-1,1)), similarityDir + "/" + template_key + ".bmp")

        if showGraph:
            plt.plot(list(range(len(target.distanceEncoding))), target.distanceEncoding, label = "Target")
            plt.plot(list(range(len(templates[template_key].distanceEncoding))), templates[template_key].distanceEncoding, label="Template")
            plt.title(target_key + "-" + template_key)
            plt.legend()
            plt.show()

            

testSimilarity('A', True, False)
print()
testSimilarity('H', True, False)

