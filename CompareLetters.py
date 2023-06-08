from ImageTemplate import Template
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

templates = {}
directory = 'Letters/Test_letters'

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
    template_key = file.replace(".png", "")
    # if template_key[0] not in ['A']:
    # if template_key not in ['A', 'B', 'H', 'R']:
    #     continue
    print("Encoding " + file)
    template = Template(os.path.join(directory, file))
    # template.EncodeMosaic(16)
    # template.EncodeCircle(20, 10)
    distanceEncoding = template.EncodeDistance(200)
    # e1 = template.EncodeDistance(200)
    # e2 = template.EncodeDistance(200, rel_centre=(0,0))
    # e3 = template.EncodeDistance(200, rel_centre=(1,0))
    # e4 = template.EncodeDistance(200, rel_centre=(0,1))
    # e5 = template.EncodeDistance(200, rel_centre=(1,1))
    # distanceEncoding = (e1+e2+e3+e4+e5)/5

    templates[file.replace(".png", "")] = template

    # cumulatedDistribution = [sum(template.distanceEncoding[:i]) for i in range(len(template.distanceEncoding))]

    # plt.plot(list(range(len(template.distanceEncoding))), cumulatedDistribution, label=template_key)
    # plt.plot(list(range(len(e1))), e1, label="Centre")
    # plt.plot(list(range(len(e2))), e2, label="LT")
    # plt.plot(list(range(len(e3))), e3, label="RB")
    # plt.plot(list(range(len(e4))), e4, label="RT")
    # plt.plot(list(range(len(e5))), e5, label="LB")

    plt.plot(list(range(len(distanceEncoding))), distanceEncoding, label=template_key)

plt.title("Comparison between radial distributions")
plt.legend()
plt.show()
quit()



# Test similarity with A
def testSimilarity(target_key, saveImage = True, showGraph = True):
    print("Testing similarity with ", target_key)

    target = templates[target_key]
    for template_key in templates:
        print(template_key)
        # similarityMatrix = target.mosaicEncoding - templates[template_key].mosaicEncoding
        # similarityMatrix = target.circleEncoding - templates[template_key].circleEncoding
        similarityMatrix = target.distanceEncoding - templates[template_key].distanceEncoding 
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

            

testSimilarity('A', False, True)
print()
testSimilarity('H', False, True)
plt.legend()
plt.title("A, H")
plt.show()

