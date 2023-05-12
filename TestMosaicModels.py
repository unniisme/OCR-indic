from ImageTemplate import Model, MosaicModel, CircleModel
import os, sys
import pandas as pd

_startTime = 0
def StartTimer():
    _startTime = os.times()[1]

def TickTimer():
    return os.times()[1] - _startTime

# Train with letters in ENG_TNR folder
_train_path = "./Letters/ENG-TNR"
_test_path = "./Letters/archive"
_base_labels = "english.csv"
_train_labels = "english_train.csv"
_predicted = "english_predictions.csv"

# Loading database and cutting it down a notch
if not os.path.isfile(os.path.join(_test_path, _train_labels)):
    labels = pd.read_csv(os.path.join(_test_path,_base_labels))
    testImages = labels[~labels["label"].isin([str(i) for i in range(10)])]
    testImages = testImages.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))
    testImages = testImages[testImages["label"].str.isupper()]
    testImages.to_csv(os.path.join(_test_path, _train_labels))

else:
    testImages = pd.read_csv(os.path.join(_test_path, _train_labels))


if not os.path.isfile(os.path.join(_test_path, _predicted)):
    train_data = pd.DataFrame(columns=['image', 'label', 'predicted_label', 'time'])
    train_data.to_csv(os.path.join(_test_path, _predicted))
else:
    train_data = pd.read_csv(os.path.join(_test_path, _predicted))


N_images = len(testImages)


# -------------------------
print("Mosaic Model")
print("n=50")
mosaic = MosaicModel(50)
print("--Training--")
mosaic.Train(_train_path, verbose=True)

print("--Testing--")
times = []
accuracy = []
for i,image in enumerate(testImages["image"]):
    if (i<len(train_data)):
        continue
    StartTimer()
    out = mosaic.Test(os.path.join(_test_path, image))
    times.append(TickTimer())
    accuracy.append(int(out == testImages[testImages["image"]==image]["label"].iloc[0]))
    train_data = pd.concat([train_data,pd.DataFrame({'image':[image], 'label': [testImages[testImages["image"]==image]["label"].iloc[0]], 'predicted_label': [out], 'time': [times[-1]]})], ignore_index=True)
    train_data.to_csv(os.path.join(_test_path, _predicted), index=False)

    print(str(100*i//N_images) + "%", end="\r")

print()
print("Accuracy: ", sum(accuracy)/len(accuracy))
print("Average Time: ", sum(times)/len(times))