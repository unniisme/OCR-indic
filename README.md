# OCR-indic
An exploration of OCR techniques and their effectivity on South Indian Lanugages.

`ImageTemplate.py` : File which contains some functions to handle bitmaps. Find the documentation in `ImageTemplate.md`.  
`CompareLetters.py` : Script to compare a specific letter with all the other letters in a given directory.

To use the following scripts, first extract `./Letters/archives.zip` into the same directory.

`TestMosaicModel.py`,
`TestCircleModel.py`,
`TestRadialModel.py` : Script to test the corresponding model on a large dataset and save the output into a csv.

`ValidateTrainingData.py` : Validate the correctness of the predictions from the previous scripts. use with file name of csv as argument.

`SegregateByOrder.md` : Segregate the letter into different categories based on their order parameter and pick representative from each category