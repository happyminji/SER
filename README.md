# SER
Deep learning driven Acoustic Emotion Recognition and Visualization Methodology

# Installation
1. Install Python 3.7.
2. Clone this repository.
3. Install the required dependencies:
    using the requirements.txt file
4. Add an audio dataset following the instructions below

## Adding an Audio Dataset        
```Bash
|           
+---datasources
|   +---EMO_DB
|   |   |   EMO_DB_KEYS.json   
|   |   \---EMO_DB_DATA
|   |       \---wav
|   |               16b10Tb.wav
|   |               16b10Td.wav
|   |               16b10Wa.wav
|   |               16b10Wb.wav
|   |               
|   +---LDC
|   |   |   LDC_KEYS.json  
|   |   \---LDC_DATA
|   |       \---wav
|   |               1AX1s0020.wav
|   |               1AX1s0021.wav
|   |               1AX1s0040.wav
|   |               1AX1s0060.wav
|   |               
|   +---RAVDESS
|   |   |   RAVDESS_KEYS.json  
|   |   \---RAVDESS_DATA
|   |       \---Audio_Speech_Actors_01-24
|   |           +---Actor_01
|   |           |       03-01-01-01-01-01-01.wav
|   |           |       03-01-01-01-01-02-01.wav
|   |           |       03-01-01-01-02-01-01.wav
|   |           |       03-01-01-01-02-02-01.wav
```

## Audio Datasets
[RAVDESS](https://zenodo.org/record/1188976#.YRJD6IhKiiM)  
[EMO_DB](http://emodb.bilderbar.info/start.html)  
[LDC](https://catalog.ldc.upenn.edu/LDC2002S28)

## Example experiment

```python
def training():

    """Raw Audio Data Labelling"""

    # Define parameters to use for labelling
    labellingFilename = "Labelled_EMO_DB_AUDIO"
    feautreOutputFilename = "mfcc_data"
    dataOriginName = "EMO_DB"

    # Check for presence or absence of the specified file (create or load file)
    if io_operations.checkIfFileExists(labellingFilename + ".csv", dataOriginName):
        dataDF = io_operations.loadDataset(labellingFilename, dataOriginName)
    else:
        # Load, Label and if needed transcribe an audio dataset
        dataDF = datasource_analysis.identifyData(dataOriginName, "AUDIO", ".csv")

        # Persist the found files and associated values to disk
        io_operations.saveDataset(dataDF, labellingFilename, dataOriginName)

    """Audio Feature Extraction"""

    # Define the list of features, and the required arguments (Originates from Librosa)
    featureSet = ["mfcc"]
    argDict = {'mfcc': {'n_mfcc': 12, 'sr': 48000}}

    # Check for presence or absence of the specified file (create or load file)
    if io_operations.checkIfFileExists(feautreOutputFilename + ".pickle", dataOriginName):
        dataDF = io_operations.loadPickle(feautreOutputFilename, dataOriginName)
    else:
        # Run the feature extraction loop function
        dataDF = feature_extraction_audio.extractFeatures(dataDF, featureSet, argDict, True, 48000, 4)

        # Persist the features to disk, in a loadable pickle form, and viewable csv
        io_operations.savePickle(dataDF, feautreOutputFilename, dataOriginName)
        io_operations.saveDataset(dataDF, feautreOutputFilename, dataOriginName)

    """Audio Model Creation"""

    # Extract the audio features from the dataframe and convert to the required shape
    featureDataFrame = dataDF['mfcc'].values.tolist()
    featureDataFrame = np.asarray(featureDataFrame)

    # Run our model code
    best_model = model_creation_audio.run_model_audio(featureDataFrame, dataDF, "emotion", 5, dataOriginName, 128, 100)  # 모델 생성 코드

    # Calculate model accuracy using the best model
    xAll = np.expand_dims(featureDataFrame, axis=2)
    all_preds = best_model.predict(xAll) # 모델에서는 all predict로 7개의 확률값이 나오고 다 합치면 1이된다.
    # print(all_preds)
    decoded_preds = np.argmax(all_preds, axis=1)  # 7개의 확률값 중에서 가장 큰 값이 있는 위치값(인덱스값)
    preds = np.max(all_preds, axis=1) 
    
    # Calculate model accuracy (using accuracy_score or any other appropriate metric)
    original_labels = dataDF['emotion'].map({'anger': 0, 'boredom': 1, 'neutral': 2, 'happiness': 3, 'anxiety/fear': 4, 'disgust': 5, 'sadness': 6}).values
    model_accuracy = np.mean(decoded_preds == original_labels)

    # Add 'Model Accuracy' column to the DataFrame
    dataDF['Model Accuracy'] = model_accuracy
    dataDF['predicts'] = preds

    # 감정에 대한 색상 이미지를 보여줍니다
    show_emotion_images(dataDF)

    # 감정의 강도를 추정하고 매핑합니다
    map_emotion_intensity(dataDF, 'mapped_emotion_intensity.txt')

    return best_model, dataDF  # 수정된 데이터프레임 반환

# experiment_1() 함수를 실행하고 그 결과를 받아옵니다
best_model, dataDF = training()
```
