# 필요한 라이브러리 가져오기
import numpy as np
import itertools
import matplotlib.pyplot as plt  # matplotlib 추가

# Project Level Imports
import datasource_analysis
import feature_extraction_audio
import io_operations
import model_creation_audio
import model_evaluation_audio
import config

# 감정 레이블을 RGB 값으로 매핑하는 함수 정의
def map_emotion_to_rgb(emotion):
    emotion_to_rgb = {
        "anger": (255, 0, 0),
        "boredom": (255, 255, 255),
        #"curious": (115, 207, 156),
        #"dignified": (206, 156, 198),
        #"elated": (157, 182, 192),
        #"hungry": (101, 146, 72),
        "neutral": (30, 40, 30),
        "happiness": (210, 255, 10),
        "anxiety/fear": (127, 127, 127),
        #"sleepy": (72, 38, 154),
        #"unconcerned": (138, 76, 111),
        #"violent": (177, 207, 64),
        "disgust": (0, 112, 192),
        "sadness": (40, 0, 80),
        # 추가적인 감정과 RGB 값 매핑을 계속 추가할 수 있음
    }
    return emotion_to_rgb.get(emotion, (0, 0, 0))  # 기본값은 검은색

def show_emotion_images(dataDF):
    # 감정 레이블에 해당하는 RGB 값을 활용하여 색상 이미지를 생성하고 표시
    emotions = dataDF['emotion'].unique()
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))

    for idx, emotion in enumerate(emotions):
        ax = axs[idx // 4, idx % 4]
        rgb = map_emotion_to_rgb(emotion)
        ax.imshow([[rgb]], extent=(0, 1, 0, 1))
        ax.set_title(emotion)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 감정의 강도를 추정하는 함수
def estimate_emotion_intensity(predicted_emotion, model_accuracy):
    # 감정 강도 추정 방식을 정의합니다
    estimated_intensity = model_accuracy if model_accuracy >= 0.7 else 0.3
    return estimated_intensity



# 감정 강도에 따른 글자 크기 스케일링 함수
def scale_font_size(intensity):
    
    # 감정 강도 값의 범위를 설정합니다 (0부터 1까지)
    min_intensity = 0.3
    max_intensity = 1

    # 글자 크기의 범위를 설정합니다 (10부터 20까지)
    min_font_size = 10
    max_font_size = 40
    
    scaled_font_size = round(min_font_size + (max_font_size - min_font_size) * ((intensity - min_intensity) / (max_intensity - min_intensity)))
    return scaled_font_size


# 예측 결과와 모델 정확도를 기반으로 감정의 강도를 추정하고, 텍스트 파일에 매핑하는 함수
def map_emotion_intensity(dataDF, output_filename):   
    with open(output_filename, 'w') as file:    
        for index, row in dataDF.iterrows():
            emotion = row['emotion']  # 예측 결과가 저장된 'emotion' 열을 사용
           # accuracy = row['Model Accuracy']
            predicts = row['predicts']
            emotion_rgb = map_emotion_to_rgb(emotion)
            font_size = scale_font_size(predicts)
           # estimated_intensity = estimate_emotion_intensity(emotion, accuracy)
            line = f"Emotion: {emotion}, Estimated Intensity: {predicts:.2f}, RGB: {emotion_rgb}, Size: {font_size}\n"
            file.write(line)
            

# 학습
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
