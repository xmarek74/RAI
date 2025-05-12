import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#directly import keras cuz pylance is blind - https://github.com/microsoft/pylance-release/issues/1066
from keras._tf_keras.keras import models ,layers ,optimizers, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import argparse
import sys
import warnings
#ignore keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
trainData = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

def trainModel(modelName):

    trainGen = trainData.flow_from_directory(
        '../public/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    valGen = trainData.flow_from_directory(
        '../public/val',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(trainGen.num_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(trainGen,
            validation_data=valGen,
            epochs=10)
    #TODO save as .keras
    model.save(modelName)
    

def evaluateModel(modelPath):
    valGen = trainData.flow_from_directory(
        '../public/val',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = models.load_model(modelPath)

    predictions = model.predict(valGen)
    predictedClasses = np.argmax(predictions, axis=1)
    trueClasses = valGen.classes
    classLabels = list(valGen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(trueClasses, predictedClasses, target_names=classLabels))

    cm = confusion_matrix(trueClasses, predictedClasses)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classLabels,
                yticklabels=classLabels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
#TODO print help
def printHelp():
    print("Help")

def countArgs(array):
    sum = 0
    for arg in array:
        if (arg):
            sum += 1
    return sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval", help="Evaluates accuracy of model", required=False)
    parser.add_argument("-t", "--train", help="Trains model and saves it as selected name, could take up to 30 minutes", required=False)
    parser.add_argument("-u", "--usage", help="Prints help on how to use the script", required=False, action="store_true")
    args = parser.parse_args()
    if (countArgs([args.eval, args.train, args.usage]) == 1):
        if args.eval:
            evaluateModel(args.eval)
        elif args.train:
            trainModel(args.train)
        else:
            printHelp()
    else:
        printHelp()