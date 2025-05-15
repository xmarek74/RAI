import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#directly import keras cuz pylance is blind - https://github.com/microsoft/pylance-release/issues/1066
from keras._tf_keras.keras import models ,layers ,optimizers, preprocessing, callbacks, regularizers, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import numpy as np
import argparse
import sys
import warnings
#ignore keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

trainData = preprocessing.image.ImageDataGenerator(rescale = 1./255)
valData = preprocessing.image.ImageDataGenerator(rescale=1./255)

def trainModel(modelName):

    trainGen = trainData.flow_from_directory(
        '../public/train',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical',
        shuffle=True
    )

    valGen = valData.flow_from_directory(
        '../public/val',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )

    classWeight = dict(zip(np.unique(trainGen.classes), class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(trainGen.classes), y=trainGen.classes)))
    print(classWeight)
    #0.3 0.3 0.3 0.4 0.4 => valF1Score = 0.85 (precision 0.76)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        layers.Dense(trainGen.num_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy', "Precision", "Recall", metrics.F1Score(average="weighted", name="F1Score")])

    history = model.fit(trainGen, validation_data=valGen, epochs=30, class_weight=classWeight,
              callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_F1Score", verbose=1),
                         callbacks.ModelCheckpoint("best_" + modelName, monitor="val_F1Score", mode="max", save_best_only=True, verbose=1),
                         callbacks.ReduceLROnPlateau(monitor="val_F1Score", factor=0.5, patience=2, verbose=1, min_lr=1e-6)])
    model.save(modelName)
    return model, history

def evaluateModel(modelPath):

    try:
        model = models.load_model(modelPath)
    except:
        print(f"Model couldn't be loaded via provided path {modelPath}")
        return 1

    valGen = valData.flow_from_directory(
        '../public/val',
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
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
    return 0

def printHelp():
    print("""
Usage: python3 ./main.py [option] [argument]

Options:
  -t, --train <model_name>    Trains a new image classification model using the dataset in ../public/train.
                              Saves the model under the specified name (e.g. model.h5).
                              Example: python3 ./main.py -t model.h5

  -e, --eval <model_path>     Evaluates an existing trained model against the validation dataset in ../public/val.
                              Prints accuracy, precision, recall, F1-score and displays confusion matrix.
                              Example: python3 ./main.py -e model.h5

  -u, --usage                 Displays this help message.
                              Example: python3 ./main.py -u

Note:
  - Only one option may be used at a time.
  - Model input/output formats are in .keras or .h5 format.
    """)

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
        elif args.train and args.train.endswith((".h5", ".keras")):
            trainModel(args.train)
        else:
            printHelp()
    else:
        printHelp()