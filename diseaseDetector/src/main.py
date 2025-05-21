import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#directly import keras cuz pylance is blind - https://github.com/microsoft/pylance-release/issues/1066
from keras._tf_keras.keras import models ,layers ,optimizers, preprocessing, callbacks, regularizers, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from datetime import datetime
import json
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
        "../public/train",
        target_size=(64, 64),
        batch_size=64,
        class_mode="categorical",
        shuffle=True
    )

    valGen = valData.flow_from_directory(
        "../public/val",
        target_size=(64, 64),
        batch_size=64,
        class_mode="categorical",
        shuffle=False
    )

    #since classes are not balanced, use class_weight => model keeps more attention to class with less images
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
        layers.Activation("relu"),
        layers.Dropout(0.4),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.4),
        layers.Dense(trainGen.num_classes, activation="softmax")
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["Accuracy", "Precision", "Recall", metrics.F1Score(average="weighted", name="F1Score")])

    #save to history to display model's evaluation
    history = model.fit(trainGen, validation_data=valGen, epochs=20, class_weight=classWeight,
              callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_F1Score", verbose=1),
                         callbacks.ModelCheckpoint("best_" + modelName, monitor="val_F1Score", mode="max", save_best_only=True, verbose=1),
                         callbacks.ReduceLROnPlateau(monitor="val_F1Score", factor=0.5, patience=3, verbose=1, min_lr=1e-6)])
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"history_{time}.json"
    with open(name, "w") as f:
        json.dump(history.history, f) 
    model.save(modelName)
    return model, history

def evaluateModel(modelPath):

    try:
        model = models.load_model(modelPath)
    except:
        print(f"Model couldn't be loaded via provided path {modelPath}")
        return 1

    valGen = valData.flow_from_directory(
        "../public/val",
        target_size=(64, 64),
        batch_size=64,
        class_mode="categorical",
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

def printHistory(path):
    with open(path, "r") as f:
        history = json.load(f)
    keys = list(history.keys())
    trainMetrics = []
    valMetrics = []
    for k in keys:
        if (not k.startswith("learning_rate")) and (not k.startswith("val_")):
            trainMetrics.append(k)
        elif k.startswith("val_"):
            valMetrics.append(k)

    #display train vs val metric as a comparison
    for metric in trainMetrics:
        plt.figure(figsize=(8, 5))
        plt.plot(history[metric], label= metric, marker="o")

        for val in valMetrics:
            if metric in val:
                plt.plot(history[val], label=f"val_{metric}", marker="o")                


        plt.title(f"{metric.capitalize()} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if ("learning_rate" in history):
        plt.figure(figsize=(8, 4))
        plt.plot(history["learning_rate"], label="Learning Rate", color="purple", marker="o")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def printHelp():
    print("""
Usage: python3 ./main.py [option] [argument]

Options:
  -t, --train <model_name>     Trains a new image classification model using the dataset in ../public/train.
                               Saves the model under the specified name (e.g. model.h5).
                               Example: python3 ./main.py -t model.h5

  -e, --eval <model_path>      Evaluates an existing trained model against the validation dataset in ../public/val.
                               Prints accuracy, precision, recall, F1-score and displays confusion matrix.
                               Example: python3 ./main.py -e model.h5

  -h, --history <history_path> Displays training history from a JSON file generated during training.
                               Shows plots comparing training and validation metrics over epochs.
                               Also displays the learning rate schedule if available.
                               Example: python3 ./main.py -h history.json

  -u, --usage                  Displays this help message.
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
    parser.add_argument("-his", "--history", help="Prints history graphs based on model's learning", required=False)
    args = parser.parse_args()
    if (countArgs([args.eval, args.train, args.usage, args.history]) == 1):
        if args.eval:
            if (not evaluateModel(args.eval)):
                sys.exit(0)
            else:
                sys.exit(1)
        elif args.train and args.train.endswith((".h5", ".keras")):
            trainModel(args.train)
            sys.exit(0)
        elif args.history and args.history.endswith(".json"):
            printHistory(args.history)
            sys.exit(1)
        else:
            printHelp()
            sys.exit(0)
    else:
        printHelp()