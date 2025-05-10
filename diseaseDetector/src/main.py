import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#directly import keras cuz pylance is blind - https://github.com/microsoft/pylance-release/issues/1066
from keras._tf_keras.keras import models ,layers ,optimizers, preprocessing
import warnings
#ignore keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
trainData = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

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
model.save('diseaseModel.h5')