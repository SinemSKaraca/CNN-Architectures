'''
TRAFFIC SIGN DATASET
* train - 4.170 images, 58 classes
* test - 1.994 images
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

base_dir = r"C:\Users\user\Desktop\SELF_LEARNING\CNN Architectures\DenseNet\traffic_sign_dataset"

datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.2)  # %80 train %20 validation

train_generator = datagen.flow_from_directory(os.path.join(base_dir, "traffic_Data/DATA"),
                                              target_size=(IMG_SIZE, IMG_SIZE),
                                              batch_size=BATCH_SIZE,
                                              class_mode="categorical",
                                              subset="training",
                                              shuffle=True)

validation_generator = datagen.flow_from_directory(os.path.join(base_dir, "traffic_Data/DATA"),
                                                   target_size=(IMG_SIZE, IMG_SIZE),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode="categorical",
                                                   subset="validation",
                                                   shuffle=False)

num_classes = train_generator.num_classes


from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import models, layers, optimizers

# Pre-trained (Base) model
base_model = DenseNet121(weights="imagenet",
                         include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Başlangıçta base modeli donduruyoruz
base_model.trainable = False

# Yeni classification head ekleyelim
model = models.Sequential([base_model,
                          layers.GlobalAveragePooling2D(),
                          layers.Dense(128, activation="relu"),
                          layers.Dropout(0.5),
                          layers.Dense(num_classes, activation="softmax")])

model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(learning_rate=1e-4),
              metrics=["accuracy"])

hist = model.fit(train_generator,
                 epochs=EPOCHS,
                 validation_data=validation_generator)

# Model yapısını görselleştirme
from tensorflow.keras.utils import plot_model

plot_model(model,
           to_file="densenet_model.png",  # PNG olarak kaydeder
           show_shapes=True,  # Katman çıktı boyutlarını gösterir
           show_layer_names=True)  # Katman isimlerini gösterir

# Accurary grafiği
plt.figure(figsize=(12, 5))

plt.plot(hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

##################################
# TEST SETİNİ DEĞERLENDİRME
##################################

'''
- Test setini değerlendirmek için flow_from_directory değil, flow_from_dataframe kullanacağız.

🔧 Plan:
** labels.csv dosyasını Pandas ile oku
** filepaths ve labels sütunlarını hazırla
** ImageDataGenerator + flow_from_dataframe ile test setini yükle
** model.evaluate() ile başarı oranını ölç
'''

import pandas as pd

labels_df = pd.read_csv("traffic_sign_dataset/labels.csv")