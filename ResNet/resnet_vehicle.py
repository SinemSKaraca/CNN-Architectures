import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

#############################
# KLASÖR YOLU VE PARAMETRELER
#############################

'''
train_size = 1400
val_size = 200
test_size = 200
'''

base_dir = r"C:\Users\user\Desktop\SELF_LEARNING\CNN Architectures\ResNet\vehicle_dataset"

IMG_SIZE = 224
BATCH_SIZE = 32

#####################################
# ImageDataGenerator ve Generator’lar
#####################################

# Augmentation da burada yapılıyor
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode="nearest")
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(os.path.join(base_dir, "train"),
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",  # mutli-class classification
                                                    shuffle=True)

validation_generator = val_datagen.flow_from_directory(os.path.join(base_dir, "val"),
                                                       target_size=(IMG_SIZE, IMG_SIZE),
                                                       batch_size=BATCH_SIZE,
                                                       class_mode="categorical",
                                                       shuffle=False)

print("Sınıf indisleri: ")
print(train_generator.class_indices)

'''
** Test klasöründe etiketli klasörler yoksa, flow_from_directory() kullanamayız.
Bu durumda:
Test setini predict() için doğrudan ImageDataGenerator(...).flow_from_dataframe() veya 
flow_from_directory(..., class_mode=None) şeklinde işleyebiliriz.
'''

#################################################
# ResNet50 Modelini Tanımlama (Transfer Learning)
#################################################

# Veri seti çok sınıflı olduğu için softmax aktivasyon ve categorical_crossentropy kullanacağız

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

base_model = ResNet50( weights="imagenet",
                       include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Base model dondurulur (Başlangıçta sadece convolution layerları eğiteceğiz)
base_model.trainable = False

##########################################
# Yeni Head (Classifier) Katmanları Ekleme
##########################################

num_classes = train_generator.num_classes

# Modeli tanımlayalım
model = models.Sequential([base_model,
                           layers.GlobalAveragePooling2D(),
                           layers.Dense(128, activation="relu"),
                           layers.Dropout(0.5),
                           layers.Dense(num_classes, activation="softmax")]) # softmax for multi-class classification

model.summary()  # resnet functional olduğundan sequential base modeller gibi katmanları detaylı bir şekilde yazılmaz

#########
# Derleme
#########

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

########
# Eğitim
########

# Bu eğitim sadece eklediğimiz sınıflayıcı katmanlarını eğitir. ResNet50 SABİTTİR!
hist = model.fit(train_generator,
                 validation_data=validation_generator,
                 epochs=5)

# Eğitim sonrası loss çok yüksek, accuracy çok düşük çıktı. Bu nedenle data augmentation yaptım.

############################################
# Modeli değerlendirme - Fine tuning olmadan
############################################

# Accuracy Grafiği:
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#############
# Fine-Tuning
#############

# Tüm katmanları eğitilebilir hale getir
base_model.trainable = True

# Sadece son 50 katmanı aç, geri kalanları dondur
for layer in base_model.layers[:-50]:
    layer.trainable = False  # Tüm katmanları açmıştık. Son 50 katman haricindekileri kapatıyoruz

from tensorflow.keras.optimizers import Adam

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=1e-5),
              metrics=["accuracy"])

# Overfitting'e karşı EarlyStopping ekleme
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(patience=3, restore_best_weights=True)

hist_finetune = model.fit(train_generator,
                          validation_data=validation_generator,
                          epochs=10,
                          callbacks=[early_stop])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_finetune.history["accuracy"], label="Train Accuracy")
plt.plot(hist_finetune.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(hist_finetune.history["loss"], label="Train Loss")
plt.plot(hist_finetune.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

##########################
# Test Verilerini İnceleme
##########################

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(base_dir,
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  batch_size=BATCH_SIZE,
                                                  classes=["test"],  # Test klasörünü belirt
                                                  class_mode=None,  # Etiket yok
                                                  shuffle=False)

# Tahmin al
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
class_labels = list(train_generator.class_indices.keys())

# Rastgele 5 görsel seç
random_indices = random.sample(range(len(predicted_classes)), 5)

plt.figure(figsize=(15, 3))

for i, idx in enumerate(random_indices):
    img_path = test_generator.filepaths[idx]

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0

    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")
    pred_class_name = class_labels[predicted_classes[idx]]
    plt.title(f"Model Tahmini: {pred_class_name}")

plt.tight_layout()
plt.show()

model.save("resnet_vehicle_model.h5")