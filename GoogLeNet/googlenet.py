import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

###########################
# Veri Yükleme ve Hazırlama
###########################

base_dir = r"C:\Users\user\Desktop\SELF_LEARNING\CNN Architectures\GoogLeNet\dataset"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    # Alttaki ikisini kodun en başında da tanımlayabilirdik
    target_size=(224, 224),  # görüntüleri modele verirken hangi boyuta yeniden boyutlandırmak istediğimiz
    batch_size=32,
    class_mode="binary",
    subset="training",
    shuffle=False  # Karışmasın, metrik analiz için gerekebilir
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False  # Karşımasın, metrik analizi için gerekebilir
)

###############################
# 2. Pre-trained modeli yükleme
###############################

from tensorflow.keras.applications import InceptionV3

base_model = InceptionV3(weights="imagenet",
                         include_top=False, # Dense layers'ı alma
                         input_shape=(224, 224, 3))

base_model.summary()

##############################
# 3. Pre-trained modeli dondur
##############################

'''
Eğer sadece transfer learning yapacaksan (yani önce sadece kendi Dense katmanlarını eğitmek istiyorsan), 
mutlaka açık şekilde: base_model.trainable = False demelisin. 
Bunu yazmadığın sürece modelin tüm parametreleri "trainable" olarak kalır.
'''
# Özellik çıkarıcı olarak kullan
base_model.trainable = False  # Başlangıçta tüm katmanları dondur

##############################
# 4. Yeni Classifier head ekle
##############################

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # binary classification
])

############
# 5. Derleme
############

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

########
# 6. Fit
########

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=5)

##########################
# 7. Başarı görselleştirme
##########################

history.history.keys()

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Eğitim ve Doğrulama Doğruluğu")
plt.legend()
plt.show()

##################################
# 8. Test verisi ile değerlendirme
##################################

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

######################################
# 9. Confusion Matrix ve görsel analiz
######################################

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Gerçek etiketler
y_true = test_generator.classes

# Tahmin olasılıkları
y_pred_probs = model.predict(test_generator)

# Tahmin sınıfları
y_pred = (y_pred_probs > 0.5).astype("int").flatten()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(test_generator.class_indices.keys())

# Görselleştirme
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

##############################################
# 10. Yanlış sınıflandırılan görselleri göster
##############################################

# Görsellerin dosya adları
filepaths = test_generator.filepaths

# Hatalı tahmin edilen indeksleri bul
wrong_idx = np.where(y_true != y_pred)[0]

# İlk 5 tanesini görselleştir
plt.figure(figsize=(15, 8))
for i, idx in enumerate(wrong_idx[:5]):
    img_path = filepaths[idx]
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Gerçek: {class_names[y_true[idx]]}\nTahmin: {class_names[y_pred[idx]]}")
plt.suptitle("Yanlış Sınıflandırılan Örnekler")
plt.tight_layout()
plt.show()

#################
# 11. Fine-Tuning
#################

# Base modeli aç
base_model.trainable = True

# Sadece son ~50-100 katmanı eğitime aç (daha öncekileri dondur)
for layer in base_model.layers[:-50]:
    layer.trainable = False  # Son 50 katmanın öncesini eğitime kapat

# Daha küçük bir learning rate ile tekrar derle
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              metrics=["accuracy"])

# Eğitim
history_fine = model.fit(train_generator,
                          validation_data=validation_generator,
                          epochs=5)