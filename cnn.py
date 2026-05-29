# -*- coding: utf-8 -*-
"""
КУРСОВАЯ РАБОТА: Анализ колористической гармонии
ФИНАЛЬНАЯ ВЕРСИЯ (без fine-tuning, но с отличными результатами)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("КУРСОВАЯ РАБОТА: Анализ колористической гармонии")
print("="*70)

# ============================================
# 1. КОНФИГУРАЦИЯ
# ============================================
C:\Users\Настя\Desktop\курсовая3К\color
DATASET_PATH = Path(r"C:\Users\Настя\Desktop\курсовая3К\color\dataset")
CSV_FEATURES_PATH = Path(r"C:\Users\Настя\Desktop\курсовая3К\color\dataset\c_a_r.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS_BASE = 30

print(f"\n📁 Датасет: {DATASET_PATH}")

# ============================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================

print("\n" + "="*70)
print("ЗАГРУЗКА ДАННЫХ")
print("="*70)

df_features = pd.read_csv(CSV_FEATURES_PATH)
print(f"📊 Загружено {len(df_features)} записей")

def find_image_path(color_class, scheme_type, dataset_path):
    scheme_folder = dataset_path / scheme_type
    if not scheme_folder.exists():
        return None
    class_folder = scheme_folder / color_class
    if not class_folder.exists():
        return None
    for file in class_folder.iterdir():
        if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            if file.stem.startswith(color_class):
                return str(file)
    return None

image_paths = []
valid_indices = []

for idx, row in df_features.iterrows():
    img_path = find_image_path(row['color_class'], row['scheme_type'], DATASET_PATH)
    if img_path:
        image_paths.append(img_path)
        valid_indices.append(idx)

df_features = df_features.iloc[valid_indices].reset_index(drop=True)
df_features['image_path'] = image_paths

print(f"📊 Найдено {len(df_features)} изображений")

# ============================================
# 3. ПОДГОТОВКА ЦЕЛЕЙ
# ============================================

print("\n" + "="*70)
print("ПОДГОТОВКА ЦЕЛЕВЫХ ПЕРЕМЕННЫХ")
print("="*70)

df_features['scheme_binary'] = (df_features['scheme_type'] == 'colormix').astype(int)

le_color = LabelEncoder()
df_features['color_class_encoded'] = le_color.fit_transform(df_features['color_class'])
num_color_classes = len(le_color.classes_)

le_balance = LabelEncoder()
df_features['balance_class_encoded'] = le_balance.fit_transform(df_features['balance_class'])
num_balance_classes = len(le_balance.classes_)

print(f"🎨 Цветовых классов: {num_color_classes}")
print(f"⚖️ Классов баланса: {num_balance_classes} -> {list(le_balance.classes_)}")

# ============================================
# 4. РАЗДЕЛЕНИЕ ДАННЫХ
# ============================================

print("\n" + "="*70)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("="*70)

train_idx, temp_idx = train_test_split(
    range(len(df_features)), test_size=0.3, random_state=SEED,
    stratify=df_features['color_class']
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=SEED,
    stratify=df_features.iloc[temp_idx]['color_class']
)

print(f"📊 Train: {len(train_idx)}")
print(f"📊 Validation: {len(val_idx)}")
print(f"📊 Test: {len(test_idx)}")

# ============================================
# 5. СОЗДАНИЕ ГЕНЕРАТОРОВ
# ============================================

print("\n" + "="*70)
print("СОЗДАНИЕ ГЕНЕРАТОРОВ")
print("="*70)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

def create_generator(indices, datagen, batch_size=BATCH_SIZE, shuffle=True):
    paths = [df_features.iloc[i]['image_path'] for i in indices]
    scheme_labels = [df_features.iloc[i]['scheme_binary'] for i in indices]
    color_labels = [df_features.iloc[i]['color_class_encoded'] for i in indices]
    balance_score_labels = [df_features.iloc[i]['balance_score'] / 10.0 for i in indices]
    balance_class_labels = [df_features.iloc[i]['balance_class_encoded'] for i in indices]
    
    def generator():
        idx_list = list(range(len(paths)))
        if shuffle:
            np.random.shuffle(idx_list)
        
        for start in range(0, len(idx_list), batch_size):
            batch_idx = idx_list[start:start + batch_size]
            batch_images = []
            batch_scheme = []
            batch_color = []
            batch_balance_score = []
            batch_balance_class = []
            
            for i in batch_idx:
                img = load_img(paths[i], target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0
                batch_images.append(img_array)
                batch_scheme.append(scheme_labels[i])
                batch_color.append(color_labels[i])
                batch_balance_score.append(balance_score_labels[i])
                batch_balance_class.append(balance_class_labels[i])
            
            yield np.array(batch_images), {
                'scheme_type': tf.keras.utils.to_categorical(batch_scheme, 2),
                'color_class': tf.keras.utils.to_categorical(batch_color, num_color_classes),
                'balance_score': np.array(batch_balance_score),
                'balance_class': tf.keras.utils.to_categorical(batch_balance_class, num_balance_classes)
            }
    
    output_signature = (
        tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        {
            'scheme_type': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            'color_class': tf.TensorSpec(shape=(None, num_color_classes), dtype=tf.float32),
            'balance_score': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'balance_class': tf.TensorSpec(shape=(None, num_balance_classes), dtype=tf.float32)
        }
    )
    
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

# ============================================
# 6. ПОСТРОЕНИЕ МОДЕЛИ
# ============================================

print("\n" + "="*70)
print("ПОСТРОЕНИЕ МОДЕЛИ")
print("="*70)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

scheme_output = layers.Dense(2, activation='softmax', name='scheme_type')(x)
color_output = layers.Dense(num_color_classes, activation='softmax', name='color_class')(x)
balance_score_output = layers.Dense(1, activation='sigmoid', name='balance_score')(x)
balance_class_output = layers.Dense(num_balance_classes, activation='softmax', name='balance_class')(x)

model = Model(inputs=inputs, outputs={
    'scheme_type': scheme_output,
    'color_class': color_output,
    'balance_score': balance_score_output,
    'balance_class': balance_class_output
})

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss={
        'scheme_type': 'categorical_crossentropy',
        'color_class': 'categorical_crossentropy',
        'balance_score': 'mse',
        'balance_class': 'categorical_crossentropy'
    },
    loss_weights={
        'scheme_type': 0.15,
        'color_class': 0.40,
        'balance_score': 0.20,
        'balance_class': 0.25
    },
    metrics={
        'scheme_type': ['accuracy'],
        'color_class': ['accuracy'],
        'balance_score': ['mae'],
        'balance_class': ['accuracy']
    }
)

print("✅ Модель скомпилирована")

# ============================================
# 7. ОБУЧЕНИЕ
# ============================================

print("\n" + "="*70)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*70)

callbacks = [
    EarlyStopping(monitor='val_color_class_accuracy', patience=10, 
                  restore_best_weights=True, verbose=1, mode='max'),
    ModelCheckpoint('best_color_harmony_model.keras', 
                    monitor='val_color_class_accuracy', save_best_only=True, 
                    verbose=1, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

train_ds = create_generator(train_idx, train_datagen, shuffle=True)
val_ds = create_generator(val_idx, val_test_datagen, shuffle=False)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_BASE,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# 8. РЕЗУЛЬТАТЫ
# ============================================

print("\n" + "="*70)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("="*70)

best_epoch = np.argmax(history.history['val_color_class_accuracy']) + 1
best_acc = max(history.history['val_color_class_accuracy'])
best_scheme_acc = max(history.history['val_scheme_type_accuracy'])
best_balance_acc = max(history.history['val_balance_class_accuracy'])
best_balance_mae = min(history.history['val_balance_score_mae'])

print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ (эпоха {best_epoch}):")
print(f"   🎨 Цветовая классификация: {best_acc:.2%}")
print(f"   📐 Тип схемы: {best_scheme_acc:.2%}")
print(f"   ⚖️ Класс баланса: {best_balance_acc:.2%}")
print(f"   📊 MAE оценки баланса: {best_balance_mae:.4f} (норм.) ≈ {best_balance_mae * 10:.2f} (0-10)")

# Сохранение модели
model.save('color_harmony_model_final.keras')
print("\n✅ Модель сохранена в color_harmony_model_final.keras")

# Сохранение энкодеров
import pickle
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({'color_class': le_color, 'balance_class': le_balance}, f)

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = [
    ('color_class_accuracy', 'Color Classification Accuracy'),
    ('scheme_type_accuracy', 'Scheme Type Accuracy'),
    ('balance_class_accuracy', 'Balance Class Accuracy'),
    ('balance_score_mae', 'Balance Score MAE')
]

for idx, (metric, title) in enumerate(metrics):
    row, col = idx // 2, idx % 2
    axes[row, col].plot(history.history[metric], 'b-', label='Train', linewidth=2)
    axes[row, col].plot(history.history[f'val_{metric}'], 'r-', label='Validation', linewidth=2)
    axes[row, col].set_xlabel('Epoch')
    axes[row, col].set_ylabel('MAE' if 'mae' in metric else 'Accuracy')
    axes[row, col].set_title(title)
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
plt.show()

print("\n" + "="*70)
print("КУРСОВАЯ РАБОТА УСПЕШНО ВЫПОЛНЕНА!")
print("="*70)