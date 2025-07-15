import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image # Using Pillow for image loading
# import cv2 # OpenCV不再是必须的，除非有其他处理需要
import random
import time

# --- Configuration ---
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS_IN = 3
IMG_CHANNELS_OUT = 3
# !!! 定义要测试的 Batch Sizes 列表 !!!
BATCH_SIZES_TO_TEST = [8, 16, 32, 64] # 例如
EPOCHS = 50 # 为了快速实验，可以减少 Epochs
LEARNING_RATE = 1e-4

# --- Paths ---
# !! IMPORTANT: Update these paths to match your system !!
TRAIN_PATH_DIR = r'D:\train_PATH'
TRAIN_TF_DIR = r'D:\train_TF'
TEST_PATH_DIR = r'D:\test_PATH'
TEST_TF_DIR = r'D:\test_TF'
#MODEL_SAVE_PATH = r'D:\laser_temp_predictor_unet_rgb32.keras'
RESULTS_VIS_DIR = r'D:\prediction_results_rgb_batch_size'
INITIAL_WEIGHTS_PATH = r'D:\initial_unet_weights.weights.h5' # 存储初始权重的文件

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_VIS_DIR):
    os.makedirs(RESULTS_VIS_DIR)

# --- Normalization Parameters ---
# !!! 注意：以下参数和函数对于 RGB 目标不再适用，已被修改或移除 !!!
# TEMP_MIN_K = 298.0
# TEMP_MAX_K = 5500.0

# def normalize_temp(img_array): ... # 不再需要，由简单的 / 255.0 替代
# def denormalize_temp(normalized_array): ... # 不再需要，无法从 RGB 简单恢复温度

def load_and_preprocess_image(path, is_temp_field=False):
    """
    Loads and preprocesses a single image.
    Handles grayscale input (path) and RGB target (temperature field).
    """
    try:
        if is_temp_field: # 处理温度场图像 (彩色温度图 PNG)
            # Load image as RGB
            img = Image.open(path).convert('RGB') # <--- 确保加载为 RGB
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST) # 确保尺寸正确
            img_array = np.array(img, dtype=np.float32)

            # 将温度场RGB值从[0, 255]归一化到[0, 1]
            img_array = img_array / 255.0 # <--- 简单的像素值归一化


        else: # 处理输入图像 (彩色路径图 PNG)
            img = Image.open(path).convert('RGB') # <--- 加载为RGB
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
            img_array = np.array(img, dtype=np.float32)
            # 将路径图RGB值从[0, 255]归一化到[0, 1]
            img_array = img_array / 255.0


        return img_array

    except Exception as e:
        print(f"Error loading image {path}: {e}")
        # 返回匹配通道数的零矩阵
        if is_temp_field:
            # 目标是 3 通道
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_OUT), dtype=np.float32)
        else:
            # 输入通道数根据 IMG_CHANNELS_IN 决定
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN), dtype=np.float32)


# --- Data Generator ---
class DataGenerator(keras.utils.Sequence):

    def __init__(self, path_dir, tf_dir, batch_size, dim, n_channels_in, n_channels_out, shuffle=True):
        self.path_dir = path_dir
        self.tf_dir = tf_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle

        # 找到所有 png 文件并排序
        all_path_files = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.lower().endswith('.png')])
        all_tf_files = sorted([os.path.join(tf_dir, f) for f in os.listdir(tf_dir) if f.lower().endswith('.png')])

        # 提取 ID (假设是文件名主体或末尾数字)
        # 此处简化为直接用文件名主体（无后缀）作为ID，适用于 '0001.png' 或 'prefix_0001.png' 等
        path_ids = set(os.path.splitext(os.path.basename(f))[0] for f in all_path_files)
        tf_ids = set(os.path.splitext(os.path.basename(f))[0] for f in all_tf_files)

        if not path_ids or not tf_ids:
             raise ValueError(f"One of the directories is empty or contains no PNG files: {path_dir}, {tf_dir}")

        if path_ids != tf_ids:
             print(f"Warning: Mismatch or difference in file IDs between {path_dir} and {tf_dir}")
             common_ids = sorted(list(path_ids.intersection(tf_ids)))
             if not common_ids:
                 raise ValueError("No matching file IDs found between path and TF directories!")

             # 根据共同 ID 筛选文件列表
             # 注意：这里假设 ID 和文件名主体相同
             self.path_files = sorted([f for f in all_path_files if os.path.splitext(os.path.basename(f))[0] in common_ids])
             self.tf_files = sorted([f for f in all_tf_files if os.path.splitext(os.path.basename(f))[0] in common_ids])
             print(f"Using {len(self.path_files)} matched files based on common IDs.")
        else:
             # 如果 ID 集合完全相同，可以直接使用
             self.path_files = all_path_files
             self.tf_files = all_tf_files
             print(f"Found {len(self.path_files)} consistent files in both directories.")


        # 再次检查最终文件数是否一致 (理论上筛选后应该一致)
        if len(self.path_files) != len(self.tf_files):
            # 如果筛选后仍然不一致，说明 ID 提取或文件名有问题
            raise ValueError(f"Final number of path images ({len(self.path_files)}) and temperature images ({len(self.tf_files)}) do not match after filtering!")

        if len(self.path_files) == 0:
            raise ValueError("No usable image pairs found!")

        self.indexes = np.arange(len(self.path_files))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # Ensure at least one batch even if few samples
        if len(self.path_files) == 0: return 0
        return int(np.floor(len(self.path_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of files for the batch
        list_path_files = [self.path_files[k] for k in indexes]
        list_tf_files = [self.tf_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_path_files, list_tf_files)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.path_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_path_files, list_tf_files):
        'Generates data containing batch_size samples'
        # Initialization using correct channel numbers
        X = np.empty((self.batch_size, *self.dim, self.n_channels_in), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels_out), dtype=np.float32)

        # Generate data
        for i, (path_file, tf_file) in enumerate(zip(list_path_files, list_tf_files)):
            # Store sample (Input Path Image)
            X[i,] = load_and_preprocess_image(path_file, is_temp_field=False)
            # Store class (Target Temperature Field RGB Image)
            y[i,] = load_and_preprocess_image(tf_file, is_temp_field=True)

        return X, y

# --- Build U-Net Model ---
# 模型结构本身不需要改变，只需要确保输出层通道数为 3
def build_unet_model(input_shape, num_classes=3): # Default num_classes changed to 3
    inputs = keras.Input(shape=input_shape)

    # --- Encoder ---
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # --- Decoder ---
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # Output layer: Use 'sigmoid' for [0,1] RGB output. num_classes must be 3.
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9) # num_classes = 3

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Main Execution ---

# 1. Build the initial model structure ONCE
print("Building initial model structure...")
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN)
model_structure = build_unet_model(input_shape, num_classes=IMG_CHANNELS_OUT)
print("Saving initial model weights...")
model_structure.save_weights(INITIAL_WEIGHTS_PATH) # 保存初始随机权重
model_structure.summary() # 打印一次结构信息
del model_structure # 删除这个实例，后面循环里会创建新的

# --- Dictionary to store histories ---
all_histories = {}

# --- Loop through each batch size ---
print(f"Starting training loops for batch sizes: {BATCH_SIZES_TO_TEST}")
total_start_time = time.time()

for current_batch_size in BATCH_SIZES_TO_TEST:
    print(f"\n--- Training with Batch Size: {current_batch_size} ---")
    batch_start_time = time.time()

    # 2. Create Data Generators for the CURRENT batch size
    print(f"Creating data generators with batch size {current_batch_size}...")
    try:
        # 使用当前的 current_batch_size
        train_generator = DataGenerator(
            path_dir=TRAIN_PATH_DIR,
            tf_dir=TRAIN_TF_DIR,
            batch_size=current_batch_size, # Use current batch size
            dim=(IMG_HEIGHT, IMG_WIDTH),
            n_channels_in=IMG_CHANNELS_IN,
            n_channels_out=IMG_CHANNELS_OUT,
            shuffle=True
        )

        validation_generator = DataGenerator(
            path_dir=TEST_PATH_DIR,
            tf_dir=TEST_TF_DIR,
            batch_size=current_batch_size, # Use current batch size
            dim=(IMG_HEIGHT, IMG_WIDTH),
            n_channels_in=IMG_CHANNELS_IN,
            n_channels_out=IMG_CHANNELS_OUT,
            shuffle=False # Validation data usually doesn't need shuffling
        )

        print(f"Train generator: {len(train_generator.path_files)} samples, {len(train_generator)} steps.")
        print(f"Validation generator: {len(validation_generator.path_files)} samples, {len(validation_generator)} steps.")

        if len(train_generator) == 0:
            print(f"Warning: Training generator is empty for batch size {current_batch_size}. Skipping.")
            continue # 跳过当前 batch size
        val_data_to_pass = validation_generator if len(validation_generator) > 0 else None

    except ValueError as e:
        print(f"Error creating data generators for batch size {current_batch_size}: {e}")
        continue # 跳过当前 batch size

    # 3. Build and Compile a FRESH model instance and load INITIAL weights
    print("Building model instance...")
    # 清理之前的计算图和会话（可选，但有时有助于避免资源泄漏）
    tf.keras.backend.clear_session()

    model = build_unet_model(input_shape, num_classes=IMG_CHANNELS_OUT)

    print(f"Loading initial weights from {INITIAL_WEIGHTS_PATH}...")
    try:
      model.load_weights(INITIAL_WEIGHTS_PATH) # 加载保存的初始权重
    except Exception as e:
      print(f"Warning: Could not load initial weights ({e}). Training will start from new random weights.")


    print("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mae',
                  metrics=['mse'])

    # 4. Define Callbacks (Optional: adjust paths/patience if needed)
    # 可以为每个 batch size 保存不同的 checkpoint 文件
    current_model_save_path = os.path.join(RESULTS_VIS_DIR, f'unet_rgb_bs{current_batch_size}.keras')
    callbacks = [
        keras.callbacks.ModelCheckpoint(current_model_save_path, verbose=0, # Less verbose during loop
                                        save_best_only=True, monitor='val_loss', save_weights_only=False), # Save full model maybe
        keras.callbacks.EarlyStopping(patience=15, verbose=1, monitor='val_loss', restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1, min_lr=1e-7)
        # 可以考虑添加 TensorBoard 回调为每个 batch size 记录日志
        # keras.callbacks.TensorBoard(log_dir=os.path.join(RESULTS_VIS_DIR, f'logs/bs_{current_batch_size}'))
    ]

    # 5. Train the Model
    print(f"Starting training for batch size {current_batch_size}...")
    history = model.fit(
        train_generator,
        validation_data=val_data_to_pass,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1 # 可以设为 1 或 2 观察进度
    )

    # 6. Store the history object
    all_histories[current_batch_size] = history.history # 只存储 history dict

    batch_end_time = time.time()
    print(f"Training for Batch Size {current_batch_size} finished in {batch_end_time - batch_start_time:.2f} seconds.")

    # 显式删除模型和生成器，尝试释放内存（不一定总能成功，但值得尝试）
    del model
    del train_generator
    del validation_generator
    del callbacks
    del history

total_end_time = time.time()
print(f"\nAll training loops finished in {total_end_time - total_start_time:.2f} seconds.")


# --- 7. Plotting the Comparison ---
print("\nGenerating comparison plot...")
plt.figure(figsize=(12, 8))

# 选择要绘制的指标，例如 'val_loss' 或 'loss'
metric_to_plot = 'val_loss' # 或者 'loss', 'val_mse', 'mse'
metric_label = 'Validation MAE Loss' if metric_to_plot == 'val_loss' else 'Training MAE Loss' # 相应调整标签

for bs, history_data in all_histories.items():
    if metric_to_plot in history_data:
        epochs_ran = range(1, len(history_data[metric_to_plot]) + 1)
        plt.plot(epochs_ran, history_data[metric_to_plot], label=f'Batch Size {bs}')
    else:
        print(f"Warning: Metric '{metric_to_plot}' not found in history for batch size {bs}.")

plt.title(f'{metric_label} Comparison for Different Batch Sizes')
plt.xlabel('Epoch')
plt.ylabel(metric_label)
plt.legend()
plt.grid(True)
# 保存图像
plot_save_path = os.path.join(RESULTS_VIS_DIR, f'{metric_to_plot}_batch_size_comparison.png')
plt.savefig(plot_save_path)
print(f"Comparison plot saved to: {plot_save_path}")
plt.show() # 显示图像

print("--- Batch Size Comparison Script Finished ---")