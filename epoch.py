import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import time
import csv

# --- 自定义回调：用于周期性记录累计时间 ---
class CumulativeTimeLogger(keras.callbacks.Callback):
    def __init__(self, log_interval=50, log_file_path=None):
        super(CumulativeTimeLogger, self).__init__()
        self.log_interval = log_interval  # 每隔多少个 epoch 记录一次
        self.training_start_time = 0
        self.log_file_path = log_file_path
        self.initial_log = True # 标志位，用于写入 CSV 文件头

    def on_train_begin(self, logs=None):
        # 训练开始时记录整体开始时间
        self.training_start_time = time.time()
        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time))}")

        if self.log_file_path and self.initial_log:
            try:
                # 使用 'w' 模式，如果文件已存在则覆盖，确保每次运行脚本生成新的时间记录文件
                with open(self.log_file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Epoch', 'CumulativeTimeSeconds'])
                self.initial_log = False # 表头已写，不再重复写
                print(f"Cumulative time log header written to {self.log_file_path}")
            except IOError as e:
                print(f"Warning: Could not write header to cumulative time log {self.log_file_path}: {e}")


    def on_epoch_end(self, epoch, logs=None):
        # Keras 的 epoch 是从 0 开始计数的，所以 (epoch + 1) 是实际的周期数
        current_epoch_number = epoch + 1
        if current_epoch_number % self.log_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            print(f"Epoch {current_epoch_number}: Cumulative training time = {elapsed_time:.2f} seconds")

            if self.log_file_path:
                try:
                    # 确保表头已写 (考虑多worker或意外情况)
                    if not os.path.exists(self.log_file_path) or os.path.getsize(self.log_file_path) == 0:
                         with open(self.log_file_path, 'w', newline='') as f_reinit: # 'w' to ensure clean header
                            writer = csv.writer(f_reinit)
                            writer.writerow(['Epoch', 'CumulativeTimeSeconds'])
                         self.initial_log = False # Re-set in case it was somehow true

                    with open(self.log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([current_epoch_number, f"{elapsed_time:.4f}"])
                except IOError as e:
                    print(f"Warning: Could not append cumulative time to {self.log_file_path}: {e}")

    def on_train_end(self, logs=None):
        final_time = time.time()
        total_training_time = final_time - self.training_start_time
        print(f"Total training finished. Total elapsed time: {total_training_time:.2f} seconds.")
        # 记录最终的总时间，确保即使总 epoch 不是 log_interval 的倍数也有最终记录
        if self.log_file_path:
            # 检查是否需要在训练结束时记录最后一次数据
            # self.model.params['epochs'] 是总的epochs数
            # (epoch + 1) 在这里是最后一个epoch number
            last_recorded_epoch = 0
            if os.path.exists(self.log_file_path):
                try:
                    with open(self.log_file_path, 'r') as f_read:
                        reader = csv.reader(f_read)
                        next(reader, None) # Skip header
                        for row in reader:
                            if row: # Check if row is not empty
                                last_recorded_epoch = int(row[0])
                except Exception: # Broad exception to catch parsing errors or empty file
                    pass # Keep last_recorded_epoch as 0

            # 如果最后一个epoch没有被正常记录（例如，总epoch数不是间隔的倍数）
            if self.model and hasattr(self.model, 'params') and 'epochs' in self.model.params:
                total_epochs_planned = self.model.params['epochs']
                if total_epochs_planned > last_recorded_epoch : # Ensure we only add if not already covered by interval
                    try:
                        with open(self.log_file_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([total_epochs_planned, f"{total_training_time:.4f}"])
                            print(f"Final cumulative time for epoch {total_epochs_planned} written to {self.log_file_path}")
                    except IOError as e:
                         print(f"Warning: Could not append final cumulative time to {self.log_file_path}: {e}")


# --- Configuration ---
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS_IN = 3
IMG_CHANNELS_OUT = 3
TARGET_EPOCHS = 200
FIXED_BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOG_INTERVAL_EPOCHS = 25 # 每隔多少个周期记录一次时间

# --- Paths ---
# !! IMPORTANT: Update these paths to match your system !!
TRAIN_PATH_DIR = r'D:\train_PATH'
TRAIN_TF_DIR = r'D:\train_TF'
TEST_PATH_DIR = r'D:\test_PATH'
TEST_TF_DIR = r'D:\test_TF'
RESULTS_VIS_DIR = r'D:\prediction_results_rgb_epoch'
INITIAL_WEIGHTS_PATH = r'D:\initial_unet_weights.weights.h5'
CUMULATIVE_TIMES_FILE = os.path.join(RESULTS_VIS_DIR, f'cumulative_training_times_epochs{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.csv')
VAL_LOSS_PLOT_SINGLE_RUN_FILE = os.path.join(RESULTS_VIS_DIR, f'val_loss_epochs{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
MODEL_SAVE_PATH_SINGLE_RUN = os.path.join(RESULTS_VIS_DIR, f'unet_rgb_epochs{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.keras')

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_VIS_DIR):
    os.makedirs(RESULTS_VIS_DIR)

# --- Helper Functions ---
def load_and_preprocess_image(path, is_temp_field=False):
    global IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS_IN, IMG_CHANNELS_OUT # Ensure access to global config
    try:
        if is_temp_field:
            img = Image.open(path).convert('RGB')
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0
        else:
            img = Image.open(path).convert('RGB') # Assuming input is also RGB as per IMG_CHANNELS_IN = 3
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        if is_temp_field:
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_OUT), dtype=np.float32)
        else:
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN), dtype=np.float32)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, path_dir, tf_dir, batch_size, dim, n_channels_in, n_channels_out, shuffle=True):
        self.path_dir = path_dir
        self.tf_dir = tf_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle

        all_path_files = sorted([os.path.join(path_dir, f) for f in os.listdir(path_dir) if f.lower().endswith('.png')])
        all_tf_files = sorted([os.path.join(tf_dir, f) for f in os.listdir(tf_dir) if f.lower().endswith('.png')])

        path_ids = set(os.path.splitext(os.path.basename(f))[0] for f in all_path_files)
        tf_ids = set(os.path.splitext(os.path.basename(f))[0] for f in all_tf_files)

        if not path_ids or not tf_ids:
             raise ValueError(f"One of the directories is empty or contains no PNG files: {path_dir}, {tf_dir}")

        if path_ids != tf_ids:
             print(f"Warning: Mismatch or difference in file IDs between {path_dir} and {tf_dir}")
             common_ids = sorted(list(path_ids.intersection(tf_ids)))
             if not common_ids:
                 raise ValueError("No matching file IDs found between path and TF directories!")
             self.path_files = sorted([f for f in all_path_files if os.path.splitext(os.path.basename(f))[0] in common_ids])
             self.tf_files = sorted([f for f in all_tf_files if os.path.splitext(os.path.basename(f))[0] in common_ids])
             print(f"Using {len(self.path_files)} matched files based on common IDs.")
        else:
             self.path_files = all_path_files
             self.tf_files = all_tf_files
             print(f"Found {len(self.path_files)} consistent files in both directories.")

        if len(self.path_files) != len(self.tf_files):
            raise ValueError(f"Final number of path images ({len(self.path_files)}) and temperature images ({len(self.tf_files)}) do not match after filtering!")
        if len(self.path_files) == 0:
            raise ValueError("No usable image pairs found!")

        self.indexes = np.arange(len(self.path_files))
        self.on_epoch_end()

    def __len__(self):
        if len(self.path_files) == 0: return 0
        return int(np.floor(len(self.path_files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_path_files = [self.path_files[k] for k in indexes]
        list_tf_files = [self.tf_files[k] for k in indexes]
        X, y = self.__data_generation(list_path_files, list_tf_files)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.path_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_path_files, list_tf_files):
        X = np.empty((self.batch_size, *self.dim, self.n_channels_in), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, self.n_channels_out), dtype=np.float32)
        for i, (path_file, tf_file) in enumerate(zip(list_path_files, list_tf_files)):
            X[i,] = load_and_preprocess_image(path_file, is_temp_field=False)
            y[i,] = load_and_preprocess_image(tf_file, is_temp_field=True)
        return X, y

def build_unet_model(input_shape, num_classes=3):
    inputs = keras.Input(shape=input_shape)
    # Encoder
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
    # Decoder
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
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# --- Main Execution ---

# 1. Check/Create initial weights
if not os.path.exists(INITIAL_WEIGHTS_PATH):
    print(f"Initial weights file not found at {INITIAL_WEIGHTS_PATH}. Creating it...")
    input_shape_init = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN)
    initial_model = build_unet_model(input_shape_init, num_classes=IMG_CHANNELS_OUT)
    initial_model.save_weights(INITIAL_WEIGHTS_PATH)
    print(f"Initial weights saved to {INITIAL_WEIGHTS_PATH}")
    # initial_model.summary() # Optional: print summary
    del initial_model
else:
    print(f"Using existing initial weights from {INITIAL_WEIGHTS_PATH}")

# --- Single Training Run ---
print(f"\n--- Starting a single training run for {TARGET_EPOCHS} Epochs ---")
print(f"Using Batch Size: {FIXED_BATCH_SIZE}")
print(f"Cumulative time will be logged every {LOG_INTERVAL_EPOCHS} epochs to {CUMULATIVE_TIMES_FILE}")

# 2. Create Data Generators
print(f"Creating data generators with batch size {FIXED_BATCH_SIZE}...")
try:
    train_generator = DataGenerator(
        path_dir=TRAIN_PATH_DIR,
        tf_dir=TRAIN_TF_DIR,
        batch_size=FIXED_BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels_in=IMG_CHANNELS_IN,
        n_channels_out=IMG_CHANNELS_OUT,
        shuffle=True
    )
    validation_generator = DataGenerator(
        path_dir=TEST_PATH_DIR,
        tf_dir=TEST_TF_DIR,
        batch_size=FIXED_BATCH_SIZE,
        dim=(IMG_HEIGHT, IMG_WIDTH),
        n_channels_in=IMG_CHANNELS_IN,
        n_channels_out=IMG_CHANNELS_OUT,
        shuffle=False
    )
    print(f"Train generator: {len(train_generator.path_files)} samples, {len(train_generator)} steps per epoch.")
    print(f"Validation generator: {len(validation_generator.path_files)} samples, {len(validation_generator)} steps per epoch.")
    if len(train_generator) == 0:
        print(f"Error: Training generator is empty. Cannot proceed.")
        exit()
    val_data_to_pass = validation_generator if len(validation_generator) > 0 else None
except ValueError as e:
    print(f"Error creating data generators: {e}")
    exit()

# 3. Build and Compile Model, Load INITIAL weights
print("Building model instance...")
tf.keras.backend.clear_session()
input_shape_run = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN)
model = build_unet_model(input_shape_run, num_classes=IMG_CHANNELS_OUT)

print(f"Loading initial weights from {INITIAL_WEIGHTS_PATH}...")
try:
  model.load_weights(INITIAL_WEIGHTS_PATH)
except Exception as e:
  print(f"Error loading initial weights ({e}). Aborting training.")
  exit()

print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='mae',
              metrics=['mse'])

# 4. Define Callbacks
time_logger_callback = CumulativeTimeLogger(log_interval=LOG_INTERVAL_EPOCHS, log_file_path=CUMULATIVE_TIMES_FILE)
callbacks = [
    keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH_SINGLE_RUN, verbose=1,
                                    save_best_only=True, monitor='val_loss', save_weights_only=False),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1, min_lr=1e-7),
    time_logger_callback
    # keras.callbacks.TensorBoard(log_dir=os.path.join(RESULTS_VIS_DIR, f'logs/single_run_epochs{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}'))
]

# 5. Train the Model
print(f"Starting training for exactly {TARGET_EPOCHS} epochs...")
history = model.fit(
    train_generator,
    validation_data=val_data_to_pass,
    epochs=TARGET_EPOCHS,
    callbacks=callbacks,
    verbose=1 # Set to 1 or 2 for Keras progress, or 0 to only see custom callback output
)

# --- 7. Plotting the Validation Loss for this single run ---
print(f"\nGenerating Validation Loss plot for {TARGET_EPOCHS} epochs run...")
if history and 'val_loss' in history.history:
    plt.figure(figsize=(10, 6))
    actual_epochs_ran_plot = len(history.history['val_loss'])
    epochs_axis = range(1, actual_epochs_ran_plot + 1)

    plt.plot(epochs_axis, history.history['val_loss'], label=f'Validation MAE Loss')
    if 'loss' in history.history:
        plt.plot(epochs_axis, history.history['loss'], label=f'Training MAE Loss')

    plt.title(f'Loss Over Epochs (Target: {TARGET_EPOCHS} Epochs, BS: {FIXED_BATCH_SIZE})')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(VAL_LOSS_PLOT_SINGLE_RUN_FILE)
    print(f"Validation loss plot saved to: {VAL_LOSS_PLOT_SINGLE_RUN_FILE}")
    plt.show()
else:
    print("Warning: 'val_loss' not found in history. Cannot plot.")

# 8. Evaluate the Model on test set (Optional)
if len(validation_generator) > 0:
    print("\nEvaluating model on the test/validation set (using best weights from checkpoint)...")
    # To ensure using the best saved model by ModelCheckpoint:
    if os.path.exists(MODEL_SAVE_PATH_SINGLE_RUN):
        print(f"Loading best model weights from {MODEL_SAVE_PATH_SINGLE_RUN} for evaluation.")
        model.load_weights(MODEL_SAVE_PATH_SINGLE_RUN) # Load best weights saved by ModelCheckpoint
    else:
        print(f"Warning: Model checkpoint file {MODEL_SAVE_PATH_SINGLE_RUN} not found. Evaluating with current model weights.")

    test_loss, test_mse = model.evaluate(validation_generator, verbose=1)
    print(f"Test/Validation MAE Loss (RGB difference): {test_loss:.4f}")
    print(f"Test/Validation MSE (RGB difference): {test_mse:.4f}")

print(f"\n--- Single Training Run for {TARGET_EPOCHS} Epochs Finished ---")
if os.path.exists(CUMULATIVE_TIMES_FILE):
    print(f"Cumulative training times saved to: {CUMULATIVE_TIMES_FILE}")