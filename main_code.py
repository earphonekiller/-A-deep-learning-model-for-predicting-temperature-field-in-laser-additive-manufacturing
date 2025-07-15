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
from sklearn.metrics import r2_score
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist # 用于计算距离
import matplotlib
import traceback # Added for better error reporting in visualization

# 示例：全局设置
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 14 # Slightly smaller legend for R2 plot
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.grid'] = False

# --- 自定义回调：用于周期性记录累计时间 ---
class CumulativeTimeLogger(keras.callbacks.Callback):
    def __init__(self, log_interval=50, log_file_path=None):
        super(CumulativeTimeLogger, self).__init__()
        self.log_interval = log_interval
        self.training_start_time = 0
        self.log_file_path = log_file_path
        self.initial_log = True

    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.training_start_time))}")
        if self.log_file_path and self.initial_log:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                     os.makedirs(log_dir)
                with open(self.log_file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Epoch', 'CumulativeTimeSeconds'])
                self.initial_log = False
                print(f"Cumulative time log header written to {self.log_file_path}")
            except IOError as e:
                print(f"Warning: Could not write header to cumulative time log {self.log_file_path}: {e}")

    def on_epoch_end(self, epoch, logs=None):
        current_epoch_number = epoch + 1
        if current_epoch_number % self.log_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            print(f"Epoch {current_epoch_number}: Cumulative training time = {elapsed_time:.2f} seconds")
            if self.log_file_path:
                try:
                    # Ensure log file exists or re-create header if empty
                    if not os.path.exists(self.log_file_path) or os.path.getsize(self.log_file_path) == 0:
                         log_dir = os.path.dirname(self.log_file_path)
                         if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
                         with open(self.log_file_path, 'w', newline='') as f_reinit:
                            writer = csv.writer(f_reinit)
                            writer.writerow(['Epoch', 'CumulativeTimeSeconds'])
                         self.initial_log = False # Reset in case it was deleted
                    # Append current epoch time
                    with open(self.log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([current_epoch_number, f"{elapsed_time:.4f}"])
                except IOError as e:
                    print(f"Warning: Could not append cumulative time to {self.log_file_path}: {e}")

    def on_train_end(self, logs=None):
        final_time = time.time()
        total_training_time = final_time - self.training_start_time
        print(f"Total training finished. Total elapsed time: {total_training_time:.2f} seconds.")
        if self.log_file_path:
            last_recorded_epoch = 0
            total_epochs_planned = 0
            # Get planned epochs from model parameters if available
            if self.model and hasattr(self.model, 'params') and 'epochs' in self.model.params:
                total_epochs_planned = self.model.params['epochs']

            # Read last recorded epoch if file exists
            if os.path.exists(self.log_file_path):
                try:
                    with open(self.log_file_path, 'r') as f_read:
                        reader = csv.reader(f_read)
                        next(reader, None) # Skip header
                        for row in reader:
                            if row: # Check if row is not empty
                                try:
                                    last_recorded_epoch = int(row[0])
                                except (ValueError, IndexError):
                                    pass # Ignore malformed rows
                except (IOError, StopIteration) as e:
                    print(f"Warning: Could not read last recorded epoch from {self.log_file_path}: {e}")
                    last_recorded_epoch = 0 # Reset if read fails

            # Append final time if planned epochs > last recorded epoch
            # Or if early stopping happened, make sure the final time for the *actual* last epoch is recorded.
            actual_epochs_trained = 0
            if logs and 'epochs' in logs: # Keras might pass this
                 actual_epochs_trained = logs['epochs']
            elif self.model and hasattr(self.model, 'history') and self.model.history and 'loss' in self.model.history.history:
                 actual_epochs_trained = len(self.model.history.history['loss'])


            record_final_epoch_num = total_epochs_planned
            if actual_epochs_trained > 0 and actual_epochs_trained < total_epochs_planned:
                 record_final_epoch_num = actual_epochs_trained # If early stopped, use actual epochs

            if record_final_epoch_num > 0 and record_final_epoch_num > last_recorded_epoch:
                try:
                    # Ensure file/header exists before appending
                    if not os.path.exists(self.log_file_path) or os.path.getsize(self.log_file_path) == 0:
                         log_dir = os.path.dirname(self.log_file_path)
                         if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
                         with open(self.log_file_path, 'w', newline='') as f_reinit:
                            writer = csv.writer(f_reinit)
                            writer.writerow(['Epoch', 'CumulativeTimeSeconds'])
                    # Append final time
                    with open(self.log_file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([record_final_epoch_num, f"{total_training_time:.4f}"])
                        print(f"Final cumulative time for epoch {record_final_epoch_num} written to {self.log_file_path}")
                except IOError as e:
                     print(f"Warning: Could not append final cumulative time to {self.log_file_path}: {e}")


# --- Configuration ---
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS_IN = 3
IMG_CHANNELS_OUT = 3
TARGET_EPOCHS = 50
FIXED_BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOG_INTERVAL_EPOCHS = 10
NUM_VIS_SAMPLES = 5

# !!! --- 温度范围和色条定义 --- !!!
T_AMBIENT_K = 298.0  # Ambient temperature in Kelvin
GLOBAL_TEMP_MIN_K = T_AMBIENT_K
GLOBAL_TEMP_MAX_K = 3500.0 # Maximum temperature in Kelvin
CMAP_NAME = 'turbo' # Colormap name
CMAP_RESOLUTION = 256 # Number of colors in the colormap

# --- Paths ---
BASE_DATA_PATH = r'D:\data_sets\3200' # Base directory for dataset
RESULTS_VIS_DIR = r'D:\data_sets\test3200' # Base directory for results and visualizations
INITIAL_WEIGHTS_PATH = r'D:\data_sets\test3200\initial_5_7x7.weights.h5' # Path for initial weights

TRAIN_PATH_DIR = os.path.join(BASE_DATA_PATH, 'train_PATH')
TRAIN_TF_DIR = os.path.join(BASE_DATA_PATH, 'train_TF')
VALIDATION_PATH_DIR = os.path.join(BASE_DATA_PATH, 'val_PATH')
VALIDATION_TF_DIR = os.path.join(BASE_DATA_PATH, 'val_TF')
TEST_PATH_DIR = os.path.join(BASE_DATA_PATH, 'test_PATH')
TEST_TF_DIR = os.path.join(BASE_DATA_PATH, 'test_TF')

RESULTS_SUMMARY_CSV = os.path.join(RESULTS_VIS_DIR, 'all_runs_summary_metrics.csv')
ERROR_MAPS_VIS_DIR = os.path.join(RESULTS_VIS_DIR, 'error_maps_temperature_K') # Specific dir for temp analysis plots
PREDICTED_TEMPS_FILE = os.path.join(RESULTS_VIS_DIR, f'predicted_max_temperatures_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.csv')
CUMULATIVE_TIMES_FILE = os.path.join(RESULTS_VIS_DIR, f'cumulative_times_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.csv')
MAE_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'mae_loss_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
MSE_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'mse_loss_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
RMSE_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'rmse_loss_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
R2_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'r2_score_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
PARE_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'pare_score_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
MODEL_SAVE_PATH_SINGLE_RUN = os.path.join(RESULTS_VIS_DIR, f'unet_rgb_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.keras')

# --- 新增温度指标图表路径 ---
MAE_TEMP_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'mae_temp_final_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
MSE_TEMP_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'mse_temp_final_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
RMSE_TEMP_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'rmse_temp_final_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
PREDICTED_VS_TRUE_TEMP_SCATTER_PLOT_FILE = os.path.join(RESULTS_VIS_DIR, f'scatter_pred_true_temp_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
MODEL_SAVE_PATH_SINGLE_RUN = os.path.join(RESULTS_VIS_DIR, f'unet_rgb_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.keras')


# --- 全局变量：颜色和温度查找表 ---
TEMP_LOOKUP_TABLE = None
RGB_LOOKUP_TABLE = None

def initialize_temperature_colormap_lookup():
    """Initializes temperature-to-RGB and RGB-to-temperature lookup tables."""
    global TEMP_LOOKUP_TABLE, RGB_LOOKUP_TABLE, GLOBAL_TEMP_MIN_K, GLOBAL_TEMP_MAX_K, CMAP_NAME, CMAP_RESOLUTION
    if TEMP_LOOKUP_TABLE is not None and RGB_LOOKUP_TABLE is not None:
        return # Already initialized

    print("Initializing temperature-RGB lookup table...")
    try:
        TEMP_LOOKUP_TABLE = np.linspace(GLOBAL_TEMP_MIN_K, GLOBAL_TEMP_MAX_K, CMAP_RESOLUTION)
        cmap = plt.get_cmap(CMAP_NAME)
        norm = mcolors.Normalize(vmin=GLOBAL_TEMP_MIN_K, vmax=GLOBAL_TEMP_MAX_K)
        normalized_temps = norm(TEMP_LOOKUP_TABLE)
        RGB_LOOKUP_TABLE = cmap(normalized_temps)[:, :3] # Shape (CMAP_RESOLUTION, 3)
        print(f"Lookup table initialized with {len(RGB_LOOKUP_TABLE)} colors for temperature range [{GLOBAL_TEMP_MIN_K:.0f}K, {GLOBAL_TEMP_MAX_K:.0f}K].")
    except Exception as e:
        print(f"Error initializing lookup table: {e}. Will attempt fallback.")
        try:
            # Fallback attempt
            CMAP_NAME = 'viridis' # Use a standard, widely available colormap
            print(f"Attempting fallback with cmap '{CMAP_NAME}'...")
            cmap = plt.get_cmap(CMAP_NAME)
            TEMP_LOOKUP_TABLE = np.linspace(GLOBAL_TEMP_MIN_K, GLOBAL_TEMP_MAX_K, CMAP_RESOLUTION)
            norm = mcolors.Normalize(vmin=GLOBAL_TEMP_MIN_K, vmax=GLOBAL_TEMP_MAX_K)
            normalized_temps = norm(TEMP_LOOKUP_TABLE)
            RGB_LOOKUP_TABLE = cmap(normalized_temps)[:, :3]
            print(f"Fallback lookup table initialized with {len(RGB_LOOKUP_TABLE)} colors using '{CMAP_NAME}'.")
        except Exception as e_fallback:
            print(f"FATAL: Fallback lookup table initialization failed: {e_fallback}")
            TEMP_LOOKUP_TABLE = None
            RGB_LOOKUP_TABLE = None
            raise RuntimeError("Could not initialize temperature lookup table.") from e_fallback


def rgb_to_temperature_approx(rgb_pixel_array, rgb_lookup, temp_lookup):
    """Converts RGB pixel values to approximate temperatures using nearest neighbor lookup."""
    if rgb_lookup is None or temp_lookup is None:
        raise ValueError("Lookup tables not initialized. Call initialize_temperature_colormap_lookup() first.")
    if not isinstance(rgb_pixel_array, np.ndarray):
        rgb_pixel_array = np.array(rgb_pixel_array)

    # Ensure input is float and scaled [0, 1] if it looks like uint8 [0, 255]
    if rgb_pixel_array.dtype == np.uint8:
         rgb_pixel_array = rgb_pixel_array.astype(np.float32) / 255.0
    elif np.max(rgb_pixel_array) > 1.0 and np.max(rgb_pixel_array) <= 255.0: # Heuristic for unnormalized floats [0, 255]
         # print("Warning: rgb_to_temperature_approx received input > 1.0. Assuming scale [0, 255] and normalizing.")
         rgb_pixel_array = np.clip(rgb_pixel_array / 255.0, 0.0, 1.0)
    elif np.max(rgb_pixel_array) > 1.0: # If still > 1.0, it might be a genuine issue or different scale
         print(f"Warning: rgb_to_temperature_approx received input with max value {np.max(rgb_pixel_array)} > 1.0 and not clearly [0,255] uint8. Clipping to [0,1]. This might indicate an issue.")
         rgb_pixel_array = np.clip(rgb_pixel_array, 0.0, 1.0)


    original_shape = rgb_pixel_array.shape
    if rgb_pixel_array.ndim == 3: # (H, W, 3)
        h, w, c = original_shape
        if c != 3: raise ValueError(f"Expected 3 channels, got {c}")
        rgb_pixel_flat = rgb_pixel_array.reshape(-1, 3)
    elif rgb_pixel_array.ndim == 2 and original_shape[1] == 3: # (N, 3)
        rgb_pixel_flat = rgb_pixel_array
    elif rgb_pixel_array.ndim == 1 and original_shape[0] == 3: # Single pixel (3,)
        rgb_pixel_flat = rgb_pixel_array.reshape(1, 3)
    else:
        raise ValueError(f"Unsupported rgb_pixel_array shape: {original_shape}")

    # Compute Euclidean distances to lookup table colors
    distances = cdist(rgb_pixel_flat, rgb_lookup, metric='euclidean')
    closest_indices = np.argmin(distances, axis=1)
    temperatures_flat = temp_lookup[closest_indices]

    if rgb_pixel_array.ndim == 3:
        return temperatures_flat.reshape(h, w)
    else:
        return temperatures_flat

def generate_max_temperature_histogram_per_image(generator, dataset_name_str, output_base_dir,
                                                 rgb_lookup_table, temp_lookup_table,
                                                 global_min_k, global_max_k,
                                                 target_epochs, batch_size_config, bins=20): # Default bins set to 20
    """Generates a histogram of the maximum temperature found in each ground truth image."""
    if not generator or not hasattr(generator, 'path_files') or len(generator.path_files) == 0:
        print(f"Skipping max temperature (per image) histogram for {dataset_name_str} set: No data.")
        return

    if rgb_lookup_table is None or temp_lookup_table is None:
        print(f"Error generating max temp histogram for {dataset_name_str}: Lookup tables not initialized.")
        return

    print(f"\nGenerating Max Temperature (per image) Histogram for {dataset_name_str} Set (Bins={bins})...")
    all_max_temperatures_per_image = []
    num_batches = len(generator)
    total_samples = len(generator.path_files)
    processed_samples = 0

    print(f"Processing {total_samples} samples in {num_batches} batches for {dataset_name_str} max temperature histogram...")

    for batch_idx in range(num_batches):
        if batch_idx > 0 and batch_idx % (max(1, num_batches // 5)) == 0: # Print progress roughly 5 times
            print(f"  {dataset_name_str} Set - Processed {processed_samples}/{total_samples} samples (batch {batch_idx}/{num_batches})...")
        try:
            _, y_batch_true_rgb = generator[batch_idx] # Get ground truth RGB batch
            if y_batch_true_rgb.shape[0] == 0:
                continue

            for i in range(y_batch_true_rgb.shape[0]):
                true_rgb_image = y_batch_true_rgb[i]
                true_temp_map = rgb_to_temperature_approx(true_rgb_image, rgb_lookup_table, temp_lookup_table)
                max_temp_for_this_image = np.max(true_temp_map)
                all_max_temperatures_per_image.append(max_temp_for_this_image)
                processed_samples += 1

        except Exception as e_hist_batch:
            print(f"  Warning: Error processing batch {batch_idx} for {dataset_name_str} max temp histogram: {e_hist_batch}")
            # Optionally add traceback.print_exc() here for debugging

    if not all_max_temperatures_per_image:
        print(f"No max temperature data collected from the {dataset_name_str} set to generate histogram.")
        return

    all_max_temperatures_per_image = np.array(all_max_temperatures_per_image)
    print(f"Collected {len(all_max_temperatures_per_image)} max temperature values from the {dataset_name_str} set.")

    # Plot and save histogram
    hist_filename = f'{dataset_name_str.lower()}_set_max_temperature_per_image_histogram_ep{target_epochs}_bs{batch_size_config}.png'
    hist_save_path = os.path.join(output_base_dir, hist_filename)

    plt.figure(figsize=(12, 7))
    plt.hist(all_max_temperatures_per_image, bins=bins, color='dodgerblue', edgecolor='black', range=(global_min_k, global_max_k))
    plt.title(f'Distribution of Max Temperatures per Image in {dataset_name_str} Set\n(Epochs: {target_epochs}, Batch Size: {batch_size_config})')
    plt.xlabel('Maximum Temperature per Image (K)')
    plt.ylabel('Number of Samples (Images)')
    plt.grid(axis='y', alpha=0.75)
    plt.xlim(global_min_k, global_max_k)
    plt.tight_layout()
    try:
        # Ensure output directory exists
        os.makedirs(output_base_dir, exist_ok=True)
        plt.savefig(hist_save_path)
        print(f"{dataset_name_str} set max temperature (per image) histogram saved to: {hist_save_path}")
    except Exception as e_save_hist:
        print(f"Error saving {dataset_name_str} set max temperature (per image) histogram: {e_save_hist}")
    plt.close()


# --- Helper Functions ---
def load_and_preprocess_image(path, is_temp_field=False, for_visualization=False):
    """Loads and preprocesses an image from path."""
    global IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS_IN, IMG_CHANNELS_OUT
    try:
        # Open image and convert to RGB
        img_pil = Image.open(path).convert('RGB')
        # Resize using NEAREST to avoid introducing intermediate colors in temperature fields
        img_pil = img_pil.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
        img_array = np.array(img_pil) # uint8 array [0, 255]

        if for_visualization:
            # Return uint8 for display input, float32 [0,1] for display target/prediction
            return img_array if not is_temp_field else img_array.astype(np.float32) / 255.0
        else:
            # Return float32 [0,1] for model input/target
             return img_array.astype(np.float32) / 255.0

    except FileNotFoundError:
        print(f"Warning: Image file not found at {path}. Returning zeros.")
    except Exception as e:
        print(f"Warning: Error loading image {path}: {e}. Returning zeros.")

    # Return zero array on error
    target_channels = IMG_CHANNELS_OUT if is_temp_field else IMG_CHANNELS_IN
    dtype = np.uint8 if (for_visualization and not is_temp_field) else np.float32
    return np.zeros((IMG_HEIGHT, IMG_WIDTH, target_channels), dtype=dtype)


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras models."""
    def __init__(self, path_dir, tf_dir, batch_size, dim, n_channels_in, n_channels_out, shuffle=True, dataset_name="Unnamed"):
        self.path_dir = path_dir
        self.tf_dir = tf_dir
        self.batch_size = batch_size
        self.dim = dim # Expected (height, width)
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        self.dataset_name = dataset_name
        self.path_files = []
        self.tf_files = []
        self.indexes = np.array([])

        self._prepare_file_lists()
        self.on_epoch_end() # Initial shuffle if applicable

    def _prepare_file_lists(self):
        """Scans directories, finds matching files, and prepares lists."""
        print(f"Preparing file list for {self.dataset_name} generator...")
        if not os.path.isdir(self.path_dir):
            print(f"Warning for {self.dataset_name}: Input directory does not exist: {self.path_dir}")
            return
        if not os.path.isdir(self.tf_dir):
            print(f"Warning for {self.dataset_name}: Target directory does not exist: {self.tf_dir}")
            return

        try:
            all_path_files_basenames = {os.path.splitext(f)[0]: os.path.join(self.path_dir, f)
                                        for f in os.listdir(self.path_dir) if f.lower().endswith('.png')}
            all_tf_files_basenames = {os.path.splitext(f)[0]: os.path.join(self.tf_dir, f)
                                      for f in os.listdir(self.tf_dir) if f.lower().endswith('.png')}

            common_ids = sorted(list(all_path_files_basenames.keys() & all_tf_files_basenames.keys()))

            if not common_ids:
                print(f"Warning for {self.dataset_name}: No matching PNG files found between {self.path_dir} and {self.tf_dir}.")
                return

            self.path_files = [all_path_files_basenames[id] for id in common_ids]
            self.tf_files = [all_tf_files_basenames[id] for id in common_ids]

            print(f"{self.dataset_name} generator: Found {len(self.path_files)} matching image pairs.")
            self.indexes = np.arange(len(self.path_files))

        except Exception as e:
            print(f"Error preparing file lists for {self.dataset_name}: {e}")
            self.path_files = []
            self.tf_files = []
            self.indexes = np.array([])

    def __len__(self):
        """Denotes the number of batches per epoch."""
        if not self.path_files: return 0
        # Return number of batches, ensuring at least one batch if there are files < batch_size
        num_samples = len(self.path_files)
        if num_samples == 0: return 0
        length = int(np.floor(num_samples / self.batch_size))
        return length if length > 0 else 1 # Ensure at least one batch if samples exist

    def __getitem__(self, index):
        """Generate one batch of data."""
        if not self.path_files:
            # Return empty arrays matching expected output structure
            return np.empty((0, *self.dim, self.n_channels_in)), np.empty((0, *self.dim, self.n_channels_out))

        # Calculate batch indices, handle potential last smaller batch
        num_samples = len(self.path_files)
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, num_samples)
        actual_batch_size = end_idx - start_idx

        if actual_batch_size <= 0: # Should not happen with __len__ logic, but safeguard
             return np.empty((0, *self.dim, self.n_channels_in)), np.empty((0, *self.dim, self.n_channels_out))

        indexes_batch = self.indexes[start_idx:end_idx]

        # Get file paths for the batch
        list_path_files = [self.path_files[k] for k in indexes_batch]
        list_tf_files = [self.tf_files[k] for k in indexes_batch]

        # Initialize batch arrays
        X = np.empty((actual_batch_size, *self.dim, self.n_channels_in), dtype=np.float32)
        y = np.empty((actual_batch_size, *self.dim, self.n_channels_out), dtype=np.float32)

        # Load and preprocess images for the batch
        valid_images_in_batch = 0
        for i, (path_file, tf_file) in enumerate(zip(list_path_files, list_tf_files)):
            x_img = load_and_preprocess_image(path_file, is_temp_field=False)
            y_img = load_and_preprocess_image(tf_file, is_temp_field=True)

            # Basic check if loading failed (returned zeros)
            # More robust check might be needed if zeros are valid data
            if np.any(x_img) or np.any(y_img): # Simple check if not all zeros
                 X[valid_images_in_batch,] = x_img
                 y[valid_images_in_batch,] = y_img
                 valid_images_in_batch += 1
            else:
                 print(f"Warning: Skipping potentially invalid data from pair: {os.path.basename(path_file)}, {os.path.basename(tf_file)}")


        # Return only the validly loaded images
        if valid_images_in_batch < actual_batch_size and valid_images_in_batch > 0 : # Only print if some but not all failed
            print(f"Warning: Batch {index} for {self.dataset_name} contained {actual_batch_size - valid_images_in_batch} invalid images. Used {valid_images_in_batch} valid images.")
        elif valid_images_in_batch == 0 and actual_batch_size > 0:
             print(f"Warning: Batch {index} for {self.dataset_name} contained ALL invalid images. Returning empty batch.")
             return np.empty((0, *self.dim, self.n_channels_in)), np.empty((0, *self.dim, self.n_channels_out))


        return X[:valid_images_in_batch], y[:valid_images_in_batch]


    def on_epoch_end(self):
        """Updates indexes after each epoch, shuffles if specified."""
        if hasattr(self, 'indexes') and len(self.indexes) > 0: # Check if indexes exist
            if self.shuffle:
                np.random.shuffle(self.indexes)
                # print(f"Shuffled indexes for {self.dataset_name} generator.")


def build_unet_model(input_shape, num_classes=3): # Kernel size is (5,5) as per previous request
    inputs = keras.Input(shape=input_shape)
    kernel_s = (7,7) # Assuming you want to keep the 5x5 kernels

    # --- ENCODER (7 Levels) ---
    # Level 1
    c1 = layers.Conv2D(16, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)  # 64x64 -> 32x32

    # Level 2
    c2 = layers.Conv2D(32, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)  # 32x32 -> 16x16

    # Level 3
    c3 = layers.Conv2D(64, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)  # 16x16 -> 8x8

    # Level 4
    c4 = layers.Conv2D(128, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)  # 8x8 -> 4x4

    #Level 5
    # c5 = layers.Conv2D(256, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    # c5 = layers.Dropout(0.3)(c5)
    # c5 = layers.Conv2D(256, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    # p5 = layers.MaxPooling2D((2, 2))(c5)  # 4x4 -> 2x2
    #
    # # Level 6
    # c6 = layers.Conv2D(512, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    # c6 = layers.Dropout(0.3)(c6)  # Dropout can be adjusted
    # c6 = layers.Conv2D(512, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    # p6 = layers.MaxPooling2D((2, 2))(c6)  # 2x2 -> 1x1


    # Level 7 (Bottleneck) - Feature map is 1x1
    c7 = layers.Conv2D(256, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c7 = layers.Dropout(0.3)(c7)  # Dropout can be adjusted, often higher at bottleneck
    c7 = layers.Conv2D(256, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    # No MaxPooling after c7 as it's the bottleneck at 1x1

    # --- DECODER ---
    #Upsample from c7 (bottleneck) and Concatenate with c6
    # u_level6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c7)  # Upsample 1x1 -> 2x2
    # u_level6 = layers.concatenate([u_level6, c6])  # c6 was 2x2 (before p6)
    # c_level6 = layers.Conv2D(512, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(u_level6)
    # c_level6 = layers.Dropout(0.3)(c_level6)
    # c_level6 = layers.Conv2D(512, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c_level6)
    #
    # #Upsample from c_level6 and Concatenate with c5
    # u_level5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c_level6)  # Upsample 2x2 -> 4x4
    # u_level5 = layers.concatenate([u_level5, c5])  # c5 was 4x4 (before p5)
    # c_level5 = layers.Conv2D(256, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(u_level5)
    # c_level5 = layers.Dropout(0.3)(c_level5)
    # c_level5 = layers.Conv2D(256, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c_level5)

    #Upsample from c_level5 and Concatenate with c4
    u_level4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)  # Upsample 4x4 -> 8x8
    u_level4 = layers.concatenate([u_level4, c4])  # c4 was 8x8 (before p4)
    c_level4 = layers.Conv2D(128, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(u_level4)
    c_level4 = layers.Dropout(0.2)(c_level4)
    c_level4 = layers.Conv2D(128, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c_level4)

    # Upsample from c_level4 and Concatenate with c3
    u_level3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c_level4)  # Upsample 8x8 -> 16x16
    u_level3 = layers.concatenate([u_level3, c3])  # c3 was 16x16 (before p3)
    c_level3 = layers.Conv2D(64, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(u_level3)
    c_level3 = layers.Dropout(0.2)(c_level3)
    c_level3 = layers.Conv2D(64, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c_level3)

    # Upsample from c_level3 and Concatenate with c2
    u_level2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c_level3)  # Upsample 16x16 -> 32x32
    u_level2 = layers.concatenate([u_level2, c2])  # c2 was 32x32 (before p2)
    c_level2 = layers.Conv2D(32, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(u_level2)
    c_level2 = layers.Dropout(0.1)(c_level2)
    c_level2 = layers.Conv2D(32, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c_level2)

    # Upsample from c_level2 and Concatenate with c1
    u_level1 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c_level2)  # Upsample 32x32 -> 64x64
    u_level1 = layers.concatenate([u_level1, c1])  # c1 was 64x64 (before p1)
    c_level1 = layers.Conv2D(16, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(u_level1)
    c_level1 = layers.Dropout(0.1)(c_level1)
    c_level1 = layers.Conv2D(16, kernel_s, activation='relu', kernel_initializer='he_normal', padding='same')(c_level1)

    # Output Layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c_level1)  # Output kernel (1,1)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def calculate_pare(y_true, y_pred, epsilon=1e-8):
    """Calculates Pixel-wise Absolute Relative Error (PARE)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Avoid division by zero or near-zero for ambient temperature pixels
    # Also ensure y_true is positive before division if it can be negative
    # For temperatures in K, y_true should be >= T_AMBIENT_K > epsilon.
    denominator = np.maximum(np.abs(y_true), epsilon)
    relative_error = np.abs(y_pred - y_true) / denominator
    pare = np.mean(relative_error)
    return pare

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 定义 save_metrics_to_csv 函数的位置 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_metrics_to_csv(filepath, config_params, metrics_dict):
    """
    Saves configuration parameters and evaluation metrics to a CSV file.
    Appends a new row if the file exists, otherwise creates a new file with headers.
    """
    # Define the full set of headers
    # Order them as you'd like them to appear in the CSV
    fieldnames = [
        'Timestamp', 'TargetEpochs', 'ActualEpochs', 'BatchSize', 'LearningRate', 'ModelUsed',
        'MAE_RGB', 'MSE_RGB', 'RMSE_RGB',
        'MAE_Temp', 'MSE_Temp', 'RMSE_Temp', # <--确保这些在这里
        'R2_RGB_Pixel', 'R2_Temp_Pixel',
        'R2_MaxTemp_Image', 'R2_AvgTemp_Image',
        'PARE_Temp_Pixel'
    ]

    # Prepare the data row from the dictionaries
    row_data = {
        'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'TargetEpochs': config_params.get('target_epochs', TARGET_EPOCHS),
        'ActualEpochs': config_params.get('actual_epochs', np.nan), # Get this from history
        'BatchSize': config_params.get('batch_size', FIXED_BATCH_SIZE),
        'LearningRate': config_params.get('learning_rate', LEARNING_RATE),
        'ModelUsed': os.path.basename(config_params.get('model_path', MODEL_SAVE_PATH_SINGLE_RUN)),
        'MAE_RGB': metrics_dict.get('mae', np.nan),
        'MSE_RGB': metrics_dict.get('mse', np.nan),
        'RMSE_RGB': metrics_dict.get('rmse', np.nan),
        'MAE_Temp': metrics_dict.get('mae_temp', np.nan),     # <--确保正确映射
        'MSE_Temp': metrics_dict.get('mse_temp', np.nan),     # <--确保正确映射
        'RMSE_Temp': metrics_dict.get('rmse_temp', np.nan),   # <--确保正确映射
        'R2_RGB_Pixel': metrics_dict.get('r2_rgb', np.nan),
        'R2_Temp_Pixel': metrics_dict.get('r2_temp', np.nan),
        'R2_MaxTemp_Image': metrics_dict.get('r2_max_temp', np.nan),
        'R2_AvgTemp_Image': metrics_dict.get('r2_avg_temp', np.nan),
        'PARE_Temp_Pixel': metrics_dict.get('pare', np.nan)
    }

    try:
        file_exists = os.path.isfile(filepath)
        # Ensure directory for CSV exists
        csv_dir = os.path.dirname(filepath)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        with open(filepath, 'a', newline='') as csvfile: # Open in append mode
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(filepath) == 0: # Write header if new file or empty
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Metrics successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving metrics to CSV {filepath}: {e}")
    except Exception as e_gen:
            print(f"An unexpected error occurred while saving metrics to CSV: {e_gen}")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 函数定义结束 +++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# --- Main Execution ---

# Ensure result directories exist
os.makedirs(RESULTS_VIS_DIR, exist_ok=True)
os.makedirs(ERROR_MAPS_VIS_DIR, exist_ok=True)

# Initialize Temperature Lookup Table
initialize_temperature_colormap_lookup()
if TEMP_LOOKUP_TABLE is None or RGB_LOOKUP_TABLE is None:
     print("FATAL: Could not initialize lookup tables. Exiting.")
     exit()

# Create or load initial weights
if not os.path.exists(INITIAL_WEIGHTS_PATH):
    print(f"Initial weights file not found at {INITIAL_WEIGHTS_PATH}. Creating it...")
    try:
        temp_input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN)
        initial_model = build_unet_model(temp_input_shape, num_classes=IMG_CHANNELS_OUT)
        # Ensure directory for weights exists
        weights_dir = os.path.dirname(INITIAL_WEIGHTS_PATH)
        if weights_dir and not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        initial_model.save_weights(INITIAL_WEIGHTS_PATH)
        print(f"Initial weights saved to {INITIAL_WEIGHTS_PATH}")
        del initial_model # Free memory
    except Exception as e_weights:
         print(f"FATAL: Failed to create initial weights file: {e_weights}")
         exit()
else:
    print(f"Using existing initial weights from {INITIAL_WEIGHTS_PATH}")


# --- 1. Data Generators ---
print("\n--- Creating Data Generators ---")
train_generator = DataGenerator(TRAIN_PATH_DIR, TRAIN_TF_DIR, FIXED_BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), IMG_CHANNELS_IN, IMG_CHANNELS_OUT, shuffle=True, dataset_name="Training")
if len(train_generator.indexes) == 0:
    print(f"FATAL: Training data generator is empty. Check paths and file matching:\n - Input: {TRAIN_PATH_DIR}\n - Target: {TRAIN_TF_DIR}\nCannot proceed.")
    exit()
print(f"Train generator: {len(train_generator.indexes)} samples, {len(train_generator)} steps/epoch.")

val_data_to_pass = None
validation_generator = DataGenerator(VALIDATION_PATH_DIR, VALIDATION_TF_DIR, FIXED_BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), IMG_CHANNELS_IN, IMG_CHANNELS_OUT, shuffle=False, dataset_name="Validation")
if len(validation_generator.indexes) > 0 and len(validation_generator) > 0:
    val_data_to_pass = validation_generator
    print(f"Validation generator: {len(validation_generator.indexes)} samples, {len(validation_generator)} steps/epoch.")
else:
    print("Warning: Validation generator is empty or directories not found. Validation will be skipped.")

test_generator = DataGenerator(TEST_PATH_DIR, TEST_TF_DIR, FIXED_BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), IMG_CHANNELS_IN, IMG_CHANNELS_OUT, shuffle=False, dataset_name="Test")
if not (len(test_generator.indexes) > 0 and len(test_generator) > 0):
    print("Warning: Test generator is empty or directories not found. Final evaluation & visualization will be skipped.")
    test_generator = None # Set to None if unusable

# --- 2. Build and Compile Model ---
print("\n--- Building and Compiling Model ---")
tf.keras.backend.clear_session() # Clear previous sessions
input_shape_run = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN)
model = build_unet_model(input_shape_run, num_classes=IMG_CHANNELS_OUT)
model.summary() # Print model architecture

print(f"Loading initial weights from {INITIAL_WEIGHTS_PATH}...")
try:
    model.load_weights(INITIAL_WEIGHTS_PATH)
    print("Initial weights loaded successfully.")
except Exception as e_load:
    print(f"Warning: Could not load initial weights from {INITIAL_WEIGHTS_PATH}: {e_load}. Training will start from scratch.")

print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='mae', # Mean Absolute Error is common for regression-like image tasks
              metrics=['mse']) # Mean Squared Error

# --- 3. Callbacks ---
print("\n--- Preparing Callbacks ---")
time_logger_callback = CumulativeTimeLogger(log_interval=LOG_INTERVAL_EPOCHS, log_file_path=CUMULATIVE_TIMES_FILE)

# Monitor validation loss if available, otherwise training loss
monitor_metric = 'val_loss' if val_data_to_pass else 'loss'
print(f"Callbacks will monitor: '{monitor_metric}'")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH_SINGLE_RUN,
        verbose=1,
        save_best_only=True,
        monitor=monitor_metric,
        save_weights_only=False # Save entire model (.keras format)
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.2, # Reduce LR by factor of 5
        patience=7, # Reduce after 7 epochs of no improvement
        verbose=1,
        min_lr=1e-7
    ),
    # keras.callbacks.EarlyStopping( # EarlyStopping is REMOVED/COMMENTED OUT
    #     monitor=monitor_metric,
    #     patience=15, # Stop after 15 epochs of no improvement
    #     verbose=1,
    #     restore_best_weights=True # Load weights from the best epoch
    # ),
    time_logger_callback
]

# --- 4. Train Model ---
print(f"\n--- Starting Training for up to {TARGET_EPOCHS} Epochs ---")
history = model.fit(
    train_generator,
    validation_data=val_data_to_pass,
    epochs=TARGET_EPOCHS, # Model will train for all TARGET_EPOCHS unless manually stopped
    callbacks=callbacks,
    verbose=1 # Show progress bar
)
history_data = history.history
print("Training finished.")

# --- 5. Load Best Model for Evaluation ---
print("\n--- Loading Best Model for Evaluation ---")
eval_model_for_test_and_viz = None
if os.path.exists(MODEL_SAVE_PATH_SINGLE_RUN):
    print(f"Loading best model from {MODEL_SAVE_PATH_SINGLE_RUN}...")
    try:
        # No need to build/compile again when loading full model
        eval_model_for_test_and_viz = keras.models.load_model(MODEL_SAVE_PATH_SINGLE_RUN)
        print("Best model loaded successfully.")
    except Exception as e_load_best:
        print(f"Warning: Failed to load the best saved model from {MODEL_SAVE_PATH_SINGLE_RUN}: {e_load_best}")
        print("Using the model state at the end of training for evaluation.")
        eval_model_for_test_and_viz = model # Fallback to the model in memory
else:
    print(f"Warning: Model checkpoint file not found at {MODEL_SAVE_PATH_SINGLE_RUN}. Using current model state.")
    eval_model_for_test_and_viz = model

# --- 6. Evaluate Model & Extract Temperatures (Test Set) ---
print("\n--- Evaluating Model on Test Set & Extracting Metrics ---")

# Initialize metrics dictionary
test_set_metrics = {
    'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, # RGB based (from evaluate)
    'r2_rgb': np.nan,                            # Pixel-wise RGB R2
    'mae_temp': np.nan,                          # Pixel-wise Temperature MAE
    'mse_temp': np.nan,                          # Pixel-wise Temperature MSE
    'rmse_temp': np.nan,                         # Pixel-wise Temperature RMSE
    'r2_temp': np.nan,                           # Pixel-wise Temperature R2
    'pare': np.nan,                              # Pixel-wise Temperature PARE
    'r2_max_temp': np.nan,                       # Per-image Max Temperature R2
    'r2_avg_temp': np.nan                        # Per-image Average Temperature R2
}

predicted_max_temps_data = [] # For CSV file

if test_generator and eval_model_for_test_and_viz:
    print("Evaluating model with Keras evaluate (RGB MAE/MSE)...")
    try:
        # Keras evaluation for MAE, MSE (operates on RGB output)
        test_eval_results = eval_model_for_test_and_viz.evaluate(test_generator, verbose=1, return_dict=True)
        test_set_metrics['mae'] = test_eval_results.get('loss', np.nan)
        test_set_metrics['mse'] = test_eval_results.get('mse', np.nan)
        if not np.isnan(test_set_metrics['mse']): test_set_metrics['rmse'] = np.sqrt(test_set_metrics['mse'])
        print(f"**FINAL TEST MAE (RGB space): {test_set_metrics['mae']:.4f}**")
        print(f"**FINAL TEST MSE (RGB space): {test_set_metrics['mse']:.4f}**")
        print(f"**FINAL TEST RMSE (RGB space): {test_set_metrics['rmse']:.4f}**")
    except Exception as e_eval:
         print(f"Error during Keras evaluation on test set: {e_eval}")

    # --- Initialize lists for metric calculations ---
    y_true_list_test_rgb, y_pred_list_test_rgb = [], [] # For R2 (pixel-wise RGB)
    all_true_temp_maps_test, all_pred_temp_maps_test = [], [] # For PARE and R2 (pixel-wise Temp)
    true_max_temps_list, pred_max_temps_list = [], [] # For R2 (per-image Max)
    true_avg_temps_list, pred_avg_temps_list = [], [] # For R2 (per-image Avg)

    # --- Iterate through test set for detailed metrics ---
    print("Processing test set for PARE and Temperature R2 metrics...")
    num_test_batches = len(test_generator)
    processed_samples_count = 0
    for batch_idx in range(num_test_batches):
        if batch_idx > 0 and batch_idx % (max(1, num_test_batches // 5)) == 0:
             print(f"  Metrics processing: Batch {batch_idx}/{num_test_batches}...")
        try:
            x_batch_test, y_batch_true_rgb_test = test_generator[batch_idx]
            if x_batch_test.shape[0] == 0: continue # Skip empty batches

            # Get predictions for the batch
            y_batch_pred_rgb_test = eval_model_for_test_and_viz.predict_on_batch(x_batch_test)

            # Store RGB for pixel-wise RGB R2
            y_true_list_test_rgb.append(y_batch_true_rgb_test)
            y_pred_list_test_rgb.append(y_batch_pred_rgb_test)

            # Process each sample in the batch for temperature metrics
            for i in range(y_batch_pred_rgb_test.shape[0]):
                sample_original_index = test_generator.indexes[batch_idx * test_generator.batch_size + i] # Find original index
                # Ensure index is within bounds (safeguard against generator issues)
                if sample_original_index >= len(test_generator.path_files): continue
                base_filename = os.path.basename(test_generator.path_files[sample_original_index])

                predicted_rgb_image = y_batch_pred_rgb_test[i]
                true_rgb_image = y_batch_true_rgb_test[i]

                # Temperature Conversion
                try:
                    predicted_temp_map = rgb_to_temperature_approx(predicted_rgb_image, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
                    true_temp_map = rgb_to_temperature_approx(true_rgb_image, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
                except ValueError as e_lookup:
                    print(f"  Skipping temperature calculations for {base_filename} due to lookup error: {e_lookup}")
                    continue # Skip this image if conversion fails

                # Store maps for Pixel-wise PARE / R2 (Temp)
                all_pred_temp_maps_test.append(predicted_temp_map)
                all_true_temp_maps_test.append(true_temp_map)

                # Calculate and Store Per-Image Max/Avg Temps
                true_max = np.max(true_temp_map)
                pred_max = np.max(predicted_temp_map)
                true_avg = np.mean(true_temp_map)
                pred_avg = np.mean(predicted_temp_map)

                true_max_temps_list.append(true_max)
                pred_max_temps_list.append(pred_max)
                true_avg_temps_list.append(true_avg)
                pred_avg_temps_list.append(pred_avg)

                # Store predicted max temp for CSV file
                predicted_max_temps_data.append({'filename': base_filename, 'max_predicted_temp_K': pred_max})
                processed_samples_count += 1

        except Exception as e_batch_metric:
             print(f"Error processing batch {batch_idx} for metrics: {e_batch_metric}")
             traceback.print_exc() # Print stack trace for debugging

    print(f"Finished processing metrics for {processed_samples_count} test samples.")

    # --- Calculate Final Metrics ---

    # R² (Pixel-wise RGB)
    print("\nCalculating Final R2 Scores...")
    if y_true_list_test_rgb and y_pred_list_test_rgb:
        try:
            y_true_all_test_rgb = np.concatenate(y_true_list_test_rgb, axis=0)
            y_pred_all_test_rgb = np.concatenate(y_pred_list_test_rgb, axis=0)
            # Flatten for R2 calculation
            y_true_flat_test_rgb = y_true_all_test_rgb.reshape(-1)
            y_pred_flat_test_rgb = y_pred_all_test_rgb.reshape(-1)
            if len(y_true_flat_test_rgb) > 1:
                test_set_metrics['r2_rgb'] = r2_score(y_true_flat_test_rgb, y_pred_flat_test_rgb)
                print(f"**FINAL TEST R² Score (RGB space, pixel-wise): {test_set_metrics['r2_rgb']:.4f}**")
            else: print("Not enough samples for R2 score (RGB space, pixel-wise).")
        except Exception as e_r2_rgb:
             print(f"Error calculating pixel-wise RGB R2: {e_r2_rgb}")
    else: print("No RGB data collected for R2 calculation (RGB space, pixel-wise).")

    # PARE, Temperature MAE, MSE, RMSE and R² (Pixel-wise Temperature)
    if all_true_temp_maps_test and all_pred_temp_maps_test:
        try:
            # Flatten all temperature maps
            y_true_all_temps_flat = np.concatenate([temp_map.flatten() for temp_map in all_true_temp_maps_test])
            y_pred_all_temps_flat = np.concatenate([temp_map.flatten() for temp_map in all_pred_temp_maps_test])

            if len(y_true_all_temps_flat) > 0:
                # MAE (Temp, pixel-wise)
                test_set_metrics['mae_temp'] = np.mean(np.abs(y_true_all_temps_flat - y_pred_all_temps_flat))
                print(f"**FINAL TEST MAE (Temperature space, pixel-wise): {test_set_metrics['mae_temp']:.4f} K**")

                # MSE (Temp, pixel-wise)
                test_set_metrics['mse_temp'] = np.mean((y_true_all_temps_flat - y_pred_all_temps_flat)**2)
                print(f"**FINAL TEST MSE (Temperature space, pixel-wise): {test_set_metrics['mse_temp']:.4f} K²**")

                # RMSE (Temp, pixel-wise)
                if not np.isnan(test_set_metrics['mse_temp']):
                    test_set_metrics['rmse_temp'] = np.sqrt(test_set_metrics['mse_temp'])
                    print(f"**FINAL TEST RMSE (Temperature space, pixel-wise): {test_set_metrics['rmse_temp']:.4f} K**")

                # PARE
                test_set_metrics['pare'] = calculate_pare(y_true_all_temps_flat, y_pred_all_temps_flat)
                print(f"**FINAL TEST PARE (Temperature space, pixel-wise): {test_set_metrics['pare']:.4f}**")

                # R² (Temp, pixel-wise)
                if len(y_true_all_temps_flat) > 1:
                    test_set_metrics['r2_temp'] = r2_score(y_true_all_temps_flat, y_pred_all_temps_flat)
                    print(f"**FINAL TEST R² Score (Temperature space, pixel-wise): {test_set_metrics['r2_temp']:.4f}**")
                else:
                    print("Not enough temperature pixels for R2 score (Temperature space, pixel-wise).")
            else:
                print("Not enough temperature data for PARE or pixel-wise Temperature R2, MAE_temp, MSE_temp, RMSE_temp.")
        except Exception as e_pare_r2_temp:
             print(f"Error calculating PARE or pixel-wise Temperature metrics: {e_pare_r2_temp}")
    else:
        print("No temperature maps collected for PARE or pixel-wise Temperature metrics calculation.")

    # R² (Per-Image Max Temperature)
    if len(true_max_temps_list) > 1 and len(pred_max_temps_list) == len(true_max_temps_list):
        try:
            true_max_temps_arr = np.array(true_max_temps_list)
            pred_max_temps_arr = np.array(pred_max_temps_list)
            test_set_metrics['r2_max_temp'] = r2_score(true_max_temps_arr, pred_max_temps_arr)
            print(f"**FINAL TEST R² Score (Per-Image Max Temperature): {test_set_metrics['r2_max_temp']:.4f}**")
        except Exception as e_r2_max:
            print(f"Warning: Could not calculate R2 score for max temperature: {e_r2_max}")
    else:
        print(f"Not enough data points ({len(true_max_temps_list)}) for R2 score (Per-Image Max Temperature).")

    # R² (Per-Image Average Temperature)
    if len(true_avg_temps_list) > 1 and len(pred_avg_temps_list) == len(true_avg_temps_list):
        try:
            true_avg_temps_arr = np.array(true_avg_temps_list)
            pred_avg_temps_arr = np.array(pred_avg_temps_list)
            test_set_metrics['r2_avg_temp'] = r2_score(true_avg_temps_arr, pred_avg_temps_arr)
            print(f"**FINAL TEST R² Score (Per-Image Average Temperature): {test_set_metrics['r2_avg_temp']:.4f}**")
        except Exception as e_r2_avg:
             print(f"Warning: Could not calculate R2 score for average temperature: {e_r2_avg}")
    else:
        print(f"Not enough data points ({len(true_avg_temps_list)}) for R2 score (Per-Image Average Temperature).")

    # --- Save predicted max temperatures to CSV ---
    if predicted_max_temps_data:
        print(f"\nSaving predicted max temperatures to {PREDICTED_TEMPS_FILE}...")
        try:
            with open(PREDICTED_TEMPS_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'max_predicted_temp_K'])
                writer.writeheader()
                writer.writerows(predicted_max_temps_data)
            print(f"Predicted max temperatures saved successfully.")
        except IOError as e:
            print(f"Error writing predicted temperatures to CSV: {e}")

else:
    print("Skipping final evaluation and metric extraction as test generator or model is not available.")
    # Ensure metrics that depend on test eval are NaN
    keys_to_nan = ['mae', 'mse', 'rmse', 'r2_rgb',
                   'mae_temp', 'mse_temp', 'rmse_temp', # Added temp metrics
                   'r2_temp', 'pare', 'r2_max_temp', 'r2_avg_temp']
    for key in keys_to_nan:
        test_set_metrics[key] = np.nan


# --- 7. Plotting Loss/Metric History & Final Scores ---
print("\n--- Generating Loss/Metric Plots ---")

# Determine number of epochs actually trained
actual_epochs_trained = len(history_data.get('loss', []))
if actual_epochs_trained == 0 and TARGET_EPOCHS > 0:
    actual_epochs_trained = TARGET_EPOCHS
    print("Warning: No training history found, using target epochs for plot axis.")
elif actual_epochs_trained == 0:
    actual_epochs_trained = 1

epochs_axis = range(1, actual_epochs_trained + 1)

# MAE Plot (RGB)
if 'loss' in history_data:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, history_data['loss'], label='Training MAE (RGB Loss)')
    if val_data_to_pass and 'val_loss' in history_data:
        plt.plot(epochs_axis, history_data['val_loss'], label='Validation MAE (RGB Loss)')
    plt.title(f'MAE (RGB, Target: {TARGET_EPOCHS}, BS: {FIXED_BATCH_SIZE})')
    plt.xlabel('Epoch'); plt.ylabel('MAE (RGB space)'); plt.legend(); #plt.grid(True);
    plt.xlim(left=0, right=max(actual_epochs_trained, TARGET_EPOCHS)+1)
    plt.tight_layout()
    plt.savefig(MAE_PLOT_FILE); plt.close()
    print(f"MAE (RGB) plot saved to: {MAE_PLOT_FILE}")
else:
    print("Skipping MAE (RGB) plot: No 'loss' data in history.")

# MSE Plot (RGB)
if 'mse' in history_data:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, history_data['mse'], label='Training MSE (RGB)')
    if val_data_to_pass and 'val_mse' in history_data:
        plt.plot(epochs_axis, history_data['val_mse'], label='Validation MSE (RGB)')
    plt.title(f'MSE (RGB, Target: {TARGET_EPOCHS}, BS: {FIXED_BATCH_SIZE})')
    plt.xlabel('Epoch'); plt.ylabel('MSE (RGB space)'); plt.legend(); #plt.grid(True);
    plt.xlim(left=0, right=max(actual_epochs_trained, TARGET_EPOCHS)+1)
    plt.tight_layout()
    plt.savefig(MSE_PLOT_FILE); plt.close()
    print(f"MSE (RGB) plot saved to: {MSE_PLOT_FILE}")
else:
    print("Skipping MSE (RGB) plot: No 'mse' data in history.")

# RMSE Plot (Derived from RGB MSE)
if 'mse' in history_data:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_axis, np.sqrt(np.array(history_data['mse'])), label='Training RMSE (RGB, derived)')
    if val_data_to_pass and 'val_mse' in history_data:
        plt.plot(epochs_axis, np.sqrt(np.array(history_data['val_mse'])), label='Validation RMSE (RGB, derived)')
    plt.title(f'RMSE (RGB, Derived, Target: {TARGET_EPOCHS}, BS: {FIXED_BATCH_SIZE})')
    plt.xlabel('Epoch'); plt.ylabel('RMSE (RGB space)'); plt.legend(); #plt.grid(True);
    plt.xlim(left=0, right=max(actual_epochs_trained, TARGET_EPOCHS)+1)
    plt.tight_layout()
    plt.savefig(RMSE_PLOT_FILE); plt.close()
    print(f"RMSE (RGB) plot saved to: {RMSE_PLOT_FILE}")
else:
    print("Skipping RMSE (RGB) plot: No 'mse' data in history for derivation.")


# R2 Plot (Combined: Lines for pixel-wise, Text for per-image)
plt.figure(figsize=(12, 7))
epochs_trained_ref_r2 = max(1, actual_epochs_trained)

has_r2_line = False
if not np.isnan(test_set_metrics['r2_rgb']):
    plt.plot([1, epochs_trained_ref_r2], [test_set_metrics['r2_rgb'], test_set_metrics['r2_rgb']],
             color='r', linestyle='--', label=f'Test R² (Pixel RGB): {test_set_metrics["r2_rgb"]:.4f}')
    has_r2_line = True
if not np.isnan(test_set_metrics['r2_temp']):
     plt.plot([1, epochs_trained_ref_r2], [test_set_metrics['r2_temp'], test_set_metrics['r2_temp']],
              color='b', linestyle=':', label=f'Test R² (Pixel Temp): {test_set_metrics["r2_temp"]:.4f}')
     has_r2_line = True

text_y_start = 0.20
text_y_step = 0.10
text_x_pos = 0.98

if not np.isnan(test_set_metrics['r2_max_temp']):
    plt.text(text_x_pos, text_y_start, f'R² (Max Temp): {test_set_metrics["r2_max_temp"]:.4f}',
             ha='right', va='center', transform=plt.gca().transAxes, color='purple', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
else:
     plt.text(text_x_pos, text_y_start, 'R² (Max Temp): N/A',
              ha='right', va='center', transform=plt.gca().transAxes, color='grey', fontsize=12)
text_y_start -= text_y_step

if not np.isnan(test_set_metrics['r2_avg_temp']):
    plt.text(text_x_pos, text_y_start, f'R² (Avg Temp): {test_set_metrics["r2_avg_temp"]:.4f}',
             ha='right', va='center', transform=plt.gca().transAxes, color='green', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
else:
     plt.text(text_x_pos, text_y_start, 'R² (Avg Temp): N/A',
              ha='right', va='center', transform=plt.gca().transAxes, color='grey', fontsize=12)

if not has_r2_line and np.isnan(test_set_metrics['r2_max_temp']) and np.isnan(test_set_metrics['r2_avg_temp']):
     plt.text(0.5, 0.5, "Test R²: N/A", ha="center", va="center", transform=plt.gca().transAxes)

all_r2_values = [v for k, v in test_set_metrics.items() if 'r2' in k and not np.isnan(v)]
if all_r2_values:
    min_r2_val = min(all_r2_values) if all_r2_values else -0.1
    max_r2_val = max(all_r2_values) if all_r2_values else 1.1
    plt.ylim(min(-0.1, min_r2_val - 0.1), max(1.05, max_r2_val + 0.1))
else:
    plt.ylim(-0.1, 1.1)

plt.title(f'R² Scores (Test Set, Target: {TARGET_EPOCHS}, BS: {FIXED_BATCH_SIZE})')
plt.xlabel('Epoch (Reference for Test Value)')
plt.ylabel('R² Score')
if has_r2_line:
    plt.legend(loc='upper left', fontsize=12)
#plt.grid(True)
plt.xlim(left=0, right=max(epochs_trained_ref_r2, TARGET_EPOCHS)+1)
plt.tight_layout()
plt.savefig(R2_PLOT_FILE); plt.close()
print(f"R2 plot (test values) saved to: {R2_PLOT_FILE}")

# PARE Plot (Final Test Value)
plt.figure(figsize=(10, 6))
epochs_trained_ref_pare = max(1, actual_epochs_trained)

if not np.isnan(test_set_metrics['pare']):
    plt.plot([1, epochs_trained_ref_pare], [test_set_metrics['pare'], test_set_metrics['pare']],
             color='orange', linestyle='-.', linewidth=2, label=f'Test PARE: {test_set_metrics["pare"]:.4f}')
    plt.text(epochs_trained_ref_pare / 2, test_set_metrics['pare'],
             f'Test PARE: {test_set_metrics["pare"]:.4f}',
             color='orange', ha='center', va='bottom', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.ylim(bottom=0, top=max(0.1, test_set_metrics['pare'] * 1.5))
else:
    plt.text(0.5, 0.5, "Test PARE: N/A", ha="center", va="center", transform=plt.gca().transAxes)
    plt.ylim(0, 0.1)

plt.title(f'Pixel-wise Absolute Relative Error (PARE) (Test Set)')
plt.xlabel('Epoch (Reference for Test Value)')
plt.ylabel('PARE Score (Temperature space)')
if not np.isnan(test_set_metrics['pare']):
    plt.legend(loc='upper right')
#plt.grid(True)
plt.xlim(left=0, right=max(epochs_trained_ref_pare, TARGET_EPOCHS)+1)
plt.tight_layout()
plt.savefig(PARE_PLOT_FILE)
plt.close()
print(f"PARE plot (test value) saved to: {PARE_PLOT_FILE}")


# --- 新增：绘制最终测试集的温度 MAE ---
plt.figure(figsize=(10, 6))
epochs_trained_ref_temp_metrics = max(1, actual_epochs_trained)

if not np.isnan(test_set_metrics['mae_temp']):
    plt.plot([1, epochs_trained_ref_temp_metrics], [test_set_metrics['mae_temp'], test_set_metrics['mae_temp']],
             color='crimson', linestyle='--', linewidth=2, label=f"Test MAE (Temp): {test_set_metrics['mae_temp']:.4f} K")
    plt.text(epochs_trained_ref_temp_metrics / 2, test_set_metrics['mae_temp'],
             f"Test MAE (Temp): {test_set_metrics['mae_temp']:.4f} K",
             color='crimson', ha='center', va='bottom', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.ylim(bottom=0, top=max(1.0, test_set_metrics['mae_temp'] * 1.5 if not np.isnan(test_set_metrics['mae_temp']) else 1.0)) # Adjust y-axis
else:
    plt.text(0.5, 0.5, "Test MAE (Temp): N/A", ha="center", va="center", transform=plt.gca().transAxes)
    plt.ylim(0, 1) # Default y-limit

plt.title(f'Pixel-wise MAE (Temperature Space, Test Set)')
plt.xlabel('Epoch (Reference for Test Value)')
plt.ylabel('MAE (K)')
if not np.isnan(test_set_metrics['mae_temp']):
    plt.legend(loc='upper right')
#plt.grid(True)
plt.xlim(left=0, right=max(epochs_trained_ref_temp_metrics, TARGET_EPOCHS)+1)
plt.tight_layout()
plt.savefig(MAE_TEMP_PLOT_FILE)
plt.close()
print(f"Temperature MAE plot (test value) saved to: {MAE_TEMP_PLOT_FILE}")


# --- 新增：绘制最终测试集的温度 MSE ---
plt.figure(figsize=(10, 6))
if not np.isnan(test_set_metrics['mse_temp']):
    plt.plot([1, epochs_trained_ref_temp_metrics], [test_set_metrics['mse_temp'], test_set_metrics['mse_temp']],
             color='darkorange', linestyle='--', linewidth=2, label=f"Test MSE (Temp): {test_set_metrics['mse_temp']:.4f} K²")
    plt.text(epochs_trained_ref_temp_metrics / 2, test_set_metrics['mse_temp'],
             f"Test MSE (Temp): {test_set_metrics['mse_temp']:.4f} K²",
             color='darkorange', ha='center', va='bottom', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.ylim(bottom=0, top=max(1.0, test_set_metrics['mse_temp'] * 1.5 if not np.isnan(test_set_metrics['mse_temp']) else 1.0))
else:
    plt.text(0.5, 0.5, "Test MSE (Temp): N/A", ha="center", va="center", transform=plt.gca().transAxes)
    plt.ylim(0, 1)

plt.title(f'Pixel-wise MSE (Temperature Space, Test Set)')
plt.xlabel('Epoch (Reference for Test Value)')
plt.ylabel('MSE (K²)')
if not np.isnan(test_set_metrics['mse_temp']):
    plt.legend(loc='upper right')
#plt.grid(True)
plt.xlim(left=0, right=max(epochs_trained_ref_temp_metrics, TARGET_EPOCHS)+1)
plt.tight_layout()
plt.savefig(MSE_TEMP_PLOT_FILE)
plt.close()
print(f"Temperature MSE plot (test value) saved to: {MSE_TEMP_PLOT_FILE}")


# --- 新增：绘制最终测试集的温度 RMSE ---
plt.figure(figsize=(10, 6))
if not np.isnan(test_set_metrics['rmse_temp']):
    plt.plot([1, epochs_trained_ref_temp_metrics], [test_set_metrics['rmse_temp'], test_set_metrics['rmse_temp']],
             color='gold', linestyle='--', linewidth=2, label=f"Test RMSE (Temp): {test_set_metrics['rmse_temp']:.4f} K")
    plt.text(epochs_trained_ref_temp_metrics / 2, test_set_metrics['rmse_temp'],
             f"Test RMSE (Temp): {test_set_metrics['rmse_temp']:.4f} K",
             color='gold', ha='center', va='bottom', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.ylim(bottom=0, top=max(1.0, test_set_metrics['rmse_temp'] * 1.5 if not np.isnan(test_set_metrics['rmse_temp']) else 1.0))
else:
    plt.text(0.5, 0.5, "Test RMSE (Temp): N/A", ha="center", va="center", transform=plt.gca().transAxes)
    plt.ylim(0, 1)

plt.title(f'Pixel-wise RMSE (Temperature Space, Test Set)')
plt.xlabel('Epoch (Reference for Test Value)')
plt.ylabel('RMSE (K)')
if not np.isnan(test_set_metrics['rmse_temp']):
    plt.legend(loc='upper right')
#plt.grid(True)
plt.xlim(left=0, right=max(epochs_trained_ref_temp_metrics, TARGET_EPOCHS)+1)
plt.tight_layout()
plt.savefig(RMSE_TEMP_PLOT_FILE)
plt.close()
print(f"Temperature RMSE plot (test value) saved to: {RMSE_TEMP_PLOT_FILE}")

# --- 新增：绘制预测温度 vs 真实温度的散点图 (所有测试集像素) ---
print("\n--- Generating Predicted vs. True Temperature Scatter Plot ---")
if 'y_true_all_temps_flat' in locals() and 'y_pred_all_temps_flat' in locals() and \
   len(y_true_all_temps_flat) > 0 and len(y_pred_all_temps_flat) > 0:

    plt.figure(figsize=(10, 10)) # Square figure often looks good for these

    # 如果像素点过多，绘制散点图可能会非常慢且密集，可以考虑采样
    num_pixels_total = len(y_true_all_temps_flat)
    max_points_to_plot = 50000 # 最多绘制的点数，可调整
    if num_pixels_total > max_points_to_plot:
        print(f"Sampling {max_points_to_plot} points out of {num_pixels_total} for scatter plot.")
        indices = np.random.choice(num_pixels_total, max_points_to_plot, replace=False)
        plot_true_temps = y_true_all_temps_flat[indices]
        plot_pred_temps = y_pred_all_temps_flat[indices]
    else:
        plot_true_temps = y_true_all_temps_flat
        plot_pred_temps = y_pred_all_temps_flat

    plt.scatter(plot_pred_temps, plot_true_temps, alpha=0.1, s=5, label='Pixel Values') # s是点的大小, alpha是透明度

    # 绘制 y=x 对角线作为参考
    min_temp_val = GLOBAL_TEMP_MIN_K
    max_temp_val = GLOBAL_TEMP_MAX_K
    # 或者基于实际数据范围
    # min_val = min(np.min(plot_true_temps), np.min(plot_pred_temps))
    # max_val = max(np.max(plot_true_temps), np.max(plot_pred_temps))
    # min_temp_val = max(GLOBAL_TEMP_MIN_K, min_val - 50) # 加一点buffer
    # max_temp_val = min(GLOBAL_TEMP_MAX_K, max_val + 50) # 加一点buffer


    plt.plot([min_temp_val, max_temp_val], [min_temp_val, max_temp_val], 'r--', lw=2, label='Ideal (y=x)')

    plt.title(f'Predicted vs. True Temperature (Test Set Pixels)\nTarget: {TARGET_EPOCHS}, BS: {FIXED_BATCH_SIZE}')
    plt.xlabel('Predicted Temperature (K)')
    plt.ylabel('True Temperature (K)')
    plt.xlim(min_temp_val, max_temp_val)
    plt.ylim(min_temp_val, max_temp_val)
    plt.legend()
    #plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # 使x,y轴刻度等长，图像为正方形
    plt.tight_layout()
    try:
        plt.savefig(PREDICTED_VS_TRUE_TEMP_SCATTER_PLOT_FILE)
        print(f"Predicted vs. True Temperature scatter plot saved to: {PREDICTED_VS_TRUE_TEMP_SCATTER_PLOT_FILE}")
    except Exception as e_scatter:
        print(f"Error saving scatter plot: {e_scatter}")
    plt.close()
else:
    print("Skipping Predicted vs. True Temperature scatter plot: Data not available ('y_true_all_temps_flat' or 'y_pred_all_temps_flat').")



# --- 8. Visualize Temperature Analysis Maps for ALL Test Samples ---
if test_generator and eval_model_for_test_and_viz and len(test_generator.indexes) > 0:
    print(
        f"\n--- Generating Temperature Analysis Visualizations for ALL {len(test_generator.indexes)} Test Samples ---")
    os.makedirs(ERROR_MAPS_VIS_DIR, exist_ok=True) # Ensure output dir exists

    for sample_idx_vis in range(len(test_generator.indexes)):
        original_file_list_idx = test_generator.indexes[sample_idx_vis]
        # Double check index validity against the actual file list length
        if original_file_list_idx >= len(test_generator.path_files):
            print(f"  Warning: Skipping visualization for original index {original_file_list_idx} - out of bounds for path_files list.")
            continue

        input_path_file_vis = test_generator.path_files[original_file_list_idx]
        true_tf_file_vis = test_generator.tf_files[original_file_list_idx]
        base_filename_vis = os.path.basename(input_path_file_vis)

        if sample_idx_vis % 50 == 0: # Print progress periodically
            print(f"  Processing visualization for: {base_filename_vis} ({sample_idx_vis + 1}/{len(test_generator.indexes)})")

        try:
            # Load images specifically for visualization
            input_img_display = load_and_preprocess_image(input_path_file_vis, is_temp_field=False, for_visualization=True)  # uint8 input
            input_img_pred_for_model = load_and_preprocess_image(input_path_file_vis, is_temp_field=False, for_visualization=False) # float32 input for model
            true_tf_img_rgb_vis = load_and_preprocess_image(true_tf_file_vis, is_temp_field=True, for_visualization=True)  # float32 [0,1] true TF

            # Basic validity check (if load failed, array might be all zeros)
            if np.count_nonzero(input_img_display) == 0 and not (os.path.exists(input_path_file_vis) and Image.open(input_path_file_vis).getbbox()):
                 print(f"    Skipping {base_filename_vis}: Input image appears empty or non-existent after loading.")
                 continue
            if np.count_nonzero(true_tf_img_rgb_vis) == 0 and not (os.path.exists(true_tf_file_vis) and Image.open(true_tf_file_vis).getbbox()):
                 print(f"    Skipping {base_filename_vis}: True TF image appears empty or non-existent after loading.")
                 continue


            # Model prediction
            predicted_tf_img_rgb_vis = eval_model_for_test_and_viz.predict(np.expand_dims(input_img_pred_for_model, axis=0), verbose=0)[0]

            # Temperature Conversion
            true_temp_map = rgb_to_temperature_approx(true_tf_img_rgb_vis, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
            predicted_temp_map = rgb_to_temperature_approx(predicted_tf_img_rgb_vis, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
            temperature_error_map = predicted_temp_map - true_temp_map
            abs_temperature_error_map = np.abs(temperature_error_map)

            # Plotting
            plt.ioff() # Turn interactive mode off for saving figures
            fig, axes = plt.subplots(2, 2, figsize=(12, 11))
            fig.suptitle(f"Temperature Analysis: {base_filename_vis}", fontsize=16)

            # 1: Input Path
            axes[0, 0].imshow(input_img_display)
            axes[0, 0].set_title('Input Path'); axes[0, 0].axis('off')

            # 2: Predicted Temperature
            im_pred_temp = axes[0, 1].imshow(predicted_temp_map, cmap=CMAP_NAME, vmin=GLOBAL_TEMP_MIN_K, vmax=GLOBAL_TEMP_MAX_K)
            axes[0, 1].set_title('Predicted Temperature (K)')
            axes[0, 1].axis('off')
            fig.colorbar(im_pred_temp, ax=axes[0, 1], label='Temperature (K)', fraction=0.046, pad=0.04)

            # 3: Ground Truth Temperature
            im_true_temp = axes[1, 0].imshow(true_temp_map, cmap=CMAP_NAME, vmin=GLOBAL_TEMP_MIN_K, vmax=GLOBAL_TEMP_MAX_K)
            axes[1, 0].set_title('Ground Truth Temperature (K)')
            axes[1, 0].axis('off')
            fig.colorbar(im_true_temp, ax=axes[1, 0], label='Temperature (K)', fraction=0.046, pad=0.04)

            # 4: Absolute Temperature Error
            current_max_abs_error = np.max(abs_temperature_error_map)
            vmax_for_error_plot = max(current_max_abs_error, 1e-6) # Avoid zero vmax for colorbar
            im_err = axes[1, 1].imshow(abs_temperature_error_map, cmap='viridis', vmin=0, vmax=vmax_for_error_plot) # Use different cmap for error
            axes[1, 1].set_title('Absolute Temperature Error (K)')
            axes[1, 1].axis('off')
            fig.colorbar(im_err, ax=axes[1, 1], label=f'Abs. Temp Error (K)\nmax: {current_max_abs_error:.2f}K', fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            save_path_vis = os.path.join(ERROR_MAPS_VIS_DIR, f"temperature_analysis_{base_filename_vis}")
            plt.savefig(save_path_vis)
            plt.close(fig) # Close the figure to free memory

        except Exception as e_vis:
            print(f"  Error generating temperature analysis map for {base_filename_vis}: {e_vis}")
            traceback.print_exc() # Print detailed error traceback
else:
    print("Skipping temperature analysis visualization: Test generator or model unavailable, or test set empty.")


# --- 9. Generate Histogram of True Temperatures (All Pixels in Training Set) ---
print("\n--- Generating Histogram of True Temperatures (All Pixels) for Training Set ---")
if train_generator and len(train_generator.indexes) > 0:
    all_true_temperatures_train = []
    num_train_batches = len(train_generator)
    print(f"Processing {len(train_generator.indexes)} training samples in {num_train_batches} batches for temperature histogram...")

    for batch_idx in range(num_train_batches):
        if batch_idx > 0 and batch_idx % (max(1, num_train_batches // 5)) == 0:
            print(f"  Histogram processing: Batch {batch_idx}/{num_train_batches}...")
        try:
            _, y_batch_true_rgb_train = train_generator[batch_idx]
            if y_batch_true_rgb_train.shape[0] == 0: continue

            for i in range(y_batch_true_rgb_train.shape[0]):
                true_rgb_image_train = y_batch_true_rgb_train[i]
                true_temp_map_train = rgb_to_temperature_approx(true_rgb_image_train, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
                all_true_temperatures_train.extend(true_temp_map_train.flatten()) # Collect all pixel values
        except Exception as e_hist_batch:
            print(f"  Warning: Error processing batch {batch_idx} for pixel histogram: {e_hist_batch}")

    if all_true_temperatures_train:
        all_true_temperatures_train = np.array(all_true_temperatures_train)
        print(f"Collected {len(all_true_temperatures_train)} temperature points from the training set.")

        # Plot and save histogram
        TRAIN_TEMP_HIST_FILE = os.path.join(RESULTS_VIS_DIR, f'training_set_all_pixels_temperature_histogram_ep{TARGET_EPOCHS}_bs{FIXED_BATCH_SIZE}.png')
        plt.figure(figsize=(12, 7))
        plt.hist(all_true_temperatures_train, bins=100, color='skyblue', edgecolor='black', range=(GLOBAL_TEMP_MIN_K, GLOBAL_TEMP_MAX_K)) # More bins for pixel-level
        plt.title(f'Distribution of All Pixel Temperatures in Training Set')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Frequency (Number of Pixels)')
        #plt.grid(axis='y', alpha=0.75)
        plt.xlim(GLOBAL_TEMP_MIN_K, GLOBAL_TEMP_MAX_K)
        plt.yscale('log') # Use log scale for y-axis due to potentially large frequency differences
        plt.tight_layout()
        try:
            plt.savefig(TRAIN_TEMP_HIST_FILE)
            print(f"Training set all-pixel temperature histogram saved to: {TRAIN_TEMP_HIST_FILE}")
        except Exception as e_save_hist:
            print(f"Error saving training set all-pixel temperature histogram: {e_save_hist}")
        plt.close()
    else:
        print("No temperature data collected from the training set to generate all-pixel histogram.")
else:
    print("Skipping training set all-pixel temperature histogram: Train generator unavailable or empty.")


# --- 10. Generate Histogram of MAX Temperatures PER IMAGE for Datasets ---
print("\n--- Generating Max Temperature (per image) Histograms for Datasets ---")

# Use bins=20 as requested
hist_bins = 20

# Training Set
generate_max_temperature_histogram_per_image(
    generator=train_generator, dataset_name_str="Training", output_base_dir=RESULTS_VIS_DIR,
    rgb_lookup_table=RGB_LOOKUP_TABLE, temp_lookup_table=TEMP_LOOKUP_TABLE,
    global_min_k=GLOBAL_TEMP_MIN_K, global_max_k=GLOBAL_TEMP_MAX_K,
    target_epochs=TARGET_EPOCHS, batch_size_config=FIXED_BATCH_SIZE, bins=hist_bins
)

# Validation Set
if validation_generator: # Check if validation_generator exists
    generate_max_temperature_histogram_per_image(
        generator=validation_generator, dataset_name_str="Validation", output_base_dir=RESULTS_VIS_DIR,
        rgb_lookup_table=RGB_LOOKUP_TABLE, temp_lookup_table=TEMP_LOOKUP_TABLE,
        global_min_k=GLOBAL_TEMP_MIN_K, global_max_k=GLOBAL_TEMP_MAX_K,
        target_epochs=TARGET_EPOCHS, batch_size_config=FIXED_BATCH_SIZE, bins=hist_bins
    )
else:
    print("Skipping Validation set max temperature histogram: Validation generator not available.")


# Test Set
if test_generator: # Check if test_generator exists
    generate_max_temperature_histogram_per_image(
        generator=test_generator, dataset_name_str="Test", output_base_dir=RESULTS_VIS_DIR,
        rgb_lookup_table=RGB_LOOKUP_TABLE, temp_lookup_table=TEMP_LOOKUP_TABLE,
        global_min_k=GLOBAL_TEMP_MIN_K, global_max_k=GLOBAL_TEMP_MAX_K,
        target_epochs=TARGET_EPOCHS, batch_size_config=FIXED_BATCH_SIZE, bins=hist_bins
    )
else:
    print("Skipping Test set max temperature histogram: Test generator not available.")


# --- 11. Save Summary Metrics to CSV ---
print("\n--- Saving Summary Metrics to CSV ---")
actual_epochs_trained_for_csv = len(history_data.get('loss', []))
if actual_epochs_trained_for_csv == 0 and TARGET_EPOCHS > 0:
    actual_epochs_trained_for_csv = TARGET_EPOCHS # Fallback if no history
elif actual_epochs_trained_for_csv == 0:
    actual_epochs_trained_for_csv = 'N/A'


config_details = {
    'target_epochs': TARGET_EPOCHS,
    'actual_epochs': actual_epochs_trained_for_csv,
    'batch_size': FIXED_BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'model_path': MODEL_SAVE_PATH_SINGLE_RUN if os.path.exists(MODEL_SAVE_PATH_SINGLE_RUN) else "N/A (Not Saved or Default)"
}
save_metrics_to_csv(RESULTS_SUMMARY_CSV, config_details, test_set_metrics)


# --- Final Summary ---
print(f"\n--- Run Finished ---")
print(f"Target Epochs: {TARGET_EPOCHS}, Batch Size: {FIXED_BATCH_SIZE}")
print(f"Results and visualizations saved in: {RESULTS_VIS_DIR}")
if os.path.exists(MODEL_SAVE_PATH_SINGLE_RUN): print(f"Best model saved to: {MODEL_SAVE_PATH_SINGLE_RUN}")
if os.path.exists(CUMULATIVE_TIMES_FILE): print(f"Cumulative training times log: {CUMULATIVE_TIMES_FILE}")
if os.path.exists(PREDICTED_TEMPS_FILE): print(f"Predicted max temperatures (test set): {PREDICTED_TEMPS_FILE}")
print("\nFinal Test Set Metrics:")
for metric, value in test_set_metrics.items():
    if not np.isnan(value):
        unit = ""
        if "temp" in metric.lower() or "pare" in metric.lower():
            if "mae" in metric.lower() or "rmse" in metric.lower(): unit = " K"
            elif "mse" in metric.lower(): unit = " K²"
        print(f"  - {metric.replace('_', ' ').title()}: {value:.4f}{unit}")
    else:
        print(f"  - {metric.replace('_', ' ').title()}: N/A")

print(f"Summary metrics CSV: {RESULTS_SUMMARY_CSV}")
print("--- End of Script ---")