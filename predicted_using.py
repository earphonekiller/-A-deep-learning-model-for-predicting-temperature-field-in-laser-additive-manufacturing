import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # 如果需要重新构建模型
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.colors as mcolors # 如果需要温度转换
from scipy.spatial.distance import cdist # 如果需要温度转换
import csv # 确保导入csv

# --- 0. 配置 (与训练时保持一致或根据预测需求调整) ---
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS_IN = 3  # 输入图像的通道数
IMG_CHANNELS_OUT = 3 # 模型输出图像的通道数 (例如RGB)

# --- 路径配置 ---
# 输入数据路径
INPUT_DATA_BASE_PATH = r'D:\data_sets\general'
PREDICT_INPUT_DIR = os.path.join(INPUT_DATA_BASE_PATH, 'test_PATH')
GROUND_TRUTH_DIR = os.path.join(INPUT_DATA_BASE_PATH, 'test_TF')

# 输出结果路径
OUTPUT_RESULTS_BASE_PATH = r'D:\data_sets\epoches_results\1600_predictions_output'
PREDICTED_TF_SUBDIR = 'predicted_TF_rgb'
ERROR_MAPS_SUBDIR = 'error_maps_temp_visualization' # 修改一下名字，因为现在是温度误差
CSV_FILENAME = 'predicted_max_temperatures.csv'

PREDICT_OUTPUT_DIR = os.path.join(OUTPUT_RESULTS_BASE_PATH, PREDICTED_TF_SUBDIR)
ERROR_MAP_OUTPUT_DIR = os.path.join(OUTPUT_RESULTS_BASE_PATH, ERROR_MAPS_SUBDIR)
MAX_TEMPS_CSV_FILE = os.path.join(OUTPUT_RESULTS_BASE_PATH, CSV_FILENAME)

# 温度转换相关
T_AMBIENT_K = 298.0
GLOBAL_TEMP_MIN_K = T_AMBIENT_K
GLOBAL_TEMP_MAX_K = 5500.0
CMAP_NAME = 'turbo'
CMAP_RESOLUTION = 256
TEMP_LOOKUP_TABLE = None
RGB_LOOKUP_TABLE = None

# --- 1. 定义必要的辅助函数 ---
def build_unet_model(input_shape, num_classes=3):
    inputs = keras.Input(shape=input_shape)
    c1=layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(inputs); c1=layers.Dropout(0.1)(c1); c1=layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1); p1=layers.MaxPooling2D((2,2))(c1)
    c2=layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p1); c2=layers.Dropout(0.1)(c2); c2=layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2); p2=layers.MaxPooling2D((2,2))(c2)
    c3=layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2); c3=layers.Dropout(0.2)(c3); c3=layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3); p3=layers.MaxPooling2D((2,2))(c3)
    c4=layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3); c4=layers.Dropout(0.2)(c4); c4=layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4); p4=layers.MaxPooling2D((2,2))(c4)
    c5=layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p4); c5=layers.Dropout(0.3)(c5); c5=layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)
    u6=layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5); u6=layers.concatenate([u6,c4]); c6=layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u6); c6=layers.Dropout(0.2)(c6); c6=layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)
    u7=layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6); u7=layers.concatenate([u7,c3]); c7=layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u7); c7=layers.Dropout(0.2)(c7); c7=layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c7)
    u8=layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7); u8=layers.concatenate([u8,c2]); c8=layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8); c8=layers.Dropout(0.1)(c8); c8=layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c8)
    u9=layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8); u9=layers.concatenate([u9,c1]); c9=layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9); c9=layers.Dropout(0.1)(c9); c9=layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c9)
    outputs=layers.Conv2D(num_classes,(1,1),activation='sigmoid')(c9)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_and_preprocess_image(path, is_temp_field=False, for_visualization=False):
    global IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS_IN, IMG_CHANNELS_OUT
    try:
        img_pil = Image.open(path).convert('RGB')
        img_pil = img_pil.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
        img_array_orig = np.array(img_pil)

        if for_visualization and not is_temp_field:
            return img_array_orig

        img_array_float = np.array(img_pil, dtype=np.float32) / 255.0

        if for_visualization and is_temp_field:
             return img_array_float
        return img_array_float
    except Exception as e:
        print(f"Warning: Error loading image {path}: {e}")
        if for_visualization and not is_temp_field:
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_IN), dtype=np.uint8)
        num_channels = IMG_CHANNELS_OUT if is_temp_field else IMG_CHANNELS_IN
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, num_channels), dtype=np.float32)

def initialize_temperature_colormap_lookup():
    global TEMP_LOOKUP_TABLE, RGB_LOOKUP_TABLE
    if TEMP_LOOKUP_TABLE is not None and RGB_LOOKUP_TABLE is not None: return
    print("Initializing temperature-RGB lookup table...")
    TEMP_LOOKUP_TABLE = np.linspace(GLOBAL_TEMP_MIN_K, GLOBAL_TEMP_MAX_K, CMAP_RESOLUTION)
    try:
        cmap = plt.get_cmap(CMAP_NAME)
    except ValueError:
        print(f"Error: Colormap '{CMAP_NAME}' not found. Using 'viridis' as fallback.")
        cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=GLOBAL_TEMP_MIN_K, vmax=GLOBAL_TEMP_MAX_K)
    normalized_temps = norm(TEMP_LOOKUP_TABLE)
    RGB_LOOKUP_TABLE = cmap(normalized_temps)[:, :3]
    print(f"Lookup table initialized.")

def rgb_to_temperature_approx(rgb_pixel_array, rgb_lookup, temp_lookup):
    if rgb_lookup is None or temp_lookup is None:
        raise ValueError("Lookup tables not initialized.")
    original_shape = rgb_pixel_array.shape
    if rgb_pixel_array.ndim == 3: h, w, c = original_shape; rgb_pixel_flat = rgb_pixel_array.reshape(-1, 3)
    elif rgb_pixel_array.ndim == 2 and original_shape[1] == 3: rgb_pixel_flat = rgb_pixel_array
    elif rgb_pixel_array.ndim == 1 and original_shape[0] == 3: rgb_pixel_flat = rgb_pixel_array.reshape(1,3)
    else: raise ValueError(f"Unsupported rgb_pixel_array shape: {original_shape}")
    distances = cdist(rgb_pixel_flat, rgb_lookup, metric='euclidean')
    closest_indices = np.argmin(distances, axis=1)
    temperatures_flat = temp_lookup[closest_indices]
    if rgb_pixel_array.ndim == 3: return temperatures_flat.reshape(h, w)
    else: return temperatures_flat

# --- 2. 加载训练好的模型 ---
MODEL_PATH = r'D:\data_sets\1600\unet_rgb_ep50_bs16.keras'

print(f"Loading model from: {MODEL_PATH}")
try:
    loaded_model = keras.models.load_model(MODEL_PATH, compile=False)
    loaded_model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. 准备预测数据 ---
if not os.path.exists(OUTPUT_RESULTS_BASE_PATH):
    os.makedirs(OUTPUT_RESULTS_BASE_PATH)
if not os.path.exists(PREDICT_OUTPUT_DIR):
    os.makedirs(PREDICT_OUTPUT_DIR)
if not os.path.exists(ERROR_MAP_OUTPUT_DIR):
    os.makedirs(ERROR_MAP_OUTPUT_DIR)

if not os.path.isdir(PREDICT_INPUT_DIR):
    print(f"Error: Input directory for prediction not found at {PREDICT_INPUT_DIR}")
elif not os.path.isdir(GROUND_TRUTH_DIR):
    print(f"Error: Ground truth directory not found at {GROUND_TRUTH_DIR}. Cannot compute error maps.")
else:
    print(f"\nPredicting for all images in directory: {PREDICT_INPUT_DIR}")
    input_image_files = [os.path.join(PREDICT_INPUT_DIR, f)
                         for f in os.listdir(PREDICT_INPUT_DIR)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not input_image_files:
        print(f"No images found in {PREDICT_INPUT_DIR}")
    else:
        if RGB_LOOKUP_TABLE is None or TEMP_LOOKUP_TABLE is None:
            initialize_temperature_colormap_lookup()

        max_temps_data = []

        for img_path in input_image_files: ####### FOR LOOP START #######
            base_filename = os.path.basename(img_path)
            true_tf_path = os.path.join(GROUND_TRUTH_DIR, base_filename)

            print(f"  Processing: {base_filename}")

            if not os.path.exists(true_tf_path):
                print(f"    Warning: Ground truth file {true_tf_path} not found. Skipping error map and temp for this image.")
                input_img_for_pred_only = load_and_preprocess_image(img_path, is_temp_field=False, for_visualization=False)
                if np.count_nonzero(input_img_for_pred_only) == 0 and not Image.open(img_path).getbbox():
                    print(f"    Skipping {base_filename} as it appears to be an empty/all-black image (input path).")
                    continue
                input_batch_only = np.expand_dims(input_img_for_pred_only, axis=0)
                predicted_output_batch_only = loaded_model.predict(input_batch_only, verbose=0)
                predicted_image_rgb_only = predicted_output_batch_only[0]
                predicted_image_to_save_only = (predicted_image_rgb_only * 255).astype(np.uint8)
                pil_img_to_save_only = Image.fromarray(predicted_image_to_save_only)
                save_path_only = os.path.join(PREDICT_OUTPUT_DIR, f"predicted_{base_filename}")
                pil_img_to_save_only.save(save_path_only)
                print(f"    Predicted RGB image saved to: {save_path_only} (no ground truth for error/temp analysis)")
                continue

            ##### MAIN TRY BLOCK FOR SINGLE FILE PROCESSING START #####
            try:
                # --- 1. 加载图像 ---
                input_img_display = load_and_preprocess_image(img_path, is_temp_field=False, for_visualization=True)
                input_img_pred_for_model = load_and_preprocess_image(img_path, is_temp_field=False, for_visualization=False)
                true_tf_img_vis = load_and_preprocess_image(true_tf_path, is_temp_field=True, for_visualization=True)

                valid_input_display = input_img_display.size > 0 and (np.count_nonzero(input_img_display) > 0 or (Image.open(img_path).getbbox() is not None))
                valid_input_pred = input_img_pred_for_model.size > 0 and (np.count_nonzero(input_img_pred_for_model) > 0 or (Image.open(img_path).getbbox() is not None))
                valid_true_tf = true_tf_img_vis.size > 0 and (np.count_nonzero(true_tf_img_vis) > 0 or (Image.open(true_tf_path).getbbox() is not None))

                if not (valid_input_display and valid_input_pred and valid_true_tf):
                    print(f"    Skipping {base_filename} due to image loading error (one or more images are effectively empty/black).")
                    continue # Skip to next image in the for loop

                # --- 2. 模型预测 ---
                input_batch = np.expand_dims(input_img_pred_for_model, axis=0)
                predicted_output_batch = loaded_model.predict(input_batch, verbose=0)
                predicted_tf_img_vis = predicted_output_batch[0]

                # --- 保存预测的RGB图像 ---
                predicted_image_to_save = (predicted_tf_img_vis * 255).astype(np.uint8)
                pil_img_to_save = Image.fromarray(predicted_image_to_save)
                save_path_rgb = os.path.join(PREDICT_OUTPUT_DIR, f"predicted_{base_filename}")
                pil_img_to_save.save(save_path_rgb)
                print(f"    Predicted RGB image saved to: {save_path_rgb}")

                # --- 计算并记录最大预测温度 ---
                if RGB_LOOKUP_TABLE is not None and TEMP_LOOKUP_TABLE is not None:
                    predicted_temp_map_for_max = rgb_to_temperature_approx(predicted_tf_img_vis, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
                    max_predicted_temp = np.max(predicted_temp_map_for_max)
                    max_temps_data.append({'filename': base_filename, 'max_predicted_temp_K': f"{max_predicted_temp:.2f}"})
                    print(f"    Max predicted temperature: {max_predicted_temp:.2f} K")
                else:
                    print("    Skipping max temperature calculation as lookup tables are not initialized.")

                # --- 计算温度图和温度误差 ---
                true_temp_map = None
                predicted_temp_map = None
                temperature_error_map = None
                abs_temperature_error_map = None

                if RGB_LOOKUP_TABLE is not None and TEMP_LOOKUP_TABLE is not None:
                    true_temp_map = rgb_to_temperature_approx(true_tf_img_vis, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
                    predicted_temp_map = rgb_to_temperature_approx(predicted_tf_img_vis, RGB_LOOKUP_TABLE, TEMP_LOOKUP_TABLE)
                    temperature_error_map = predicted_temp_map - true_temp_map
                    abs_temperature_error_map = np.abs(temperature_error_map)
                else:
                    print("    Skipping temperature error calculation as lookup tables are not initialized.")

                # --- 绘图部分 (2x2布局，显示温度和温度误差) ---
                plt.ioff()
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Prediction and Temperature Error: {base_filename}", fontsize=16)

                # 子图1: 输入路径图
                axes[0, 0].imshow(input_img_display)
                axes[0, 0].set_title('Input Path'); axes[0, 0].axis('off')

                # 子图2: 预测的温度场
                if predicted_temp_map is not None:
                    im_pred_temp = axes[0, 1].imshow(predicted_temp_map, cmap=CMAP_NAME, vmin=GLOBAL_TEMP_MIN_K, vmax=GLOBAL_TEMP_MAX_K)
                    axes[0, 1].set_title('Predicted Temperature (K)')
                    fig.colorbar(im_pred_temp, ax=axes[0, 1], label='Temperature (K)', fraction=0.046, pad=0.04)
                else:
                    axes[0, 1].text(0.5, 0.5, 'Pred Temp N/A', ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0, 1].axis('off')

                # 子图3: 真实的温度场
                if true_temp_map is not None:
                    im_true_temp = axes[1, 0].imshow(true_temp_map, cmap=CMAP_NAME, vmin=GLOBAL_TEMP_MIN_K, vmax=GLOBAL_TEMP_MAX_K)
                    axes[1, 0].set_title('Ground Truth Temperature (K)')
                    fig.colorbar(im_true_temp, ax=axes[1, 0], label='Temperature (K)', fraction=0.046, pad=0.04)
                else:
                    axes[1, 0].text(0.5, 0.5, 'True Temp N/A', ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1, 0].axis('off')

                # 子图4: 绝对温度误差图
                if abs_temperature_error_map is not None:
                    # --- 修改开始 ---
                    # 如果希望误差图的颜色条与温度图的范围和颜色映射一致
                    # 警告：这可能不是信息最丰富的可视化方式，因为误差的范围通常远小于温度范围
                    im_err = axes[1, 1].imshow(abs_temperature_error_map,
                                               cmap=CMAP_NAME,  # <--- 修改这里为 CMAP_NAME ('turbo')
                                               vmin=0,  # 绝对误差从0开始
                                               vmax=np.max(abs_temperature_error_map)  # <--- 修改这里，将vmax设置为全局最大温度
                                               # 或者，如果你想让vmax是误差的最大值，但仍用turbo，则保留：
                                               # vmax=np.max(abs_temperature_error_map) or 0.1
                                               )
                    axes[1, 1].set_title('Absolute Temperature Error (K)')
                    # 颜色条的标签仍然是误差，但颜色映射现在是 'turbo'
                    fig.colorbar(im_err, ax=axes[1, 1], label='Abs. Temperature Error (K) - Turbo Colormap',
                                 fraction=0.046, pad=0.04)
                    # --- 修改结束 ---
                else:
                    axes[1, 1].text(0.5, 0.5, 'Temp Error N/A', ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1, 1].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                error_map_save_path = os.path.join(ERROR_MAP_OUTPUT_DIR, f"temp_error_visualization_{base_filename}")
                fig.savefig(error_map_save_path)
                plt.close(fig)
                print(f"    Temperature error map visualization saved to: {error_map_save_path}")

            ##### MAIN TRY BLOCK FOR SINGLE FILE PROCESSING END #####
            except Exception as e_vis_single: # 这个 except 对应上面的 try
                print(f"    Error processing file {base_filename}: {e_vis_single}")
                import traceback
                traceback.print_exc()
        ####### FOR LOOP END #######

        # --- 在循环结束后保存最大温度到CSV ---
        if max_temps_data:
            try:
                with open(MAX_TEMPS_CSV_FILE, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'max_predicted_temp_K'])
                    writer.writeheader()
                    writer.writerows(max_temps_data)
                print(f"\nPredicted max temperatures saved to {MAX_TEMPS_CSV_FILE}")
            except IOError as e_csv:
                print(f"Error writing predicted max temperatures to CSV: {e_csv}")

print("\n--- Prediction Script Finished ---")