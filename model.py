import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import segmentation_models as sm

BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Open the large image and mask
img = Image.open("soil-erosion/T36UXV_20200406T083559_TCI_10m.jp2")
mask = Image.open("soil-erosion/train/train.jp2")

# Define the patch size
patch_size = 256

# Loop over the image and mask, extracting patches
patches = []
mask_patches = []
for i in range(0, img.width, patch_size):
    for j in range(0, img.height, patch_size):
        # Extract the patch from the image and mask
        patch = img.crop((i, j, i+patch_size, j+patch_size))
        mask_patch = mask.crop((i, j, i+patch_size, j+patch_size))

        # Convert the patches to numpy arrays
        patch = np.array(patch)
        mask_patch = np.array(mask_patch)

        # Append the patches to the list
        patches.append(patch)
        mask_patches.append(mask_patch)

# Convert the patches to a numpy array
patches = np.array(patches)
mask_patches = np.array(mask_patches).astype("float32")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(patches, mask_patches, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

# Preprocess input
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(optimizer="adam", loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

print(model.summary())

history = model.fit(x_train,
                    y_train,
                    batch_size=16,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))
