# -----------------------------------
# GLOBAL FEATURE EXTRACTION
# -----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

# --------------------
# tunable-parameters
# --------------------
images_per_class = 80  # Jumlah gambar per kelas
fixed_size = tuple((500, 500))  # Ukuran gambar yang telah ditetapkan
train_path = "dataset/train"  # Path untuk dataset latihan
h5_data = 'output/data.h5'  # File untuk menyimpan data fitur
h5_labels = 'output/labels.h5'  # File untuk menyimpan label
bins = 8  # Jumlah bin untuk histogram warna

# deskriptor-fitur-1: Hu Moments


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# deskriptor-fitur-2: Haralick Texture


def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# deskriptor-fitur-3: Histogram Warna


def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# Mengambil label pelatihan dari direktori train_path
train_labels = os.listdir(train_path)

# Mengurutkan label pelatihan
train_labels.sort()
print(train_labels)

# List kosong untuk menyimpan vektor fitur global dan label
global_features = []
labels = []

# Perulangan untuk setiap sub-folder data latihan
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name

    # Perulangan untuk setiap gambar di setiap sub-folder
    for x in range(1, images_per_class + 1):
        file = dir + "/" + str(x) + ".jpg"
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Ekstraksi fitur global
        ####################################

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        ###################################
        # Menggabungkan fitur global
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# Mendapatkan ukuran vektor fitur secara keseluruhan
print("[STATUS] feature vector size {}".format(
    np.array(global_features).shape))

# Mendapatkan ukuran label pelatihan secara keseluruhan
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# Mengkodekan label target
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# Penskalaan fitur dalam rentang (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# Menyimpan vektor fitur menggunakan HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")
