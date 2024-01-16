import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import concurrent.futures
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib

def get_filepaths(dir_path):
    directory_path = dir_path
    all_files = os.listdir(directory_path)
    mp3_files = [os.path.join(directory_path, file) for file in all_files if file.endswith('.mp3')]
    audio_data_list = []
    for mp3 in mp3_files:
        audio_data_list.append(mp3)
    return audio_data_list

def load_files(audio_data_list):
    loaded_data_list = []
    for mp3 in audio_data_list:
        y, sr = librosa.load(mp3)
        loaded_data_list.append((y, sr))
    return loaded_data_list

def calculate_audio_features(y, sr):
    # Calculate Chroma Short-Time Fourier Transform (STFT)
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))

    # Calculate Root Mean Square (RMS)
    rms_mean = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))

    # Calculate Spectral Centroid
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Calculate Spectral Bandwidth
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Calculate Rolloff
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Calculate Zero Crossing Rate
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y=y))

    # Calculate Harmony
    harmony = librosa.effects.harmonic(y=y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)

    # Calculate Perceptr
    perceptr = librosa.effects.percussive(y=y)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)

    # Calculate Tempo
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]

    # Calculate MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Calculate 20 MFCCs
    mfcc_means = np.mean(mfccs, axis=1)  # Calculate means of MFCCs
    mfcc_vars = np.var(mfccs, axis=1)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Calculate mean and variance for each MFCC coefficient
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_variances = np.var(mfccs, axis=1)

    # Assign values to individual variables
    mfcc1_mean, mfcc2_mean, mfcc3_mean, mfcc4_mean, mfcc5_mean, \
    mfcc6_mean, mfcc7_mean, mfcc8_mean, mfcc9_mean, mfcc10_mean, \
    mfcc11_mean, mfcc12_mean, mfcc13_mean, mfcc14_mean, mfcc15_mean, \
    mfcc16_mean, mfcc17_mean, mfcc18_mean, mfcc19_mean, mfcc20_mean = mfcc_means

    mfcc1_var, mfcc2_var, mfcc3_var, mfcc4_var, mfcc5_var, \
    mfcc6_var, mfcc7_var, mfcc8_var, mfcc9_var, mfcc10_var, \
    mfcc11_var, mfcc12_var, mfcc13_var, mfcc14_var, mfcc15_var, \
    mfcc16_var, mfcc17_var, mfcc18_var, mfcc19_var, mfcc20_var = mfcc_variances

    return (chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
            spectral_centroid_mean, spectral_centroid_var,
            spectral_bandwidth_mean, spectral_bandwidth_var,
            rolloff_mean, rolloff_var,
            zero_crossing_rate_mean, zero_crossing_rate_var,
            harmony_mean, harmony_var,
            perceptr_mean, perceptr_var, tempo,
            mfcc1_mean, mfcc1_var,
            mfcc2_mean, mfcc2_var,
            mfcc3_mean, mfcc3_var,
            mfcc4_mean, mfcc4_var,
            mfcc5_mean, mfcc5_var,
            mfcc6_mean, mfcc6_var,
            mfcc7_mean, mfcc7_var,
            mfcc8_mean, mfcc8_var,
            mfcc9_mean, mfcc9_var,
            mfcc10_mean, mfcc10_var,
            mfcc11_mean, mfcc11_var,
            mfcc12_mean, mfcc12_var,
            mfcc13_mean, mfcc13_var,
            mfcc14_mean, mfcc14_var,
            mfcc15_mean, mfcc15_var,
            mfcc16_mean, mfcc16_var,
            mfcc17_mean, mfcc17_var,
            mfcc18_mean, mfcc18_var,
            mfcc19_mean, mfcc19_var,
            mfcc20_mean, mfcc20_var
            )

def calculate_audio_features_wrapper(args):
    return calculate_audio_features(*args)

def extract_feature_matrix(dir_path, cache_file):
    if os.path.exists(cache_file):
        print("Loading feature matrix from cache...")
        feature_matrix = joblib.load(cache_file)
        return feature_matrix
    
    audio_data_list = get_filepaths(dir_path)

    feature_matrix = np.empty((len(audio_data_list), 57))  # num_features is the number of features you have

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(calculate_audio_features_wrapper, (y, sr)) for _,y, sr in audio_data_list]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            features = future.result()
            feature_matrix[i, :] = features
            print('Feature calculated')


    feature_matrix = np.array(feature_matrix)
    print('Feature Calculation finished!')
    return feature_matrix

def extract_feature_matrix_batch(dir_path, cache_file, batch_size=10):

    if os.path.exists(cache_file):
        print("Loading feature matrix from cache...")
        feature_matrix = joblib.load(cache_file)
        return feature_matrix

    audio_data_list = get_filepaths(dir_path)
    num_features = 57

    feature_matrix = np.zeros((len(audio_data_list), num_features))

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for start in range(0, len(audio_data_list), batch_size):
            batch_data = audio_data_list[start:start + batch_size]
            loaded_data = load_files(batch_data)
            futures = [executor.submit(calculate_audio_features_wrapper, (y, sr)) for y, sr in loaded_data]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                features = future.result()
                feature_matrix[start + i, :] = features
                print('Feature calculated')

    feature_matrix = np.array(feature_matrix)
    print('Feature Calculation finished!')
    return feature_matrix

def preprocessing(feature_matrix):
    scaler = StandardScaler(with_mean=False, with_std=False)
    scaled_data = scaler.fit_transform(feature_matrix)
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    print('Preprocessing finished')
    return pca_result

def main():
    dir_path = r'C:\Users\Shark\Music\Music\TEST\Oldschool'
    cache_file = f"feature_matrix_cache_v{dir_path}.joblib"

    feature_matrix = extract_feature_matrix_batch(dir_path, cache_file)
    joblib.dump(feature_matrix, cache_file) 
    pca_result = preprocessing(feature_matrix)

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result)
    print(cluster_labels)

    # Create directories for each cluster
    cluster_dirs = [os.path.join(dir_path, f"Cluster_{i}") for i in range(n_clusters)]
    for cluster_dir in cluster_dirs:
        os.makedirs(cluster_dir, exist_ok=True)

    # Move files to respective cluster directories
    for i, (mp3) in enumerate(get_filepaths(dir_path, load = False)):
        cluster_dir = cluster_dirs[cluster_labels[i]]
        shutil.move(mp3, os.path.join(cluster_dir, os.path.basename(mp3)))

if __name__ == '__main__':
    main()
        

