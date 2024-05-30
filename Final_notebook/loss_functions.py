import numpy as np

# Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Root Mean Square Error (RMSE)
def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Normalized Root Mean squared Error (NRMSE)
def normalized_root_mean_squared_error(y_true, y_pred):
    c = 1/np.abs(np.mean(y_true))
    return np.sqrt(np.mean((y_true - y_pred) ** 2))/c

# Mean Absolute Error (MAE)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))/np.max(y_pred)

# Peak Signal to Noise Ratio (PSNR)
def peak_signal_to_noise_ratio(y_true, y_pred):
    return 10*np.log10((np.max(y_true) ** 2)/np.mean((y_true - y_pred) ** 2))

# Normalized Mutual Information (NMI)
def normalized_mutual_information(y_true, y_pred):

    y_true_flatten = y_true.flatten()
    y_true_histogram, _ = np.histogram(y_true_flatten, bins=y_true_flatten.size, range=(0, 256))
    y_true_entropy = -np.sum(y_true_histogram * np.log2(y_true_histogram + 1e-10))

    y_pred_flatten = y_true.flatten()
    y_pred_histogram, _ = np.histogram(y_pred_flatten, bins=y_pred_flatten.size, range=(0, 256))
    y_pred_entropy = -np.sum(y_pred_histogram * np.log2(y_pred_histogram + 1e-10))

    if y_true.size != y_pred.size:
        raise ValueError("The images must have the same size")
    
    joint_histogram, _, _ = np.histogram2d(y_true.ravel(), y_pred.ravel(), bins=y_true.shape)
    joint_histogram = joint_histogram / np.sum(joint_histogram)
    joint_entropy = -np.sum(joint_histogram * np.log2(joint_histogram + 1e-10))

    return (y_true_entropy + y_pred_entropy)/joint_entropy
