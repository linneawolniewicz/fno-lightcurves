import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

# Create bandpass filtering functions
def ButterBandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def ButterBandpassFilter(data, lowcut, highcut, fs, order=5):
    b, a = ButterBandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Create a transform function that will randomly sample 400 out of the 500 timesteps
class RandomSample(object):
    def __init__(self, n_sample=400):
        self.n_sample = n_sample

    def __call__(self, sample):
        keep_indices = np.sort(np.random.choice(sample.shape[1], self.n_sample, replace=False))
        return sample[:, keep_indices]

# Create a transform function that will randomly translate the data in time (leaving edges at 0)
class RandomTimeTranslateFill0(object):
    def __init__(self, max_shift=100):
        self.max_shift = max_shift

    def __call__(self, sample):
        shift = np.random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return sample
        
        result = np.zeros_like(sample)
        if shift > 0:
            result[:, shift:] = sample[:, :-shift]
        elif shift < 0:
            result[:, :shift] = sample[:, -shift:]
        return result

# Create a transform function that will randomly translate the data in time (reflecting the data at edges)
class RandomTimeTranslateReflect(object):
    def __init__(self, max_shift=100):
        self.max_shift = max_shift

    def __call__(self, sample):
        shift = np.random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return sample
        
        result = np.zeros_like(sample)
        if shift > 0:
            result[:, shift:] = sample[:, :-shift]
            result[:, :shift] = sample[:, :shift][::-1]
        elif shift < 0:
            result[:, :shift] = sample[:, -shift:]
            result[:, shift:] = sample[:, shift:][::-1]
        return result
    
# Create a transform function that will randomly add gaussian noise to the data
class RandomNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        noise = np.random.normal(self.mean, self.std, sample.shape)
        return sample + noise

# Define a custom Dataset class for your data
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]

        if self.transform:
            x = self.transform(x)

        return x, y