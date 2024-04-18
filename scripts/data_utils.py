import numpy as np
from numpy.random import randint, seed, normal
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

# Define global max and min variables for freqs = [1,0.01,0.1,0.005] and noise_std=0.3
GLOBAL_MAX = 1.5
GLOBAL_MIN = -1.5

# Define a function to generate the signal
def generate_signal(start, freqs=[1,0.01,0.1,0.005], noise_std=0.05, class_label=0, time_dim=100):
    # Generate the signal
    if class_label == 0:
        signal = np.sin(2 * np.pi * freqs[0] * np.arange(start, start+time_dim, 1)) + (1/2)*np.sin(2 * np.pi * freqs[1] * np.arange(start, start+time_dim, 1))
    else:
        signal = np.sin(2 * np.pi * freqs[2] * np.arange(start, start+time_dim, 1)) + (1/2)*np.sin(2 * np.pi * freqs[3] * np.arange(start, start+time_dim, 1))

    # Add noise to the signal
    noise = normal(0, noise_std, time_dim)
    signal += noise

    return signal

# Define a function to reshape, canonacalize, and normalize the signal
def preprocess_signal(signal):
    # Reshape the signal
    signal = signal.reshape((1, len(signal)))

    # Normalize the signal
    signal = (signal - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)

    # TODO: Analyze how this is different from before:
    # x -= x[:, :, 0].reshape(-1, 1, 1)
    # Canonicalize the data (pass through 0 at the origin)
    # signal -= signal[:, 0].reshape(1, 1)

    return signal

# Create a SignalDataset
class SignalDataset(Dataset):
    def __init__(self, generate=True, idx_range=[0, 1000], seq_length=100, transform=None, data_seed=None):
        self.generate = generate
        self.idx_range = idx_range
        self.num_samples = idx_range[1] - idx_range[0]
        self.seq_length = seq_length
        self.transform = transform
        self.data = []
        self.labels = []

        # If not generate on the fly, generate data ahead of time
        if self.generate == False:
            for i in range(self.num_samples):
                # If we are over halfway through num_samples, set label = 1
                if i < int(self.num_samples/2): 
                    label = 0
                else: 
                    label = 1

                if data_seed != None:
                    seed(data_seed+i)

                # Generate the signal
                start = randint(idx_range[0], idx_range[1])
                signal = generate_signal(start=start, class_label=label, time_dim=self.seq_length)
                # Preprocess the signal
                signal = preprocess_signal(signal)

                # Append the signal to the data
                self.data.append(signal)
                self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # If generate is True, generate a sample on the fly
        if self.generate == True:
            label = randint(0, 2)
            start = randint(self.idx_range[0], self.idx_range[1])
            signal = generate_signal(start=start, class_label=label, time_dim=self.seq_length)

            # Preprocess the signal
            signal = preprocess_signal(signal)
        else: 
            signal = self.data[idx]
            label = self.labels[idx]

        if self.transform:
            signal = self.transform(signal)
        
        return signal, label

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