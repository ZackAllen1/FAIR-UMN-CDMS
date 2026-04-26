import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import PchipInterpolator

class augmentor:
    def __init__(self,dataset_csv_path):
        self.path = dataset_csv_path
        data = pd.read_csv(self.path)
        self.columns = data.columns
        self.data_array = np.array(data)
        self.detectors = ['A','B','C','D','F']
        self.indices = {}
        self.amplitudes = {}
        
        self.chrono_suffixes = [
            'r10','r20','r30','r40', 'r50', 'r60', 'r70', 'r80', 'r90','r95', 'r100', # rise
            'f95', 'f90', 'f80', 'f40', 'f20' # fall
        ]
        for detector in self.detectors:
            self.amplitudes[detector] = np.where(data.columns == f"P{detector}amp")[0][0]
            
            detector_timing_indices = []
            for suffix in self.chrono_suffixes:
                col_name = f"P{detector}{suffix}"
                detector_timing_indices.append(np.where(self.columns == col_name)[0][0])
            self.indices[detector] = detector_timing_indices

            
        self.rows = []
        for row in self.data_array:
            self.rows.append(pulses(row,self.detectors,self.indices,self.amplitudes))
            
        self.noisyData = None
        self.noiseIntensities = None
        self.noiseSNR = None
            
            
    def getPulses(self,row):
        return self.rows[row]
    
    def createNoisyData(self, timesteps, weights, intensities=[1e-9, 1e-8, 5e-8], max_tries=100):
        self.noisyData = None
        out_data = []
        agg_snr = 0.0

        for i in range(len(self.rows)):
            aRiseOffset = 0.0
            out_row = np.zeros(87) 
            
            # Preserve Metadata
            out_row[0] = self.data_array[i][0]    # Row ID
            out_row[86] = self.data_array[i][86]  # Target y

            snr = 0.0
            for j in self.detectors:
                signal = self.rows[i].getSignal(j)
                
                # Grab the original, un-noised values from the raw data array.
                # If the noise loop fails 100 times, these clean values will be used.
                kt = self.data_array[i][self.indices[j]].copy()
                amplitude = self.data_array[i][self.amplitudes[j]]
                
                tries = 0
                while tries < max_tries:
                    try:
                        signal.add_noise(timesteps, weights, intensities)
                        # Only add to SNR if successful
                        current_snr = signal.getSNR() 
                        
                        # Attempt to extract the 16 markers
                        kt_noisy, amp_noisy = signal.getKeyTimes()
                        
                        # If successful, overwrite the fallback values and break
                        kt = kt_noisy
                        amplitude = amp_noisy
                        snr += current_snr
                        break
                    except (IndexError, ValueError):
                        tries += 1
                        
                # Use Sensor A's 10% Rise Time as the global 'Start' reference
                if j == 'A':
                    aRiseOffset = kt[0] 

                # Map back to the CSV indices
                out_row[self.indices[j]] = kt - aRiseOffset
                out_row[self.amplitudes[j]] = amplitude
                
            out_data.append(out_row.tolist())
            agg_snr += snr / len(self.detectors)
            
        self.noiseSNR = 10 * np.log10(agg_snr / len(self.rows))
        print(f"Average dB SNR: {self.noiseSNR:.2f} dB")
        self.noiseIntensities = intensities
        self.noisyData = pd.DataFrame(out_data, columns=self.columns)

class pulses:
    def __init__(self, data_row, detectors, indices, amplitudes):
        self.pulses  = {}
        for detector in detectors:
            times = data_row[indices[detector]]
            amplitude = data_row[amplitudes[detector]]
            self.pulses[detector] = signal(times, amplitude)
            
    def getSignal(self, pulse):
        return self.pulses[pulse]
    
class signal:
    def __init__(self, times, amplitude):
        # 16 standard voltages (11 Rise, 5 Fall)
        voltages_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 
                         0.95, 0.9, 0.8, 0.4, 0.2]
        base_voltages = amplitude * np.array(voltages_list)
        
        self.voltages = np.append(base_voltages, 0.01 * amplitude)
        
        tail_time = times[-1] + abs(times[-1] * 0.5) + 1e-3
        self.times = np.append(times, tail_time)
        
        for idx in range(1, len(self.times)):
            if self.times[idx] <= self.times[idx-1]:
                self.times[idx] = self.times[idx-1] + 1e-5

        self.spline = PchipInterpolator(self.times, self.voltages)
        self.noisyTimes = None
        self.noisyVoltages = None
        
    def getOriginalSignal(self):
        return self.times, self.voltages
        
    def sampleFromSpline(self, timesteps):
        times = np.linspace(self.times[0], self.times[-1], timesteps)
        voltages = self.spline(times)
        return times, voltages
    
    def add_noise(self, timesteps, weights, intensities=[1e-9, 1e-8, 5e-8]):
        times, voltages = self.sampleFromSpline(timesteps)
        new_voltages = np.zeros(len(voltages))
        new_voltages += weights[0] * noise.add_shot_noise(voltages, intensities[0])
        new_voltages += weights[1] * noise.add_pink_noise(voltages, intensities[1])
        new_voltages += weights[2] * noise.add_brown_noise(voltages, intensities[2])
        self.noisyTimes = times
        self.noisyVoltages = new_voltages
        
    def getSNR(self):
        if self.noisyVoltages is None:
            return 0.0
        t, signal = self.sampleFromSpline(len(self.noisyVoltages))
        noise_diff = self.noisyVoltages - signal
        p_signal = np.sqrt(np.mean(signal**2))
        p_noise = np.sqrt(np.mean(noise_diff**2))
        return (p_signal / p_noise)**2
    
    def getKeyTimes(self):
        if self.noisyVoltages is None or self.noisyTimes is None:
            raise ValueError("Noise must be added before extracting Key Times.")
        
        v = self.noisyVoltages
        t = self.noisyTimes
        amplitude = np.max(v)
        peak_idx = np.argmax(v)
        
        rise_v = v[:peak_idx+1]
        rise_t = t[:peak_idx+1]
        fall_v = v[peak_idx:]
        fall_t = t[peak_idx:]
        
        levels_rise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        levels_fall = [0.95, 0.9, 0.8, 0.4, 0.2]
        
        new_kt = []
        
        for l in levels_rise:
            try:
                idx = np.where(rise_v >= l * amplitude)[0][0]
                new_kt.append(rise_t[idx])
            except IndexError:
                new_kt.append(rise_t[-1] if len(new_kt) == 0 else new_kt[-1])
                
        for l in levels_fall:
            try:
                idx = np.where(fall_v <= l * amplitude)[0][0]
                new_kt.append(fall_t[idx])
            except IndexError:
                new_kt.append(fall_t[-1] if len(new_kt) == 0 else new_kt[-1])
                
        return np.array(new_kt), amplitude
    
class noise:
    @staticmethod
    def add_shot_noise(signal,events=1e-9):
        return np.random.poisson(signal/events)*events
    
    @staticmethod
    def add_pink_noise(signal, intensity=1e-8):
        n = len(signal)
        white_noise = np.random.randn(n)
        white_noise_fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n)
        filter_mag = np.zeros_like(freqs)
        filter_mag[1:] = 1.0 / np.sqrt(freqs[1:])
        pink_noise_fft = white_noise_fft * filter_mag
        pink_noise = np.fft.irfft(pink_noise_fft, n)
        pink_noise /= np.std(pink_noise)
        return signal + (pink_noise * intensity)
    
    @staticmethod
    def add_brown_noise(signal, intensity=5e-8):
        white_noise = np.random.normal(0, 1, len(signal))
        brown_noise = np.cumsum(white_noise)
        brown_noise = brown_noise / np.max(np.abs(brown_noise))
        return signal + (intensity * brown_noise)

if __name__ == "__main__":
    # Define the dataset path (we will just use Train set)
    train_filepath = "../data/full_train_data.csv"
    output_file = "../data/full_train_augmented_data.csv"
    
    print(f"Loading original training data from {train_filepath}...")
    
    original_df = pd.read_csv(train_filepath)
    all_dataframes = [original_df]
    
    # Initialize the augmentor
    dataset = augmentor(train_filepath)
    
    # Combining these 6 with the 1 original would essentially give 7x expansion
    profiles = [
        ([0.2, 0.2, 0.6], [5e-9, 2e-8, 8e-8]),
        ([0.8, 0.1, 0.1], [1e-9, 7e-8, 3e-7]),
        ([0.1, 0.8, 0.1], [1e-9, 5e-8, 1e-8]),
        ([0.33, 0.33, 0.33], [2e-9, 2e-8, 5e-8]),
        ([0.4, 0.3, 0.3], [5e-10, 5e-9, 1e-8]),
        ([0.5, 0.2, 0.3], [1e-8, 1e-7, 5e-7])
    ]
    
    # Now generate the augmented arrays
    for i, (weights, intensities) in enumerate(profiles):
        print(f"\nGenerating noisy dataset {i+1}/6...")
        
        # Used 150 like Bobby had in his jupyter
        dataset.createNoisyData(150, weights, intensities=intensities)
        
        if dataset.noisyData is None:
            raise ValueError("No Noise created!")
        
        # append to the all_dataframes
        all_dataframes.append(dataset.noisyData)
        
    # Concat the all_dataframes
    augmented_df = pd.concat(all_dataframes, ignore_index = True)
    
    # save this as the new training dataset
    augmented_df.to_csv(output_file, index = False)
    print(f"\nSUCCESS! Saved the expanded dataset to {output_file}")
    print(f"Original rows: {len(original_df)} -> New rows: {len(augmented_df)}")