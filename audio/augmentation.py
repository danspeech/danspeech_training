from abc import ABC, abstractmethod

import numpy as np
import random

import librosa
import pyroomacoustics as pra


class DataAugmenter(ABC):
    """
    Abstract class for data augmentations
    """

    @abstractmethod
    def augment(self, recording):
        pass


class DanSpeechAugmenter(DataAugmenter):
    """
    Class that implements the DanSpeech Augmentation scheme
    """

    def __init__(self, sampling_rate, augmentation_list=None):
        self.sampling_rate = sampling_rate

        # Allow user to specify a list of augmentations
        # otherwise default to danspeech schema.
        if not augmentation_list:
            self.augmentations_list = [
                self.speed_perturb,
                self.room_reverb,
                self.volume_perturb,
                self.add_wn,
                self.shift_perturb
            ]
        else:
            self.augmentations_list = [getattr(self, augmentation) for augmentation in augmentation_list]

        print("Using the following augmentations: {}".format(", ".join([x.__name__ for x in self.augmentations_list])))

    def augment(self, recording):
        scheme = self.choose_augmentation_scheme()
        # Apply the chosen augmentations
        for augmentation_function in scheme:
            recording = augmentation_function(recording, self.sampling_rate)

        return recording

    def choose_augmentation_scheme(self):
        """
        Chooses a valid danspeech augmentation based on the ordered
        list of augmentations

        :param list_of_augmentations: Ordered list of augmentation functions
        :return: A valid danspeech augmentation scheme
        """
        n_augments = random.randint(0, len(self.augmentations_list))
        augmentations_to_apply = random.sample(
            self.augmentations_list, n_augments)

        augmentation_scheme = []
        for augmentation in self.augmentations_list:
            if augmentation in augmentations_to_apply:
                augmentation_scheme.append(augmentation)

        return augmentation_scheme

    @staticmethod
    def speed_perturb(recording, sampling_rate, *args):
        """
        Select up/down-sampling randomly between 90% and 110% of original sample rate

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """

        new_sample_rate = sampling_rate * random.choice([0.9, 1.1])
        return librosa.core.resample(recording, sampling_rate, new_sample_rate)

    @staticmethod
    def shift_perturb(recording, sampling_rate, *args):
        """
        Shifts the audio recording randomly in time.

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """

        shift_ms = np.random.randint(low=-50, high=50)
        shift_samples = int(shift_ms * sampling_rate / 1000)

        if shift_samples > 0:
            # time advance
            recording[:-shift_samples] = recording[shift_samples:]
            recording[-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            recording[-shift_samples:] = recording[:shift_samples]
            recording[:-shift_samples] = 0
        return recording

    @staticmethod
    def room_reverb(recording, sampling_rate, *args):
        """
        Perturb signal with room reverberations in a randomly generated shoebox room.

        :param recording: Recording to be augmented
        :param sampling_rate: Sampling rate of the recording
        :return: Augmented recording
        """

        # generate random room specifications, including absorption factor.
        alpha = random.uniform(0, 0.4)

        room_length = np.random.uniform(2, 12)
        room_width = np.random.uniform(2, 6)
        room_height = 3.0 + np.random.uniform(-0.5, 0.5)

        microphone_x = np.random.uniform(0.5, room_width - 0.5)
        microphone_y = np.random.uniform(0.5, room_length - 0.5)
        microphone_height = 1.50 + np.random.uniform(-0.5, 0.5)

        r = 0.5 * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 1) * 2 * np.pi
        source_x = microphone_x + r * np.cos(theta)
        source_y = microphone_y + r * np.sin(theta)
        source_height = 1.80 + np.random.uniform(-0.25, 0.25)

        # create the room based on the specifications simulated above
        room = pra.ShoeBox([room_width, room_length, room_height],
                           fs=sampling_rate,
                           max_order=17,
                           absorption=alpha)

        # add recording at source, and a random microphone to room
        room.add_source([source_x, source_y, source_height], signal=recording)
        R = np.array([[microphone_x], [microphone_y], [microphone_height]])
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
        room.image_source_model()
        room.simulate()

        # return the reverberation convolved signal
        return room.mic_array.signals[0, :]

    @staticmethod
    def volume_perturb(recording, *args):
        """
        Select a gain in decibels randomly and add to recording

        :param recording:
        :return: Augmented recording
        """
        gain = np.random.randint(low=5, high=30)
        recording *= 10. ** (gain / 20.)
        return recording

    @staticmethod
    def add_wn(recording, *args):
        """
        Add wn white noise with random variance to recording

        :param recording:
        :return: Augmented recording
        """
        # Normalize recording before adding wn
        mean = np.mean(recording)
        std = np.std(recording)

        recording = (recording - mean) / std

        variance = np.random.uniform(low=0.5, high=1.8)
        noise = np.random.normal(0, random.uniform(0, variance), len(recording))

        # append noise to signal
        return recording + noise
