import numpy as np

def split_data(data_x : np.array, data_y : np.array, train_ratio : float, generator = None):
    """
    Split data into training and testing sets.
    :param data_x: input data
    :param data_y: output data
    :param split_ratio: ratio of training data
    :return: training and testing data
    """
    if generator is None:
        generator = np.random.default_rng()
    generator.shuffle(data_x)
    generator.shuffle(data_y)
    split_idx = int(len(data_x) * train_ratio)
    return data_x[:split_idx], data_x[split_idx:], data_y[:split_idx], data_y[split_idx:]