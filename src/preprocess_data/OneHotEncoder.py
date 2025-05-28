import numpy as np


class OneHotEncoder:
    _instance = None

    def __new__(cls, args):
        if not cls._instance:
            cls._instance = super(OneHotEncoder, cls).__new__(cls)
        return cls._instance

    def __init__(self, args):
        if not hasattr(self, 'initialized'):
            self.args = args
            self.one_hot_mapping = {
                'PFOTS-Si-water'       : np.array([1, 0, 0, 0, 0, 0]),
                '40-Glycerol-Teflon-Au': np.array([0, 1, 0, 0, 0, 0]),
                '20-Glycerol-Teflon-Au': np.array([0, 0, 1, 0, 0, 0]),
                '30-Glycerol-Teflon-Au': np.array([0, 0, 0, 1, 0, 0]),
                'Fthiols_Au'           : np.array([1, 0, 0, 0, 1, 0]),
                'Fthiols-Au'           : np.array([0, 0, 0, 0, 0, 1]),
                'PFOTS-Si-water_drop'  : np.array([1, 1, 0, 0, 0, 0]),
                'other'                : np.array([0, 0, 0, 0, 0, 0])
            }
            self.args.num_dim_one_hot = len(self.one_hot_mapping['other'])
            self.initialized = True

    def add_one_hot_to_frame(self, df):
        for i in range(self.args.num_dim_one_hot):
            df[f'OneHot_{i}'] = (
                df[self.args.system_id_column].apply(
                    lambda x: self.map_to_one_hot(x, i)))

    def map_to_one_hot(self, name, i):
        return self.one_hot_mapping.get(name, self.one_hot_mapping['other'])[i]
