NUM_CLASSES = {
    'Images130k': 11,
    'Imagenet1k': 1000,
    'Google-Landmark-2021': 81313,
    'Product10k': 9691,
    'Clothing-Dataset': 17,
    'HnM-Fashion-Dataset': 20
}

class DataConfigType:
    # dataset selection
    # 1. Images130k
    # 2. Imagenet1k
    # 3. Google-Landmark-2021
    # 4. Product10k
    # 5. Clothing-Dataset
    # 6. HnM-Fashion-Dataset
    data_name: str = ""
    data_dir: str = ""

    def __getitem__(self, key):
        return getattr(self, key, None)