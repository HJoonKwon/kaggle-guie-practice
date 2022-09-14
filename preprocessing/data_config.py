class DataConfigType:
    # dataset selection
    # 1. Images130k
    # 2. Imagenet1k
    # 3. Google-Landmark-2021
    # 4. Product10k
    # 5. Clothing-Dataset
    # 6. HnM-Fashion-Dataset
    # 7. iFood
    # 8. MET
    # 9. Furniture-Images
    # 10. Bonn-Furniture-Styles-Dataset
    data_name: str
    data_dir: str # root path to the data directory
    label_column: str # which column to use as labels (default is 'label')

    def __init__(self, data_name: str="", data_dir: str="", label_column: str="label"):
        self.data_name = data_name
        self.data_dir = data_dir
        self.label_column = label_column

    def __getitem__(self, key):
        return getattr(self, key, None)