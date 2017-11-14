# from db.hdf5 import FileStructure

class Indexer(object):
    """
    Indexer for handling Data structure
    & saves to HDF5 file
    """

    def __init__(self, img_list, images, hdf5_obj):
        self.vectores = []
        self._img_list = img_list
        self._images = images
        self._hdf5 = hdf5_obj

    def dump(self):
        for img_path in self._img_list:
            img = self._images(img_path)
            self._hdf5.add_dataset(img.name, img.des)

            for i in range(len(img.des)):  # <-----------------------------enumerate
                self.vectores.append((img, i))

        self._hdf5.close()
        return self.vectores


# def index(imgList):
#     features = []
# from db.hdf5 import FileStructure

# #     HDF5 = FileStructure()

#     for img_path in imgList:
#         img = images(img_path)
#         HDF5.add_dataset(img.name, img.des)

#         for i in range(len(img.des)):  # <-----------------------------enumerate
#             # features.append((img.name, i))
#             features.append((img, i))

#     HDF5.close()
#     return features


def getVal(val):
    img_obj, idx = val
    # return hdf5_test_obj.test(img_obj, idx)
    # data = obj.get_des_val(img_obj, idx)
    # return data
    return img_obj.des[idx]
