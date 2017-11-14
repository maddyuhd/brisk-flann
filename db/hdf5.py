import numpy as np 
import h5py
import cv2
from time import time

def create_folder(name, file_access):
    access = file_access.create_group(name)
    return access

def creat_data(file_access, name, data_, size, type_):
    file_access.create_dataset(name, size, dtype=type_,
                               maxshape=(None, size[1]), data=data_)

def get_access(file_access, path):
    file_access = file_access[path]
    return file_access


class FileStructure(object):

    def __init__(self, mode="r+", name="snapar", uuid="snapon"):
        self.uuid = uuid
        self.path = "/" + self.uuid + "/dataset"

        try:
            self.file_access = h5py.File(name+".hdf5", mode)

        except Exception as e:
            self.file_access = h5py.File(name+".hdf5", "w")
            create_folder(self.path, self.file_access)

    def creat_db(self):
        if self.path not in self.file_access:
            create_folder(self.path, self.file_access)

    def add_dataset(self, name, data, type_="uint8"):
        dataset_access = self.path + "/" + name not in self.file_access

        if dataset_access:
            size = data.shape[:]
            db_access = get_access(self.file_access, self.path)
            creat_data(db_access, name, data, size, type_)

    def list_of_dataset(self, uuid):
        dic = {}
        temp_path = "/" + uuid + "/dataset"

        names = get_access(self.file_access, temp_path)

        for name in names:
            data = self.file_access[self.path + "/"+ name][:]
            dic[name] = data
        return dic

    def test(self, img_name, idx):
        return self.file_access[self.path + "/" + img_name][idx]

    def test1(self, img_name):
        # img_name = img_name.name
        return self.file_access[self.path + "/" + img_name][:]

    def close(self):
        self.file_access.close()

# img = images("/home/smacar/Desktop/test.jpg", resize=400, src=False)

# for i in img.des:
#     print i
# obj = FileStructure()
# obj.add_dataset("name",img.des)
# # dataset = obj.list_of_dataset("snapon")
# obj.close()

# print str(time()-s)+" sec."

#     def append_data(self, file_access, data):
#         inc_by = len(data)
#         end, dim = file_access.shape[:]
#         file_access.resize((end + inc_by, dim))
#         file_access[end:end + inc_by] = data


# f = h5py.File("snapar.hdf5","r+")

# # group1 = f.create_group("dbId")

# datasetname = "dataset"
# # dataset = f.create_dataset(datasetname,(10,),dtype="i")
# # data=np.arange(10)
# # dataset[...] = data

# grp3 = f['/dbId/dataset']
# # dataset = grp3.create_dataset(datasetname,(6,10),dtype="uint8",maxshape=(None, 10))
# data=np.arange(10)
# grp3[1] = [1,1,1,1,1,1,1,1,1,1]
# # print grp3.shape[0]
# grp3.resize((grp3.shape[0]+5, 10))
# # for i in grp3:
# # 	print i
# f.close()

# # f=h5py.File("test.hdf5","r")
# # dataset=f[datasetname]
# # data = dataset[...]
# # print (data)
