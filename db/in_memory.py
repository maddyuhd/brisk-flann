import redis
import sys
from time import time

class Memory(object):
    def __init__(self, reconstruct=False, pipeline=False):
        self.r = redis.Redis(host='localhost', port=6379, db=0)

        if pipeline:
            self.temp = self.r
            self.r = self.r.pipeline()
        if reconstruct:
            self.reset_node_idx()

    def pipeline_close(self, send=False, switch=False):
        if send:
            val = self.r.execute()
            if switch:
                self.r = self.temp
            return val

        else:
            self.r.execute()

    def if_dict_exists(self, key, val):
        return self.r.hexists(key, val)

    def set_des(self, name, val):
        for idx, des in enumerate(val):
            self.r.hset(name, idx, des)
        pass

    def set_dict_value(self, key, val, idx=None, entire=True):
        if entire:
            self.r.hmset(key, val)
        else:
            self.r.hset(key, idx, val)

    def update_dict_value(self, key, idx, val):
        new_val = self.get_dict_value(key, idx, entire=False)
        new_val.append(val)
        self.set_dict_value(key, new_val, idx, entire=False)

    def get_dict_value(self, key, idx=None, entire=True):
        if entire:
            return self.r.hgetall(key)
        else:
            if key == "tree":
                return map(str, eval(self.r.hget(key, idx)))
            
            return eval(self.r.hget(key, idx))

    def delete(self, key, idx=None, entire=False):
        if entire:
            self.r.delete(key)
        else:
            self.r.hdel(key, idx)

    def get_node_idx(self):
        return self.r.incr("nodeIndex")

    def reset_node_idx(self, val=-1):
        self.r.set("nodeIndex", val)


class RedisHandler(object):
    def __init__(self, pipeline=False):
        self.pipeline = pipeline
        self.r = redis.Redis(host='localhost', port=6379, db=0)
        if self.pipeline:
            self.temp, self.r = self.r, self.r.pipeline()

            # self.r = self.r.pipeline()

    def _close(self, return_=False):
        if self.pipeline:
            if return_:
                return self.r.execute()
            else:
                self.r.execute()


import numpy as np

class ImagetoMemory(RedisHandler):
    def __init__(self, pipeline=True):
        RedisHandler.__init__(self, pipeline=pipeline)

    def dump_des(self, name, val):
        for idx, des in enumerate(val):
            self.r.hset(name, idx, des)

    def get_des_val(self, name, idx):
        self.r = self.temp
        val = self.r.hmget(name, idx)
        # val[0][1:-1].split()
        data = np.asarray(val[0][1:-1].split())
        return data.astype("uint8")

        # return eval(val.replace(" ", ","))

    def entire_des(self, name):
        # val = self.r.hvals(name)
        val = []
        for i in self.r.hvals(name):
            val.append(i[1:-1].split())
        # val = a
        val = np.asarray(val)
        val = val.astype("uint8")
        return val

    def close(self, send=False):
        self._close(return_=send)


class Manage_del_dis(RedisHandler):

    def disable(self, val):
        if self.r.sismember("disable", val):
            self.r.srem("disable", val)
            return 0

        else:
            self.r.sadd("disable", val)
            return 1

    def remove(self, val):
        if self.r.sismember("disable", val):
            self.r.smove("disable", "remove", val)
            return 1

        elif self.r.sismember("remove", val):
            self.r.srem("remove", val)
            return 0

        else:
            self.r.sadd("remove", val)
            return 1

    def stuff_to_ignore(self):
        return self.r.sunion("remove", "disable")


# r=redis.Redis(host='localhost',port=6379,db=0)

# for i in val:
#     print i,type(i)
# results = map(str, results)

# # # to delete all the keys in currently connecting db
# # r.flushdb() 

# # # to delete all db
# # r.flushall()

# # to delete the particular key
# r.delete('nodes')

# # to delete the particular field in a key
# r.hdel('tree','1')

# # to list all the keys 
# r.scan()

# # to print all the keys matching specified pattern
# keys = r.scan_iter(match='aa*')
