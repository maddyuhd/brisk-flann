from heapq import heappop
import collections

class analyse():
    def __init__(self):
        self.c = collections.Counter()
        # self.setVal = []

    def update(self, result, top_n=5):
        for _ in range(top_n):
            top_result = heappop(result)
            self.c.update([top_result[1][0]])
            # self.c.update([top_result[1][0].name])

    """
    Clean me
    """
    def output(self, qdes, top_n=5):
        self.c = self.c.most_common(top_n)

        h = Temp()
        arr = []

        for idx, i in enumerate(self.c):
            i, _ = i
            count = h.match(qdes, i.des)
            arr.append((count, i.name))

        # import heapq
        # arr = heapq.nlargest(5, arr, key=lambda x: x[1])

        arr = max(arr)

        if arr[0] >= 5:  # arr[0][1] >= 5:
            # print arr
            # arr = arr[0][0]
            return (1, arr[1])
        else:
            return (0, arr)

        # setVal = [x for x in self.c if x[1] >= 10 and not in remove] <

"""
Clean me
"""
class Temp(object):
    def __init__(self):
        import cv2

        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2

        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(
            index_params, search_params)

    def match(self, qdes, sdes):
        count = 0
        matches = self.flann.knnMatch(qdes, sdes, k=2)
        # ratio test as per Lowe's paper
        for _, val in enumerate(matches):
            if len(val) == 2:
                m, n = val[0], val[1]
                if m.distance < 0.7 * n.distance:
                    count += 1
        return count
