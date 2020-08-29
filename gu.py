from math import sin, cos, sqrt, atan2, radians,degrees
# from numpy.linalg import norm
# import numpy as np
# import pandas as pd
# from shapely.geometry import Point, Polygon
# from sklearn.neighbors import NearestNeighbors


def get_distance(X1, Y1, X2, Y2, R=6371137.0):
    # X/Y 格式：[longitude, latitude]
    # 减少getitem的消耗
    # R = 6371137.0  # 地球半径(m)
    lng_1 = radians(X1)
    lat_1 = radians(Y1)
    lng_2 = radians(X2)
    lat_2 = radians(Y2)

    dlon = lng_2 - lng_1
    dlat = lat_2 - lat_1

    a = sin(dlat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def get_distance_lnglat(x,y):
    return get_distance(*x,*y)

class Trans_coor(object):


    def __init__(self, startll=(114.06,22.56)):
        self.lng, self.lat, = tuple(map(radians,startll))
        self.R = 6371137.0  # 地球半径(m)

    def lnglat_to_plane_coor(self,lng,lat):
        lng, lat = tuple(map(radians, (lng, lat)))
        dlng = lng - self.lng
        dlat = lat - self.lat
        y = dlat*self.R

        R_factor = cos(lat)
        x = dlng*self.R*R_factor
        plane_coor = x, y
        return plane_coor

    def plane_to_lnglat_coor(self, plane_coor):
        x, y = plane_coor

        dlat = y/self.R
        lat = self.lat+dlat

        R_factor = cos(lat)
        dlng = x/(self.R*R_factor)
        lng = self.lng+dlng
        lng, lat = tuple(map(degrees, (lng, lat)))

        return lng, lat

    def get_distance_ll(self,X1, Y1, X2, Y2):
        return get_distance(X1, Y1, X2, Y2,R=self.R)

    def get_distance_plane(self,X1, Y1, X2, Y2):
        return sqrt((X1-X2)**2+(Y1-Y2)**2)

def sample_mae(ll,ll_p,tc,sample_num=1000):
    length = len(ll)
    from random import randint
    from tqdm import tqdm
    error = []
    p_time = 0
    l_time = 0
    from time import time
    for i in tqdm(range(sample_num)):
        ind_ = randint(0,length-1)
        ind = randint(0,length-1)

        # t = time()
        # dist_p = tc.get_distance_plane(*ll_p[ind],*ll_p[ind_])
        # p_time += (time() - t)

        t = time()
        dist_ll = tc.get_distance_ll(*ll[ind],*ll_p[ind])
        l_time += (time() - t)

        error.append(abs(dist_ll))
    print("time compare: ",p_time,l_time)
    print("max error",max(error))
    return sum(error)/sample_num

def loaddata(filename = "poi_data/meilin_business_poi.csv"):
    poi_df_ = pd.read_csv(filename)
    x = poi_df_["wgs_lat"].to_list()
    y = poi_df_["wgs_lng"].to_list()
    x_y = list(zip(y,x))
    return np.array(x_y)

def isInsidePolygon(pt, poly):
    res = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l - 1:
        i += 1
        # print(i, poly[i], j, poly[j])
        if ((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or (
                poly[j][0] <= pt[0] and pt[0] < poly[i][0])): # 在线段中
            if (pt[1] <= (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (
                poly[j][0] - poly[i][0]) + poly[i][1]):
                res = not res
        j = i
    return res



def poi_prepocess(ba, poi_df,factor=0):
    """
    根据BA的范围过滤poi条目 \n
    factor ba范围膨胀比例 0不膨胀 1 边界往外膨胀 1/10距离
    """

    wgs_lng = poi_df['wgs_lng'].to_list(); wgs_lat = poi_df['wgs_lat'].to_list()

    coor = list(zip(wgs_lng, wgs_lat))
    ebound = ba.extend_bound_array(factor=factor)
    # bound = ba.bound_array
    label = [1 if (isInsidePolygon(x_y,ebound)) else 0  for x_y in coor  ]
    poi_df["l"] = label
    poi_df_ = poi_df[poi_df["l"]==1]
    return poi_df_

def get_distance_point2polygen(pts,polygen):
    polygen = np.array(polygen)
    x = polygen[:,0].mean()
    y = polygen[:,1].mean()
    tc = Trans_coor((x,y))
    polygen = [ tc.lnglat_to_plane_coor(*p) for p in polygen]
    pts = tc.lnglat_to_plane_coor(*pts)
    pts = Point(pts)
    polygon = Polygon(polygen)
    return pts.distance(polygon)


from math import floor, cos
import time
import numpy as np

PI = 3.1415926
EARTH_RADIUS = 6371000.0
DEGREE_HALF_CIRCLE = 180.0
DEGREE_CIRCLE = 360.0
CHINA_NORTHEST = 54.0
CHINA_SOUTHEST = 3.5
CHINA_WESTEST = 73.5
CHINA_EASTEST = 135.5
EQUATOR_LENGTH = 40076000.0
TILE_ID_COEF = 1100000
D_X = 5
D_Y = 5
RSSI_EMPTY = -120
POSITIONING_FP_NUM = 3
MINI_VALUE = 0.00001


def GetTileID(lng,  lat, d_x=5, d_y=5):
    if ((lat > CHINA_NORTHEST) and (lat < CHINA_SOUTHEST) and(lng > CHINA_EASTEST) and (lng < CHINA_WESTEST)):
        return None

    latID = 0
    lngID = 0

    expandIndex = 1000000
    latDivideIndex = d_y * 10
    latInt = int(lat * expandIndex)
    lngInt = int(lng * expandIndex)
    latStartPoint = int(CHINA_NORTHEST * expandIndex)
    latID = int((latStartPoint - latInt) / latDivideIndex)
    latBand = floor(float(latInt) / float(expandIndex)) + 0.5
    latBandLength = float(EQUATOR_LENGTH * cos(latBand * PI / DEGREE_HALF_CIRCLE))

    lngID = (int((lng - CHINA_WESTEST)/DEGREE_CIRCLE * (latBandLength / d_x)))

    tileID = lngID * TILE_ID_COEF + latID

    return tileID, lngID, latID


# /*函数：根据网格ID计算网格的经纬度边界*/
def TileIDAnalysis(tileID, d_x=5, d_y=5):

    latMinVal = 0.0
    latMaxVal = 0.0
    lngMinVal = 0.0
    lngMaxVal = 0.0

    # 解析latID和lonID
    latID = 0
    lngID = 0

    lngID = floor(tileID / TILE_ID_COEF)
    latID = tileID - lngID * TILE_ID_COEF
    latPerMeter = 0.00001

    latMaxVal = float(CHINA_NORTHEST - latID * d_y * latPerMeter)
    latMinVal = float(latMaxVal - d_y * latPerMeter)

    latBand = float(floor(float(latMinVal + latMaxVal) /2.0) + 0.5)
    latBandLength = float(EQUATOR_LENGTH * cos(float(latBand *PI / DEGREE_HALF_CIRCLE)))

    lngMinVal = float(lngID * d_x / latBandLength * DEGREE_CIRCLE + CHINA_WESTEST)
    lngMaxVal = float(lngMinVal + d_x * latPerMeter / cos(float(latBand) * PI / DEGREE_HALF_CIRCLE))

    return (latMinVal, latMaxVal), (lngMinVal, lngMaxVal)


def GetNerbTileID(tileID, long_x,long_y,d_x=5, d_y=5):
    """
    long_*  ：Extend distance by one side
    """
    lngID = floor(tileID / TILE_ID_COEF)
    latID = tileID - lngID * TILE_ID_COEF
    dlngs = int(long_x/d_x)
    dlats = int(long_y/d_y)
    for dlng in range(-dlngs, dlngs+1):
        for dlat in range(-dlats, dlats+1):
            yield (lngID+dlng) * TILE_ID_COEF + (latID+dlat)



def getstrtime(timeStamp):
    timeStamp = float(timeStamp)/1000
    timeArray = time.localtime(timeStamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", timeArray)



    # 编辑距离函数
def levenshtein_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1],
                                   matrix[x, y - 1] + 1)
            else:
                matrix[x, y] = min(matrix[x - 1, y] + 1,
                                   matrix[x - 1, y - 1] + 1,
                                   matrix[x, y - 1] + 1)
    return matrix[size_x - 1, size_y - 1]




if __name__ == "__main__":
    for tileid in GetNerbTileID(917593129096, 1000,500):
        pass



if __name__ == "__main__":

    print(get_distance(*[114.06094917667023, 22.52515499827182]
,*[114.06027794195909, 22.52514618249599]
))

