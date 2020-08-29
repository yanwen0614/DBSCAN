import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from geography_utils import Trans_coor, get_distance_lnglat
import pickle
from sklearn.neighbors import NearestNeighbors
from multiprocess_df import Consumer, TaskTracker
import multiprocessing
import time

def landmark_index(row, neighbors_model=None,namelist=None,addresslist=None):
    ind,row = row
    neighborhoods = neighbors_model.radius_neighbors([(row.wgs_lng,row.wgs_lat)],return_distance=False)
    nbh = neighborhoods[0]
    key = row["name"]
    namelist_ = namelist[nbh]
    addresslist_ = addresslist[nbh]
    bool_ = sum([1 if (key in str(i) or key in str(j)) else 0 for i,j in zip(namelist_,addresslist_)])

    near =  nbh[np.array([True if (key in str(i) or key in str(j)) else False for i,j in zip(namelist_,addresslist_)])]

    return ind,bool_,near



def multi(num_process=None):
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()
    start_time = time.time()

    filename = "hz_poi_all.csv"
    poi_df_ = pd.read_csv(filename)
    neighbors_model = NearestNeighbors(radius=1000,leaf_size=100, metric=get_distance_lnglat, n_jobs=8)
    neighbors_model.fit(poi_df_[["wgs_lng","wgs_lat"]])
    print("neighbors_model fit done")

    namelist = np.array(poi_df_["name"].tolist())
    addresslist = np.array(poi_df_["address"].tolist())
    landmark_index_ = []
    print("imput row to task")
    for row in tqdm(poi_df_.iterrows()):
        tasks.put(row)

    for i in range(num_process):
        tasks.put(None)

    args = {'neighbors_model': neighbors_model,'namelist':namelist,'addresslist': addresslist}
    print('Create {} processes'.format(num_process))
    consumers = [Consumer(landmark_index, tasks, results, error_queue, **args)
                 for i in range(num_process)]
    for w in consumers:
        w.start()
    # Add a task tracking process
    task_tracker = TaskTracker(tasks, True)
    task_tracker.start()
    # Wait for all input data to be processed
    tasks.join()
    # If there is any error in any process, output the error messages
    num_error = error_queue.qsize()
    if num_error > 0:
        for i in range(num_error):
            print(error_queue.get())
        raise RuntimeError('Multi process jobs failed')
    else:
        # Collect results
        result_table = []
        while results.qsize():
            result_table.append(results.get())
            num_task -= 1
        print("Jobs finished in {0:.2f}s".format(
            time.time()-start_time))
        return result_table



if __name__ == "__main__":
    result_table = multi(20)
    pickle.dump(result_table, open("landmark_index","wb"))
