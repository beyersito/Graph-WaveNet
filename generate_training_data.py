from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.spatial.distance import cdist
import pickle
import seaborn as sns

from matplotlib import rcParams
rcParams['figure.figsize'] = 15,15


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def subset_by_iqr(df, column, whisker_width=1.5):
    """Remove outliers from a dataframe by column, including optional 
       whiskers, removing rows for which the column value are 
       less than Q1-1.5IQR or greater than Q3+1.5IQR.
    Args:
        df (`:obj:pd.DataFrame`): A pandas dataframe to subset
        column (str): Name of the column to calculate the subset from.
        whisker_width (float): Optional, loosen the IQR filter by a
                               factor of `whisker_width` * IQR.
    Returns:
        (`:obj:pd.DataFrame`): Filtered dataframe
    """
    # Calculate Q1, Q2 and IQR
    q1 = df[column].quantile(0.25)                 
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR, including optional whiskers
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter]                                                     


def generate_adj_dist(df, normalized_k=0.01):
    coord = df[['lat', 'long']].values
    dist_mx = cdist(coord, coord,
                   lambda u, v: geodesic(u, v).kilometers)
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx

def generate_adj_matrix_from_trips(df, stations_ids, normalized_k = 0.1, weight_by_time = False, step_duration_minutes = 60):
    num_stations = len(stations_ids)
    adj_mx = np.zeros((num_stations, num_stations), dtype=np.float32)
   
    
    station_id_to_ind = {}
    for i, station_id in enumerate(stations_ids):
        station_id_to_ind[station_id] = i
        
    for i in range(num_stations):
        for j in range(num_stations):
            if weight_by_time:
                adj_mx[i, j] = (df[(df['start_station_id'] == stations_ids[i]) &
                                             (df['end_station_id'] == stations_ids[j])].duration /
                                                                        (step_duration_minutes * 60)).sum()
            else:
                adj_mx[i, j] = df[(df['start_station_id'] == stations_ids[i]) &
                                             (df['end_station_id'] == stations_ids[j])].id.count()
            if adj_mx[i, j] == 0.0:
                adj_mx[i,j] = np.inf
            else:
                adj_mx[i,j] = 1 / adj_mx[i,j]

    values = adj_mx[~np.isinf(adj_mx)].flatten()
    std = values.std()
    adj_mx_raw = adj_mx.copy()
    adj_mx = np.exp(-np.square(adj_mx / std))
    
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx, adj_mx_raw


def generate_train_val_test(args):
    df = pd.read_csv(args.status_df_filename)

    df.time = pd.to_datetime(df.time).dt.round('min')
    print("Pivoting")
    df_m = df.pivot_table(index='time', columns='station_id', values=args.output_column_name, aggfunc=np.min)
    print("Resampling")
    df_mr = df_m.resample(args.resample_time).mean()

    #null treatment
    null_quantile = df_mr.isnull().sum().quantile(0.85)
    threshold_null = len(df_mr.index) - null_quantile
    print('Threshold of null rows per column', null_quantile)

    print('Columns to be removed', (df_mr.isnull().sum() > null_quantile).sum())

    df_mrn = df_mr.dropna(thresh=threshold_null, axis='columns', how='all').interpolate()

    print('Null values remaining', df_mrn.isnull().sum().sum())

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df_mrn,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
        
    
    # ADJ MX
    print("Generating adj mx")
    
    stations_df = pd.read_csv(args.station_df_filename)
    st_df = stations_df[stations_df.id.isin(list(df_mrn.columns))].reset_index(drop=True)

    if args.adj_mx_mode == "distance":
        adj_dist = generate_adj_dist(st_df)
        with open(os.path.join(args.output_dir, "adj_dist.pkl"), 'wb') as f:
            pickle.dump(adj_dist, f, protocol=2)
            
        mask = np.zeros_like(adj_dist)
        mask[np.triu_indices_from(mask)] = True

        ax_labels = [st_df.iloc[i].id for i in range(adj_dist.shape[0])]
        sns.heatmap(adj_dist, mask=mask, cmap="YlGnBu", xticklabels=ax_labels, yticklabels=ax_labels).get_figure().savefig(os.path.join(args.output_dir, "adj_dist.png"))
    elif args.adj_mx_mode == "trips":
        trips_df = pd.read_csv(args.trips_df_filename)
        trips_df_filt = subset_by_iqr(trips_df, 'duration', whisker_width=5)
        stations_ids = list(df_mrn.columns)
        adj_trips, _ = generate_adj_matrix_from_trips(trips_df_filt, stations_ids, weight_by_time=False)
        with open(os.path.join(args.output_dir, "adj_trips.pkl"), 'wb') as f:
            pickle.dump(adj_trips, f, protocol=2)
            
        ax_labels = [st_df.iloc[i].id for i in range(adj_trips.shape[0])]
        sns.heatmap(adj_trips, cmap="YlGnBu", xticklabels=ax_labels, yticklabels=ax_labels).get_figure().savefig(os.path.join(args.output_dir, "adj_trips.png"))


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/SF-BIKE", help="Output directory."
    )
    parser.add_argument(
        "--status_df_filename",
        type=str,
        default="data/sf-bay-area-bike-share/status.csv",
    )
    parser.add_argument(
        "--station_df_filename",
        type=str,
        default="data/sf-bay-area-bike-share/station.csv",
    )
    parser.add_argument(
        "--trips_df_filename",
        type=str,
        default="data/sf-bay-area-bike-share/trip.csv",
    )
    parser.add_argument("--output_column_name", type=str, default="bikes_available")
    parser.add_argument("--adj_mx_mode", type=str, default="distance")
    parser.add_argument("--resample_time", type=str, default="5min")

    args = parser.parse_args()
    print(args.__dict__)
    main(args)

