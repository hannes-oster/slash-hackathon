import time
from model import model
from os import times_result
import pandas
from pathlib import Path
import argparse
import logging
from utils import check_paths, set_logging, read_json
from geometric_utils import haversine
import math

from sklearn.neighbors import NearestNeighbors
import numpy as np

S2S_RADIUS = 50
MAX_SPEED = 5.0
MAX_COURSE_DIVERGENCE = 20
EARTH_RADIUS = 6371000

def construct_kd_tree(ship_data_orig):
    # print(ship_data.columns, "lon" in ship_data.columns)
    # print(ship_data)
    # print(ship_data.loc[:, "lon"])
    ship_data = ship_data_orig.copy()
    logging.debug(f"constructing kd tree from {len(ship_data)} rows")
    # print(ship_data[['lat', 'lon']])
    ship_data = ship_data[(ship_data.lat != "---") & (ship_data.lon != "---")]
    bt = NearestNeighbors(n_neighbors=1, radius=S2S_RADIUS / EARTH_RADIUS, metric='haversine').fit(np.deg2rad(ship_data[['lat', 'lon']].astype(float).values))
    logging.debug("tree created")
    return bt

def get_close_ship_pairs(kd_tree: NearestNeighbors):
    logging.debug("getting close pairs...")
    close_pairs = kd_tree.radius_neighbors(return_distance=False)

    logging.debug(f"found close pairs, shape: {len(close_pairs)}x{len(close_pairs[0])}")
    
    logging.debug("extracting point pairs")
    pairs = set()
    for i in range(len(close_pairs)):
        for j in close_pairs[i]:
            pairs.add((min(i, j), max(i, j)))
            # print(haversine(kd_tree.))
    logging.debug(f"{len(pairs)} pairs extracted!")
    return list(pairs)

def filter_suspicious(ship_pairs, positional_ship_data, per_ship_data, config):
    # print(config)
    suspicios_source_ports = config["suspicious_source_ports"]
    # sanctioning_destination_ports = config["sanctioning_destination_ports"]
    
    filtered_pairs = []
    sd = positional_ship_data[(positional_ship_data.lat != "---") & (positional_ship_data.lon != "---")]
    # radiants = np.deg2rad(sd[['lat', 'lon']].astype(float).values)
    # raw = sd[['lat', 'lon']].astype(float).values
    # for (i, j) in close_ship_pairs:
    for i, j in ship_pairs:
        # print(haversine(radiants[i], radiants[j]), S2S_RADIUS / EARTH_RADIUS, raw[i], raw[j])
        position_row_1, position_row_2 = positional_ship_data.iloc[i], positional_ship_data.iloc[j]
        # print(position_row_1)
        # row_1, row_2 = per_ship_data.loc[per_ship_data["MMSI"] == position_row_1["mmsi"]], per_ship_data.loc[per_ship_data["MMSI"] == position_row_2["mmsi"]]
        # print(row_1)
        suspicious_receiver, suspicious_deliverer = position_row_1, position_row_2
        if position_row_1["mmsi"] == position_row_2["mmsi"]:
            continue
        if position_row_1["source_port"] in suspicios_source_ports:
            suspicious_deliverer = position_row_1
            if position_row_2["dest_port"] in suspicios_source_ports:
                continue
            suspicious_receiver = position_row_2
        elif position_row_2["source_port"] in suspicios_source_ports:
            suspicious_deliverer = position_row_2
            if position_row_1["dest_port"] in suspicios_source_ports:
                continue
            suspicious_receiver = position_row_1
        else:
            continue
        # if position_row_1["source_port"] in suspicios_source_ports or position_row_1["dest_port"] in suspicios_source_ports:
        #     suspicious_deliverer = position_row_1
        #     if position_row_2["source_port"] in suspicios_source_ports or position_row_2["dest_port"] in suspicios_source_ports:
        #         continue
        #     suspicious_receiver = position_row_2
        # elif position_row_2["source_port"] in suspicios_source_ports or position_row_2["dest_port"] in suspicios_source_ports:
        #     suspicious_deliverer = position_row_2
        #     if position_row_1["source_port"] in suspicios_source_ports or position_row_1["dest_port"] in suspicios_source_ports:
        #         continue
        #     suspicious_receiver = position_row_1
        # else:
        #     continue
        
        if position_row_2["course"] != "---" and position_row_1["course"] != "---":
            if abs(float(position_row_2["course"]) - float(position_row_1["course"])) > MAX_COURSE_DIVERGENCE:
                continue
        
        if position_row_2["speed"] != "---" and position_row_1["speed"] != "---":
            speed_1, speed_2 = float(position_row_1["speed"].replace("Knots", "")), float(position_row_2["speed"].replace("Knots", ""))
            if speed_1 > MAX_SPEED or speed_2 > MAX_SPEED :
            # if speed_1 > MAX_SPEED or speed_2 > MAX_SPEED or speed_1 < 0.2 or speed_2 < 0.2:
                continue
        
        # suspicious_deliverer, suspicious_receiver = row_1, row_2
        
        logging.debug(f"linking {suspicious_deliverer['source_port']} to {suspicious_receiver['dest_port']}")
        filtered_pairs.append((
            list(suspicious_deliverer[["mmsi", "lon", "lat", "speed", "course", "source_port", "dest_port"]]),
            list(suspicious_receiver[["mmsi", "lon", "lat", "speed", "course", "source_port", "dest_port"]]),
        ))
        # print("now added:", filtered_pairs[-1])
    return filtered_pairs
    # return ship_pairs

def find_suspicios_pairs(positional_data_path, per_ship_data_path, config):
    positional_data_path = Path(positional_data_path)
    per_ship_data_path = Path(per_ship_data_path)
    check_paths(positional_data_path, per_ship_data_path)
    
    ship_data = pandas.read_csv(positional_data_path)
    ship_data = ship_data[(ship_data.lat != "---") & (ship_data.lon != "---")]
    per_ship_data = pandas.read_csv(per_ship_data_path)

    ship_kd_tree = construct_kd_tree(ship_data)
    close_ship_pairs = get_close_ship_pairs(ship_kd_tree)
    filtered_ship_pairs = filter_suspicious(close_ship_pairs, ship_data, per_ship_data, config)
    # print(filtered_ship_pairs)

    # print(filtered_ship_pairs)
    timestamps = ship_data[["timestamp"]].astype(str).dropna().values[1:]
    timestamps = [timestamp for timestamp in timestamps if timestamp[0] != "nan"]
    # print(timestamps, max(timestamps))
    return filtered_ship_pairs, max(timestamps)[0]

def get_ship_name(per_ship_data, mmsi):
    name = list(per_ship_data[per_ship_data["MMSI"] == mmsi]["name"].values)
    if len(name) > 0:
        name = name[0]
    if isinstance(name, list) and len(name) == 0:
        name = "Unknown Ship Name"
    # print(name)
    return name

def main(args):
    set_logging(args.log_file, args.log_level, args.log_stdout)
    config = read_json(args.config)
    positional_data_path = Path(args.positional_data_dir)
    per_ship_data_path = Path(args.per_ship_data_dir)
    check_paths(positional_data_path, per_ship_data_path)
    
    positional_data_files = sorted(list(positional_data_path.iterdir()))
    per_ship_data_files = sorted(list(per_ship_data_path.iterdir()))
    
    # if args.history_file is None:
    #     args.history_file = Path("hist.csv")
    # args.history_file = Path(args.history_file)
    # if not args.history_file.exists():
    #     with args.history_file.open("w") as hist_file:
    #         hist_file.write("deliverer_mmsi,receiver_mmsi,lon,lat,speed,course,source_deliverer,dest_deliverer,source_receiver,dest_receiver,suspiciousness_duration")
    upload_model = model()
    
    for positional_data in positional_data_files:
        
        per_ship_data = pandas.read_csv(per_ship_data_files[0])
        suspicios_ship_pairs, last_update = find_suspicios_pairs(positional_data, per_ship_data_files[0], config)
        logging.warning(f"final suspicios ship pairs: {len(suspicios_ship_pairs)}")
        suspicious_incidents = pandas.DataFrame(columns=[
            "deliverer_name",
            "receiver_name",
            "deliverer_mmsi",
            "receiver_mmsi",
            "lon_deliverer",
            "lat_deliverer",
            "speed_deliverer",
            "course_deliverer",
            "source_deliverer",
            "dest_deliverer",
            "lon_receiver",
            "lat_receiver",
            "speed_receiver",
            "course_receiver",
            "source_receiver",
            "dest_receiver",
            "suspiciousness_duration"
        ])
        for pair in suspicios_ship_pairs:
            deliverer_mmsi, lon_deliverer, lat_deliverer, speed_deliverer, course_deliverer, source_deliver, dest_deliver = pair[0]
            receiver_mmsi, lon_recv, lat_recv, speed_recv, course_recv, source_recv, dest_recv = pair[1]
            # print(deliverer_mmsi)
            if deliverer_mmsi not in suspicious_incidents[["deliverer_mmsi"]].values:
                suspicious_incidents.loc[len(suspicious_incidents)] = [
                    get_ship_name(per_ship_data, deliverer_mmsi),
                    get_ship_name(per_ship_data, receiver_mmsi),
                    deliverer_mmsi,
                    receiver_mmsi,
                    lon_deliverer,
                    lat_deliverer,
                    speed_deliverer,
                    course_deliverer,
                    source_deliver,
                    dest_deliver,
                    lon_recv,
                    lat_recv,
                    speed_recv,
                    course_recv,
                    source_recv,
                    dest_recv,
                    0
                ]
                continue
            deliverer_rows = suspicious_incidents.loc[suspicious_incidents["deliverer_mmsi"] == deliverer_mmsi]
            if receiver_mmsi not in deliverer_rows[["receiver_mmsi"]].values:
                suspicious_incidents.loc[len(suspicious_incidents)] = [
                    get_ship_name(per_ship_data, deliverer_mmsi),
                    get_ship_name(per_ship_data, receiver_mmsi),
                    deliverer_mmsi,
                    receiver_mmsi,
                    lon_deliverer,
                    lat_deliverer,
                    speed_deliverer,
                    course_deliverer,
                    source_deliver,
                    dest_deliver,
                    lon_recv,
                    lat_recv,
                    speed_recv,
                    course_recv,
                    source_recv,
                    dest_recv,
                    0
                ]
                continue
            
            indices = (suspicious_incidents["deliverer_mmsi"] == deliverer_mmsi) & (suspicious_incidents["receiver_mmsi"] == receiver_mmsi)
            indices = np.flatnonzero(indices.to_numpy())
            suspicious_incidents.at[indices[0], "suspiciousness_duration"] += 1
        # args.history_file.unlink()
        # print(last_update)
        suspicious_file_name = Path(f"incidents/suspicious_incidents_{last_update}.csv")
        suspicious_file_name.parent.mkdir(parents=True, exist_ok=True)
        if suspicious_file_name.exists():
            suspicious_file_name.unlink()
        suspicious_file_name.touch()
        suspicious_incidents.to_csv(suspicious_file_name, index=False)
        suspicious_incidents.to_json(suspicious_file_name.parent / (suspicious_file_name.stem + ".json"), index=True, orient="records", indent=4)
        suspicious_incidents = suspicious_incidents.fillna(value="")
        suspicios_data_json = suspicious_incidents.to_dict(orient="records")
        upload_model.write_suspicious_data(suspicios_data_json)
        time.sleep(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("s2s detection")
    parser.add_argument("--log-level", type=str, default="debug",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    parser.add_argument("--log-file", type=str, default="log.log",
                     help="output file path for logging. default to stdout")
    parser.add_argument("--log-stdout", action="store_true", default=True,
                     help="toggles force logging to stdout. if a log file is specified, logging will be "
                     "printed to both the log file and stdout")
    parser.add_argument("--config", help="path to positional data csv")
    parser.add_argument("--positional-data-dir", help="path to positional data csv")
    parser.add_argument("--per-ship-data-dir", help="path to per ship data data csv")
    parser.add_argument("--history-file", help="path to per ship data data csv")
    args = parser.parse_args()
    main(args)
