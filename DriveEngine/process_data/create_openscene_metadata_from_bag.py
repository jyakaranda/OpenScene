import argparse
import os
from os import listdir
from os.path import isfile, join
from typing import Any, List, Dict, Optional, Union

import multiprocessing
import munch
import numpy as np
import pickle
from pyquaternion import Quaternion
from tqdm import tqdm

from helpers.multiprocess_helper import get_scenes_per_thread

import fastbag
from pluspy import topic_utils

from localization_pb2 import LocalizationEstimation
from obstacle_detection_pb2 import ObstacleDetection, PerceptionObstacle

filtered_classes = ["traffic_cone", "barrier", "czone_sign", "generic_object"]


def get_hash_value(data: Union[int, float, str, List, Dict]) -> str:
    import hashlib
    # Convert non-hashable types (like dict) to hashable types
    if isinstance(data, dict):
        import json
        data = json.dumps(data, sort_keys=True)

    # Convert all data to bytes and hash it
    data_bytes = str(data).encode('utf-8')
    hash_value = hashlib.sha256(data_bytes).hexdigest()

    return hash_value[:16]


def open_bag(bag_file: str) -> Optional[fastbag.Reader]:
    if not os.path.exists(bag_file):
        raise Exception(f"The input bag file {bag_file} doesn't  exist")

    if bag_file.endswith('.db'):
        bag = fastbag.Reader(bag_file)
        bag.open()
    else:
        raise Exception(f'Input bag file: {bag_file} not in supported format, only support fastbag(.db)')

    return bag


def convert_object_type(obs_type: PerceptionObstacle.Type) -> Optional[str]:
    plus_type_to_nuplan_dict: Dict[PerceptionObstacle.Type, str] = {
        PerceptionObstacle.CAR: "vehicle",
        PerceptionObstacle.TRUCK: "vehicle",
        PerceptionObstacle.BUS: "vehicle",
        PerceptionObstacle.SUV: "vehicle",
        PerceptionObstacle.LIGHTTRUCK: "vehicle",
        PerceptionObstacle.VIRTUAL_CAR: "vehicle",
        PerceptionObstacle.PEDESTRIAN: "pedestrian",
        PerceptionObstacle.BICYCLE: "bicycle",
        PerceptionObstacle.MOTO: "bicycle",
        PerceptionObstacle.CONE: "traffic_cone",
        PerceptionObstacle.BARRIER: "barrier",
        PerceptionObstacle.CONSTRUCTION_AHEAD: "czone_sign",
        PerceptionObstacle.ROAD_WORK_AHEAD: "czone_sign",
        PerceptionObstacle.UNKNOWN: "generic_object",
        # "ego" type is not incluced in plus yet
    }
    if obs_type not in plus_type_to_nuplan_dict.keys():
        # print(f"not processed obs_type: {PerceptionObstacle.Type.Name(obs_type)}, ignore it")
        return None
    return plus_type_to_nuplan_dict[obs_type]


def populate_single_frame(frame: munch.Munch, frame_idx: int, localization_state: LocalizationEstimation, obstacle_detection: ObstacleDetection):
    # basic info
    # this is foreign key in nuplan tables which is unuseful if we don't use nuplan scheme
    frame.token = get_hash_value([frame.log_name, frame_idx])
    frame.frame_idx = frame_idx
    frame.timestamp = obstacle_detection.header.timestamp_msec * 1e-3
    # temporarily using log_name as scene_name
    frame.scene_name = frame.log_name
    # this is foreign key in nuplan tables which is unuseful if we don't use nuplan scheme
    frame.scene_token = frame.log_token

    # localization info
    frame.can_bus = [
        localization_state.position.x,
        localization_state.position.y,
        localization_state.position.z,
        localization_state.orientation.qx,
        localization_state.orientation.qy,
        localization_state.orientation.qz,
        localization_state.orientation.qw,
        localization_state.linear_acceleration.x,
        localization_state.linear_acceleration.y,
        localization_state.linear_acceleration.z,
        localization_state.linear_velocity.x,
        localization_state.linear_velocity.y,
        localization_state.linear_velocity.z,
        localization_state.angular_velocity.x,
        localization_state.angular_velocity.y,
        localization_state.angular_velocity.z,
        # yaw in radians
        0.0,
        # yaw in degrees
        0.0,
    ]
    frame.ego2global_translation = frame.can_bus[:3]
    frame.ego2global_rotation = frame.can_bus[3:7]
    frame.ego_dynamic_state = [
        localization_state.linear_velocity.x,
        localization_state.linear_velocity.y,
        localization_state.linear_acceleration.x,
        localization_state.linear_acceleration.y,
    ]

    e2g_r_mat = Quaternion(frame.ego2global_rotation).rotation_matrix
    e2g = np.eye(4)
    e2g[:3, :3] = e2g_r_mat
    e2g[:3, -1] = frame.ego2global_translation
    frame.ego2global = e2g

    # unknown driving command by default
    frame.driving_command = [0, 0, 0, 1]

    # calibration info
    frame.lidar2ego = np.eye(4)
    frame.lidar2ego_translation = np.zeros(3)
    frame.lidar2ego_rotation = [0.0, 0.0, 0.0, 1.0]
    frame.lidar2global = frame.ego2global

    # perception info
    anns = frame.anns
    obstacles_with_type = list(map(lambda obs: (obs, convert_object_type(obs.type)), obstacle_detection.obstacle))
    filtered_obstacles_with_type = list(filter(lambda obs_with_type: obs_with_type[1] is not None, obstacles_with_type))
    filtered_obstacles = list(map(lambda obs_with_type: obs_with_type[0], filtered_obstacles_with_type))
    # in local imu(lidar frame in nuplan) frame
    obstacle_velocities = list(map(lambda obs: np.array([obs.motion.vx, obs.motion.vy, 0.0]) @ np.linalg.inv(e2g_r_mat).T, filtered_obstacles))
    obstacle_positions = list(map(lambda obs: [obs.motion.xrel, obs.motion.yrel, obs.motion.zrel], filtered_obstacles))
    obstacle_rots = list(map(lambda obs: [obs.motion.yaw_rel], filtered_obstacles))
    obstacle_dims = list(map(lambda obs: [obs.length, obs.width, obs.height], filtered_obstacles))

    anns.track_tokens = list(map(lambda obs: obs.id, filtered_obstacles))
    # this is foreign key in nuplan tables which is unuseful if we don't use nuplan scheme
    anns.instance_tokens = list(map(lambda obs: get_hash_value([frame.log_name, frame_idx, obs.id]), filtered_obstacles))
    anns.gt_names = list(map(lambda obs_with_type: obs_with_type[1], filtered_obstacles_with_type))
    anns.gt_velocity_3d = obstacle_velocities
    anns.gt_boxes = np.concatenate([obstacle_positions, obstacle_dims, obstacle_rots], axis=-1)
    # print(f"original obstacles: {len(obstacles_with_type)}, filtered obstacles: {len(filtered_obstacles)}")
    pass


def create_nuplan_info_from_db(args, db_name: str, msg_decoder: topic_utils.MessageDecoder):
    tqdm.write(f"{multiprocessing.current_process().name}: {db_name}")
    pkl_file_path = f"{args.out_dir}/{db_name[:-3]}.pkl"
    if os.path.exists(pkl_file_path) and not args.ignore_existed:
        return

    db_path = os.path.join(args.dataset_db_path, db_name)
    bag = open_bag(db_path)
    topics = ['/localization/state', '/perception/obstacles', '/perception/calibrations']
    frame_template = munch.Munch(
        token=None,
        frame_idx=None,
        timestamp=None,
        log_name=db_name,
        log_token=get_hash_value(db_name),
        scene_name=None,
        scene_token=None,
        vehicle_name=bag.get_vehicle(),
        can_bus=[],
        # duplicated with can_bus
        ego2global=None,
        ego2global_translation=[],
        ego2global_rotation=[],
        ego_dynamic_state=[],
        driving_command=[],
        anns=munch.Munch(
            gt_boxes=[],
            gt_names=[],
            gt_velocity_3d=[],
            instance_tokens=[],
            track_tokens=[],
        ),
        # need map convertion
        map_location='sg-one-north',
        roadblock_ids=[],
        # don't have yet
        traffic_lights=[],
        # unused
        cams={
            "cam_f0": None,
            "cam_l0": None,
            "cam_l1": None,
            "cam_l2": None,
            "cam_r0": None,
            "cam_r1": None,
            "cam_r2": None,
            "cam_b0": None,
        },
        lidar2ego=None,
        lidar2global=None,
        lidar_path="",
        lidar2ego_translation=[],
        lidar2ego_rotation=[],
        sample_prev=None,
        sample_next=None,
    )

    frame_infos = []
    latest_loc_state: LocalizationEstimation = None
    for topic, raw_msg, t in bag.read_messages(topics=topics, ros_time=False, raw=True):
        if topic == '/localization/state':
            latest_loc_state = msg_decoder.decode(topic, raw_msg[1], raw=True)
        elif topic == '/perception/obstacles':
            latest_obs_detection: ObstacleDetection = msg_decoder.decode(topic, raw_msg[1], raw=True)
            frame = frame_template.copy()
            # 处理populate异常情况：1. empty loc state; 2. unsynced messages
            populate_single_frame(frame, len(frame_infos), latest_loc_state, latest_obs_detection)
            frame_infos.append(frame)
        else:
            continue
    
    with open(pkl_file_path, "wb") as f:
        pickle.dump(frame_infos, f, protocol=pickle.HIGHEST_PROTOCOL)

    # tqdm.write(f"{frame_infos[0]}")


def create_nuplan_info(args):
    # get all db files & assign db files for current thread.
    dataset_db_path = args.dataset_db_path
    db_names_with_extension = [
        f for f in listdir(dataset_db_path) if isfile(join(dataset_db_path, f))]
    db_names_with_extension.sort()
    db_names_splited, start = get_scenes_per_thread(db_names_with_extension, args.thread_num)
    os.makedirs(args.out_dir, exist_ok=True)

    # For each sequence...
    msg_decoder = topic_utils.MessageDecoder()
    for log_db_name in tqdm(db_names_splited, dynamic_ncols=True):
        create_nuplan_info_from_db(args, log_db_name, msg_decoder)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument(
        "--thread-num", type=int, default=16, help="number of threads for multi-processing."
    )

    # directory configurations.
    parser.add_argument("--dataset-root-path", type=str, default=None, help="the path to dataset root path.")
    parser.add_argument("--dataset-db-path", type=str, default=None, help="the dir saving dataset db.")
    parser.add_argument("--dataset-sensor-path", type=str, default=None, help="the dir to dataset sensor data.")
    parser.add_argument("--dataset-map-version", type=str, default=None, help="dataset mapping dataset version.")
    parser.add_argument("--dataset-map-root", type=str, default=None, help="path to dataset map data.")
    parser.add_argument("--out-dir", type=str, default=None, help="output path.")

    # nuplan数据是20hz，这里10 interval就是down sample成2hz
    parser.add_argument(
        "--sample-interval", type=int, default=10, help="interval of key frame samples."
    )
    parser.add_argument("--ignore-existed", default=False, action="store_true", help="Ignore existed pickle file and override it with latest dump")

    # split.
    parser.add_argument("--is-test", default=False, action="store_true", help="Dealing with Test set data.")
    parser.add_argument(
        "--filter-instance", default=False, action="store_true", help="Ignore instances in filtered_classes."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    threads = []
    for x in range(args.thread_num):
        t = multiprocessing.Process(
            target=create_nuplan_info,
            name=str(x),
            args=(args,),
        )
        threads.append(t)
    for thr in threads:
        thr.start()
    for thr in threads:
        if thr.is_alive():
            thr.join()
