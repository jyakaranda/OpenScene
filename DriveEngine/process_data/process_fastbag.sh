# Scripts for processing fastbag data.

# Please install https://github.com/autonomousvision/navsim.
# This is used for generating High-Level Driving Commands, used in E2E AD.
# export NAVSIM_DEVKIT_ROOT=/home/henry.zhang/navsim
# export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}

export DATASET_PATH=/home/plusai/workspace/plusai/gameformer/dataset/plus/dataset
export DATASET_DB_PATH=${DATASET_PATH}/splits/mini
export DATASET_SENSOR_PATH=None
export DATASET_MAP_VERSION=nuplan-maps-v1.0
export DATASET_MAPS_ROOT=/home/plusai/workspace/plusai/gameformer/nuplan/dataset/maps

OUT_DIR=/home/plusai/workspace/plusai/gameformer/dataset/plus/exp

python create_openscene_metadata_from_bag.py \
  --dataset-root-path ${DATASET_PATH} \
  --dataset-db-path ${DATASET_DB_PATH} \
  --dataset-sensor-path ${DATASET_SENSOR_PATH} \
  --dataset-map-version ${DATASET_MAP_VERSION} \
  --dataset-map-root ${DATASET_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --ignore-existed \
  --thread-num 32
