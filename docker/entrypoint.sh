#!/bin/bash
# Точка входа контейнера: подключает ROS 2, собирает пакет sortbot_ros и запускает команду.
set -e
source /opt/ros/jazzy/setup.bash

# Рабочее пространство ROS живёт внутри контейнера. Исходники в папке проекта /repo
# подключены только для чтения, поэтому сборка работает с их копией.
# Пакет на чистом Python собирается несколько секунд. Второй терминал того же
# контейнера (docker exec) сборку не повторяет.
if [ ! -d /ws/install ]; then
    mkdir -p /ws/src
    cp -r /repo/ros/sortbot_ros /ws/src/sortbot_ros
    cd /ws
    colcon build --packages-select sortbot_ros > /tmp/colcon_build.log 2>&1 \
        || { cat /tmp/colcon_build.log; exit 1; }
fi
source /ws/install/setup.bash

cd /repo
exec "$@"
