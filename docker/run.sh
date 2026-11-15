#!/bin/bash
# Запуск проекта в контейнере с ROS 2 Jazzy.
#
#   docker/run.sh                    запустить сортировку: мост с MuJoCo, зрение, логика
#   docker/run.sh bash               открыть терминал в контейнере
#   docker/run.sh ros2 topic list    выполнить любую команду ROS 2
#
# Во второй терминал того же контейнера: docker exec -it sortbot /entrypoint.sh bash
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=sortbot-ros:jazzy

# Образ собирается при первом запуске (несколько минут) и после правки Dockerfile.
docker build -q -t "$IMAGE" "$REPO/docker" > /dev/null

if [ $# -eq 0 ]; then
    set -- ros2 launch sortbot_ros sorting.launch.py
fi

# --init: чтобы Ctrl+C и остановка контейнера доходили до узлов.
# Папка проекта подключается только для чтения; картинки примеров пишутся внутрь контейнера.
FLAGS=(--rm --init --name sortbot -v "$REPO:/repo:ro" -e SORTBOT_OUT=/tmp/sortbot_out)
[ -t 0 ] && FLAGS+=(-it)
# Видеокарта для рендера камеры. Без неё MuJoCo рендерит на процессоре, медленнее.
[ -e /dev/dri ] && FLAGS+=(--device /dev/dri)

exec docker run "${FLAGS[@]}" "$IMAGE" "$@"
