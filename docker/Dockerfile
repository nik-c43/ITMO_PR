# Образ для запуска проекта под ROS 2 Jazzy (LTS для Ubuntu 24.04).
# Внутри: ROS 2, MuJoCo для симуляции и библиотеки рендера без окна.
# Сам код проекта в образ не копируется: он подключается папкой при запуске (см. run.sh),
# поэтому после правки кода образ пересобирать не нужно.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-numpy python3-opencv python3-pil python3-pytest \
        ros-jazzy-vision-msgs ros-jazzy-tf2-ros ros-jazzy-message-filters \
        libegl1 libgl1 libgles2 libgl1-mesa-dri libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo ставится из PyPI: в репозиториях Ubuntu и ROS его нет.
# --break-system-packages нужен, потому что ROS использует системный Python без venv.
RUN pip install --no-cache-dir --break-system-packages "mujoco>=4.0,<5"

# Камера симулятора рендерит без окна через EGL.
ENV MUJOCO_GL=egl
# Пакеты ядра (sortbot_vision и другие) берутся прямо из подключённой папки проекта.
ENV PYTHONPATH=/repo
ENV PYTHONUNBUFFERED=1

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
