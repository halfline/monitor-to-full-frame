#!/usr/bin/python3

import cv2
import numpy as np
import subprocess
import threading
import traceback
from collections import deque
from sortedcontainers import SortedDict
from itertools import takewhile
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import queue
import sys
from enum import Enum, auto

import json

input_file = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
output_file = sys.argv[2] if len(sys.argv) > 2 else 'output.mp4'
audio_loop = GLib.MainLoop()

class ZoomState(Enum):
    ZOOMED_OUT = auto()
    ZOOMING_IN = auto()
    ZOOMED_IN = auto()
    ZOOMING_OUT = auto()

def ease_out_quad(time):
    return 1.0 - (1.0 - time)**2

def on_message(bus, message):
    if message.type == Gst.MessageType.EOS:
        audio_loop.quit()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print('Warning: %s' % err, debug)
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print('Error: %s' % err, debug)
        audio_loop.quit()
    return True

def on_level(bus, message, audio_levels, condition):
    if message.get_structure().get_name() == 'level':
        audio_level = message.get_structure().get_value('peak')[0]
        timestamp = message.get_structure().get_value('timestamp')
        audio_levels[timestamp] = audio_level
        with condition:
            condition.notify_all()
    return True

def get_rotation(video_file):
    command = ['exiftool', '-Rotation', '-json', video_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()

    metadata = json.loads(stdout)[0]
    return int(metadata.get('Rotation'))

def extract_audio(audio_levels, condition):
    print("Generating audio waveform data")

    pipeline = Gst.parse_launch(f'filesrc location={input_file} ! decodebin ! audioconvert ! audio/x-raw,channels=1 ! level ! fakesink')

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', on_message)
    bus.connect('message::element', on_level, audio_levels, condition)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        audio_loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)

    with condition:
        condition.notify_all()

def order_points(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
    points = points[np.argsort(angles)]

    return np.roll(points, shift=-1, axis=0)

def draw_biggest_object(image):
    image_height, image_width = image.shape[:2]

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, quantized_image = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY)
    objects, _ = cv2.findContours(quantized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_object = max(objects, key=cv2.contourArea)
    hull = cv2.convexHull(biggest_object)
    epsilon = 0.01 * cv2.arcLength(hull, True)
    hull = cv2.approxPolyDP(hull, epsilon, True)

    if len(hull) > 4 and len(hull) < 8:
        unaligned_box_around_object = cv2.minAreaRect(hull)
        hull = np.intp(cv2.boxPoints(unaligned_box_around_object))

    cv2.drawContours(image, [hull], -1, (0, 255, 0), 3)
    return image

def draw_outline_on_where_to_zoom(image, matrix, width, height):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
    corners = corners.reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, np.linalg.inv(matrix))
    warped_corners = warped_corners.astype(int)
    cv2.drawContours(image, [warped_corners], -1, (0, 255, 255), 3)

    return image

def calculate_transform_matrix(image, width, height):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, quantized_image = cv2.threshold(grayscale_image, 100, 255, cv2.THRESH_BINARY)
    objects, _ = cv2.findContours(quantized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_object = max(objects, key=cv2.contourArea)
    hull = cv2.convexHull(biggest_object)
    epsilon = 0.01 * cv2.arcLength(hull, True)
    hull = cv2.approxPolyDP(hull, epsilon, True)

    if len(hull) > 4 and len(hull) < 8:
        unaligned_box_around_object = cv2.minAreaRect(hull)
        hull = np.intp(cv2.boxPoints(unaligned_box_around_object))

    if len(hull) != 4:
        return None

    input_points = order_points(hull.reshape(4, 2))
    output_points = order_points(np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]))

    input_rectangle = np.array(input_points, dtype=np.float32)
    output_rectangle = np.array(output_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(input_rectangle, output_rectangle)

    return matrix

def find_monitor_in_frames(input_file, transformation_matrices, condition, min_queue_length):
    video = cv2.VideoCapture(input_file)
    frame_number = 0

    while True:
        success, image = video.read()
        if not success:
            break
        transform_matrix = calculate_transform_matrix(image, width, height)

        transformation_matrices[frame_number] = transform_matrix
        with condition:
            if len(transformation_matrices) >= min_queue_length:
                condition.notify()
        frame_number += 1

    video.release()

def rotate_image (image, rotation):
    if rotation == 180:
        image = cv2.rotate (image, cv2.ROTATE_180)
    elif rotation == 90:
        image = cv2.rotate (image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270:
        image = cv2.rotate (image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

def draw_audio_bars(image, width, height, audio_level, minimum_audio_level, maximum_audio_level):
    audio_bars_width = int(0.10 * width)
    audio_bars_height = int(0.025 * height)
    audio_bars_image = np.zeros((audio_bars_height, audio_bars_width, 3), dtype=np.uint8)

    max_audio_bars = 20
    number_of_audio_bars = int(audio_level * max_audio_bars)

    audio_bar_thickness = audio_bars_width // max_audio_bars
    audio_bar_thickness = audio_bar_thickness // 3

    for i in range(number_of_audio_bars):
        x1 = int (i * (audio_bar_thickness * 2))
        x2 = x1 + audio_bar_thickness
        y1 = 0
        y2 = audio_bars_height

        progress = (1.0 * i) / number_of_audio_bars
        color = (int(128 * progress), 64 + int(128 * progress), int(128 * progress))
        cv2.rectangle(audio_bars_image, (x1, y1), (x2, y2), color, -1)

    x_offset = width - audio_bars_width - 20
    y_offset = height - audio_bars_height - 20

    image[y_offset:y_offset+audio_bars_height, x_offset:x_offset+audio_bars_width] = audio_bars_image

    return image

def get_audio_levels_for_timestamp(audio_levels, timestamp, state, condition):
    with condition:
        while not audio_levels or timestamp > audio_levels.peekitem(-1)[0]:
            condition.wait()

    if not audio_levels or timestamp > audio_levels.peekitem(-1)[0]:
        raise Exception("No audio frames left")

    index = audio_levels.bisect_left(timestamp)

    if index != 0 and (index == len(audio_levels) or audio_levels.iloc[index] - timestamp > timestamp - audio_levels.iloc[index - 1]):
        index -= 1

    closest_timestamp = audio_levels.iloc[index]
    audio_level = audio_levels[closest_timestamp]

    if state is None:
        state = {'minimum-audio-level': audio_level, 'maximum-audio-level': audio_level }

    minimum_audio_level = min(state['minimum-audio-level'], min(list(audio_levels.values())))
    maximum_audio_level = max(state['maximum-audio-level'], max(list(audio_levels.values())))

    if maximum_audio_level != minimum_audio_level:
        audio_level = (audio_level - minimum_audio_level) / (maximum_audio_level - minimum_audio_level)
    else:
        audio_level = 0.0

    stale_indices = list(audio_levels.islice(stop=index))

    for index in stale_indices:
        del audio_levels[index]

    state['minimum-audio-level'] = minimum_audio_level
    state['maximum-audio-level'] = maximum_audio_level

    return (audio_level, minimum_audio_level, maximum_audio_level, state)

Gst.init(None)

output = None
video = cv2.VideoCapture(input_file)

success = video.isOpened()

if success:
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(f'appsrc ! videoconvert ! x264enc ! mpegtsmux ! filesink location={output_file}', cv2.VideoWriter_fourcc(*'x264'), fps, (width,height), isColor=True)
    success = output.isOpened()

    fps = int(fps)

audio_levels = SortedDict()
audio_levels_changed_condition = threading.Condition()

if success:
    print("Starting audio extraction")
    audio_thread = threading.Thread(target=extract_audio, args=(audio_levels, audio_levels_changed_condition))
    audio_thread.start()

warp_time = 2.5
steady_time = 2.5
min_transformation_matrices_length = int(fps * (warp_time * 2 + steady_time))
transforms_condition = threading.Condition()
transformation_matrices = {}
if success:
    print("Analyzing frames for monitor")
    analyzer_thread = threading.Thread(target=find_monitor_in_frames, args=(f"{input_file}", transformation_matrices, transforms_condition, min_transformation_matrices_length))
    analyzer_thread.start()

i = 0
rotation_matrices = []
cropping_rects = []
transformations_buffer = deque(maxlen=fps)
audio_state = None
zoom_state = ZoomState.ZOOMED_OUT
last_zoomed_out_frame = 0
untransformed_matrix = np.eye(3, dtype=np.float32)

try:
    rotation = get_rotation(input_file);
except Exception as e:
    print(e)
    rotation = 0

try:
    while success:
        success, pristine_image = video.read()

        if not success:
            break

        try:
            print("Drawing frame " + str(i));

            image = pristine_image
            #image = draw_biggest_object (image)
            with transforms_condition:
                while len(transformation_matrices) <= min_transformation_matrices_length:
                    print(f"\rWaiting for matrices [{len(transformation_matrices)}/{min_transformation_matrices_length}]")
                    transforms_condition.wait()

            #if transformation_matrices[i] is not None:
            #    image = draw_outline_on_where_to_zoom (image, transformation_matrices[i], width, height)

            if zoom_state == ZoomState.ZOOMED_OUT:
                consecutive_matrices = list(takewhile(lambda matrix: matrix is not None, transformation_matrices))
                if len(consecutive_matrices) >= min_transformation_matrices_length:
                    last_zoomed_out_frame = i
                    zoom_state = ZoomState.ZOOMING_IN
                    print("Zooming in")
            elif zoom_state == ZoomState.ZOOMING_IN:
                fraction = (i - last_zoomed_out_frame) / (warp_time * fps)
                fraction = ease_out_quad (fraction)
                print (f"zooming in {fraction * 100}%")
                matrix = untransformed_matrix * (1 - fraction) + transformation_matrices[i] * fraction
                image = cv2.warpPerspective (image, matrix, (width, height))
                if fraction >= 1.0:
                    zoom_state = ZoomState.ZOOMED_IN
                    print("Done zooming in")
            elif zoom_state == ZoomState.ZOOMED_IN:
                image = cv2.warpPerspective (image, transformation_matrices[i], (width, height))

                consecutive_matrices = list(takewhile(lambda matrix: matrix is not None, transformation_matrices))
                if len(consecutive_matrices) < fps * (steady_time + warp_time):
                    last_zoomed_in_frame = i
                    zoom_state = ZoomState.ZOOMING_OUT
                    print("Zooming out")
            elif zoom_state == ZoomState.ZOOMING_OUT:
                fraction = (i - last_zoomed_in_frame) / (warp_time * fps)
                matrix = transformation_matrices[i] * (1 - fraction) + untransformed_matrix * fraction
                image = cv2.warpPerspective (image, matrix, (width, height))
                if fraction >= 1.0:
                    zoom_state = ZoomState.ZOOMED_OUT
                    print("Done zooming out")

            del transformation_matrices[i]

            image = rotate_image (image, rotation)

            timestamp = video.get(cv2.CAP_PROP_POS_MSEC) * 1e6

            if audio_loop.is_running():
                audio_level, minimum_audio_level, maximum_audio_level, audio_state = get_audio_levels_for_timestamp (audio_levels, timestamp, audio_state, audio_levels_changed_condition)
                image = draw_audio_bars(image, width, height, audio_level, minimum_audio_level, maximum_audio_level);

        except Exception as e:
            print(e)
            traceback.print_exc()
            image = pristine_image

        output.write(image)

        i = i + 1
except:
    pass

if output:
    output.release()

if audio_thread:
    if audio_loop.is_running():
        audio_loop.quit()
    audio_thread.join()

if analyzer_thread:
    analyzer_thread.join()
