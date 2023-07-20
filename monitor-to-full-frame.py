#!/usr/bin/python3

import cv2
import numpy as np
import subprocess
import threading
import traceback
from collections import deque
from sortedcontainers import SortedDict
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import queue
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
output_file = sys.argv[2] if len(sys.argv) > 2 else 'output.mp4'

def on_message(bus, message, loop):
    if message.type == Gst.MessageType.EOS:
        loop.quit()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print('Warning: %s' % err, debug)
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print('Error: %s' % err, debug)
        loop.quit()
    return True

def on_level(bus, message, audio_levels, condition):
    if message.get_structure().get_name() == 'level':
        audio_level = message.get_structure().get_value('peak')[0]
        timestamp = message.get_structure().get_value('timestamp')
        audio_levels[timestamp] = audio_level
        with condition:
            condition.notify_all()
    return True

def extract_audio(audio_levels, condition):
    print("Generating audio waveform data")

    loop = GLib.MainLoop()
    pipeline = Gst.parse_launch(f'filesrc location={input_file} ! decodebin ! audioconvert ! audio/x-raw,channels=1 ! level ! fakesink')

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', on_message, loop)
    bus.connect('message::element', on_level, audio_levels, condition)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)

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

def transform_image_to_be_axis_aligned(image, width, height, transformation_matrices, max_matrices):
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

    if len(hull) == 4:
        input_points = order_points(hull.reshape(4, 2))
        output_points = order_points(np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]))

        input_rectangle = np.array(input_points, dtype=np.float32)
        output_rectangle = np.array(output_points, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(input_rectangle, output_rectangle)

        transformation_matrices.append(matrix)
        if len(transformation_matrices) > max_matrices:
            transformation_matrices.pop(0)

    if len (transformation_matrices) == 0:
        return image

    average_matrix = np.mean(transformation_matrices, axis=0)

    corrected_image = cv2.warpPerspective(image, average_matrix, (width, height))

    return corrected_image

def stabilize_image(image, transformations_buffer, max_transformations, state):
    orb = cv2.ORB_create()

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if state is None:
        key_points, descriptors = orb.detectAndCompute(grayscale_image, None)
        state = {'key-points': key_points, 'descriptors': descriptors}
        return image, state

    key_points, descriptors = orb.detectAndCompute(grayscale_image, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(state['descriptors'], descriptors)

    if len(matches) < 200:
        state['key-points'] = key_points
        state['descriptors'] = descriptors
        transformations_buffer.clear()
        return image, state

    matches = sorted(matches, key=lambda x: x.distance)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = state['key-points'][match.queryIdx].pt
        points2[i, :] = key_points[match.trainIdx].pt

    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    transformations_buffer.append(h)

    if len(transformations_buffer) > max_transformations:
        transformations_buffer.pop(0)

    average_transform = np.mean(transformations_buffer, axis=0)

    stabilized_image = cv2.warpPerspective(image, average_transform, (image.shape[1], image.shape[0]))

    state['key-points'] = key_points
    state['descriptors'] = descriptors

    return stabilized_image, state

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

def get_audio_levels_for_timestamp(audio_levels, timestamp, state):
    with condition:
        while not audio_levels or timestamp > audio_levels.peekitem(-1)[0]:
            condition.wait()

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
video = cv2.VideoCapture(sys.argv[1])

success = video.isOpened()

if success:
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(f'appsrc ! videoconvert ! x264enc ! mpegtsmux ! filesink location={output_file}', cv2.VideoWriter_fourcc(*'x264'), fps, (width,height), isColor=True)
    success = output.isOpened()

audio_levels = SortedDict()
condition = threading.Condition()

if success:
    print("Starting audio extraction")
    thread = threading.Thread(target=extract_audio, args=(audio_levels, condition))
    thread.start()

i = 0
rotation_matrices = []
cropping_rects = []
transformation_matrices = []
transformations_buffer = deque(maxlen=15)
stabilization_state = None
audio_state = None

try:
    while success:
        success, pristine_image = video.read()

        if not success:
            break

        try:
            print("Drawing frame " + str(i));
            image = pristine_image
            #image = draw_biggest_object (image)
            number_of_frames_to_smooth=fps
            image, stabilization_state = stabilize_image (image, transformations_buffer, number_of_frames_to_smooth, stabilization_state)
            image = transform_image_to_be_axis_aligned(image, width, height, transformation_matrices, number_of_frames_to_smooth)

            timestamp = video.get(cv2.CAP_PROP_POS_MSEC) * 1e6

            audio_level, minimum_audio_level, maximum_audio_level, audio_state = get_audio_levels_for_timestamp (audio_levels, timestamp, audio_state)

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
