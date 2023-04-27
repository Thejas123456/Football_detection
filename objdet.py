import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from motrackers import CentroidKF_Tracker 
from motrackers.utils import draw_tracks
from deep_sort_realtime.deepsort_tracker import DeepSort

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
tf.get_logger().setLevel('ERROR') 


VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'testt.mp4')
video_path_out = '{}_out.mp4'.format(video_path)
YOLO_MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train27', 'weights', 'last.pt')
TF_MODEL_CFG = "exported-models/footmodel/pipeline.config"
TF_MODEL_LABELS = "data/object-detection.pbtxt"
TF_MODEL_CKPT = "exported-models/footmodel/checkpoint"


yolo_model = YOLO(YOLO_MODEL_PATH)
deepsort_tracker = DeepSort(max_age=5)
previous_tracks = {}
def get_center_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    

    frame_height, frame_width, _ = frame.shape


    center_x = max(0, min(center_x, frame_width - 1))
    center_y = max(0, min(center_y, frame_height - 1))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv[center_y, center_x]
def get_team_by_color(color):

    team1_lower = np.array([150, 150, 160])
    team1_upper = np.array([190, 210, 225])

    team2_lower = np.array([40, 6, 200])
    team2_upper = np.array([110, 20, 220])


    team1_midpoint = (team1_lower + team1_upper) / 2
    team2_midpoint = (team2_lower + team2_upper) / 2


    team1_distance = np.linalg.norm(color - team1_midpoint)
    team2_distance = np.linalg.norm(color - team2_midpoint)


    if team1_distance < team2_distance:
        return "team1"
    else:
        return "team2"


def find_closest_old_track(new_track_bbox, previous_tracks, distance_threshold=10):
    for old_track_id, old_track_positions in previous_tracks.items():
        if not old_track_positions:
            continue
        avg_position = np.mean(old_track_positions, axis=0)
        distance = np.linalg.norm(np.array(new_track_bbox[:2]) - avg_position[:2])
        if distance < distance_threshold:
            return old_track_id
    return None
def is_too_close_to_existing_label(new_track_bbox, previous_tracks, distance_threshold=10):
    for old_track_positions in previous_tracks.values():
        if not old_track_positions:
            continue
        avg_position = np.mean(old_track_positions, axis=0)
        distance = np.linalg.norm(np.array(new_track_bbox[:2]) - avg_position[:2])
        if distance < distance_threshold:
            return True
    return False
def create_birds_eye_view(frame, src_points, dst_points):
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    aerial_view = cv2.warpPerspective(frame, matrix, (int(dst_points[2][0]), int(dst_points[2][1])))
    return aerial_view

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at:", x, y)
def is_overlapping_with_old_track(new_track_bbox, previous_tracks, iou_threshold=0.7):
    for old_track_positions in previous_tracks.values():
        if not old_track_positions:
            continue

        for old_track_bbox in old_track_positions:

            left1, top1, right1, bottom1 = new_track_bbox
            left2, top2, right2, bottom2 = old_track_bbox[:4]
            area1 = (right1 - left1) * (bottom1 - top1)
            area2 = (right2 - left2) * (bottom2 - top2)

            intersection_left = max(left1, left2)
            intersection_top = max(top1, top2)
            intersection_right = min(right1, right2)
            intersection_bottom = min(bottom1, bottom2)

            intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
            union_area = area1 + area2 - intersection_area

            iou = intersection_area / union_area

            if iou > iou_threshold:
                return True

    return False
@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

configs = config_util.get_configs_from_pipeline_file(TF_MODEL_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(TF_MODEL_CKPT, 'ckpt-0')).expect_partial()

def get_nearby_removed_track_id(new_track_bbox, last_frame_positions, distance_threshold=50):
    for track_id, track_positions in last_frame_positions.items():
        if not track_positions:
            continue
        avg_position = np.mean(track_positions, axis=0)
        distance = np.linalg.norm(np.array(new_track_bbox[:2]) - avg_position[:2])
        if distance < distance_threshold:
            return track_id
    return None


category_index = label_map_util.create_category_index_from_labelmap(TF_MODEL_LABELS, use_display_name=True)


cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))


threshold = 0.5
class_name_dict = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
cv2.namedWindow('Combined Detections')
cv2.setMouseCallback('Combined Detections', mouse_click)
ball_track_id = None
tracker = CentroidKF_Tracker ()


last_15_frames_positions = {i: [] for i in range(30)}
frame_number = 0
no_output_counter = 0 
previous_position = None
out_of_radius_frames = 0
previous_tracks2 = {}
while ret:

    frame_number += 1
    yolo_results = yolo_model(frame)[0]
    last_15_positions = [pos for positions in last_15_frames_positions.values() for pos in positions]
    average_position = np.mean(last_15_positions, axis=0) if last_15_positions else np.array([0, 0, 0, 0])
    bbs = []
    for result in yolo_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            bbs.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))

    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    detection_bboxes = []
    detection_confidences = []
    detection_class_ids = []
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    print(detections['num_detections'])
    for i in range(detections['detection_boxes'].shape[0]):
        ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
        score = detections['detection_scores'][i]
        class_id = detections['detection_classes'][i] - 1

        if score > threshold:
            left, top, right, bottom = xmin * W, ymin * H, xmax * W, ymax * H
            detection_bboxes.append([left, top, right - left, bottom - top])
            detection_confidences.append(score)
            detection_class_ids.append(class_id)

    tracks = deepsort_tracker.update_tracks(bbs, frame=frame)


    previous_track_ids = set(previous_tracks.keys())
    print(previous_track_ids)
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.det_class

        if track_id not in previous_track_ids and is_overlapping_with_old_track(ltrb, previous_tracks):
            continue


        if track_id in previous_tracks:
            previous_tracks[track_id].append(ltrb)
        else:
            previous_tracks[track_id] = [ltrb]


        if class_id > 0:
            left, top, right, bottom = ltrb
            center_color = get_center_color(frame, (int(left), int(top), int(right), int(bottom)))
            print("center" +str(center_color))
            team = get_team_by_color(center_color)

            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            label = f"{team} - {class_id} : {track_id}"
            cv2.putText(frame, label, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    detection_bboxes = np.array(detection_bboxes)
    detection_confidences = np.array(detection_confidences)
    detection_class_ids = np.array(detection_class_ids)
    ball_detection_bboxes = []
    ball_detection_confidences = []
    ball_detection_class_ids = []

    for i in range(len(detection_bboxes)):
        bbox = detection_bboxes[i]
        confidence = detection_confidences[i]
        class_id = detection_class_ids[i]
        if class_id == 0:  
            ball_detection_bboxes.append(bbox)
            ball_detection_confidences.append(confidence)
            ball_detection_class_ids.append(class_id)

    tracks2 = tracker.update(np.array(detection_bboxes), np.array(detection_confidences), np.array(detection_class_ids))


    for track_tuple in tracks2:
        frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = track_tuple
        ltrb = [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height]


        if previous_position is not None:
            position_difference = np.linalg.norm(np.array(ltrb[:2]) - previous_position[:2])
        else:
            position_difference = float("inf")


        if position_difference < 20 or previous_position is None:
            previous_position = np.array(ltrb[:2])  
            out_of_radius_frames = 0 


            if ball_track_id is None:
                ball_track_id = track_id  

            if track_id == ball_track_id:
                frame = draw_tracks(frame, [track_tuple])
                print("here1")
            else:
                track_id = ball_track_id
                track_tuple = frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                frame = draw_tracks(frame, [track_tuple])
                print("here2")
        else:
            out_of_radius_frames += 1 


            if out_of_radius_frames >= 3:
                previous_position = np.array(ltrb[:2])  
                out_of_radius_frames = 0  

                if ball_track_id is None:
                    ball_track_id = track_id  

                if track_id == ball_track_id:
                    frame = draw_tracks(frame, [track_tuple])
                    print("here1")
                else:
                    track_id = ball_track_id
                    track_tuple = frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                    frame = draw_tracks(frame, [track_tuple])
                    print("here2")
        

            
    cv2.imshow('Combined Detections', cv2.resize(frame, (1920, 1080)))
    src_points = np.float32([[1010, 185], [2, 253], [1538, 724], [132, 828]])  
    dst_points = np.float32([[0, 0], [1000, 0], [1000, 500], [0, 500]])

    aerial_view = create_birds_eye_view(frame, src_points, dst_points)

    cv2.imshow('Aerial View', cv2.resize(aerial_view, (1920, 1080)))


    out.write(frame)
    ret, frame = cap.read()


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
