from ultralytics import YOLO
import os
import numpy as np
import cv2
import argparse
import configparser
import json
import numpy as np
from tqdm import tqdm


ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# 32 Colors for keypoints
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (218, 170, 34), # Gold
    (0, 128, 128),  # Dark Cyan
    (176, 224, 230),# Light Blue
    (189, 183, 117),# Chartreuse
    (128, 128, 0),  # Olive
    (255, 192, 203),# Pink
    (0, 255, 0),    # Lime
    (255, 140, 0),  # Dark Orange
    (0, 0, 128),    # Navy Blue
    (255, 69, 0),   # Red-Orange
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (128, 0, 0),    # Brown
    (255, 255, 255),# White
    (192, 192, 192),# Light Gray
    (0, 0, 0),      # Black
    (70, 130, 180), # Steel Blue
    (255, 99, 71),  # Tomato
    (0, 128, 255),  # Royal Blue
    (255, 20, 147), # Deep Pink
    (255, 215, 0),  # Gold
    (0, 255, 128),  # Spring Green
    (139, 69, 19),  # Saddle Brown
    (205, 92, 92)   # Indian Red
]

# Create the OUTPUT_DIR folder
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def calculate_iou(box1, box2):
    """
    计算两个边界框之间的IOU
    """
    # 确保输入是一维数组
    box1 = np.array(box1).flatten()
    
    # 处理 ground truth 格式 [[x1, y1], [x2, y2]]
    if isinstance(box2, np.ndarray) and len(box2.shape) == 2:
        box2 = np.array([
            box2[0][0],  # x1
            box2[0][1],  # y1
            box2[1][0],  # x2
            box2[1][1]   # y2
        ])
    
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # 计算IOU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def solve_image(model, image, gt_boxes=None):
    """
    Keypoints detection for the image according to the model.
    """
    # 使用YOLOv11进行推理
    results = model(image, verbose=False)
    
    # 处理结果
    result = results[0]
    
    # 获取边界框和类别信息
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()
    cls_names = [result.names[int(cls_id)] for cls_id in cls_ids]

    # 处理关键点
    xy = []
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        xy = [[[idx, kp[0], kp[1]] for idx, kp in enumerate(kps) 
               if not (kp[0] == 0 and kp[1] == 0)] for kps in keypoints]

    # 计算IOU 
    iou_scores = []
    if gt_boxes is not None:
        for box in boxes:
            max_iou = 0
            for gt_box in gt_boxes:
                iou = calculate_iou(box, gt_box)
                if iou > max_iou:
                    max_iou = iou
            iou_scores.append(max_iou)

    return {
        'keypoints': xy,
        'bbox': boxes,
        'score': scores,
        'category': cls_names,
        'iou': iou_scores
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str,
                        help='Path of the source (can be image or video)')
    parser.add_argument('--model', help='Path of model weights', required=True)
    parser.add_argument("--render", help='Save the detections to another image or video', 
                       action='store_true')
    parser.add_argument('--gt', type=str, default=None,
                       help='Path to ground truth data file')
    args = parser.parse_args()

    # 加载 ground truth 数据
    gt_data = None
    gt_path = r"C:\Users\zhangfan.LAPTOP-0PV7OOP0\Desktop\graduation\ES-EOT-main\data\bus_change_lane\labels.npy"
    # Add the check after loading ground truth data
    if os.path.exists(gt_path):
        gt_data = np.load(gt_path, allow_pickle=True)
        print(f"已加载 ground truth 数据: {gt_path}")
        print(f"数据格式: {gt_data[0].keys()}")
        # Add format check here
        if gt_data is not None and len(gt_data) > 0:
            print("Ground truth bbox 格式示例:", gt_data[0]['bboxes'][0])
            print("Ground truth bbox 形状:", gt_data[0]['bboxes'].shape)

    if not os.path.exists(args.model):
        print('Model path not exist!')
        exit(0)

    # Load a model
    model = YOLO(args.model)  # pretrained model

    # Load source
    source = args.source
    cap = None
    image = None
    if os.path.exists(source):
        if source.endswith(('mp4', 'avi')):
            cap = cv2.VideoCapture(source)
        elif source.endswith(('jpg', 'png', 'bmp')):
            image = cv2.imread(source)
        else:
            print('Bad file format.')
            exit(0)
    else:
        print('Source path not exists.')
        exit(0)

    rets = []
    # Solve video
    if cap is not None:
        ext = os.path.splitext(os.path.basename(source))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)

        if args.render:
            video_path = os.path.join(
                OUTPUT_DIR, '{0}-keypoints.mp4'.format(ext[0]))
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            # 使用对应帧的ground truth boxes
            frame_gt_boxes = gt_data[cnt]['bboxes'] if gt_data is not None and cnt < len(gt_data) else None
            ret = solve_image(model, frame, frame_gt_boxes)
            rets.append(ret)

            if args.render:
                for i, box in enumerate(ret['bbox']):
                    cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)

                # 显示IOU
                if 'iou' in ret and i < len(ret['iou']):
                    iou = ret['iou'][i]
                    text = f"Current IOU={iou:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                    center_x, center_y = 30 + text_width // 2, 30 + text_height // 2
                    cv2.rectangle(frame, (center_x - text_width // 2, center_y - text_height // 2 - baseline), (center_x + text_width // 2, center_y + text_height // 2 + baseline), (255, 255, 204), -1)
                    cv2.putText(frame, text, (center_x - text_width // 2, center_y + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

                # 显示检测到的车辆数量
                car_text = f"Detected: {len(ret['bbox'])} car(s)"
                (car_text_width, car_text_height), car_baseline = cv2.getTextSize(car_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                car_center_x = center_x
                car_center_y = center_y + text_height + car_text_height + 20
                cv2.rectangle(frame, (car_center_x - car_text_width // 2, car_center_y - car_text_height // 2 - car_baseline), (car_center_x + car_text_width // 2, car_center_y + car_text_height // 2 + car_baseline), (255, 255, 204), -1)
                cv2.putText(frame, car_text, (car_center_x - car_text_width // 2, car_center_y + car_text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

                # 显示预测速度
                if cnt == 0:  # 只在第一帧时加载速度数据
                    speed_data = {}
                    res_dir_path = r'C:\Users\zhangfan.LAPTOP-0PV7OOP0\Desktop\graduation\ES-EOT-main\ES-EOT-CTRA\res'
                    if not os.path.exists(res_dir_path):
                        print(f"路径不存在: {res_dir_path}")
                        exit(0)
                    res_file_list = [f for f in os.listdir(res_dir_path) if f.split('-')[0] == 'bus_change_lane']
                    for res_file in res_file_list:
                        file_path = os.path.join(res_dir_path, res_file)
                        res = np.load(file_path, allow_pickle=True)
                        for frame_idx, frame_data in enumerate(res):
                            if frame_idx not in speed_data:
                                speed_data[frame_idx] = []
                            speed_data[frame_idx].append(np.abs(frame_data['x_ref'][3]))
                
                # 使用当前帧的速度数据
                if cnt in speed_data:
                    v = np.mean(speed_data[cnt])/3
                    # 使用平均速度
                    speed_text = f"Speed: {v:.2f} m/s"
                    overspeed_count = 1 if v >= 11 else 0
                (speed_text_width, speed_text_height), speed_baseline = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                speed_center_x = car_center_x
                speed_center_y = car_center_y + car_text_height + speed_text_height + 20
                cv2.rectangle(frame, (speed_center_x - speed_text_width // 2, speed_center_y - speed_text_height // 2 - speed_baseline), (speed_center_x + speed_text_width // 2, speed_center_y + speed_text_height // 2 + speed_baseline), (255, 255, 204), -1)
                cv2.putText(frame, speed_text, (speed_center_x - speed_text_width // 2, speed_center_y + speed_text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

                # 显示交通情况
                traffic_status = "Congested" if len(ret['bbox']) >= 8 else "Clear"
                traffic_text = f"Traffic: {traffic_status}"
                (traffic_text_width, traffic_text_height), traffic_baseline = cv2.getTextSize(traffic_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                traffic_center_x = speed_center_x
                traffic_center_y = speed_center_y + speed_text_height + traffic_text_height + 20
                cv2.rectangle(frame, (traffic_center_x - traffic_text_width // 2, traffic_center_y - traffic_text_height // 2 - traffic_baseline), (traffic_center_x + traffic_text_width // 2, traffic_center_y + traffic_text_height // 2 + traffic_baseline), (255, 255, 204), -1)
                cv2.putText(frame, traffic_text, (traffic_center_x - traffic_text_width // 2, traffic_center_y + traffic_text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

                # 显示超速车辆数量

                overspeed_text = f"Overspeed: {overspeed_count} car(s)"
                (overspeed_text_width, overspeed_text_height), overspeed_baseline = cv2.getTextSize(overspeed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                overspeed_center_x = traffic_center_x
                overspeed_center_y = traffic_center_y + traffic_text_height + overspeed_text_height + 20
                cv2.rectangle(frame, (overspeed_center_x - overspeed_text_width // 2, overspeed_center_y - overspeed_text_height // 2 - overspeed_baseline), (overspeed_center_x + overspeed_text_width // 2, overspeed_center_y + overspeed_text_height // 2 + overspeed_baseline), (255, 255, 204), -1)
                cv2.putText(frame, overspeed_text, (overspeed_center_x - overspeed_text_width // 2, overspeed_center_y + overspeed_text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

                for kps in ret['keypoints']:
                    for kp in kps:
                        cv2.circle(frame, tuple(map(int, kp[1:])), 6, colors[int(kp[0])][::-1], -1)
                out.write(frame)

            print('Frame {} finished'.format(cnt))
            cnt += 1

        npy_path = os.path.join(OUTPUT_DIR, '{}-keypoints.npy'.format(ext[0]))
        np.save(npy_path, rets)
        print('Save predictions to {}'.format(npy_path))
        if args.render:
            out.release()
            print('Save rendered video to {}'.format(video_path))

    # Solve image
    if image is not None:
        ext = os.path.splitext(os.path.basename(source))

        ret = solve_image(model, image)
        rets.append(ret)

        npy_path = os.path.join(OUTPUT_DIR, '{}-keypoints.npy'.format(ext[0]))
        np.save(npy_path, rets)
        print('Save predictions to {}'.format(npy_path))

        if args.render:
            for box in ret['bbox']:
                cv2.rectangle(image, tuple(map(int, box[:2])), tuple(
                    map(int, box[2:])), (0, 0, 255), 2)
                if 'iou' in ret:
                    iou = ret['iou'][0]
                    text = f"Current IOU={iou:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                    center_x, center_y = 30 + text_width // 2, 30 + text_height // 2
                    cv2.rectangle(image, (center_x - text_width // 2, center_y - text_height // 2 - baseline),
                                  (center_x + text_width // 2, center_y + text_height // 2 + baseline),
                                  (255, 255, 204), -1)
                    cv2.putText(image, text, (center_x - text_width // 2, center_y + text_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)
                # 显示检测到的车辆数量
                car_text = f"Detected: {len(ret['bbox'])} car(s)"
                (car_text_width, car_text_height), car_baseline = cv2.getTextSize(car_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                car_center_x = center_x
                car_center_y = center_y + text_height + car_text_height + 20
                cv2.rectangle(frame, (car_center_x - car_text_width // 2, car_center_y - car_text_height // 2 - car_baseline), (car_center_x + car_text_width // 2, car_center_y + car_text_height // 2 + car_baseline), (255, 255, 204), -1)
                cv2.putText(frame, car_text, (car_center_x - car_text_width // 2, car_center_y + car_text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

                # 显示预测速度
                if cnt == 0:  # 只在第一帧时加载速度数据
                    speed_data = {}
                    res_dir_path = r'C:\Users\zhangfan.LAPTOP-0PV7OOP0\Desktop\graduation\ES-EOT-main\ES-EOT-CTRA\res'
                    if not os.path.exists(res_dir_path):
                        print(f"路径不存在: {res_dir_path}")
                        exit(0)
                    res_file_list = [f for f in os.listdir(res_dir_path) if f.split('-')[0] == 'bus_change_lane']
                    for res_file in res_file_list:
                        file_path = os.path.join(res_dir_path, res_file)
                        res = np.load(file_path, allow_pickle=True)
                        for frame_idx, frame_data in enumerate(res):
                            if frame_idx not in speed_data:
                                speed_data[frame_idx] = []
                            speed_data[frame_idx].append(np.abs(frame_data['x_ref'][3]))
                
                # 使用当前帧的速度数据
                if cnt in speed_data:
                    v = np.mean(speed_data[cnt])/3  # 使用平均速度
                    speed_text = f"Speed: {v:.2f} m/s"
                (speed_text_width, speed_text_height), speed_baseline = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                speed_center_x = car_center_x
                speed_center_y = car_center_y + car_text_height + speed_text_height + 20
                cv2.rectangle(frame, (speed_center_x - speed_text_width // 2, speed_center_y - speed_text_height // 2 - speed_baseline), (speed_center_x + speed_text_width // 2, speed_center_y + speed_text_height // 2 + speed_baseline), (255, 255, 204), -1)
                cv2.putText(frame, speed_text, (speed_center_x - speed_text_width // 2, speed_center_y + speed_text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

                # 显示交通情况
                traffic_status = "Congested" if len(ret['bbox']) >= 8 else "Clear"
                traffic_text = f"Traffic: {traffic_status}"
                (traffic_text_width, traffic_text_height), traffic_baseline = cv2.getTextSize(traffic_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
                traffic_center_x = speed_center_x
                traffic_center_y = speed_center_y + speed_text_height + traffic_text_height + 20
                cv2.rectangle(frame, (traffic_center_x - traffic_text_width // 2, traffic_center_y - traffic_text_height // 2 - traffic_baseline), (traffic_center_x + traffic_text_width // 2, traffic_center_y + traffic_text_height // 2 + traffic_baseline), (255, 255, 204), -1)
                cv2.putText(frame, traffic_text, (traffic_center_x - traffic_text_width // 2, traffic_center_y + traffic_text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 3)

            for kps in ret['keypoints']:
                for kp in kps:
                    cv2.circle(image, tuple(
                        map(int, kp[1:])), 6, colors[int(kp[0])][::-1], -1)

            img_path = os.path.join(
                OUTPUT_DIR, '{0}-keypoints{1}'.format(ext[0], ext[1]))
            cv2.imwrite(img_path, image)
            print('Save rendered image to {}'.format(img_path))
