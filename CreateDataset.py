#create the dataset from 3 frames of the video
from ultralytics import SAM
import cv2
import os
import shutil

def extractFrames(video_path, frame_counts = 5):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return []
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = [int(i * total_frames /frame_counts) for i in range(frame_counts)]
    frame_paths = []

    cnt = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if cnt in frame_ids:
            frame_path = f"frame_{cnt}.jpg"
            frame_paths.append(frame_path)
            cv2.imwrite(frame_path, frame)
        
        cnt += 1
    video.release()

    return frame_paths

def annotate(frame_path, points):
    model = SAM("sam2.1_b.pt")
    model.to("cpu") 

    results = model(frame_path, points=points)
    
    # Get image dimensions for normalization
    label_path = frame_path.replace(".jpg", ".txt")
    
    with open(label_path, "w", encoding="utf8") as f:
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x, y, w, h = box.xywhn[0]  # center_x, center_y, width, height
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def createDataset(frame_paths, train_ratio =  0.6, val_ratio = 0.2, test_ratio = 0.2, output_dir = "dataset"):
    num = len(frame_paths)
    train_frames = frame_paths[:int(train_ratio * num)]
    val_frames = frame_paths[int(train_ratio * num):int((train_ratio + val_ratio) * num)]
    test_frames = frame_paths[int((train_ratio + val_ratio) * num):]

    #create YOLO dataset structure
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    img_types = ["train", "val", "test"]
    for img_type in img_types:
        os.makedirs(os.path.join(output_dir, img_type, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, img_type, "labels"), exist_ok=True)
        if img_type == "train":
            for frame in train_frames:
                shutil.move(frame, os.path.join(output_dir, img_type, "images"))
                shutil.move(frame.replace("jpg", "txt"), os.path.join(output_dir, img_type, "labels"))
        elif img_type == "val":
            for frame in val_frames:
                shutil.move(frame, os.path.join(output_dir, img_type, "images"))
                shutil.move(frame.replace("jpg", "txt"), os.path.join(output_dir, img_type, "labels"))
        elif img_type == "test":
            for frame in test_frames:
                shutil.move(frame, os.path.join(output_dir, img_type, "images"))
                shutil.move(frame.replace("jpg", "txt"), os.path.join(output_dir, img_type, "labels"))

def createYaml(abs_path):
    with open("data.yaml", "w", encoding="utf8") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("nc: 1\n")
        f.write("names: \n\t0: car\n")
    
    f.close()
    


if __name__ == "__main__":
    frame_paths = extractFrames("input_vid.mov") 

    annotate("frame_0.jpg", points = [[226, 713], [635, 702], [1085, 685], [1455, 651], [1769, 623], [2207, 640], [2067, 39], [2320, 236], [2550, 399], [2814, 528], [2589, 662], [3005, 601], [3358, 640], [3656, 921], [3386, 853], [3049, 741], [2634, 848], [2106, 702], [1517, 769], [697, 814], [248, 842], [254, 1190], [809, 1263], [1006, 1140], [1118, 1055], [1360, 870], [1769, 820], [2106, 808], [2348, 982], [3072, 926], [3482, 966], [3791, 1252], [3431, 1246], [2381, 1151], [2263, 1117], [2252, 1336], [1702, 1353], [1281, 1128], [1141, 1156], [1219, 1314], [602, 1325], [231, 1280], [360, 1482], [596, 1426], [798, 1566], [1281, 1538], [1371, 1443], [1820, 1443], [2033, 1577], [2207, 1443], [2746, 1359], [3027, 1336], [3420, 1347], [3802, 1448], [3375, 1443], [1601, 2066], [1478, 1926], [1326, 1785], [1118, 1662]])
    annotate("frame_150.jpg", points = [[98, 741], [540, 707], [797, 741], [1234, 682], [1554, 728], [1849, 720], [2152, 114], [2287, 413], [2489, 349], [2771, 518], [3550, 779], [3365, 893], [3032, 935], [2855, 779], [2658, 627], [2527, 859], [2241, 707], [1891, 817], [1575, 842], [544, 909], [110, 1149], [26, 1474], [274, 1276], [965, 1284], [994, 1137], [1087, 1133], [1171, 1187], [1289, 1124], [1512, 985], [1394, 1339], [468, 1385], [371, 1490], [645, 1516], [877, 1579], [1058, 1495], [1180, 1427], [1205, 1709], [1470, 1903], [1752, 1970], [1487, 1663], [1639, 1448], [1782, 1364], [2055, 1444], [2304, 1330], [2274, 1137], [2350, 989], [2405, 1002], [2481, 1069], [2586, 1389], [2674, 1272], [3327, 1352], [3630, 1478]])
    annotate("frame_300.jpg", points =[[321, 808], [877, 796], [1386, 766], [1660, 632], [1967, 737], [2266, 644], [2485, 366], [3222, 775], [3714, 930], [3306, 985], [3112, 846], [2607, 884], [2531, 754], [2051, 821], [1020, 901], [599, 930], [291, 1170], [140, 1246], [26, 1326], [375, 1469], [460, 1364], [557, 1175], [624, 1318], [590, 1495], [902, 1562], [978, 1486], [999, 1128], [1087, 1133], [1171, 1175], [1289, 1179], [1512, 1090], [1799, 1352], [1786, 1440], [2253, 1301], [2291, 1133], [2342, 994], [2413, 998], [2489, 1036], [2994, 1263], [3411, 1259], [3794, 1373], [3714, 1495], [3083, 1457], [2338, 1457], [1660, 1545], [1188, 1663], [1428, 1869], [1752, 2076], [1634, 2122]] )
    annotate("frame_450.jpg", points = [[39, 695], [418, 813], [973, 779], [1079, 627], [1601, 632], [2481, 669], [3028, 745], [3504, 712], [3693, 1036], [3306, 884], [3247, 981], [2944, 846], [2413, 762], [2034, 825], [1618, 741], [994, 909], [312, 897], [270, 1263], [380, 1469], [792, 1326], [910, 1469], [994, 1621], [1260, 1423], [1003, 1133], [1100, 1137], [1163, 1179], [1293, 1187], [1517, 1086], [1512, 1330], [1786, 1453], [2102, 1537], [2190, 1343], [2287, 1133], [2346, 998], [2413, 1010], [2489, 1040], [2439, 1154], [2418, 1318], [2451, 1427], [2523, 1516], [3074, 1474], [3491, 1267], [3778, 1495], [1352, 1794], [1567, 2004]])
    annotate("frame_600.jpg", points = [[135, 678], [893, 632], [1575, 619], [2131, 105], [2426, 754], [3180, 775], [3567, 914], [3058, 842], [2860, 897], [2291, 842], [1946, 829], [1415, 737], [847, 787], [199, 808], [308, 1284], [649, 1410], [906, 1570], [1003, 1145], [1095, 1141], [1167, 1175], [1289, 1175], [1512, 1154], [1554, 964], [2287, 1162], [2338, 1023], [2413, 1006], [2489, 1027], [2426, 1166], [2834, 1377], [3597, 1276], [3744, 1406], [3698, 1520], [1980, 1461], [1837, 1352], [1293, 1322], [1340, 1427], [1323, 1756]])

    createDataset(frame_paths, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, output_dir="dataset")
    createYaml("dataset")

    createYaml("/media/minh/Ubuntu Data/DL for CV/Exercises/traffic_monitoring/dataset")