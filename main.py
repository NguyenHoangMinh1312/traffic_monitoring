from ultralytics import YOLO
import cv2
import numpy as np

class CarRegionTracking:
    def __init__(self, model_path, regions, vid_src, save_path = "output_vid.mp4"):
        self.model = YOLO(model_path)
        self.regions = regions
        self.vid_src = vid_src
        self.save_path = save_path

       
        self.hist_status = {}   #all track ids used to be in a region from the 1st frame to current - 1 frame
        self.cur_status = {}    #all track ids in each region in the current frame

        #Store the number of cars that entered and exited each region
        self.in_counts = {}
        self.out_counts = {}

        for region_name in self.regions.keys():
            self.hist_status[region_name] = set()
            self.cur_status[region_name] = set()
            self.in_counts[region_name] = 0
            self.out_counts[region_name] = 0
        
        # Define colors for each region
        self.region_colors = [
            (0, 255, 0),    
            (255, 0, 0),    
            (0, 0, 255),    
            (255, 255, 0),  
            (255, 0, 255),  
            (0, 255, 255), 
            (128, 0, 128),  
            (255, 165, 0),  
            (0, 128, 255) 
        ]
        
            
    def isInRegion(self, point, region):
        x, y = point
        n = len(region)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = region[i]
            xj, yj = region[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside

    #draw the regions
    def drawRegions(self, frame):
        overlay = frame.copy()
        
        for i, (region_name, coords) in enumerate(self.regions.items()):
            color = self.region_colors[i]
            
            pts = np.array(coords, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Draw the border
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            
            # Fill the region with lighter color (for transparency effect)
            cv2.fillPoly(overlay, [pts], color)


        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    #draw the bounding box and track id of cars
    def drawBoundingBox(self, frame, box, track_id):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame
    
    #compare the current and previous status of regions
    def compareRegions(self):
        for region_name in self.regions.keys():
            for track_id in self.cur_status[region_name]:
                if track_id not in self.hist_status[region_name]:
                    self.in_counts[region_name] += 1
                
                self.out_counts[region_name] = len(self.hist_status[region_name] - self.cur_status[region_name])



    def process(self):
        video = cv2.VideoCapture(self.vid_src)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.save_path, fourcc, fps, (frame_width, frame_height))
        
        if not video.isOpened():
            print("Error: Could not open video.")
            return
        
        while True:
            ret, frame = video.read()
            if not ret: 
                break
            
            # Perform tracking
            results = self.model.track(source = frame, show = False, persist = True)
            result = results[0] if results else None

            if result and result.boxes:
                for iter, box in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = box.tolist()
                    track_id  = int(result.boxes.id[iter]) if result.boxes.id is not None else -1
                    x_center = int((x1 + x2) // 2)
                    y_center = int((y1 + y2) // 2)

                    # Draw bounding box and track ID
                    frame = self.drawBoundingBox(frame, box, track_id)

                    #detect which region the car is in
                    for region_name, coords in self.regions.items():
                        if self.isInRegion((x_center, y_center), coords):
                            self.cur_status[region_name].add(track_id)
                            break
            
            # Compare current and previous status
            self.compareRegions()

            # Draw regions on the frame
            frame = self.drawRegions(frame)

            #draw the counts of cars that entered and exited each region in the frame
            for i, region_name in enumerate(self.regions.keys()):
                in_count = self.in_counts[region_name]
                out_count = self.out_counts[region_name]
                cv2.putText(frame, f"{region_name}: In {in_count}, Out {out_count}", 
                            (10, 40 + 40 * list(self.regions.keys()).index(region_name)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.region_colors[i], 3)
            
            # Write the frame to output video
            out.write(frame)
            
            # Optional: Display the frame (comment out if running headless)
            cv2.imshow('Traffic Monitoring', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Update status
            for region_name in self.hist_status.keys():
                for track_id in self.cur_status[region_name]:
                    if track_id not in self.hist_status[region_name]:
                        self.hist_status[region_name].add(track_id)
            self.cur_status = {region_name: set() for region_name in self.regions.keys()}
            
        video.release()
        out.release()  # Don't forget to release the output video writer
        cv2.destroyAllWindows()
        print(f"Output video saved to: {self.save_path}")


if __name__ == "__main__":
    model_path = "runs/detect/train2/weights/best.pt"
    regions = {
        "region-01": [[8, 594], [716, 594], [716, 984], [8, 984]],
        "region-02": [[8, 1115], [682, 1132], [670, 1565], [4, 1544]],
        "region-03": [[2884, 594], [2879, 1624], [767, 1590], [826, 547]],
        "region-04": [[3821, 577], [3825, 1056], [2964, 997], [2985, 513]],
        "region-05": [[3821, 1209], [3825, 1620], [2981, 1595], [2977, 1192]],
        "region-06": [[1314, 8], [1543, -4], [1518, 543], [1013, 530], [1204, 314], [1289, 144]],
        "region-07": [[2146, 4], [2871, 505], [2871, 585], [2095, 560], [1768, -4]],
        "region-08": [[873, 1616], [1717, 1624], [1849, 2146], [1564, 2150], [1446, 1959], [1170, 1743]],
        "region-09": [[2129, 1629], [2850, 1637], [2570, 1764], [2366, 2002], [2294, 2146], [2027, 2146]]
    }

    vid_src = "input_vid.mov"
    output_path = "traffic_monitoring_output.mp4"  
    tracker = CarRegionTracking(model_path, regions, vid_src, output_path)
    tracker.process()
        