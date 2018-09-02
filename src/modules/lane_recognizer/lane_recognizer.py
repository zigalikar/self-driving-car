import os
import cv2
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

import other.util as util
from modules.lane_recognizer.line import Line

from modules.module_base import ModuleBase

class LaneRecognizer(ModuleBase):

    def __init__(self, module_name, loader, config):
        super().__init__(module_name, loader, config)

        self.process_images()
        pass
    
    # Processes all the images in the data folder
    def process_images(self):
        plt.ion()
        module_root_dir = util.construct_dataset_paths(self.module_name, self.config, True)
        test_images = [path.join(module_root_dir, name) for name in os.listdir(module_root_dir)]

        for img in test_images:
            if path.isfile(img):
                self.module_log('Processing image: ' + img)

                out_path = path.join(module_root_dir, 'output', path.basename(img))
                in_image = cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                out_image = self.color_frame_pipeline([in_image], solid_lines=True)
                cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
                plt.imshow(out_image)
                plt.waitforbuttonpress()
        
        plt.close('all')
    
    # Returns a processed image from given list of frames (1 frame == image)
    def color_frame_pipeline(self, frames, solid_lines=True, temporal_smoothing=True):
        is_video = len(frames) > 0
        img_h, img_w = frames[0].shape[0], frames[0].shape[1]

        lane_lines = []
        for t in range(0, len(frames)):
            inferred_lanes = self.get_lane_lines(color_image=frames[t], solid_lines=solid_lines)
            lane_lines.append(inferred_lanes)

        if temporal_smoothing and solid_lines:
            lane_lines = self.smoothen_over_time(lane_lines)
        else:
            lane_lines = lane_lines[0]

        # Lines mask
        line_img = np.zeros(shape=(img_h, img_w))

        for lane in lane_lines:
            lane.draw(line_img)

        # Masking the image
        vertices = np.array([[(50, img_h), (450, 310), (490, 310), (img_w - 50, img_h)]], dtype=np.int32)
        img_masked, _ = self.region_of_interest(line_img, vertices)

        img_color = frames[-1] if is_video else frames[0]
        img_blend = self.weighted_img(img_masked, img_color, α=0.8, β=1., λ=0.)

        return img_blend

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image, mask
    
    def hough_lines_detection(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        return lines


    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        img = np.uint8(img)
        if len(img.shape) is 2:
            img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

        return cv2.addWeighted(initial_img, α, img, β, λ)

    def compute_lane_from_candidates(self, line_candidates, img_shape):
        pos_lines = [l for l in line_candidates if l.slope > 0]
        neg_lines = [l for l in line_candidates if l.slope < 0]

        neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
        neg_slope = np.median([l.slope for l in neg_lines])
        x1, y1 = 0, neg_bias
        x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
        left_lane = Line(x1, y1, x2, y2)

        lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
        lane_right_slope = np.median([l.slope for l in pos_lines])
        x1, y1 = 0, lane_right_bias
        x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
        right_lane = Line(x1, y1, x2, y2)

        return left_lane, right_lane

    def get_lane_lines(self, color_image, solid_lines=True):
        color_image = cv2.resize(color_image, (960, 540))
        img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)
        img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)
        detected_lines = self.hough_lines_detection(img=img_edge, rho=2, theta=np.pi / 180, threshold=1, min_line_len=15, max_line_gap=5)

        detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

        if solid_lines:
            candidate_lines = []
            for line in detected_lines:
                    if 0.5 <= np.abs(line.slope) <= 2:
                        candidate_lines.append(line)
            lane_lines = self.compute_lane_from_candidates(candidate_lines, img_gray.shape)
        else:
            lane_lines = detected_lines

        return lane_lines

    def smoothen_over_time(self, lane_lines):
        avg_line_lt = np.zeros((len(lane_lines), 4))
        avg_line_rt = np.zeros((len(lane_lines), 4))

        for t in range(0, len(lane_lines)):
            avg_line_lt[t] += lane_lines[t][0].get_coords()
            avg_line_rt[t] += lane_lines[t][1].get_coords()

        return Line(*np.mean(avg_line_lt, axis=0)), Line(*np.mean(avg_line_rt, axis=0))