import cv2
import numpy as np




def project_waypoints_to_fisheye_image_with_polygon_new(
        waypoints, intrinsic_params, image_path, save_path='projected_waypoints_fisheye.jpg', 
        camera_height=0.41, save_fig=False, color=None, base_alpha=0.15, 
        text_to_visualize=None, plotted_number=None, camera_pitch_deg=0.0):
    """
    New fisheye projection function with gradient transparency and side bars.
    
    Parameters:
    - waypoints: Array of shape (N, 2) with x,y coordinates in meters
    - intrinsic_params: [f, cx, cy, k] where f is focal length, (cx,cy) is principal point, k is distortion coefficient
    - image_path: Path to fisheye image or numpy array
    - save_path: Output path for saving the result
    - camera_height: Height of camera above ground (meters)
    - save_fig: Whether to save the result image
    - color: Custom color for the path (BGR format)
    - camera_pitch_deg: Visualization-only camera pitch in degrees. Positive
      values tilt the virtual camera upward and move projected ground paths
      lower in the image.
    
    Returns:
    - img: Image with projected waypoints as navigation path
    """

    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints should be of shape (N, 2)")
    if len(intrinsic_params) != 4:
        raise ValueError("intrinsic_params should be [f, cx, cy, k]")
    
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
    else:
        print(f'Unsupported image_path type: {type(image_path)}')

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    H, W = img.shape[:2]
    f, cx, cy, k = intrinsic_params

    # ========== step1: Generate left and right offset waypoints ==========
    # Make the path wider to match navigation style
    path_width = 0.22  # Increased width for navigation path
    offsets_left, offsets_right = [], []
    
    for i in range(len(waypoints)-1):
        p1, p2 = waypoints[i], waypoints[i+1]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        dir_unit = direction / norm
        # Normal vectors (left: [-dy, dx], right: [dy, -dx])
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(p1 + path_width * normal)
        offsets_right.append(p1 - path_width * normal)
    
    # Add the last point as well
    if len(waypoints) > 1:
        direction = waypoints[-1] - waypoints[-2]
        dir_unit = direction / np.linalg.norm(direction)
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(waypoints[-1] + path_width * normal)
        offsets_right.append(waypoints[-1] - path_width * normal)

    offsets_left = np.array(offsets_left)
    offsets_right = np.array(offsets_right)

    # ========== step2: Improved fisheye projection function ==========
    def project_points_fisheye(points):
        num_points = points.shape[0]
        ego_points = np.zeros((num_points, 3))
        ego_points[:, :2] = points
        cam_translation = np.array([-0.1, 0, camera_height])
        R = np.array([
            [0, -1, 0],   
            [0, 0, -1],   
            [1, 0, 0]     
        ])
        cam_points = (R @ (ego_points - cam_translation).T).T
        if abs(camera_pitch_deg) > 1e-6:
            # Positive "pitch up" should make the ground path appear lower in
            # the image. In this camera frame (+y down, +z forward), that is a
            # negative rotation around the camera x-axis.
            pitch = -np.deg2rad(camera_pitch_deg)
            cp, sp = np.cos(pitch), np.sin(pitch)
            pitch_R = np.array([
                [1, 0, 0],
                [0, cp, -sp],
                [0, sp, cp],
            ])
            cam_points = (pitch_R @ cam_points.T).T
        valid_mask = cam_points[:, 2] > 0
        cam_points = cam_points[valid_mask]
        if cam_points.shape[0] == 0:
            return np.zeros((0,2))
        
        # Normalize to unit sphere (perspective projection)
        x = cam_points[:, 0] / cam_points[:, 2]
        y = cam_points[:, 1] / cam_points[:, 2]
        
        # Calculate distance from optical center
        r = np.sqrt(x**2 + y**2)
        
        # Fisheye projection: theta = atan(r)
        theta = np.arctan(r)
        
        # Apply fisheye distortion model
        theta_d = theta * (1 + k * theta**2)
        
        # Handle division by zero for points at optical center
        scaling = np.ones_like(r)
        mask = r > 1e-8
        scaling[mask] = theta_d[mask] / r[mask]
        
        # Apply scaling to get distorted coordinates
        x_distorted = x * scaling
        y_distorted = y * scaling
        
        # Project to image plane using focal length and principal point
        u = f * x_distorted + cx
        v = f * y_distorted + cy
        
        return np.column_stack((u, v))

    points_center = project_points_fisheye(waypoints)
    points_left = project_points_fisheye(offsets_left)
    points_right = project_points_fisheye(offsets_right)

    # ========== step3: Create navigation path with gradient transparency ==========
    if len(points_left) > 1 and len(points_right) > 1 and waypoints[-1, 0] > 1.:
        # Create polygon for the path
        polygon = np.vstack([points_left, points_right[::-1]])
        polygon = polygon.astype(np.int32)
        
        # Navigation green color (similar to the reference image)
        nav_green = (65, 121, 76)  # BGR format - forest green
        if color is not None:
            nav_green = color
        
        # Create mask for the path region
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        
        # Create gradient mask (stronger near bottom, weaker towards top)
        gradient_mask = np.zeros((H, W), dtype=np.float32)
        
        # For each pixel in the mask, calculate gradient based on distance along path
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) > 0:
            # Calculate normalized distance from bottom (1.0 at bottom, 0.0 at top)
            # Using y-coordinate for simple vertical gradient
            normalized_y = y_coords / H
            
            # Create gradient that's stronger at bottom (closer to camera)
            # Use exponential or power function for smooth gradient
            alpha_values = base_alpha + 0.45 * (normalized_y ** 1.5)  # Range from 0.15 to 0.6
            
            # Apply gradient to mask
            gradient_mask[y_coords, x_coords] = alpha_values
        
        # Apply the navigation path with gradient transparency
        overlay = img.copy()
        
        # Apply color to overlay where mask is active
        overlay[mask > 0] = nav_green
        
        # Blend with gradient transparency
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - gradient_mask) + overlay[:, :, c] * gradient_mask
        
        # ========== step4: Add narrow dark bars on left and right edges ==========
        # Calculate positions for edge bars based on the polygon bounds
        if len(points_left) > 0 and len(points_right) > 0 and waypoints[-1, 0] > 2.:
            # Create slightly offset lines for the edge bars
            bar_width = 5  # Very narrow bars (2-3 pixels)
            bar_color = (int(nav_green[0] * 0.7), int(nav_green[1] * 0.7), int(nav_green[2] * 0.7))  # Darker green/gray color
            
            # Left edge bar - follow the left boundary
            left_points = points_left.astype(np.int32)
            # Filter valid points
            valid_left = left_points[
                (left_points[:, 0] >= 0) & (left_points[:, 0] < W) &
                (left_points[:, 1] >= 0) & (left_points[:, 1] < H)
            ]
            
            if len(valid_left) > 1:
                # Draw polyline for left edge
                cv2.polylines(img, [valid_left], False, bar_color, bar_width, cv2.LINE_AA)
            
            # Right edge bar - follow the right boundary
            right_points = points_right.astype(np.int32)
            # Filter valid points
            valid_right = right_points[
                (right_points[:, 0] >= 0) & (right_points[:, 0] < W) &
                (right_points[:, 1] >= 0) & (right_points[:, 1] < H)
            ]
            
            if len(valid_right) > 1:
                # Draw polyline for right edge
                cv2.polylines(img, [valid_right], False, bar_color, bar_width, cv2.LINE_AA)
    
    # ========== step5: Optional - Add subtle center guidance ==========
    # Add the vehicle starting point at bottom center
    start_point = np.array([[W//2, H-1]])
    points_center = np.vstack((start_point, points_center))
    
    # Filter points within image bounds
    points_center = points_center[
        (points_center[:, 0] >= 0) & (points_center[:, 0] < W) &
        (points_center[:, 1] >= 0) & (points_center[:, 1] < H)
    ]
    
    # Optional: Add very subtle center line with gradient opacity
    # print(waypoints, waypoints.shape)
    if points_center.shape[0] > 1 and waypoints[-1, 0] > 2.:
        for i in range(len(points_center) - 1):
            pt1 = tuple(points_center[i].astype(int))
            pt2 = tuple(points_center[i+1].astype(int))
            
            # Calculate opacity based on y-position (stronger at bottom)
            y_avg = (pt1[1] + pt2[1]) / 2
            opacity = 0.1 + 0.2 * (y_avg / H)  # Very subtle, 0.1 to 0.3 opacity
            
            # Draw on overlay and blend
            overlay = img.copy()
            cv2.line(overlay, pt1, pt2, (int(nav_green[0] * 0.8), int(nav_green[1] * 0.8), int(nav_green[2] * 0.8)), 2, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, opacity, img, 1-opacity, 0)
    
    # ========== step6: Add color bar and text if text_to_visualize is provided ==========
    if text_to_visualize is not None and plotted_number is not None and color is not None:
        # Color bar dimensions
        bar_width = 20
        bar_height = 45
        bar_x = 10
        # Position color bar vertically based on plotted_number (0 = bottom)
        bar_y = H - 10 - bar_height - (plotted_number * (bar_height + 2))
        
        # Ensure color is a tuple of Python integers in BGR format
        # Matplotlib colormap returns RGB, need to convert to BGR for OpenCV
        if isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
            # Convert to Python ints and swap RGB to BGR
            color_bgr = (int(color[0]), int(color[1]), int(color[2]))
        else:
            color_bgr = (255, 255, 255)  # Default white
        
        # Ensure coordinates are Python ints
        bar_x_int = int(bar_x)
        bar_y_int = int(bar_y)
        bar_width_int = int(bar_width)
        bar_height_int = int(bar_height)
        
        # Draw color bar (vertical rectangle)
        cv2.rectangle(img, (bar_x_int, bar_y_int), (bar_x_int + bar_width_int, bar_y_int + bar_height_int), color_bgr, -1)
        # Add border to color bar
        cv2.rectangle(img, (bar_x_int, bar_y_int), (bar_x_int + bar_width_int, bar_y_int + bar_height_int), (255, 255, 255), 2)
        
        # Text positioning - stack vertically from bottom, aligned with color bar
        text_x = int(bar_x_int + bar_width_int + 10)  # Right of the color bar
        text_y = int(bar_y_int + bar_height_int - 5)  # Aligned with bottom of color bar
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        
        # Add black background for text readability
        (text_width, text_height), baseline = cv2.getTextSize(text_to_visualize, font, font_scale, font_thickness)
        cv2.rectangle(img, 
                     (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + baseline + 5), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, text_to_visualize, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if save_fig:
        cv2.imwrite(save_path, img)
    
    return img
#

def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=10):
    """Draw a dashed line between two points"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Calculate line length and direction
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx*dx + dy*dy)
    
    if line_length == 0:
        return
    
    # Normalize direction
    dx_norm = dx / line_length
    dy_norm = dy / line_length
    
    # Draw dashed segments
    current_length = 0
    draw_dash = True
    
    while current_length < line_length:
        # Calculate next dash endpoint
        next_length = min(current_length + dash_length, line_length)
        
        start_x = int(x1 + current_length * dx_norm)
        start_y = int(y1 + current_length * dy_norm)
        end_x = int(x1 + next_length * dx_norm)
        end_y = int(y1 + next_length * dy_norm)
        
        if draw_dash:
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        
        current_length = next_length
        draw_dash = not draw_dash

def draw_solid_centerline(img, points, color, thickness):
    """Draw solid centerline"""
    for i in range(len(points) - 1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
        cv2.line(img, pt1, pt2, color, thickness)


def draw_dashed_centerline(img, points, color, thickness, dash_length=15, gap_length=10):
    """Draw dashed centerline"""
    for i in range(len(points) - 1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
        draw_dashed_line_segment(img, pt1, pt2, color, thickness, dash_length, gap_length)


def draw_dotted_centerline(img, points, color, thickness, dot_spacing=8):
    """Draw dotted centerline"""
    for i in range(len(points) - 1):
        x1, y1 = int(points[i][0]), int(points[i][1])
        x2, y2 = int(points[i+1][0]), int(points[i+1][1])
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx*dx + dy*dy)
        
        if line_length == 0:
            continue
            
        # Number of dots along this segment
        num_dots = int(line_length / dot_spacing)
        
        for j in range(num_dots + 1):
            t = j / max(num_dots, 1)
            dot_x = int(x1 + t * dx)
            dot_y = int(y1 + t * dy)
            cv2.circle(img, (dot_x, dot_y), max(2, thickness//2), color, -1)


def draw_dashed_line_segment(img, pt1, pt2, color, thickness, dash_length=15, gap_length=10):
    """Draw a dashed line between two points"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx*dx + dy*dy)
    
    if line_length == 0:
        return
    
    dx_norm = dx / line_length
    dy_norm = dy / line_length
    
    current_length = 0
    draw_dash = True
    
    while current_length < line_length:
        if draw_dash:
            next_length = min(current_length + dash_length, line_length)
        else:
            next_length = min(current_length + gap_length, line_length)
        
        start_x = int(x1 + current_length * dx_norm)
        start_y = int(y1 + current_length * dy_norm)
        end_x = int(x1 + next_length * dx_norm)
        end_y = int(y1 + next_length * dy_norm)
        
        if draw_dash:
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        
        current_length = next_length
        draw_dash = not draw_dash
def draw_solid_centerline_alpha(img, points, color, thickness, alpha=0.7):
    """Draw solid centerline with alpha transparency"""
    overlay = img.copy()
    for i in range(len(points) - 1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
        cv2.line(overlay, pt1, pt2, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
def project_multiple_centerlines_to_fisheye(
        multiple_waypoints, intrinsic_params, image_path, save_path='multiple_centerlines.jpg', 
        camera_height=0.61, save_fig=False, line_colors=None, line_styles=None, i=0):
    """
    Project multiple centerlines (trajectories) onto fisheye image
    
    Args:
        multiple_waypoints: List of waypoint arrays, each of shape (N, 2)
        intrinsic_params: [f, cx, cy, k] camera parameters
        image_path: Path to image or numpy array
        save_path: Output save path
        camera_height: Camera height in meters
        save_fig: Whether to save the result
        line_colors: List of colors (BGR) for each centerline, if None uses default colors
        line_styles: List of line styles ('solid', 'dashed', 'dotted') for each centerline
    
    Returns:
        Processed image with multiple centerlines
    """
    mode_i = i
    
    if not isinstance(multiple_waypoints, list):
        raise ValueError("multiple_waypoints should be a list of waypoint arrays")
    
    if len(intrinsic_params) != 4:
        raise ValueError("intrinsic_params should be [f, cx, cy, k]")
    
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
    else:
        print(f'Unsupported image_path type: {type(image_path)}')

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    H, W = img.shape[:2]
    f, cx, cy, k = intrinsic_params
    
    # Default colors for multiple centerlines (BGR format)
    default_colors = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue  
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (255, 255, 255),  # White
        (128, 128, 128),  # Gray
    ]
    
    # Default line styles
    default_styles = ['solid'] * len(multiple_waypoints)
    
    # Use provided colors/styles or defaults
    if line_colors is None:
        line_colors = default_colors[:len(multiple_waypoints)]
    if line_styles is None:
        line_styles = default_styles[:len(multiple_waypoints)]
    
    # Ensure we have enough colors and styles
    while len(line_colors) < len(multiple_waypoints):
        line_colors.extend(default_colors)
    while len(line_styles) < len(multiple_waypoints):
        line_styles.extend(['solid'])
    if mode_i == 3:
        line_colors[0] = (34, 215, 70)
    elif mode_i == 2:
        line_colors[0] = (34, 180, 70)
    elif mode_i == 1:
        line_colors[0] = (34, 150, 70)
    else:
        line_colors[0] = (34, 120, 70)
    line_colors[1:] = [(170, 110, 100) for _ in range(len(multiple_waypoints) - 1)]
    # ========== Projection function ==========
    def project_points(points):
        if points.shape[0] == 0:
            return np.zeros((0, 2))
            
        num_points = points.shape[0]
        ego_points = np.zeros((num_points, 3))
        ego_points[:, :2] = points
        
        cam_translation = np.array([-0.3, 0, camera_height])
        R = np.array([
            [0, -1, 0],   
            [0, 0, -1],   
            [1, 0, 0]     
        ])
        
        cam_points = (R @ (ego_points - cam_translation).T).T
        valid_mask = cam_points[:, 2] > 0
        cam_points = cam_points[valid_mask]
        
        if cam_points.shape[0] == 0:
            return np.zeros((0, 2))
        
        x = cam_points[:, 0] / cam_points[:, 2]
        y = cam_points[:, 1] / cam_points[:, 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(r)
        theta_d = theta * (1 + k * theta**2)
        
        scaling = np.zeros_like(r)
        mask = r > 1e-8
        scaling[mask] = theta_d[mask] / r[mask]
        scaling[~mask] = 1.0
        
        x_distorted = x * scaling
        y_distorted = y * scaling
        u = f * x_distorted + cx
        v = f * y_distorted + cy
        
        return np.column_stack((u, v))
    
    # ========== Process each centerline ==========
    line_thickness = max(3, int(H/150))  # Adaptive line thickness
    
    for traj_idx, waypoints in enumerate(multiple_waypoints):
        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            print(f"Skipping trajectory {traj_idx}: Invalid shape {waypoints.shape}")
            continue
            
        # Project waypoints to image coordinates
        points_2d = project_points(waypoints)
        
        if points_2d.shape[0] == 0:
            print(f"Skipping trajectory {traj_idx}: No valid projected points")
            continue
        
        # Add vehicle starting point (bottom center of image) as first point
        start_point = np.array([[W//2, H-1]])
        points_2d = np.vstack((start_point, points_2d))
        
        # Filter points within image bounds
        valid_points = points_2d[
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        ]
        
        if valid_points.shape[0] < 2:
            print(f"Skipping trajectory {traj_idx}: Not enough valid points")
            continue
        
        # Get color and style for this trajectory
        color = line_colors[traj_idx]
        style = line_styles[traj_idx]
        
        # Draw centerline based on style
        if style == 'solid':
            if traj_idx == 0:
                draw_solid_centerline(img, valid_points, color, line_thickness)
            else:
                draw_solid_centerline_alpha(img, valid_points, color, line_thickness, alpha=0.4)
        elif style == 'dashed':
            draw_dashed_centerline(img, valid_points, color, line_thickness)
        elif style == 'dotted':
            draw_dotted_centerline(img, valid_points, color, line_thickness)
        else:
            # Default to solid
            draw_solid_centerline(img, valid_points, color, line_thickness)
    
        # Add trajectory number label at the end point
        if valid_points.shape[0] > 0:
            end_point = valid_points[-1]
            label_x, label_y = int(end_point[0]), int(end_point[1])
            
            # Add small circle and number at end point
            # cv2.circle(img, (label_x, label_y), max(8, int(H/80)), color, -1)
            cv2.circle(img, (label_x, label_y), max(8, int(H/80)), color, -1) if traj_idx == 0 else draw_alpha_circle_manual(img, (label_x, label_y), max(8, int(H/80)), color, -1, alpha=0.4)
            # cv2.putText(img, str(traj_idx + 1), (label_x - 5, label_y + 5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # break
    # for traj_idx, waypoints in enumerate(multiple_waypoints):
    #     if waypoints.ndim != 2 or waypoints.shape[1] != 2:
    #         print(f"Skipping trajectory {traj_idx}: Invalid shape {waypoints.shape}")
    #         continue
            
    #     # Project waypoints to image coordinates
    #     points_2d = project_points(waypoints)
        
    #     if points_2d.shape[0] == 0:
    #         print(f"Skipping trajectory {traj_idx}: No valid projected points")
    #         continue
        
    #     # Add vehicle starting point (bottom center of image) as first point
    #     start_point = np.array([[W//2, H-1]])
    #     points_2d = np.vstack((start_point, points_2d))
        
    #     # Filter points within image bounds
    #     valid_points = points_2d[
    #         (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
    #         (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
    #     ]
        
    #     if valid_points.shape[0] < 2:
    #         print(f"Skipping trajectory {traj_idx}: Not enough valid points")
    #         continue
        
    #     # Get color and style for this trajectory
    #     color = line_colors[traj_idx]
    #     style = line_styles[traj_idx]
        
    #     # Draw centerline based on style
    #     if style == 'solid':
    #         if traj_idx == 0:
    #             draw_solid_centerline(img, valid_points, color, line_thickness)
    #         else:
    #             draw_solid_centerline_alpha(img, valid_points, color, line_thickness, alpha=0.4)
    #     elif style == 'dashed':
    #         draw_dashed_centerline(img, valid_points, color, line_thickness)
    #     elif style == 'dotted':
    #         draw_dotted_centerline(img, valid_points, color, line_thickness)
    #     else:
    #         # Default to solid
    #         draw_solid_centerline(img, valid_points, color, line_thickness)
    
    #     # Add trajectory number label at the end point
    #     if valid_points.shape[0] > 0:
    #         end_point = valid_points[-1]
    #         label_x, label_y = int(end_point[0]), int(end_point[1])
            
    #         # Add small circle and number at end point
    #         # cv2.circle(img, (label_x, label_y), max(8, int(H/80)), color, -1)
    #         cv2.circle(img, (label_x, label_y), max(8, int(H/80)), color, -1) if traj_idx == 0 else draw_alpha_circle_manual(img, (label_x, label_y), max(8, int(H/80)), color, -1, alpha=0.4)
    #         # cv2.putText(img, str(traj_idx + 1), (label_x - 5, label_y + 5), 
    #         #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #     break

    # waypoints = multiple_waypoints[0]
    for idx, waypoints in enumerate(multiple_waypoints):
        path_width = 0.3  # Increased width for navigation path
        offsets_left, offsets_right = [], []
        
        for i in range(len(waypoints)-1):
            p1, p2 = waypoints[i], waypoints[i+1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            dir_unit = direction / norm
            # 法向量 (左: [-dy, dx], 右: [dy, -dx])
            normal = np.array([-dir_unit[1], dir_unit[0]])
            offsets_left.append(p1 + path_width * normal)
            offsets_right.append(p1 - path_width * normal)
        
        # 最后一个点也加上
        if len(waypoints) > 1:
            direction = waypoints[-1] - waypoints[-2]
            dir_unit = direction / np.linalg.norm(direction)
            normal = np.array([-dir_unit[1], dir_unit[0]])
            offsets_left.append(waypoints[-1] + path_width * normal)
            offsets_right.append(waypoints[-1] - path_width * normal)

        offsets_left = np.array(offsets_left)
        offsets_right = np.array(offsets_right)

        # ========== step2: 投影函数 ==========
        def project_points(points):
            num_points = points.shape[0]
            ego_points = np.zeros((num_points, 3))
            ego_points[:, :2] = points
            cam_translation = np.array([-0.3, 0, camera_height])
            R = np.array([
                [0, -1, 0],   
                [0, 0, -1],   
                [1, 0, 0]     
            ])
            cam_points = (R @ (ego_points - cam_translation).T).T
            valid_mask = cam_points[:, 2] > 0
            cam_points = cam_points[valid_mask]
            if cam_points.shape[0] == 0:
                return np.zeros((0,2))
            
            x = cam_points[:, 0] / cam_points[:, 2]
            y = cam_points[:, 1] / cam_points[:, 2]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan(r)
            theta_d = theta * (1 + k * theta**2)
            scaling = np.zeros_like(r)
            mask = r > 1e-8
            scaling[mask] = theta_d[mask] / r[mask]
            scaling[~mask] = 1.0
            x_distorted = x * scaling
            y_distorted = y * scaling
            u = f * x_distorted + cx
            v = f * y_distorted + cy
            return np.column_stack((u, v))

        points_center = project_points(waypoints)
        points_left = project_points(offsets_left)
        points_right = project_points(offsets_right)

        # ========== step3: Create navigation-style path overlay ==========
        if len(points_left) > 1 and len(points_right) > 1:
            # Create smooth polygon for the path
            polygon = np.vstack([points_left, points_right[::-1]])
            polygon = polygon.astype(np.int32)
            
            # Create overlay with navigation green color
            overlay = img.copy()
            
            # Navigation green color (similar to the reference image)
            if mode_i == 3:
                nav_green = (65, 200, 80)  # BGR format - forest green
            elif mode_i == 2:
                nav_green = (65, 175, 80) # BGR format - forest green
            elif mode_i == 1:
                nav_green = (65, 140, 80)  # BGR format - forest green
            else:
                nav_green = (65, 101, 76)  # BGR format - forest green

            if idx != 0:
                nav_green = (160, 110, 110)
            
            # Fill the path polygon
            cv2.fillPoly(overlay, [polygon], color=nav_green)
            
            # Blend with original image for semi-transparency
            alpha = 0.7  # Higher alpha for more prominent path
            img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        
        # Add subtle border/outline to the path
        # cv2.polylines(img, [polygon], isClosed=True, color=(0, 100, 0), thickness=2)
    for idx, waypoints in enumerate(multiple_waypoints):
        path_width = 0.3  # Increased width for navigation path
        offsets_left, offsets_right = [], []
        
        for i in range(len(waypoints)-1):
            p1, p2 = waypoints[i], waypoints[i+1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            dir_unit = direction / norm
            # 法向量 (左: [-dy, dx], 右: [dy, -dx])
            normal = np.array([-dir_unit[1], dir_unit[0]])
            offsets_left.append(p1 + path_width * normal)
            offsets_right.append(p1 - path_width * normal)
        
        # 最后一个点也加上
        if len(waypoints) > 1:
            direction = waypoints[-1] - waypoints[-2]
            dir_unit = direction / np.linalg.norm(direction)
            normal = np.array([-dir_unit[1], dir_unit[0]])
            offsets_left.append(waypoints[-1] + path_width * normal)
            offsets_right.append(waypoints[-1] - path_width * normal)

        offsets_left = np.array(offsets_left)
        offsets_right = np.array(offsets_right)

        # ========== step2: 投影函数 ==========
        def project_points(points):
            num_points = points.shape[0]
            ego_points = np.zeros((num_points, 3))
            ego_points[:, :2] = points
            cam_translation = np.array([-0.3, 0, camera_height])
            R = np.array([
                [0, -1, 0],   
                [0, 0, -1],   
                [1, 0, 0]     
            ])
            cam_points = (R @ (ego_points - cam_translation).T).T
            valid_mask = cam_points[:, 2] > 0
            cam_points = cam_points[valid_mask]
            if cam_points.shape[0] == 0:
                return np.zeros((0,2))
            
            x = cam_points[:, 0] / cam_points[:, 2]
            y = cam_points[:, 1] / cam_points[:, 2]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan(r)
            theta_d = theta * (1 + k * theta**2)
            scaling = np.zeros_like(r)
            mask = r > 1e-8
            scaling[mask] = theta_d[mask] / r[mask]
            scaling[~mask] = 1.0
            x_distorted = x * scaling
            y_distorted = y * scaling
            u = f * x_distorted + cx
            v = f * y_distorted + cy
            return np.column_stack((u, v))

        points_center = project_points(waypoints)
        points_left = project_points(offsets_left)
        points_right = project_points(offsets_right)

        # ========== step3: Create navigation-style path overlay ==========
        if len(points_left) > 1 and len(points_right) > 1:
            # Create smooth polygon for the path
            polygon = np.vstack([points_left, points_right[::-1]])
            polygon = polygon.astype(np.int32)
            
            # Create overlay with navigation green color
            overlay = img.copy()
            
            # Navigation green color (similar to the reference image)
            if mode_i == 3:
                nav_green = (65, 200, 80)  # BGR format - forest green
            elif mode_i == 2:
                nav_green = (65, 175, 80) # BGR format - forest green
            elif mode_i == 1:
                nav_green = (65, 140, 80)  # BGR format - forest green
            else:
                nav_green = (65, 101, 76)  # BGR format - forest green

            if idx != 0:
                nav_green = (150, 110, 110)
            
            # Fill the path polygon
            cv2.fillPoly(overlay, [polygon], color=nav_green)
            
            # Blend with original image for semi-transparency
            alpha = 0.7  # Higher alpha for more prominent path
            img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        break
    # ========== step4: Add center line for guidance ==========
    # Add the vehicle starting point at bottom center
    # start_point = np.array([[W//2, H-1]])
    # points_center = np.vstack((start_point, points_center))
    
    # # Filter points within image bounds
    # points_center = points_center[
    #     (points_center[:, 0] >= 0) & (points_center[:, 0] < W) &
    #     (points_center[:, 1] >= 0) & (points_center[:, 1] < H)
    # ]
    
    # # Draw center guidance line with dashed style
    # if points_center.shape[0] > 1:
    #     # Create a more prominent center line
    #     center_color = (34, 34, 170)  # White center line
    #     line_thickness = max(6, int(H/200))
        
    #     for pt_id in range(points_center.shape[0]-1):
    #         x1, y1 = int(np.round(points_center[pt_id][0])), int(np.round(points_center[pt_id][1]))
    #         x2, y2 = int(np.round(points_center[pt_id+1][0])), int(np.round(points_center[pt_id+1][1]))
            
    #         if all(0 <= x < W and 0 <= y < H for x, y in [(x1, y1), (x2, y2)]):
    #             # Draw dashed line for center guidance
    #             draw_dashed_line(img, (x1, y1), (x2, y2), center_color, line_thickness)

    # ========== step5: Add distance markers (optional) ==========
    # Add small markers at regular intervals along the path
    if points_center.shape[0] > 2:
        marker_interval = max(1, len(points_center) // 5)  # 5 markers max
        for i in range(0, len(points_center), marker_interval):
            if i < len(points_center):
                x, y = int(points_center[i][0]), int(points_center[i][1])
                if 0 <= x < W and 0 <= y < H:
                    if mode_i == 3:
                        color = (34, 255, 70)
                    elif mode_i == 2:
                        color = (34, 220, 70)
                    elif mode_i == 1:
                        color = (34, 190, 70)
                    else:
                        color = (34, 150, 70)
                    cv2.circle(img, (x, y), max(3, int(H/150)), color, -1)
    for traj_idx, waypoints in enumerate(multiple_waypoints):
        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            print(f"Skipping trajectory {traj_idx}: Invalid shape {waypoints.shape}")
            continue
            
        # Project waypoints to image coordinates
        points_2d = project_points(waypoints)
        
        if points_2d.shape[0] == 0:
            print(f"Skipping trajectory {traj_idx}: No valid projected points")
            continue
        
        # Add vehicle starting point (bottom center of image) as first point
        start_point = np.array([[W//2, H-1]])
        points_2d = np.vstack((start_point, points_2d))
        
        # Filter points within image bounds
        valid_points = points_2d[
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        ]
        
        if valid_points.shape[0] < 2:
            print(f"Skipping trajectory {traj_idx}: Not enough valid points")
            continue
        
        # Get color and style for this trajectory
        color = line_colors[traj_idx]
        style = line_styles[traj_idx]
        
        # Draw centerline based on style
        if style == 'solid':
            if traj_idx == 0:
                draw_solid_centerline(img, valid_points, color, line_thickness)
            else:
                draw_solid_centerline_alpha(img, valid_points, color, line_thickness, alpha=0.4)
        elif style == 'dashed':
            draw_dashed_centerline(img, valid_points, color, line_thickness)
        elif style == 'dotted':
            draw_dotted_centerline(img, valid_points, color, line_thickness)
        else:
            # Default to solid
            draw_solid_centerline(img, valid_points, color, line_thickness)
    
        # Add trajectory number label at the end point
        if valid_points.shape[0] > 0:
            end_point = valid_points[-1]
            label_x, label_y = int(end_point[0]), int(end_point[1])
            
            # Add small circle and number at end point
            # cv2.circle(img, (label_x, label_y), max(8, int(H/80)), color, -1)
            cv2.circle(img, (label_x, label_y), max(8, int(H/80)), color, -1) if traj_idx == 0 else draw_alpha_circle_manual(img, (label_x, label_y), max(8, int(H/80)), color, -1, alpha=0.4)
        break
    # Resize output
    img_resized = img#cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
    
    if save_fig:
        cv2.imwrite(save_path, img_resized)
    
    return img_resized
def draw_alpha_circle_manual(img, center, radius, color, thickness, alpha=0.5):
    """
    Method 2: Manual pixel-by-pixel blending
    More control but slower for large circles
    """
    # Create temporary image for the circle
    temp_img = np.zeros_like(img)
    cv2.circle(temp_img, center, radius, color, thickness)
    
    # Create mask where circle exists
    mask = np.any(temp_img > 0, axis=2)
    
    # Blend only where mask is true
    img[mask] = (alpha * temp_img[mask] + (1 - alpha) * img[mask]).astype(np.uint8)


def project_multiple_horizons_to_fisheye(
        pose_data, orientation_data, horizons, intrinsic_params, image_path, 
        save_path='multiple_horizons.jpg', camera_height=0.41, save_fig=False,
        line_colors=None, line_styles=None, start_frame=0):
    """
    Project trajectories with multiple horizons onto fisheye image
    
    Args:
        pose_data: Array of shape (T, 3) containing vehicle poses
        orientation_data: Array of shape (T, 4) containing quaternion orientations
        horizons: List of horizon lengths to plot (e.g., [20, 40, 80, 160])
        intrinsic_params: [f, cx, cy, k] camera parameters
        image_path: Path to image or numpy array
        save_path: Output save path
        camera_height: Camera height in meters
        save_fig: Whether to save the result
        line_colors: List of colors (BGR) for each horizon, if None uses default colors
        line_styles: List of line styles ('solid', 'dashed', 'dotted') for each horizon
        start_frame: Starting frame index
    
    Returns:
        Processed image with multiple horizon trajectories
    """
    from e2enav.datasets.nav_dataset import quaternions_to_euler_angles, rotate_points_along_z
    
    # Convert orientations to euler angles
    angles = quaternions_to_euler_angles(orientation_data)
    
    # Prepare trajectories for each horizon
    multiple_waypoints = []
    
    for h in horizons:
        if start_frame + h + 1 > len(pose_data):
            print(f"Warning: Horizon {h} exceeds available data length")
            continue
            
        # Extract waypoints for this horizon
        waypoints = pose_data[start_frame:start_frame+h+1]
        waypoints = waypoints[:, :2]  # Take only x, y coordinates
        
        # Transform to vehicle frame
        waypoints = waypoints - waypoints[0]  # Center at origin
        waypoints = rotate_points_along_z(
            waypoints.reshape(1, -1, 2), 
            -angles[start_frame:start_frame+1, 2]
        )[0]
        
        multiple_waypoints.append(waypoints)
    
    if not multiple_waypoints:
        print("No valid trajectories to plot")
        return None
    
    # Default colors for different horizons (BGR format)
    default_colors = [
        (0, 255, 0),      # Green - shortest horizon
        (255, 0, 0),      # Blue  
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
    ]
    
    # Default line styles - make shorter horizons more prominent
    default_styles = []
    for i, h in enumerate(horizons):
        if i == 0:  # Shortest horizon - solid and thick
            default_styles.append('solid')
        elif i < len(horizons) // 2:  # Medium horizons - dashed
            default_styles.append('dashed')
        else:  # Longest horizons - dotted
            default_styles.append('dotted')
    
    # Use provided colors/styles or defaults
    if line_colors is None:
        line_colors = default_colors[:len(horizons)]
    if line_styles is None:
        line_styles = default_styles[:len(horizons)]
    
    # Ensure we have enough colors and styles
    while len(line_colors) < len(horizons):
        line_colors.extend(default_colors)
    while len(line_styles) < len(horizons):
        line_styles.extend(['solid'])
    
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        img = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
    else:
        print(f'Unsupported image_path type: {type(image_path)}')
        return None

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    H, W = img.shape[:2]
    f, cx, cy, k = intrinsic_params
    
    # ========== Projection function ==========
    def project_points(points):
        if points.shape[0] == 0:
            return np.zeros((0, 2))
            
        num_points = points.shape[0]
        ego_points = np.zeros((num_points, 3))
        ego_points[:, :2] = points
        
        cam_translation = np.array([-0.3, 0, camera_height])
        R = np.array([
            [0, -1, 0],   
            [0, 0, -1],   
            [1, 0, 0]     
        ])
        
        cam_points = (R @ (ego_points - cam_translation).T).T
        valid_mask = cam_points[:, 2] > 0
        cam_points = cam_points[valid_mask]
        
        if cam_points.shape[0] == 0:
            return np.zeros((0, 2))
        
        x = cam_points[:, 0] / cam_points[:, 2]
        y = cam_points[:, 1] / cam_points[:, 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(r)
        theta_d = theta * (1 + k * theta**2)
        
        scaling = np.zeros_like(r)
        mask = r > 1e-8
        scaling[mask] = theta_d[mask] / r[mask]
        scaling[~mask] = 1.0
        
        x_distorted = x * scaling
        y_distorted = y * scaling
        u = f * x_distorted + cx
        v = f * y_distorted + cy
        
        return np.column_stack((u, v))
    
    # ========== Process each horizon trajectory ==========
    for horizon_idx, (waypoints, horizon_length) in enumerate(zip(multiple_waypoints, horizons)):
        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            print(f"Skipping horizon {horizon_length}: Invalid shape {waypoints.shape}")
            continue
            
        # Project waypoints to image coordinates
        points_2d = project_points(waypoints)
        
        if points_2d.shape[0] == 0:
            print(f"Skipping horizon {horizon_length}: No valid projected points")
            continue
        
        # Add vehicle starting point (bottom center of image) as first point
        start_point = np.array([[W//2, H-1]])
        points_2d = np.vstack((start_point, points_2d))
        
        # Filter points within image bounds
        valid_points = points_2d[
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        ]
        
        if valid_points.shape[0] < 2:
            print(f"Skipping horizon {horizon_length}: Not enough valid points")
            continue
        
        # Get color and style for this horizon
        color = line_colors[horizon_idx]
        style = line_styles[horizon_idx]
        
        # Adaptive line thickness based on horizon length (shorter = thicker)
        max_thickness = max(8, int(H/100))
        min_thickness = max(2, int(H/200))
        line_thickness = int(max_thickness - (horizon_idx / max(1, len(horizons)-1)) * (max_thickness - min_thickness))
        
        # Draw trajectory based on style
        if style == 'solid':
            draw_solid_centerline(img, valid_points, color, line_thickness)
        elif style == 'dashed':
            draw_dashed_centerline(img, valid_points, color, line_thickness)
        elif style == 'dotted':
            draw_dotted_centerline(img, valid_points, color, line_thickness)
        else:
            draw_solid_centerline(img, valid_points, color, line_thickness)
        
        # Add horizon label at the end point
        if valid_points.shape[0] > 0:
            end_point = valid_points[-1]
            label_x, label_y = int(end_point[0]), int(end_point[1])
            
            # Add circle at end point
            cv2.circle(img, (label_x, label_y), max(6, int(H/120)), color, -1)
            
            # Add horizon length text
            text = f"{horizon_length}"
            font_scale = max(0.4, H/1000)
            thickness = max(1, int(H/500))
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Position text near the end point
            text_x = label_x + max(10, int(H/100))
            text_y = label_y - max(5, int(H/200))
            
            # Ensure text stays within image bounds
            text_x = min(text_x, W - text_size[0] - 5)
            text_y = max(text_y, text_size[1] + 5)
            
            # Draw text with background for better visibility
            cv2.rectangle(img, (text_x-2, text_y-text_size[1]-2), 
                         (text_x+text_size[0]+2, text_y+2), (0, 0, 0), -1)
            cv2.putText(img, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Add legend
    legend_y_start = 30
    legend_x_start = 30
    legend_spacing = 25
    
    for horizon_idx, horizon_length in enumerate(horizons):
        if horizon_idx < len(multiple_waypoints):
            color = line_colors[horizon_idx]
            y_pos = legend_y_start + horizon_idx * legend_spacing
            
            # Draw legend line
            cv2.line(img, (legend_x_start, y_pos), (legend_x_start + 30, y_pos), color, 3)
            
            # Draw legend text
            text = f"Horizon {horizon_length}"
            cv2.putText(img, text, (legend_x_start + 40, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if save_fig:
        cv2.imwrite(save_path, img)
    
    return img
# ========== Example Usage for Multiple Horizons ==========


# ========== Original Code ==========
import tqdm
intrinsic_params = [790, 1920 /2, 1080 /2, 0.0]
horizons = [20, 40, 80, 160]
horizons = [100]
scenario_name = '2025-07-29__C11170__f08a7628-8912-4e2a-89b3-8895ffda735a__3'
import os
import copy
scenario_name_list = os.listdir(f'/bigdata/datasets/coco_converted_50h/mcap/')
scenario_name_list.sort()
scenario_name_list = scenario_name_list[-300:] #[::2]
# scenario_name_list = os.listdir('/bigdata/honglin/vnav-test/a')
# scenario_name_list = [i.split('_pred')[0] for i in scenario_name_list]
import json
interesting_scenes = json.load(open('/bigdata/honglin/interesting_scene.json', 'r'))
interesting_scenes = [k for k, v in interesting_scenes.items() if np.sum(v) == 6]
scenario_name_list = [i for i in scenario_name_list if i in interesting_scenes]
temp = copy.deepcopy(scenario_name_list)

scenario_name_list = os.listdir('/bigdata/honglin/vis_traj_video')
scenario_name_list = [i.split('_pred')[0] for i in scenario_name_list] + temp
scenario_name_list = list(set(scenario_name_list))

scenario_name_list = os.listdir('/bigdata/honglin/S2E/saved_preds_ours')
scenario_name_list = [i.split('_1_ours.pkl')[0] for i in scenario_name_list if i.endswith('_1_ours.pkl')]
scenario_name_list = list(set(scenario_name_list))
# print(scenario_name_list, len(scenario_name_list))
# import sys; sys.exit()
# scenario_name_list = ['2025-08-08__C10702__dbbe1cbc-4a96-42de-a09a-547afa07ecc1__8']

scenario_name_list.sort()

# scenario_name_list = [
#     '2025-09-07__C10672__346d0544-dd9c-407e-a151-493e44222f5b__1757278181465230336',
#             '2025-09-07__C10808__8c7e0e22-37fc-4ae0-b50c-f6b82f7160b4__1757273255120427008',
#             '2025-09-08__C10019__6a0fbd18-9628-4199-8e80-5b173e0103bb__1757290035446450176',
#             '2025-09-08__C10341__01104c98-0938-4e3b-8324-b3df38b8caab__1757304149419397632',
#             '2025-09-08__C10595__f7b887ff-b3ca-44f0-a09a-d77e5e653865__1757366983827275520',
# ]
# scenario_name_list = np.random.choice(scenario_name_list, size=50, replace=False).tolist()
# scenario_name_list = ['2025-07-30__C11067__2cae1d80-095a-47d4-9843-e2af64b5082f__1']  + scenario_name_list
# '2025-08-03__C10849__7fc9b97d-a798-4c5f-90dd-5e6f6ea34777__8', 
# '2025-08-01__C10777__a9a8034b-c9d2-49b7-ad77-4b88b2e052ea__5', 
# '2025-08-02__C11160__586d0bdf-7e49-48dd-97e6-6479b265f720__18']


# cluster_centers = np.load('cluster_centers_64_coco_very_long.npy')
# cluster_centers = cluster_centers.reshape(64, 40, 3)[..., :2] * 0.25
# cluster_centers_very_long = cluster_centers
# cluster_centers_long = np.load('cluster_centers_64_coco_long.npy')
# cluster_centers_long = cluster_centers_long.reshape(64, 20, 3)[..., :2] * 0.25
# cluster_centers_mid = np.load('cluster_centers_64_coco_mid.npy')
# cluster_centers_mid = cluster_centers_mid.reshape(64, 10, 3)[..., :2] * 0.25
# cluster_centers_short = np.load('cluster_centers_64_coco_short.npy')
# cluster_centers_short = cluster_centers_short.reshape(64, 5, 3)[..., :2] * 0.25

# cluster_centers = np.load('cluster_centers_64.npy')
# cluster_centers = cluster_centers.reshape(64, 10, 3)
# plan_anchor = []
# for center in cluster_centers:
#     trajectory = center.reshape(10, 3)
#     x, y, heading = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
#     if np.all(x < 0.1):
#         x, y, heading = np.zeros_like(x), np.zeros_like(y), np.zeros_like(heading)
#     elif np.all(x < 0.3):
#         y = np.zeros_like(y)
#     else:
#         heading = np.arctan2(trajectory[1:, 1] - trajectory[:-1, 1], trajectory[1:, 0] - trajectory[:-1, 0])
#         heading = np.concatenate([heading[0:1], heading])
    
#     if np.any(np.abs(heading) > np.pi / 2):
#         continue
#     else:
#         plan_anchor.append(np.stack([x, y, heading], axis=1))
# cluster_centers = np.stack(plan_anchor)[..., :2] * 0.25
# ========== Example: Plot Multiple Horizons in One Image ==========
def plot_multiple_horizons_example():
    """
    Example showing how to plot multiple horizons in one image
    """
    horizons_to_plot = [20, 40, 80, 160]  # Different prediction horizons
    horizons_to_plot = [15, 40, 80, 160]  # Different prediction horizons

    for scenario_name in tqdm.tqdm(scenario_name_list):
        for t in [0]:  # tqdm.trange(15):
            start_frame = t * 5
            image_path = f'/bigdata/datasets/coco_converted_50h/5Hz_pinhole_images/{scenario_name}/{start_frame:04d}.png'
            pose = np.load(f'/bigdata/datasets/coco_converted_50h/mcap/{scenario_name}/frame_positions.npy')
            orientation = np.load(f'/bigdata/datasets/coco_converted_50h/mcap/{scenario_name}/frame_orientations.npy')
            
            # Plot multiple horizons in one image
            save_path_multiple_horizons = f'./vis_traj/{scenario_name}_{start_frame:04d}_multiple_horizons.png'
            
            try:
                project_multiple_horizons_to_fisheye(
                    pose_data=pose,
                    orientation_data=orientation,
                    horizons=horizons_to_plot,
                    intrinsic_params=intrinsic_params,
                    image_path=image_path,
                    save_path=save_path_multiple_horizons,
                    save_fig=True,
                    start_frame=start_frame
                )
                print(f"Multiple horizons plot saved: {save_path_multiple_horizons}")
            except Exception as e:
                print(f"Error plotting multiple horizons for {scenario_name}: {e}")
                continue

# ========== Original Code (Individual Horizon Plots) ==========
from torchcodec.decoders import VideoDecoder
def slerp_2d(v1, v2, t):
    """
    2D SLERP between two vectors
    
    Args:
        v1, v2: 2D vectors
        t: interpolation parameter [0, 1]
    """
    # Normalize
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    if n1 < 1e-6 or n2 < 1e-6:
        return (1 - t) * v1 + t * v2
    
    u1 = v1 / n1
    u2 = v2 / n2
    
    # Calculate angle
    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    omega = np.arccos(abs(dot))
    
    # if omega < 1e-6:
    #     # Vectors are parallel
    #     result_dir = (1 - t) * u1 + t * u2
    #     result_mag = (1 - t) * n1 + t * n2
    #     return result_dir / np.linalg.norm(result_dir) * result_mag
    if omega < 1e-6:
        # For parallel vectors, use linear interpolation of magnitudes only
        result_mag = (1 - t) * n1 + t * n2
        
        # Choose the direction based on which vector has larger magnitude
        if n1 >= n2:
            return u1 / np.linalg.norm(u1) * result_mag
        else:
            return u2 / np.linalg.norm(u2) * result_mag
    
    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega
    
    result_dir = a * u1 + b * u2
    result_mag = (1 - t) * n1 + t * n2
    
    return result_dir * result_mag

def simple_slerp_trajectory(trajectory, target_velocity=None, smoothness=0.5):
    """
    Simple SLERP-based constant velocity trajectory
    
    Args:
        trajectory: np.array of shape (N, 2) or (N, 3)
        target_velocity: desired velocity per step
        smoothness: 0=straight line, 1=follow original path closely
    
    Returns:
        new_trajectory: smoothed trajectory with constant velocity
        actual_velocity: achieved velocity
    """
    
    if len(trajectory) < 2:
        return trajectory, 0.0
    
    positions = trajectory[:, :2]
    start_pos = positions[0]
    end_pos = positions[-1]
    
    total_distance = np.linalg.norm(end_pos - start_pos)
    
    if total_distance < 1e-6:
        # Handle stationary case
        new_trajectory = trajectory.copy()
        for i in range(1, len(new_trajectory)):
            new_trajectory[i, 0] = new_trajectory[0, 0] + i * 0.01
        return new_trajectory, 0.01
    
    # Calculate target velocity
    if target_velocity is None:
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        target_velocity = np.mean(distances[distances > 1e-6])
    
    # Extract key waypoints from original trajectory for SLERP
    waypoints = []
    waypoint_indices = []
    
    # Always include start and end
    waypoints.append(start_pos)
    waypoint_indices.append(0)
    
    # Add intermediate waypoints based on curvature
    for i in range(2, len(positions) - 1):
        # Calculate curvature using three consecutive points
        p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Cross product magnitude indicates curvature
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            cross = np.cross(v1, v2)
            curvature = abs(cross) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            # Add waypoint if curvature is significant
            if curvature > 0.1:  # Threshold for significant curvature
                waypoints.append(p2)
                waypoint_indices.append(i)
    
    waypoints.append(end_pos)
    waypoint_indices.append(len(positions) - 1)
    
    # Generate new trajectory using SLERP between waypoints
    new_positions = np.zeros_like(positions)
    new_positions[0] = start_pos
    
    current_waypoint_idx = 0
    
    for i in range(1, len(positions)):
        t_global = i / (len(positions) - 1)
        
        # Find which waypoint segment we're in
        while (current_waypoint_idx < len(waypoint_indices) - 1 and 
               i > waypoint_indices[current_waypoint_idx + 1]):
            current_waypoint_idx += 1
        
        if current_waypoint_idx < len(waypoints) - 1:
            # SLERP between current and next waypoint
            start_waypoint = waypoints[current_waypoint_idx]
            end_waypoint = waypoints[current_waypoint_idx + 1]
            
            start_idx = waypoint_indices[current_waypoint_idx]
            end_idx = waypoint_indices[current_waypoint_idx + 1]
            
            # Local t within this segment
            if end_idx > start_idx:
                t_local = (i - start_idx) / (end_idx - start_idx)
            else:
                t_local = 0
            
            # SLERP between waypoints
            direction_to_waypoint = end_waypoint - start_waypoint
            direct_direction = end_pos - new_positions[i-1]
            
            # Blend between waypoint direction and direct direction
            if np.linalg.norm(direction_to_waypoint) > 1e-6 and np.linalg.norm(direct_direction) > 1e-6:
                blended_direction = slerp_2d(direction_to_waypoint, direct_direction, smoothness)
            else:
                blended_direction = direction_to_waypoint
            
            # Normalize and apply constant velocity
            if np.linalg.norm(blended_direction) > 1e-6:
                unit_direction = blended_direction / np.linalg.norm(blended_direction)
                step = unit_direction * target_velocity
            else:
                step = np.array([target_velocity, 0])
            
            new_positions[i] = new_positions[i-1] + step
        else:
            # Final segment - head directly to end
            direction_to_end = end_pos - new_positions[i-1]
            if np.linalg.norm(direction_to_end) > target_velocity:
                step = direction_to_end / np.linalg.norm(direction_to_end) * target_velocity
            else:
                step = direction_to_end
            
            new_positions[i] = new_positions[i-1] + step
    
    # Final adjustment to ensure we end at the target
    new_positions[-1] = end_pos
    
    # Reconstruct full trajectory
    if trajectory.shape[1] == 3:
        new_trajectory = np.zeros_like(trajectory)
        new_trajectory[:, :2] = new_positions
        
        # Smooth angle transitions
        for i in range(len(new_trajectory)):
            if i == 0:
                new_trajectory[i, 2] = trajectory[0, 2]
            else:
                direction = new_positions[i] - new_positions[i-1]
                if np.linalg.norm(direction) > 1e-6:
                    new_angle = np.arctan2(direction[1], direction[0])
                    
                    # Smooth angle transition
                    if i > 1:
                        prev_angle = new_trajectory[i-1, 2]
                        angle_diff = np.arctan2(np.sin(new_angle - prev_angle), 
                                              np.cos(new_angle - prev_angle))
                        new_trajectory[i, 2] = prev_angle + 0.8 * angle_diff
                    else:
                        new_trajectory[i, 2] = new_angle
                else:
                    new_trajectory[i, 2] = new_trajectory[i-1, 2]
    else:
        new_trajectory = new_positions
    
    return new_trajectory, target_velocity
import torch
def trajectory_nms(trajectory_raw, score_raw, max_trajectories=6, distance_threshold=2.0):
    """
    Apply NMS to trajectories based on endpoint distance
    
    Args:
        trajectory_raw: [B, N, T, 3] - batch, num_trajectories, time, (x,y,z)
        score_raw: [B, N] - confidence scores
        max_trajectories: int - number of trajectories to keep
        distance_threshold: float - minimum distance between trajectory endpoints
    
    Returns:
        selected_trajectories: [B, max_trajectories, T, 3]
        selected_scores: [B, max_trajectories]
        selected_indices: [B, max_trajectories]
    """
    B, N, T, _ = trajectory_raw.shape
    
    selected_trajectories = []
    selected_scores = []
    selected_indices = []
    
    for b in range(B):
        # Get trajectories and scores for this batch
        trajs = trajectory_raw[b]  # [N, T, 3]
        scores = score_raw[b]      # [N]
        
        # Sort by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        
        # NMS selection
        keep_indices = []
        
        for i in sorted_indices:
            if len(keep_indices) >= max_trajectories:
                break
                
            # Check if this trajectory is too close to any kept trajectory
            should_keep = True
            
            for kept_idx in keep_indices:
                # Calculate distance between trajectory endpoints
                endpoint_dist = torch.norm(trajs[i][-1, :2] - trajs[kept_idx][-1, :2])
                
                if endpoint_dist < distance_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep_indices.append(i)
        
        # Pad if we don't have enough trajectories
        while len(keep_indices) < max_trajectories:
            keep_indices.append(keep_indices[-1] if keep_indices else 0)
        
        keep_indices = torch.tensor(keep_indices[:max_trajectories])
        
        selected_trajectories.append(trajs[keep_indices])
        selected_scores.append(scores[keep_indices])
        selected_indices.append(keep_indices)
    
    return (torch.stack(selected_trajectories),
            torch.stack(selected_scores))
import random
scenario_name_list_all = os.listdir('/bigdata/datasets/coco_converted_50h/mcap')[1500:2000]
scenario_name_list = random.sample(scenario_name_list_all, 15)
# scenario_name_list = [
#         # '2025-07-16__C10959__67d8a3e8-e848-4866-b92f-659e643ecbaa__28',
#         # '2025-07-14__C10246__c9d8835b-c5bf-45df-ad79-a9b73ec83ba7__5',
#         # '2025-07-28__C10223__f08b99b1-4e01-4b8a-890d-379f68d8584b__7',   
#         # '2025-08-01__C11056__71efbb3e-c8fc-48e9-a705-f6af227eaf3e__16',
#         # '2025-07-31__C10981__5390d3d7-3b16-4844-b5dc-ab6b87ea5c9c__7',
#         # '2025-07-19__C11169__ad1d31e3-0d62-4c5c-ab05-071e009b2637__4',
#         # '2025-08-02__C11243__35bd5362-452f-446d-8b79-1a15ba651bb7__2',
#         # '2025-07-15__C11046__2d9c37c1-100b-4e59-a050-f852ca176995__19',
#         '2025-07-13__C10374__9d03c56b-81ff-4f6e-a49d-84d19512115f__1'
# ]

# 配置开关：是否保存预测结果（True=保存GT+预测, False=只保存GT）
SAVE_PREDICTIONS = True

for scenario_name in tqdm.tqdm(scenario_name_list):
    h=horizons[0]

    decoder = VideoDecoder(f'/bigdata/datasets/coco_converted_50h/mcap/{scenario_name}/front.mp4', dimension_order="NHWC")
    # image_path = f'/bigdata/datasets/coco_converted_50h/5Hz_pinhole_images/{scenario_name}/{start_frame:04d}.png'
    pose = np.load(f'/bigdata/datasets/coco_converted_50h/mcap/{scenario_name}/frame_positions.npy')
    orientation = np.load(f'/bigdata/datasets/coco_converted_50h/mcap/{scenario_name}/frame_orientations.npy')
    # print(pose.shape, orientation.shape)
    from e2enav.datasets.nav_dataset import quaternions_to_euler_angles, rotate_points_along_z
    angles_all = quaternions_to_euler_angles(orientation)
    for t in tqdm.trange(200*5):
        start_frame = t
        

        h=horizons[0]
        waypoints = pose[start_frame:start_frame+h+1]
        angles = angles_all[start_frame:start_frame+h+1]
        waypoints_z = waypoints[:, 2]
        # if waypoints_z[0] != 0.:
        #     print(waypoints_z[0], '\n')
        waypoints = waypoints[:, :2]
        waypoints = waypoints - waypoints[0]
        waypoints = rotate_points_along_z(waypoints.reshape(1, -1, 2), -angles[0:1, 2])[0]

        # active_mode = np.argmin(np.linalg.norm(waypoints[0::4,].reshape(1, -1, 2)[:, :-1] - cluster_centers[:], axis=1).mean(1), axis=0)
        image_path = decoder.get_frames_at(indices=[t]).data.squeeze().numpy()

        save_path = f'./vis_waypoint/vis_traj_all/{scenario_name}_{start_frame:04d}_{h:02d}.png'
        pred_save_path = f'./vis_waypoint/vis_traj_all/{scenario_name}_{start_frame:04d}_{h:02d}_pred.png'
        pred_1_save_path = f'./vis_waypoint/vis_traj_all/{scenario_name}_{start_frame:04d}_{h:02d}_pred_1.png'
        pred_2_save_path = f'./vis_waypoint/vis_traj_all/{scenario_name}_{start_frame:04d}_{h:02d}_pred_2.png'
        pred_3_save_path = f'./vis_waypoint/vis_traj_all/{scenario_name}_{start_frame:04d}_{h:02d}_pred_3.png'
        project_waypoints_to_fisheye_image_with_polygon_new(
            waypoints, intrinsic_params, image_path, save_path, save_fig=True, color=(100, 230, 100)
        )
        plot_multiple_horizons_example(scenario_name, start_frame, h)
        
        import pickle

        # 只有当SAVE_PREDICTIONS为True时才处理预测结果
        if SAVE_PREDICTIONS:
            try:
                image_path = decoder.get_frames_at(indices=[t]).data.squeeze().numpy()
                waypoints_pred = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_ours/{scenario_name}_{t // 4}_ours.pkl', 'rb'))[..., :2] * 0.25
                scores_pred = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_ours/{scenario_name}_{t // 4}_ours_scores.pkl', 'rb'))

                scores_pred = torch.from_numpy(scores_pred)
                waypoints_pred = torch.from_numpy(waypoints_pred)
                # scores_pred = torch.softmax(scores_pred, dim=-1)
                # print(scores_pred)
                
                waypoints_pred, scores_pred = trajectory_nms(waypoints_pred.unsqueeze(0), scores_pred.unsqueeze(0), max_trajectories=6, distance_threshold=2.)
                waypoints_pred_all = waypoints_pred.numpy()[0]
                scores_pred = scores_pred.numpy()[0]
                scores_pred = scores_pred - np.max(scores_pred)
                scores_pred = np.exp(scores_pred)
                scores_pred = scores_pred / scores_pred.sum(axis=-1, keepdims=True)
                scores_pred_normalized = scores_pred / scores_pred.sum(axis=-1, keepdims=True)

                threshold = 0.1

                import matplotlib.cm as cm
                cmap = cm.get_cmap("inferno", waypoints_pred_all.shape[0])  # 按照 mode 数量生成
                top_modes = np.argsort(scores_pred)[::-1]  # 所有 mode，从大到小
                down_modes = np.argsort(scores_pred) #[::1]  # 所有 mode，从小到大

                plotted_count = 0  # Track the number of actually plotted items
                for idx, active_mode_pred in enumerate(down_modes):
                    # print('plotted_count0', plotted_count)
                # # for active_mode_pred in range(waypoints_pred_all.shape[0]):
                # if True:
                    # active_mode_pred = np.argmax(scores_pred)
                    if scores_pred_normalized[active_mode_pred] < threshold:
                        continue
                    # print('plotted_count', plotted_count)

                    score_to_visualize = f'{scores_pred_normalized[active_mode_pred]:.4f}' + f' ({scores_pred[active_mode_pred]:.4f})'
                    waypoints_pred = waypoints_pred_all[active_mode_pred]
                    waypoints_pred = np.concatenate([np.zeros((1, 2)), waypoints_pred], axis=0)
                    color = tuple((np.array(cmap(idx)[:3]) * 255).astype(int))
                    image_path = project_waypoints_to_fisheye_image_with_polygon_new(
                        waypoints_pred, intrinsic_params, image_path, pred_save_path, save_fig=(idx == len(top_modes) - 1), color=color, base_alpha=0.35 if idx == len(top_modes) - 1 else 0.15, text_to_visualize=score_to_visualize, plotted_number=plotted_count#(255, 153, 51)
                    )[..., ::-1]
                    plotted_count += 1
                    # print('plotted_count', plotted_count)

                image_path = decoder.get_frames_at(indices=[t]).data.squeeze().numpy()
                waypoints_pred_all = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_commai/{scenario_name}_{t // 4}_ours.pkl', 'rb'))[..., :2] * 0.25
                scores_pred = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_commai/{scenario_name}_{t // 4}_ours_scores.pkl', 'rb'))

                scores_pred = torch.from_numpy(scores_pred)
                waypoints_pred = torch.from_numpy(waypoints_pred_all)
                waypoints_pred, scores_pred = trajectory_nms(waypoints_pred.unsqueeze(0), scores_pred.unsqueeze(0), max_trajectories=6, distance_threshold=2.0)
                waypoints_pred_all = waypoints_pred.numpy()[0]
                scores_pred = scores_pred.numpy()[0]
                scores_pred = scores_pred - np.max(scores_pred)
                scores_pred = np.exp(scores_pred)
                scores_pred = scores_pred / scores_pred.sum(axis=-1, keepdims=True)
                scores_pred_normalized = scores_pred / scores_pred.sum(axis=-1, keepdims=True)

                threshold = 0.1

                import matplotlib.cm as cm
                cmap = cm.get_cmap("inferno", waypoints_pred_all.shape[0])  # 按照 mode 数量生成
                top_modes = np.argsort(scores_pred)[::-1]  # 所有 mode，从大到小
                down_modes = np.argsort(scores_pred) #[::1]  # 所有 mode，从小到大

                plotted_count = 0  # Track the number of actually plotted items
                for idx, active_mode_pred in enumerate(down_modes):
                    # print('plotted_count0', plotted_count)
                # # for active_mode_pred in range(waypoints_pred_all.shape[0]):
                # if True:
                    # active_mode_pred = np.argmax(scores_pred)
                    if scores_pred_normalized[active_mode_pred] < threshold:
                        continue
                    # print('plotted_count', plotted_count)

                    score_to_visualize = f'{scores_pred_normalized[active_mode_pred]:.4f}' + f' ({scores_pred[active_mode_pred]:.4f})'
                    waypoints_pred = waypoints_pred_all[active_mode_pred]
                    waypoints_pred = np.concatenate([np.zeros((1, 2)), waypoints_pred], axis=0)
                    color = tuple((np.array(cmap(idx)[:3]) * 255).astype(int))
                    image_path = project_waypoints_to_fisheye_image_with_polygon_new(
                        waypoints_pred, intrinsic_params, image_path, pred_1_save_path, save_fig=(idx == len(top_modes) - 1), color=color, base_alpha=0.35 if idx == len(top_modes) - 1 else 0.15, text_to_visualize=score_to_visualize, plotted_number=plotted_count#(255, 153, 51)
                    )[..., ::-1]
                    plotted_count += 1

                image_path = decoder.get_frames_at(indices=[t]).data.squeeze().numpy()
                waypoints_pred_all = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_ours_1/{scenario_name}_{t // 4}_ours.pkl', 'rb'))[..., :2] * 0.25
                scores_pred = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_ours_1/{scenario_name}_{t // 4}_ours_scores.pkl', 'rb'))

                scores_pred = torch.from_numpy(scores_pred)
                waypoints_pred = torch.from_numpy(waypoints_pred_all)
                waypoints_pred, scores_pred = trajectory_nms(waypoints_pred.unsqueeze(0), scores_pred.unsqueeze(0), max_trajectories=6, distance_threshold=2.0)
                waypoints_pred_all = waypoints_pred.numpy()[0]
                scores_pred = scores_pred.numpy()[0]
                scores_pred = scores_pred - np.max(scores_pred)
                scores_pred = np.exp(scores_pred)
                scores_pred = scores_pred / scores_pred.sum(axis=-1, keepdims=True)
                scores_pred_normalized = scores_pred / scores_pred.sum(axis=-1, keepdims=True)

                threshold = 0.1

                import matplotlib.cm as cm
                cmap = cm.get_cmap("inferno", waypoints_pred_all.shape[0])  # 按照 mode 数量生成
                top_modes = np.argsort(scores_pred)[::-1]  # 所有 mode，从大到小
                down_modes = np.argsort(scores_pred) #[::1]  # 所有 mode，从小到大

                plotted_count = 0  # Track the number of actually plotted items
                for idx, active_mode_pred in enumerate(down_modes):
                    # print('plotted_count0', plotted_count)
                # # for active_mode_pred in range(waypoints_pred_all.shape[0]):
                # if True:
                    # active_mode_pred = np.argmax(scores_pred)
                    if scores_pred_normalized[active_mode_pred] < threshold:
                        continue
                    # print('plotted_count', plotted_count)

                    score_to_visualize = f'{scores_pred_normalized[active_mode_pred]:.4f}' + f' ({scores_pred[active_mode_pred]:.4f})'
                    waypoints_pred = waypoints_pred_all[active_mode_pred]
                    waypoints_pred = np.concatenate([np.zeros((1, 2)), waypoints_pred], axis=0)
                    color = tuple((np.array(cmap(idx)[:3]) * 255).astype(int))
                    image_path = project_waypoints_to_fisheye_image_with_polygon_new(
                        waypoints_pred, intrinsic_params, image_path, pred_2_save_path, save_fig=(idx == len(top_modes) - 1), color=color, base_alpha=0.35 if idx == len(top_modes) - 1 else 0.15, text_to_visualize=score_to_visualize, plotted_number=plotted_count#(255, 153, 51)
                    )[..., ::-1]
                    plotted_count += 1

                image_path = decoder.get_frames_at(indices=[t]).data.squeeze().numpy()
                waypoints_pred_all = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_ours_gl/{scenario_name}_{t // 4}_ours.pkl', 'rb'))[..., :2] * 0.25
                scores_pred = pickle.load(open(f'/bigdata/honglin/S2E/saved_preds_ours_gl/{scenario_name}_{t // 4}_ours_scores.pkl', 'rb'))

                scores_pred = torch.from_numpy(scores_pred)
                waypoints_pred = torch.from_numpy(waypoints_pred_all)
                waypoints_pred, scores_pred = trajectory_nms(waypoints_pred.unsqueeze(0), scores_pred.unsqueeze(0), max_trajectories=6, distance_threshold=2.0)
                waypoints_pred_all = waypoints_pred.numpy()[0]
                scores_pred = scores_pred.numpy()[0]
                scores_pred = scores_pred - np.max(scores_pred)
                scores_pred = np.exp(scores_pred)
                scores_pred = scores_pred / scores_pred.sum(axis=-1, keepdims=True)
                scores_pred_normalized = scores_pred / scores_pred.sum(axis=-1, keepdims=True)

                threshold = 0.1

                import matplotlib.cm as cm
                cmap = cm.get_cmap("inferno", waypoints_pred_all.shape[0])  # 按照 mode 数量生成
                top_modes = np.argsort(scores_pred)[::-1]  # 所有 mode，从大到小
                down_modes = np.argsort(scores_pred) #[::1]  # 所有 mode，从小到大

                plotted_count = 0  # Track the number of actually plotted items
                for idx, active_mode_pred in enumerate(down_modes):
                    # print('plotted_count0', plotted_count)
                # # for active_mode_pred in range(waypoints_pred_all.shape[0]):
                # if True:
                    # active_mode_pred = np.argmax(scores_pred)
                    if scores_pred_normalized[active_mode_pred] < threshold:
                        continue
                    # print('plotted_count', plotted_count)

                    score_to_visualize = f'{scores_pred_normalized[active_mode_pred]:.4f}' + f' ({scores_pred[active_mode_pred]:.4f})'
                    waypoints_pred = waypoints_pred_all[active_mode_pred]
                    waypoints_pred = np.concatenate([np.zeros((1, 2)), waypoints_pred], axis=0)
                    color = tuple((np.array(cmap(idx)[:3]) * 255).astype(int))
                    image_path = project_waypoints_to_fisheye_image_with_polygon_new(
                        waypoints_pred, intrinsic_params, image_path, pred_3_save_path, save_fig=(idx == len(top_modes) - 1), color=color, base_alpha=0.35 if idx == len(top_modes) - 1 else 0.15, text_to_visualize=score_to_visualize, plotted_number=plotted_count#(255, 153, 51)
                    )[..., ::-1]
                    plotted_count += 1

            except Exception as e:
                print(f"❌ Test failed: {e}")
                break

        # break
    
    # 只有当SAVE_PREDICTIONS为True时才生成预测视频
    if SAVE_PREDICTIONS:
        full_video_path = f'./vis_traj_video_new/{scenario_name}_pred.mp4'
        os.system(f'ffmpeg -y -framerate 20 -i ./vis_waypoint/vis_traj_all/{scenario_name}_%4d_{h:02d}_pred.png -c:v libx264 -pix_fmt yuv420p {full_video_path}')
        full_video_path = f'./vis_traj_video_new/{scenario_name}_pred_1.mp4'
        os.system(f'ffmpeg -y -framerate 20 -i ./vis_waypoint/vis_traj_all/{scenario_name}_%4d_{h:02d}_pred_1.png -c:v libx264 -pix_fmt yuv420p {full_video_path}')
        full_video_path = f'./vis_traj_video_new/{scenario_name}_pred_2.mp4'
        os.system(f'ffmpeg -y -framerate 20 -i ./vis_waypoint/vis_traj_all/{scenario_name}_%4d_{h:02d}_pred_2.png -c:v libx264 -pix_fmt yuv420p {full_video_path}')
        full_video_path = f'./vis_traj_video_new/{scenario_name}_pred_3.mp4'
        os.system(f'ffmpeg -y -framerate 20 -i ./vis_waypoint/vis_traj_all/{scenario_name}_%4d_{h:02d}_pred_3.png -c:v libx264 -pix_fmt yuv420p {full_video_path}')
    
    # 始终生成GT视频
    full_video_path = f'./vis_waypoint/vis_traj_video_new/{scenario_name}_gt.mp4'
    os.system(f'ffmpeg -y -framerate 20 -i ./vis_waypoint/vis_traj_all/{scenario_name}_%4d_{h:02d}.png -c:v libx264 -pix_fmt yuv420p {full_video_path}')
    print('video saved = {full_video_path}')
    continue
    
