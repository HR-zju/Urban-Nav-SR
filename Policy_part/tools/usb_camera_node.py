import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')
        
        # --- 1. 初始化 ROS 2 发布者 ---
        # 发布到 /image_raw 话题，队列深度为 10
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        
        # 创建 CvBridge 对象用于格式转换
        self.bridge = CvBridge()
        
        # --- 2. 摄像头配置---
        # 强制指定 V4L2 后端，索引 0 代表第一个摄像头
        self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # 关键配置：设置像素格式为 MJPG，解决带宽不足导致的卡顿
        # 注意：VideoWriter_fourcc 返回的是整数，直接传给 set 即可
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 设置分辨率 480p 720p
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 设置帧率
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        # 检查摄像头是否打开
        if not self.capture.isOpened():
            self.get_logger().error("无法打开摄像头，请检查连接或权限 (如: sudo chmod 666 /dev/video0)")
            # 初始化失败，虽然节点启动了但无法工作
            return

        self.get_logger().info("摄像头启动，发布图像：/image_raw")
        
        # --- 3. 创建定时器 ---
        # ROS 2 推荐使用定时器来触发回调，而不是死循环
        # camera 10Hz
        self.timer = self.create_timer(0.05, self.timer_callback)
 
    def timer_callback(self):
        ret, image = self.capture.read()
        if ret:
            # 镜像翻转
            #image = cv2.flip(image, 1)
            #cv2.imshow("USB Camera", image)
            #cv2.waitKey(1)
            
            # --- 核心步骤：OpenCV 转 ROS 2 消息 ---
            try:
                # cv2_to_imgmsg 将 numpy 数组转换为 sensor_msgs/Image
                # encoding="bgr8" 对应 OpenCV 的默认 BGR 格式
                image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                image_msg.header.stamp = self.get_clock().now().to_msg()
                image_msg.header.frame_id = "camera_link"
                
                # 发布消息
                self.publisher_.publish(image_msg)
            except Exception as e:
                self.get_logger().error(f"图像转换失败: {e}")
        else:
            self.get_logger().warn("无法接收帧")

    def destroy_node(self):
        # 节点销毁时释放摄像头资源
        if self.capture.isOpened():
            self.capture.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

