import cv2

def get_available_cameras():
    """ 사용 가능한 카메라 목록을 동적으로 반환 """
    available_cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:  # 프레임이 읽히지 않으면 종료
            break
        available_cameras.append(f"CAM{index+1}")
        cap.release()
        index += 1
    return available_cameras

def generate_frames(camera_index=0):
    """ 웹캠에서 프레임을 생성하여 스트리밍 """
    cap = cv2.VideoCapture(camera_index)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
