from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('./runs/train/weights/best.pt')

# 設定資料夾路徑
source_folder = './plate'  # 替換為你的資料夾路徑
output_folder = './output'  # 預測結果輸出的目錄

# 預測資料夾中的所有圖片
# model.predict(source=source_folder, project=output_folder, name='batch_predict', save=True)

# Predict a video
model.predict(source='test_video.mp4', project='output/video', name='predict', save=True)

# -----------------------------------------------------------------

# # Open the video
# cap = cv2.VideoCapture('test_video.mp4')
#
# # Define the output video parameters
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
#
# while cap.isOpened():
#     ret, frame = cap.read()  # Read each frame
#     if not ret:
#         break
#
#     # Predict using YOLO
#     results = model(frame)  # Predict the current frame
#
#     # Render the results on the frame
#     frame_with_results = results[0].plot()  # 使用 `plot()` 方法繪製偵測結果
#
#     # Show the frame with the predictions
#     cv2.imshow('Prediction', frame_with_results)
#
#     # Write the frame to output video
#     out.write(frame_with_results)
#
#     # Press 'q' to quit the display window
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()