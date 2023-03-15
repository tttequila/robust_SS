import cv2

video_path = '/mntnfs/lee_data1/liuqinghua/code/gse_scrip_server/baseline_masking/vox2_sample.mp4'
captureObj = cv2.VideoCapture(video_path)
roiSize = 112
i = 0
while (captureObj.isOpened()):
    ret, frame = captureObj.read()
    grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayed = cv2.resize(grayed, (roiSize * 2, roiSize * 2))
    roi = grayed[int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2)),
        int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2))]
    i += 1
    if i == 50:
        cv2.imwrite('/mntnfs/lee_data1/liuqinghua/code/gse_scrip_server/baseline_masking/roi.jpg', roi)
        cv2.imwrite('/mntnfs/lee_data1/liuqinghua/code/gse_scrip_server/baseline_masking/frame.jpg', frame)
        break
    # cv2.imwrite('/mntnfs/lee_data1/liuqinghua/code/gse_scrip_server/baseline_masking/roi.jpg', roi)
    # cv2.imwrite('/mntnfs/lee_data1/liuqinghua/code/gse_scrip_server/baseline_masking/frame.jpg', frame)
