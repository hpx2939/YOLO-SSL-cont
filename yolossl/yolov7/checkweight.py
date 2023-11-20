import torch

# 設置權重文件的路徑
weight_file = 'D://yolov7/weights/aitod4h.pt'

# 讀取權重文件
model_state_dict = torch.load(weight_file)

# 查看權重文件的內容
print(model_state_dict.keys())  # 列印所有權重的名稱

# 可以進一步檢查特定權重的值
print(model_state_dict)