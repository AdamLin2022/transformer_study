import torch

if torch.cuda.is_available():
    print("恭喜！GPU 已经准备就绪！")
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
else:
    print("注意：目前只能使用 CPU，未检测到 GPU。")