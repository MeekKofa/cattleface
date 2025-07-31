$ python main.py --data cattleface --arch vgg_yolov8 --depth '{"vgg_yolov8": [16]}' --train_batch 32 --epochs 2 --lr 0.0001 --drop 0.5 --num_workers 4 --pin_memory --gpu-ids 0 --task_name normal_training --optimizer adam
WARNING:root:torchmetrics.detection.MeanAveragePrecision not available.
Torch version:  2.5.1+cu121
Torchvision version:  0.20.1+cu121
CUDA available:  True
Device ID: 0
  Name: NVIDIA GeForce RTX 3050 Laptop GPU
  Compute Capability: 8.6
  Total Memory: 4.00 GB
  Multi-Processor Count: 16
  Is Integrated: 0
  Is Multi-GPU Board: 0
  L2_cache_size: 1572864
  gcnArchName: NVIDIA GeForce RTX 3050 Laptop GPU
  is_integrated: 0
  is_multi_gpu_board: 0
  major: 8
  max_threads_per_multi_processor: 1536
  minor: 6
  multi_processor_count: 16
  name: NVIDIA GeForce RTX 3050 Laptop GPU
  regs_per_multiprocessor: 65536
  total_memory: 4294443008
  uuid: 6ad7d66c-da3b-b3e7-7b75-3f578b7b3df9
  warp_size: 32

INFO - Logger initialized. Logging to: out\normal_training\cattleface\vgg_yolov8_16\training.log
INFO - Main script started.
INFO - ModelLoader initialized with models: resnet, densenet, vgg, vgg_myccc, vgg_yolov8, meddef1
INFO - OptimizerLoader initialized with optimizers: sgd, adam, rmsprop, adagrad
INFO - LRSchedulerLoader initialized with schedulers: StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
INFO - ModelLoader initialized with models: resnet, densenet, vgg, vgg_myccc, vgg_yolov8, meddef1
INFO - Starting normal training task
Created transform pipeline: ['RandomHorizontalFlip', 'Resize', 'ToTensor']
Created transform pipeline: ['Resize', 'ToTensor']
INFO - Loading cattleface object detection dataset:
INFO - Found 4562 training images
INFO - Found 977 validation images
INFO - Found 979 test images
INFO - Classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383]
INFO - Looking for normal checkpoint in: out/normal_training/cattleface/vgg_yolov8_16/save_model
INFO - ModelLoader: No checkpoint directory found for vgg_yolov8_16 in out/normal_training/cattleface/vgg_yolov8_16/save_model
INFO - ModelLoader: Created a new model: vgg_yolov8_16
INFO - Loading Conv2d parameters...
INFO - Loading BatchNorm2d parameters...
INFO - Successfully loaded model using memory-efficient wrapper
INFO - Created adam optimizer with learning rate 0.0001
INFO - Object detection model (vgg_yolov8) detected. Using CrossEntropyLoss for classes and SmoothL1Loss for boxes.
INFO - Object detection model detected. Loss is computed inside the model.
INFO - Timer initialized.
INFO - Using matplotlib backend: True
INFO - Applying dropout with rate: 0.5
INFO - Metrics will be saved to runs\normal_training\cattleface\vgg_yolov8_16\20250704-181418
INFO - Metrics will be saved to runs\normal_training\cattleface\vgg_yolov8_16\20250704-181418
INFO - Training vgg_yolov8_16...
INFO - Initial model parameters: 18.35M
WARNING - torchmetrics.detection.MeanAveragePrecision not available.
ERROR - Error initializing training components for vgg_yolov8_16: stack expects each tensor to be equal size, but got [9, 4] at entry 0 and [8, 4] at entry 2

