# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolov8n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolov8n.pt                 # PyTorch
                          yolov8n.torchscript        # TorchScript
                          yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolov8n_openvino_model     # OpenVINO
                          yolov8n.engine             # TensorRT
                          yolov8n.mlpackage          # CoreML (macOS-only)
                          yolov8n_saved_model        # TensorFlow SavedModel
                          yolov8n.pb                 # TensorFlow GraphDef
                          yolov8n.tflite             # TensorFlow Lite
                          yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolov8n_paddle_model       # PaddlePaddle
                          yolov8n.mnn                # MNN
                          yolov8n_ncnn_model         # NCNN
"""

import json  # 用于处理 JSON 文件
import time  # 提供时间相关工具
from pathlib import Path  # 用于文件路径操作

import numpy as np  # 数值计算库
import torch  # PyTorch 深度学习框架

# 导入 YOLO 框架中的工具模块
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    BaseValidator.

    一个用于创建验证器的基类。

    Attributes:
        args (SimpleNamespace): 验证器的配置参数。
        dataloader (DataLoader): 用于验证的数据加载器。
        pbar (tqdm): 验证过程中的进度条。
        model (nn.Module): 用于验证的模型。
        data (dict): 数据集相关的字典。
        device (torch.device): 验证过程中使用的设备。
        batch_i (int): 当前批次的索引。
        training (bool): 模型是否处于训练模式。
        names (dict): 类别名称。
        seen: 验证过程中已处理的图像数。
        stats: 用于存储验证统计数据的占位符。
        confusion_matrix: 混淆矩阵占位符。
        nc: 类别数。
        iouv (torch.Tensor): IoU 阈值，从 0.50 到 0.95，每次递增 0.05。
        jdict (dict): 用于存储 JSON 格式的验证结果。
        speed (dict): 记录验证过程中每个阶段的耗时（毫秒）。
        save_dir (Path): 保存结果的目录。
        plots (dict): 用于存储可视化的绘图数据。
        callbacks (dict): 存储各种回调函数。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        初始化一个 BaseValidator 实例。

        Args:
            dataloader (torch.utils.data.DataLoader): 用于验证的数据加载器。
            save_dir (Path, optional): 保存验证结果的目录。
            pbar (tqdm.tqdm): 用于显示验证进度的进度条。
            args (SimpleNamespace): 验证器的配置参数。
            _callbacks (dict): 存储各种回调函数的字典。
        """
        self.args = get_cfg(overrides=args)  # 解析并加载验证配置
        self.dataloader = dataloader  # 设置验证数据加载器
        self.pbar = pbar  # 初始化进度条
        self.stride = None  # 模型步幅占位符
        self.data = None  # 数据集信息占位符
        self.device = None  # 设备占位符
        self.batch_i = None  # 当前批次索引占位符
        self.training = True  # 初始状态为训练模式
        self.names = None  # 类别名称占位符
        self.seen = None  # 已验证的图像数
        self.stats = None  # 统计数据占位符
        self.confusion_matrix = None  # 混淆矩阵占位符
        self.nc = None  # 类别数量
        self.iouv = None  # IoU 阈值
        self.jdict = None  # JSON 验证结果占位符
        self.speed = {  # 初始化速度统计
            "preprocess": 0.0,  # 预处理时间
            "inference": 0.0,  # 推理时间
            "loss": 0.0,  # 损失计算时间
            "postprocess": 0.0,  # 后处理时间
        }

        # 设置保存目录
        self.save_dir = save_dir or get_save_dir(self.args)  # 获取或创建保存目录
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # 创建必要子目录
        if self.args.conf is None:
            self.args.conf = 0.001  # 默认置信度阈值为 0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)  # 检查并设置输入图像尺寸

        # 初始化绘图和回调
        self.plots = {}  # 绘图存储字典
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 设置默认回调

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        执行验证过程，在数据加载器上运行推理并计算性能指标。

        Args:
            trainer: 如果模型处于训练模式，则传入训练器实例。
            model: 如果验证未处于训练模式，则传入待验证的模型。

        Returns:
            训练模式下返回性能指标字典，测试模式下返回最终性能统计。
        """

        self.training = trainer is not None  # 判断是否处于训练模式
        augment = self.args.augment and (not self.training)  # 数据增强仅在非训练模式下启用
        if self.training:
            # 如果处于训练模式，从训练器中获取设备和数据集信息
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp  # 是否使用半精度（AMP 模式）
            model = trainer.ema.ema or trainer.model  # 使用 EMA 模型或原始模型
            model = model.half() if self.args.half else model.float()  # 模型切换到半精度或全精度
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)  # 初始化损失为 0
            # 如果即将结束训练，启用绘图
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()  # 切换模型到验证模式
        else:
            # 非训练模式下加载模型
            if str(self.args.model).endswith(".yaml"):  # 如果指定的是未训练的模型配置
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)  # 添加集成回调
            model = AutoBackend(  # 加载后端支持的模型
                weights=model or self.args.model,  # 模型权重
                device=select_device(self.args.device, self.args.batch),  # 选择设备
                dnn=self.args.dnn,  # 是否使用 DNN 推理
                data=self.args.data,  # 数据集路径
                fp16=self.args.half,  # 是否使用半精度推理
            )
            self.device = model.device  # 更新设备信息
            self.args.half = model.fp16  # 更新精度信息
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine  # 获取模型属性
            imgsz = check_imgsz(self.args.imgsz, stride=stride)  # 检查图像尺寸是否与模型步幅对齐
            if engine:  # 如果使用 TensorRT 推理引擎
                self.args.batch = model.batch_size  # 更新批量大小
            elif not pt and not jit:  # 非 PyTorch 模型
                self.args.batch = model.metadata.get("batch", 1)  # 默认批量大小为 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            # 根据任务类型加载数据集
            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:  # 检测任务数据集
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":  # 分类任务数据集
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:  # 未知任务类型抛出异常
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:  # 如果是 CPU 或 Apple MPS 设备
                self.args.workers = 0  # 数据加载线程数设为 0，加速推理
            if not pt:  # 非 PyTorch 模型禁用矩形推理
                self.args.rect = False
            self.stride = model.stride  # 保存模型步幅
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)  # 获取数据加载器

            model.eval()  # 切换模型到验证模式
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # 模型预热

        self.run_callbacks("on_val_start")  # 触发验证开始回调事件
        dt = (
            Profile(device=self.device),  # 初始化阶段计时器
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))  # 创建进度条
        self.init_metrics(de_parallel(model))  # 初始化性能指标
        self.jdict = []  # 清空 JSON 验证结果
        for batch_i, batch in enumerate(bar):  # 遍历每个批次的数据
            self.run_callbacks("on_val_batch_start")  # 触发批次开始回调
            self.batch_i = batch_i  # 更新当前批次索引

            # 预处理
            with dt[0]:
                batch = self.preprocess(batch)

            # 推理
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # 损失计算（仅训练模式下）
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # 后处理
            with dt[3]:
                preds = self.postprocess(preds)

            # 更新性能指标
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:  # 若启用绘图且当前批次数小于 3
                self.plot_val_samples(batch, batch_i)  # 绘制验证样本
                self.plot_predictions(batch, preds, batch_i)  # 绘制预测结果

            self.run_callbacks("on_val_batch_end")  # 触发批次结束回调

        stats = self.get_stats()  # 获取性能统计
        self.check_stats(stats)  # 检查性能统计
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))  # 计算阶段耗时
        self.finalize_metrics()  # 完成指标计算
        self.print_results()  # 打印结果
        self.run_callbacks("on_val_end")  # 触发验证结束回调

        if self.training:  # 若处于训练模式
            model.float()  # 将模型切换回全精度
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}  # 汇总结果
            return {k: round(float(v), 5) for k, v in results.items()}  # 返回结果，保留 5 位小数
        else:
            # 打印速度信息
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            LOGGER.info(f'FPS:{(1000/sum(self.speed.values())):.2f}')
            # 保存 JSON 格式的预测结果
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # 保存 JSON 文件
                stats = self.eval_json(stats)  # 更新统计结果
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")  # 打印保存路径信息
            return stats  # 返回性能统计

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        根据 IoU 和类别匹配预测与真实目标。

        Args:
            pred_classes (torch.Tensor): 预测类别索引，形状为 (N, )。
            true_classes (torch.Tensor): 真实类别索引，形状为 (M, )。
            iou (torch.Tensor): N x M 的 IoU 矩阵，表示预测框和真实框的 IoU。
            use_scipy (bool): 是否使用 scipy 进行匹配（精度更高）。

        Returns:
            torch.Tensor: 匹配结果矩阵，形状为 (N, 10)，表示对每个 IoU 阈值的匹配情况。
        """
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)  # 初始化匹配结果矩阵
        correct_class = true_classes[:, None] == pred_classes  # 检查类别是否匹配
        iou = iou * correct_class  # 将类别不匹配的 IoU 置为 0
        iou = iou.cpu().numpy()  # 转为 numpy 数组以加速处理
        for i, threshold in enumerate(self.iouv.cpu().tolist()):  # 遍历每个 IoU 阈值
            if use_scipy:  # 使用 scipy 进行优化匹配
                import scipy  # 局部导入 scipy
                cost_matrix = iou * (iou >= threshold)  # 构建代价矩阵，仅保留满足阈值的 IoU
                if cost_matrix.any():  # 如果有满足条件的匹配
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0  # 检查是否有效
                    if valid.any():  # 若有有效匹配
                        correct[detections_idx[valid], i] = True
            else:  # 使用 numpy 进行匹配
                matches = np.nonzero(iou >= threshold)  # 找到满足 IoU 阈值的匹配
                matches = np.array(matches).T
                if matches.shape[0]:  # 如果存在匹配
                    if matches.shape[0] > 1:  # 多对多匹配
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]  # 按 IoU 排序
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 每个真实目标只匹配一次
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 每个预测框只匹配一次
                    correct[matches[:, 1].astype(int), i] = True  # 更新匹配结果
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)  # 转为 Tensor 返回


    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Preprocesses the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
