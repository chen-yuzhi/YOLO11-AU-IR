# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc  # 用于垃圾回收，释放不再使用的内存
import math  # 提供数学计算工具
import os  # 提供操作系统相关功能，如文件路径操作
import subprocess  # 用于运行子进程
import time  # 提供时间相关的功能
import warnings  # 用于管理警告信息
from copy import copy, deepcopy  # 用于对象的浅拷贝和深拷贝
from datetime import datetime, timedelta  # 提供日期和时间操作工具
from pathlib import Path  # 用于更方便地操作文件路径

# 科学计算与深度学习库
import numpy as np  # 提供高效的多维数组操作
import torch  # PyTorch 的核心模块，用于深度学习
from torch import distributed as dist  # 用于分布式训练
from torch import nn, optim  # `nn` 用于构建神经网络，`optim` 提供优化器

# 导入 YOLO 框架的核心工具
from ultralytics.cfg import get_cfg, get_save_dir  # 获取配置和保存路径
from ultralytics.data.utils import check_cls_dataset, check_det_dataset  # 数据集检查工具
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights  # 加载模型权重
from ultralytics.utils import (  # 实用工具模块
    DEFAULT_CFG,  # 默认配置
    LOCAL_RANK,  # 本地进程的 GPU ID
    LOGGER,  # 日志记录工具
    RANK,  # 进程的全局 ID
    TQDM,  # 进度条工具
    __version__,  # 版本号
    callbacks,  # 回调管理
    clean_url,  # 处理 URL 的工具
    colorstr,  # 格式化字符串，支持颜色输出
    emojis,  # 表情符号支持
    yaml_save,  # 将数据保存为 YAML 文件
)
from ultralytics.utils.autobatch import check_train_batch_size  # 动态调整适合的批量大小
from ultralytics.utils.checks import (  # 用于训练前检查和配置
    check_amp,  # 检查是否启用混合精度训练
    check_file,  # 检查文件路径
    check_imgsz,  # 验证输入图像大小
    check_model_file_from_stem,  # 验证模型文件名
    print_args,  # 打印训练参数
)
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command  # 分布式训练相关工具
from ultralytics.utils.files import get_latest_run  # 获取最新的运行记录
from ultralytics.utils.torch_utils import (  # PyTorch 实用工具
    TORCH_2_4,  # 检查 PyTorch 版本
    EarlyStopping,  # 提前停止训练的机制
    ModelEMA,  # 模型的指数移动平均
    autocast,  # 自动混合精度
    convert_optimizer_state_dict_to_fp16,  # 将优化器状态转为 FP16 精度
    init_seeds,  # 初始化随机种子
    one_cycle,  # 余弦学习率调度器
    select_device,  # 选择计算设备（CPU/GPU）
    strip_optimizer,  # 清理优化器状态以减小模型体积
    torch_distributed_zero_first,  # 确保分布式训练的第一个进程优先运行
)


class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): 配置参数，例如训练超参数、设备等。
        validator (BaseValidator): 验证器实例，用于评估模型性能。
        model (nn.Module): 模型实例，定义了要训练的神经网络。
        callbacks (defaultdict): 回调函数字典，用于在特定训练事件中触发函数。
        save_dir (Path): 结果保存的目录路径。
        wdir (Path): 权重保存的目录路径。
        last (Path): 最近一次保存的检查点路径。
        best (Path): 保存性能最佳的检查点路径。
        save_period (int): 设置每隔多少个 epoch 保存一次检查点（若 <1 则禁用）。
        batch_size (int): 训练时的批量大小。
        epochs (int): 训练的总 epoch 数。
        start_epoch (int): 训练的起始 epoch。
        device (torch.device): 指定使用的设备（CPU 或 GPU）。
        amp (bool): 是否启用自动混合精度 (AMP)。
        scaler (amp.GradScaler): AMP 的梯度缩放器实例。
        data (str): 数据集路径或描述文件。
        trainset (torch.utils.data.Dataset): 训练数据集实例。
        testset (torch.utils.data.Dataset): 测试数据集实例。
        ema (nn.Module): 模型的指数移动平均 (EMA) 实例。
        resume (bool): 是否从检查点恢复训练。
        lf (nn.Module): 损失函数实例。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        best_fitness (float): 当前训练过程中达到的最佳性能指标。
        fitness (float): 当前 epoch 的性能指标。
        loss (float): 当前 epoch 的损失值。
        tloss (float): 累计损失值。
        loss_names (list): 损失项的名称列表。
        csv (Path): 保存结果的 CSV 文件路径。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): 配置文件路径或默认配置。默认为 DEFAULT_CFG。
            overrides (dict, optional): 配置的覆盖参数。默认为 None。
        """
        self.args = get_cfg(cfg, overrides)  # 加载并解析训练配置
        self.check_resume(overrides)  # 检查是否从检查点恢复
        self.device = select_device(self.args.device, self.args.batch)  # 选择训练设备（CPU 或 GPU）
        self.validator = None  # 验证器初始化为 None
        self.metrics = None  # 存储评估指标
        self.plots = {}  # 用于存储绘图数据
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)  # 初始化随机种子

        # 设置保存目录
        self.save_dir = get_save_dir(self.args)  # 确定保存路径
        self.args.name = self.save_dir.name  # 更新名称以便于日志记录
        self.wdir = self.save_dir / "weights"  # 权重保存的子目录
        if RANK in {-1, 0}:  # 仅主进程创建保存路径
            self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
            self.args.save_dir = str(self.save_dir)  # 更新配置中的保存路径
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # 将参数保存为 YAML 文件
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 设置检查点路径
        self.save_period = self.args.save_period  # 保存周期设置

        # 初始化训练相关参数
        self.batch_size = self.args.batch  # 批量大小
        self.epochs = self.args.epochs or 100  # 总 epoch 数，若为 None 则默认 100
        self.start_epoch = 0  # 起始 epoch 为 0
        if RANK == -1:  # 仅在单 GPU 或 CPU 环境下打印参数
            print_args(vars(self.args))

        # 设备相关设置
        if self.device.type in {"cpu", "mps"}:  # 若设备为 CPU 或 MPS
            self.args.workers = 0  # 将数据加载器的 worker 数量设置为 0，以提升效率

        # 模型和数据集设置
        self.model = check_model_file_from_stem(self.args.model)  # 加载模型文件路径
        with torch_distributed_zero_first(LOCAL_RANK):  # 在分布式训练中避免多次下载数据集
            self.trainset, self.testset = self.get_dataset()  # 加载训练和测试数据集
        self.ema = None  # 初始化 EMA 为 None

        # 初始化优化工具
        self.lf = None  # 损失函数
        self.scheduler = None  # 学习率调度器

        # 记录训练过程的指标
        self.best_fitness = None  # 最佳指标
        self.fitness = None  # 当前指标
        self.loss = None  # 当前损失
        self.tloss = None  # 累计损失
        self.loss_names = ["Loss"]  # 损失项的名称
        self.csv = self.save_dir / "results.csv"  # 结果 CSV 文件路径
        self.plot_idx = [0, 1, 2]  # 绘图索引

        # 集成支持
        self.hub_session = None  # 集成会话初始化为 None

        # 回调函数设置
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 获取默认回调
        if RANK in {-1, 0}:  # 主进程添加集成回调
            callbacks.add_integration_callbacks(self)


    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        # 为特定事件添加新的回调函数
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        # 覆盖特定事件的所有回调函数，只保留新的回调函数
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        # 运行指定事件的所有回调函数
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # 确定分布式训练的设备数量
        if isinstance(self.args.device, str) and len(self.args.device):  # 设备参数为字符串（如 '0,1,2'）
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # 设备参数为列表（如 [0, 1]）
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # 设备为 CPU 或 Apple 的 MPS
            world_size = 0
        elif torch.cuda.is_available():  # GPU 可用，设备默认为 1
            world_size = 1
        else:  # 设备未指定且没有 GPU
            world_size = 0

        # 如果使用分布式训练 (DDP)
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # 确保特定参数在 DDP 中的设置
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # 生成 DDP 命令并运行
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)  # 执行分布式训练命令
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))  # 清理分布式进程
        else:
            # 若非分布式训练，直接开始训练
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        # 初始化学习率调度器
        if self.args.cos_lr:  # 余弦调度
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # 生成余弦调度器函数
        else:  # 线性调度
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        # 将调度器绑定到优化器
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        # 设置分布式训练参数
        torch.cuda.set_device(RANK)  # 为当前进程分配 GPU
        self.device = torch.device("cuda", RANK)  # 确定当前设备为 CUDA
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # 强制超时等待
        # 初始化分布式训练
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",  # NCCL 或 Gloo 后端
            timeout=timedelta(seconds=10800),  # 超时时间 3 小时
            rank=RANK,  # 当前进程的 rank
            world_size=world_size,  # 总进程数
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        # 初始化训练所需的 dataloader 和优化器
        self.run_callbacks("on_pretrain_routine_start")  # 触发 "on_pretrain_routine_start" 回调事件
        ckpt = self.setup_model()  # 加载或创建模型
        self.model = self.model.to(self.device)  # 将模型移至指定设备
        self.set_model_attributes()  # 设置模型的属性

        # 冻结特定层
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )  # 根据配置决定冻结哪些层
        always_freeze_names = [".dfl"]  # 默认冻结的层（如 DFL）
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):  # 判断当前参数是否需要冻结
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False  # 冻结参数
            elif not v.requires_grad and v.dtype.is_floating_point:  # 确保冻结的浮点参数可以梯度更新
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # 检查 AMP（自动混合精度）
        self.amp = torch.tensor(self.args.amp).to(self.device)  # 初始化 AMP 标志
        if self.amp and RANK in {-1, 0}:  # 若启用 AMP，且当前为主进程
            callbacks_backup = callbacks.default_callbacks.copy()  # 备份回调
            self.amp = torch.tensor(check_amp(self.model), device=self.device)  # 检查 AMP 支持
            callbacks.default_callbacks = callbacks_backup  # 恢复回调
        if RANK > -1 and world_size > 1:  # 若为分布式训练
            dist.broadcast(self.amp, src=0)  # 广播 AMP 设置至所有进程
        self.amp = bool(self.amp)  # 转为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )  # 初始化梯度缩放器
        if world_size > 1:  # 若分布式训练，封装模型为 DDP 模式
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # 验证输入图像尺寸
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 确定最大步幅
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)  # 验证图像尺寸
        self.stride = gs  # 保存步幅

        # 调整批量大小
        if self.batch_size < 1 and RANK == -1:  # 单 GPU 下估算最佳批量大小
            self.args.batch = self.batch_size = self.auto_batch()

        # 创建数据加载器
        batch_size = self.batch_size // max(world_size, 1)  # 根据设备数量调整批量大小
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")  # 训练加载器
        if RANK in {-1, 0}:  # 主进程创建验证加载器
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size * 2, rank=-1, mode="val"  # 验证加载器，批量大小为训练的两倍
            )
            self.validator = self.get_validator()  # 初始化验证器
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")  # 获取验证指标
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # 初始化验证指标为 0
            self.ema = ModelEMA(self.model)  # 初始化 EMA 模型
            if self.args.plots:
                self.plot_training_labels()  # 绘制训练标签分布

        # 初始化优化器
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # 累积梯度更新的次数
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # 调整权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs  # 迭代总次数
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )  # 创建优化器

        # 初始化学习率调度器
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False  # 提前停止机制
        self.resume_training(ckpt)  # 检查是否恢复训练
        self.scheduler.last_epoch = self.start_epoch - 1  # 设置调度器的起始 epoch
        self.run_callbacks("on_pretrain_routine_end")  # 触发 "on_pretrain_routine_end" 回调事件
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        # 训练过程的实际执行
        if world_size > 1:  # 若使用分布式训练
            self._setup_ddp(world_size)  # 设置分布式训练
        self._setup_train(world_size)  # 设置训练过程（如优化器、数据加载器等）

        nb = len(self.train_loader)  # 训练集的批次数
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 预热迭代次数
        last_opt_step = -1  # 上次优化的步骤
        self.epoch_time = None  # 初始化 epoch 计时
        self.epoch_time_start = time.time()  # 记录 epoch 开始时间
        self.train_time_start = time.time()  # 记录训练开始时间
        self.run_callbacks("on_train_start")  # 触发 "on_train_start" 回调事件
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb  # 关闭 mosaic 增强时的索引
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])  # 绘图的索引
        epoch = self.start_epoch  # 设置 epoch 起始值
        self.optimizer.zero_grad()  # 清空优化器的梯度
        while True:
            self.epoch = epoch  # 当前 epoch
            self.run_callbacks("on_train_epoch_start")  # 触发 "on_train_epoch_start" 回调事件
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略某些警告
                self.scheduler.step()  # 学习率更新

            self.model.train()  # 设置模型为训练模式
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)  # 设置数据加载器的 epoch
            pbar = enumerate(self.train_loader)  # 枚举训练数据加载器
            if epoch == (self.epochs - self.args.close_mosaic):  # 若达到关闭 mosaic 增强的条件
                self._close_dataloader_mosaic()  # 关闭 mosaic 增强
                self.train_loader.reset()  # 重置数据加载器

            if RANK in {-1, 0}:  # 仅在主进程显示进度条
                LOGGER.info(self.progress_string())  # 打印训练进度
                pbar = TQDM(enumerate(self.train_loader), total=nb)  # 使用 TQDM 显示进度条
            self.tloss = None  # 初始化当前 epoch 的总损失
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")  # 触发 "on_train_batch_start" 回调事件
                # Warmup：学习率预热
                ni = i + nb * epoch  # 当前迭代的步数
                if ni <= nw:
                    xi = [0, nw]  # 预热区间
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))  # 调整累积步数
                    for j, x in enumerate(self.optimizer.param_groups):  # 更新学习率
                        x["lr"] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # 前向传播
                with autocast(self.amp):  # 自动混合精度前向传播
                    batch = self.preprocess_batch(batch)  # 数据预处理
                    self.loss, self.loss_items = self.model(batch)  # 计算损失
                    if RANK != -1:  # 分布式训练中放大损失
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )  # 累计损失

                # 反向传播
                self.scaler.scale(self.loss).backward()  # 梯度缩放后的反向传播

                # 优化
                if ni - last_opt_step >= self.accumulate:  # 根据累积步数进行优化
                    self.optimizer_step()
                    last_opt_step = ni

                    # 根据时间停止训练
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)  # 超过指定时间则停止训练
                        if RANK != -1:  # 若是分布式训练，广播停止信号
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # 广播停止信号
                            self.stop = broadcast_list[0]
                        if self.stop:  # 如果超过训练时间，则停止训练
                            break

                # 日志记录
                if RANK in {-1, 0}:  # 主进程记录日志
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # 显示 GPU 内存使用情况
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 损失
                            batch["cls"].shape[0],  # 当前批次的大小
                            batch["img"].shape[-1],  # 当前图像尺寸
                        )
                    )
                    self.run_callbacks("on_batch_end")  # 触发 "on_batch_end" 回调事件
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)  # 绘制训练样本

                self.run_callbacks("on_train_batch_end")  # 触发 "on_train_batch_end" 回调事件

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 记录每个参数组的学习率
            self.run_callbacks("on_train_epoch_end")  # 触发 "on_train_epoch_end" 回调事件
            if RANK in {-1, 0}:  # 主进程进行模型验证与保存
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # 验证
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()  # 验证模型性能
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})  # 保存指标
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch  # 判断是否提前停止
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)  # 根据时间停止

                # 保存模型
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")  # 触发 "on_model_save" 回调事件

            # 学习率调度
            t = time.time()
            self.epoch_time = t - self.epoch_time_start  # 当前 epoch 的时间
            self.epoch_time_start = t  # 更新 epoch 开始时间
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)  # 估算平均 epoch 时间
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)  # 调整训练总 epoch
                self._setup_scheduler()  # 重新初始化学习率调度器
                self.scheduler.last_epoch = self.epoch  # 确保不调整 epoch
                self.stop |= epoch >= self.epochs  # 判断是否超过最大训练 epoch
            self.run_callbacks("on_fit_epoch_end")  # 触发 "on_fit_epoch_end" 回调事件
            self._clear_memory()  # 清理内存

            # 提前停止
            if RANK != -1:  # 分布式训练
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # 广播停止信号
                self.stop = broadcast_list[0]
            if self.stop:  # 如果达到停止条件，则退出
                break
            epoch += 1  # 增加 epoch

        if RANK in {-1, 0}:  # 结束训练后进行最终评估
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()  # 进行最终评估
            if self.args.plots:
                self.plot_metrics()  # 绘制训练过程中的指标图
            self.run_callbacks("on_train_end")  # 触发 "on_train_end" 回调事件
        self._clear_memory()  # 清理内存
        self.run_callbacks("teardown")  # 触发 "teardown" 回调事件

    def auto_batch(self, max_num_obj=0):
        """Get batch size by calculating memory occupation of model."""
        # 动态调整批量大小以适应模型的内存占用
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # 返回推荐的批量大小

    def _get_memory(self):
        """Get accelerator memory utilization in GB."""
        # 获取设备的内存使用情况
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9  # 转换为 GB

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        # 清理显存或内存
        gc.collect()  # 垃圾回收
        if self.device.type == "mps":
            torch.mps.empty_cache()  # 清理 MPS 显存
        elif self.device.type == "cpu":
            return  # 如果是 CPU，不做任何操作
        else:
            torch.cuda.empty_cache()  # 清理 CUDA 显存



    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        # 使用 pandas 读取训练结果 CSV 文件并返回为字典格式
        import pandas as pd  # 延迟导入以提高初始化速度
        return pd.read_csv(self.csv).to_dict(orient="list")  # 读取并转换为字典，按列存储

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        # 保存训练模型的检查点文件，并附加额外的元数据
        import io  # 用于在内存中操作字节流

        # 将检查点数据序列化到内存中
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,  # 当前训练的 epoch
                "best_fitness": self.best_fitness,  # 当前最佳的 fitness 值
                "model": None,  # 模型，保存时不包含模型本身
                "ema": deepcopy(self.ema.ema).half(),  # EMA 模型，半精度保存
                "updates": self.ema.updates,  # EMA 更新次数
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),  # 优化器状态，转换为 FP16
                "train_args": vars(self.args),  # 保存训练时的参数
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},  # 训练过程中的指标
                "train_results": self.read_results_csv(),  # 训练结果
                "date": datetime.now().isoformat(),  # 当前时间
                "version": __version__,  # 当前 YOLO 版本
                "license": "AGPL-3.0 (https://ultralytics.com/license)",  # 许可证
                "docs": "https://docs.ultralytics.com",  # 文档链接
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # 获取序列化的检查点字节流

        # 保存检查点文件
        self.last.write_bytes(serialized_ckpt)  # 保存最新的检查点
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 若当前 fitness 是最佳值，则保存最佳检查点
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 按照保存周期保存 epoch 检查点
        # 如果设置了 mosaic 并且当前 epoch 是最后一个要关闭 mosaic 的 epoch，则保存相应检查点
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        # 加载数据集并返回训练集和验证集的路径
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)  # 分类任务
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)  # 检测任务
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # 确保数据集路径是 YAML 文件
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data  # 将数据集存储到实例变量
        return data["train"], data.get("val") or data.get("test")  # 返回训练集和验证集路径

    def setup_model(self):
        """Load/create/download model for any task."""
        # 加载、创建或下载模型
        if isinstance(self.model, torch.nn.Module):  # 如果模型已经是一个 nn.Module 实例，则不需要重新加载
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):  # 如果模型是以 .pt 结尾，则加载权重
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):  # 如果有预训练权重
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # 加载模型
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # 执行优化器的单步更新，带有梯度裁剪和 EMA 更新
        self.scaler.unscale_(self.optimizer)  # 去除梯度缩放
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # 梯度裁剪，防止梯度爆炸
        self.scaler.step(self.optimizer)  # 优化器步进
        self.scaler.update()  # 更新缩放器
        self.optimizer.zero_grad()  # 清空优化器的梯度
        if self.ema:
            self.ema.update(self.model)  # 更新 EMA 模型

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        # 执行批量数据的预处理
        return batch  # 默认不做预处理，直接返回 batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        # 在验证集上运行模型验证
        metrics = self.validator(self)  # 调用验证器进行验证
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # 获取 fitness，若没有则使用负损失值作为 fitness
        if not self.best_fitness or self.best_fitness < fitness:  # 更新最佳 fitness
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        # 获取模型实例
        raise NotImplementedError("This task trainer doesn't support loading cfg files")  # 若未实现此方法，则抛出异常

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        # 返回验证器实例
        raise NotImplementedError("get_validator function not implemented in trainer")  # 如果未实现此方法，抛出异常

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        # 获取数据加载器实例
        raise NotImplementedError("get_dataloader function not implemented in trainer")  # 如果未实现此方法，抛出异常

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        # 构建数据集实例
        raise NotImplementedError("build_dataset function not implemented in trainer")  # 如果未实现此方法，抛出异常

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        # 返回带标签的损失项字典
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        # 设置或更新模型的参数
        self.model.names = self.data["names"]  # 设置模型的类别名称

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        # 构建训练目标张量
        pass  # 此方法尚未实现

    def progress_string(self):
        """Returns a string describing training progress."""
        # 返回描述训练进度的字符串
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        # 绘制训练样本
        pass  # 尚未实现

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        # 绘制训练标签
        pass  # 尚未实现

    def save_metrics(self, metrics):
        """保存训练过程中生成的指标到 CSV 文件。"""
        # 获取指标的键和值
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # 包括 epoch 和 time 两列
        # 如果 CSV 文件不存在，则添加表头，否则为空
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")
        t = time.time() - self.train_time_start  # 计算当前训练时间
        # 打开 CSV 文件并追加数据
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")  # 写入 epoch、时间和指标

    def plot_metrics(self):
        """可视化训练指标（未实现）。"""
        pass

    def on_plot(self, name, data=None):
        """注册绘图数据（可用于回调）。"""
        path = Path(name)  # 确定绘图路径
        self.plots[path] = {"data": data, "timestamp": time.time()}  # 存储绘图数据及时间戳

    def final_eval(self):
        """执行最终的评估和验证操作，适用于 YOLO 对象检测模型。"""
        ckpt = {}  # 用于存储检查点数据
        for f in self.last, self.best:  # 遍历最新检查点和最佳检查点
            if f.exists():  # 如果文件存在
                if f is self.last:  # 对最近的检查点进行处理
                    ckpt = strip_optimizer(f)  # 清理优化器状态
                elif f is self.best:  # 对最佳检查点进行处理
                    k = "train_results"  # 更新最佳检查点中的训练结果
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)  # 处理最佳检查点文件
                    LOGGER.info(f"\nValidating {f}...")  # 打印验证信息
                    self.validator.args.plots = self.args.plots  # 设置验证器是否生成绘图
                    self.metrics = self.validator(model=f)  # 使用验证器对模型进行验证
                    self.metrics.pop("fitness", None)  # 移除 fitness 指标
                    self.run_callbacks("on_fit_epoch_end")  # 触发 fit_epoch_end 回调事件

    def check_resume(self, overrides):
        """检查是否需要从检查点恢复训练，并更新相应的参数。"""
        resume = self.args.resume  # 检查是否指定了恢复训练的标志
        if resume:
            try:
                # 确定检查点路径是否存在
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())  # 获取最新运行的检查点

                # 检查数据集的 YAML 文件是否存在，不存在则重新下载
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data  # 更新为当前数据集路径

                resume = True  # 标记为恢复训练
                self.args = get_cfg(ckpt_args)  # 更新训练配置
                self.args.model = self.args.resume = str(last)  # 更新模型路径和恢复路径
                # 根据 overrides 参数覆盖某些训练参数
                for k in ("imgsz", "batch", "device", "close_mosaic"):
                    if k in overrides:
                        setattr(self.args, k, overrides[k])  # 动态更新参数

            except Exception as e:
                # 如果恢复检查点失败，抛出错误提示
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume  # 更新恢复标志

    def resume_training(self, ckpt):
        """从指定的检查点恢复 YOLO 模型的训练。"""
        if ckpt is None or not self.resume:  # 如果没有指定检查点或不需要恢复训练
            return
        best_fitness = 0.0  # 初始化最佳 fitness 值
        start_epoch = ckpt.get("epoch", -1) + 1  # 获取从哪一轮开始恢复训练
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # 恢复优化器状态
            best_fitness = ckpt["best_fitness"]  # 恢复最佳 fitness
        if self.ema and ckpt.get("ema"):  # 如果启用了 EMA，则恢复其状态
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # 加载 EMA 模型
            self.ema.updates = ckpt["updates"]  # 更新 EMA 更新次数
        assert start_epoch > 0, (  # 确保训练没有完全结束
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:  # 如果目标训练轮数小于恢复的起始轮数
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # 增加目标轮数
        self.best_fitness = best_fitness  # 更新最佳 fitness
        self.start_epoch = start_epoch  # 设置起始轮数
        if start_epoch > (self.epochs - self.args.close_mosaic):  # 如果恢复的轮数已经超过关闭 mosaic 的条件
            self._close_dataloader_mosaic()  # 关闭 mosaic 数据增强

    def _close_dataloader_mosaic(self):
        """关闭数据加载器中的 Mosaic 数据增强。"""
        if hasattr(self.train_loader.dataset, "mosaic"):  # 如果数据集支持 mosaic 属性
            self.train_loader.dataset.mosaic = False  # 禁用 mosaic
        if hasattr(self.train_loader.dataset, "close_mosaic"):  # 如果有关闭 mosaic 的方法
            LOGGER.info("Closing dataloader mosaic")  # 打印日志信息
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))  # 执行关闭操作

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        根据模型构建优化器，并支持指定学习率、动量、权重衰减等参数。

        Args:
            model (torch.nn.Module): 要优化的模型实例。
            name (str, optional): 优化器的名称，支持 'SGD', 'Adam', 'AdamW' 等，默认自动选择。
            lr (float, optional): 学习率，默认 0.001。
            momentum (float, optional): 动量系数，默认 0.9。
            decay (float, optional): 权重衰减系数，默认 1e-5。
            iterations (float, optional): 迭代次数，用于调整优化器参数，默认 1e5。

        Returns:
            torch.optim.Optimizer: 构建好的优化器实例。
        """
        g = [], [], []  # 参数组：g[0] 有权重衰减，g[1] 无权重衰减，g[2] 偏置参数
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 标准化层（如 BatchNorm2d）
        if name == "auto":  # 如果优化器选择为自动模式
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically..."
            )
            nc = getattr(model, "nc", 10)  # 获取模型的类别数，默认为 10
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # 根据公式自动计算学习率
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)  # 根据迭代次数选择优化器
            self.args.warmup_bias_lr = 0.0  # 若使用 Adam，bias 的 warmup 学习率不超过 0.01

        for module_name, module in model.named_modules():  # 遍历模型的模块
            for param_name, param in module.named_parameters(recurse=False):  # 遍历模块的参数
                fullname = f"{module_name}.{param_name}" if module_name else param_name  # 完整参数名称
                if "bias" in fullname:  # 偏置参数（无权重衰减）
                    g[2].append(param)
                elif isinstance(module, bn):  # 标准化层权重（无权重衰减）
                    g[1].append(param)
                else:  # 其他参数（有权重衰减）
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}  # 支持的优化器列表
        name = {x.lower(): x for x in optimizers}.get(name.lower())  # 标准化优化器名称
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)  # 初始化优化器
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)  # 使用 RMSProp
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)  # 使用 SGD 并启用 Nesterov 动量
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
            )

        # 添加参数组到优化器
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # 有权重衰减的参数组
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # 无权重衰减的权重
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer  # 返回优化器实例
