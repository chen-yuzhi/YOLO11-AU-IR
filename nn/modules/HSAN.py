import torch
import torch.nn as nn
import torch.nn.functional as F
import math   
from timm.models.layers import trunc_normal_
from .conv import Conv



class HSAN(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=4, reduction=16):
        super().__init__()

        # 卷积层
        self.cv1 = nn.Conv2d(c1, c2, 1, 1, groups=g)
        self.cv2 = nn.Conv2d(c2, c2, 1, 1, groups=g)

        # 特殊张量分解
        self.num_groups = g
        self.group_channels = c2 // g

        # 全局池化
        self.avg_pool_h = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool_h = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, 1))

        # 多尺度卷积模块
        self.multi_scale_attention = nn.ModuleList([
            self.MSCM(self.group_channels) 
            for _ in range(g)
        ])

        # 共享卷积层
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels // reduction, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.group_channels // reduction, self.group_channels, 1, 1)
            ) for _ in range(g)
        ])

        # 注意力权重计算
        self.coord_reduce_h = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.coord_reduce_w = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid_channel = nn.Sigmoid()

        # 注意力应用
        self.shortcut = shortcut and c1 == c2

    def MSCM(self, dim):
        """构建多尺度卷积模块"""
        return nn.Sequential(
            # 5x5卷积
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            
            # 多尺度卷积分支
            nn.Sequential(
                # 1x7和7x1卷积
                nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
                nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim),
                
                # 1x11和11x1卷积
                nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim),
                nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim),
                
                # 1x21和21x1卷积
                nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim),
                nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
            ),
            
            # 1x1卷积整合特征
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        # 主分支卷积
        identity = x
        x = self.cv1(x)
        x = self.cv2(x)

        # 分组
        x_grouped = x.view(batch_size, self.num_groups, self.group_channels, height, width)

        # 对每组应用注意力
        processed_groups = []
        for i in range(self.num_groups):
            group_feat = x_grouped[:, i, :, :, :]
            
            # 多尺度注意力
            group_feat = self.multi_scale_attention[i](group_feat)

            # 通道注意力
            avg_pool_channel = self.avg_pool_h(group_feat)
            max_pool_channel = self.max_pool_h(group_feat)
            channel_att = self.channel_attention[i](avg_pool_channel + max_pool_channel)
            channel_att = self.sigmoid_channel(channel_att)

            # 高度方向的空间注意力
            h_avg_pool = torch.mean(group_feat, dim=1, keepdim=True)
            h_max_pool, _ = torch.max(group_feat, dim=1, keepdim=True)
            h_coord_feat = torch.cat([h_avg_pool, h_max_pool], dim=1)
            h_att = self.coord_reduce_h(h_coord_feat)
            h_att = self.sigmoid_h(h_att)

            # 宽度方向的空间注意力
            w_avg_pool = torch.mean(group_feat, dim=1, keepdim=True)
            w_max_pool, _ = torch.max(group_feat, dim=1, keepdim=True)
            w_coord_feat = torch.cat([w_avg_pool, w_max_pool], dim=1)
            w_att = self.coord_reduce_w(w_coord_feat)
            w_att = self.sigmoid_w(w_att)

            # 结合注意力权重
            group_feat = group_feat * channel_att * h_att * w_att
            processed_groups.append(group_feat)

        # 重组分组结果
        x = torch.stack(processed_groups, dim=1).view(batch_size, -1, height, width)

        # 如果启用shortcut，则将原始输入与处理后的特征相加
        if self.shortcut:
            x = x + identity

        return x
