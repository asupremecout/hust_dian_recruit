import torch
import torch.nn as nn
import torch.nn.functional as F
import shorter_transformer


class PatchEmbedding(nn.Module):
    """将图像分割为patch并嵌入"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2    #计算patch的数量

        # 使用卷积层实现patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        #为了适配Transformer架构的输入要求，需要对patch进行变换
        #目标是[Batch_size, seq_len, d_model]，d_model: 每个token的维度，seq_len: 序列长度（token数量）
        #初始时输入: [Batch_size, 3, 224, 224]

    def forward(self, x):
        # [Batch_size, in-Channel, H, W] -> [Batch_size, embed_dim, num_patches_h, num_patches_w]
        #channel从in-Channel变成embed_dim
        #尺寸从H*W变成num_patches_h*num_patches_w
        x = self.proj(x)
        # 展平到第2个维度: [Batch_size, embed_dim, num_patches_h, num_patches_w] -> [Batch_size, embed_dim, num_patches]
        x = x.flatten(2)
        # 转置: [Batch_size, embed_dim, num_patches] -> [Batch_size, num_patches, embed_dim]
        x = x.transpose(1, 2)
        #最终x为[Batch_size, num_patches, embed_dim]符合ransformer架构的输入要求
        return x


class VisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 分类token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #利用 Parameter创造可学习参数
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))


    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 添加分类token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #每个token都会通过自注意力与其他所有token交互。CLS token作为序列的第一个token，会与所有图像patch token进行交互
        #第一个维度扩展为 x.shape[0]（即 batch_size）后面两个是-1，保持后面两个维度保持不变。
        x = torch.cat((cls_tokens, x), dim=1)  # [Batch_size, num_patches+1, embed_dim]
        # 在序列维度（dim=1）上拼接，得到(batch_size, num_patches+1, embed_dim)
        # 添加位置编码
        x = x + self.pos_embed




