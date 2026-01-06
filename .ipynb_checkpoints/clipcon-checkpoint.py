import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

from src.diffusion.ddpm import DDPM
from src.diffusion.unet import UNet
from src.data.mnist import get_mnist_loader_and_transform


class MNISTCLIPGenerator:
    def __init__(self, model_path, T=1000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.T = T
        
        data = get_mnist_loader_and_transform()
        
        # 添加引导开始的时间步
        self.guidance_start_step = 300
        
        # 1. 加载MNIST DDPM模型
        print("Loading MNIST DDPM model...")
        self.ddpm = DDPM(
            T=T,
            eps_model=UNet(
                in_channels=data.in_channels,
                out_channels=data.out_channels,
                T=T+1,
                steps=data.recommended_steps,
                attn_step_indexes=data.recommended_attn_step_indexes
            ),
            device=device
        )
        
        # 加载训练好的权重
        self.ddpm.load_state_dict(torch.load(model_path, map_location=device))
        self.ddpm.to(device)
        self.ddpm.eval()
        
        # 2. 加载CLIP模型
        print("Loading CLIP model...")
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.clip_model.eval().requires_grad_(False)
        
        # 3. CLIP预处理参数
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        print("模型初始化完成!")
    
    def mnist_to_clip(self, mnist_images):
        """
        将MNIST图像(1,28,28)转换为CLIP需要的格式(3,224,224)
        """
        # 1. 从[-1, 1]转换到[0, 1]
        images = (mnist_images + 1) / 2
        
        # 2. 灰度转RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # 3. 上采样到224x224
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 4. 应用CLIP标准化
        return self.normalize(images)
    
    def compute_clip_loss(self, images, text_features):
        """改进的CLIP损失"""
        clip_input = self.mnist_to_clip(images)
        image_features = self.clip_model.encode_image(clip_input)
        
        # 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 使用负对数似然损失
        similarity = (image_features * text_features).sum(dim=-1)
        loss = -similarity.mean()
        
        return loss
    
    def guided_sampling_step(self, x_t, t, text_features, clip_guidance_scale=1000):
        """改进的引导采样步骤"""
        t_tensor = torch.tensor([t] * x_t.shape[0], device=self.device, dtype=torch.long)

        # 预测噪声
        with torch.no_grad():
            eps = self.ddpm.eps_model(x_t, t_tensor)

        if t > 0:
            alpha_t = self.ddpm.alpha_t_schedule[t]
            sqrt_one_minus_alpha_cumprod_t = self.ddpm.sqrt_minus_bar_alpha_t_schedule[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)

            # 预测x0
            pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * eps) / self.ddpm.sqrt_bar_alpha_t_schedule[t]
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # 应用CLIP引导到预测的x0上
            if t <= self.guidance_start_step:
                pred_x0_guided = pred_x0.detach().requires_grad_(True)

                # 计算CLIP损失
                clip_loss = self.compute_clip_loss(pred_x0_guided, text_features)

                # 计算梯度
                grad = torch.autograd.grad(clip_loss, pred_x0_guided)[0]

                # 动态调整引导强度
                guidance_scale = clip_guidance_scale * (t / self.guidance_start_step)
                pred_x0 = pred_x0 - guidance_scale * grad
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                # 从引导后的x0重新计算噪声
                eps = (x_t - self.ddpm.sqrt_bar_alpha_t_schedule[t] * pred_x0) / sqrt_one_minus_alpha_cumprod_t

            # 使用原来的采样公式
            beta_t = self.ddpm.beta_schedule[t]
            mean = (1.0 / sqrt_alpha_t) * (x_t - (1.0 - alpha_t) * eps / sqrt_one_minus_alpha_cumprod_t)

            # 计算方差
            alpha_cumprod_t = self.ddpm.bar_alpha_t_schedule[t]
            if t > 1:
                alpha_cumprod_t_prev = self.ddpm.bar_alpha_t_schedule[t-1]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=self.device)

            posterior_variance = beta_t * (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)

            noise = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(posterior_variance) * noise
        else:
            pred_x0 = x_t
            x_t_minus_1 = pred_x0

        return x_t_minus_1, pred_x0
    
    def generate(self, text_prompt, n_samples=1, timesteps=1000, 
                 clip_guidance_scale=50, show_progress=True):
        """
        生成图像
        """
        # 1. 编码文本提示
        text_input = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
        
        # 2. 初始化噪声
        shape = (n_samples, 1, 28, 28)
        x_t = torch.randn(shape, device=self.device)
        
        # 3. 扩散采样循环（反向过程）
        iterator = tqdm(range(timesteps, 0, -1), desc=f"'{text_prompt}'") if show_progress else range(timesteps, 0, -1)
        
        for t in iterator:
            x_t, pred_x0 = self.guided_sampling_step(
                x_t, t, text_features, clip_guidance_scale
            )
        
        # 4. 转换为图像列表
        final_images = []
        for i in range(n_samples):
            img = self.tensor_to_image(x_t[i].cpu())
            final_images.append(img)
        
        return final_images
    
    def tensor_to_image(self, tensor):
        """转换tensor为PIL图像"""
        tensor = tensor.detach().cpu()
        img = tensor.squeeze().numpy()
        img = np.clip(img, -1, 1)
        img = (img + 1) / 2
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)


def create_grid_image(all_images, save_path, n_digits=10, n_samples=6):
    """
    创建网格图：10行（每个数字）× 6列（每个数字的样本）
    
    Args:
        all_images: 列表的列表 [[digit0_samples], [digit1_samples], ...]
        save_path: 保存路径
        n_digits: 数字个数（0-9）
        n_samples: 每个数字的样本数
    """
    fig, axes = plt.subplots(n_digits, n_samples, figsize=(n_samples * 2, n_digits * 2))
    
    for digit in range(n_digits):
        for sample in range(n_samples):
            ax = axes[digit, sample]
            ax.imshow(all_images[digit][sample], cmap='gray')
            
            # 只在第一列显示数字标签
            if sample == 0:
                ax.set_ylabel(f'Digit {digit}', fontsize=12, fontweight='bold')
            
            # 只在第一行显示样本编号
            if digit == 0:
                ax.set_title(f'Sample {sample + 1}', fontsize=10)
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Grid image saved to: {save_path}")


def main():
    # ==================== 参数设置 ====================
    model_path = "model.pth"
    
    clip_guidance_scale = 100  # CLIP引导强度
    n_samples_per_digit = 6   # 每个数字生成6个样本
    
    # 文本提示模板（可以尝试不同的描述）
    text_templates = [
        "the number {}",  # 默认模板
        # "a handwritten digit {}",
        # "digit {}",
        # "the number {} written in black ink",
    ]
    
    template = text_templates[0]  # 选择使用的模板
    
    # 创建输出目录
    output_dir = "CLIP引导100"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("批量生成 MNIST 数字 (0-9) - CLIP引导")
    print("="*70)
    print(f"输出目录: {output_dir}/")
    print(f"每个数字样本数: {n_samples_per_digit}")
    print(f"总共图像数: {10 * n_samples_per_digit}")
    print(f"CLIP引导强度: {clip_guidance_scale}")
    print(f"文本模板: '{template}'")
    print("="*70 + "\n")
    
    # ==================== 创建生成器 ====================
    generator = MNISTCLIPGenerator(model_path, T=1000)
    
    # ==================== 生成所有数字 ====================
    print("\n开始生成图像...")
    print("="*70)
    
    all_images = []  # 存储所有生成的图像
    
    for digit in range(10):
        # 构造文本提示
        text_prompt = template.format(digit)
        
        print(f"\n生成数字 {digit} ({n_samples_per_digit} 个样本)...")
        print(f"文本提示: '{text_prompt}'")
        
        # 生成该数字的多个样本
        digit_images = generator.generate(
            text_prompt=text_prompt,
            n_samples=n_samples_per_digit,
            timesteps=1000,
            clip_guidance_scale=clip_guidance_scale,
            show_progress=True
        )
        
        all_images.append(digit_images)
        print(f"  ✓ 数字 {digit} 生成完成")
    
    # ==================== 创建网格图 ====================
    print("\n" + "="*70)
    print("创建网格图...")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(output_dir, f"all_digits_grid_{timestamp}.png")
    
    create_grid_image(all_images, grid_path, n_digits=10, n_samples=n_samples_per_digit)
    
    # ==================== 保存单独的图像 ====================
    print("\n保存单独的图像...")
    for digit in range(10):
        for sample_idx, img in enumerate(all_images[digit]):
            img_path = os.path.join(output_dir, f"digit{digit}_sample{sample_idx + 1}.png")
            img.save(img_path)
    print(f"✓ 保存了 {10 * n_samples_per_digit} 张单独的图像")
    
    # ==================== 保存配置信息 ====================
    config_path = os.path.join(output_dir, f"config_{timestamp}.txt")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("MNIST CLIP引导生成配置\n")
        f.write("="*70 + "\n\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"CLIP引导强度: {clip_guidance_scale}\n")
        f.write(f"每个数字样本数: {n_samples_per_digit}\n")
        f.write(f"总图像数: {10 * n_samples_per_digit}\n")
        f.write(f"文本模板: '{template}'\n")
        f.write(f"DDPM模型: {model_path}\n")
        f.write(f"引导开始步数: {generator.guidance_start_step}\n\n")
        
        f.write("生成的文本提示:\n")
        f.write("-"*70 + "\n")
        for digit in range(10):
            text_prompt = template.format(digit)
            f.write(f"数字 {digit}: '{text_prompt}'\n")
    
    print(f"\n✓ 配置信息已保存: {config_path}")
    
    # ==================== 完成 ====================
    print("\n" + "="*70)
    print("生成完成!")
    print("="*70)
    print(f"输出目录: {output_dir}/")
    print(f"文件:")
    print(f"  - all_digits_grid_{timestamp}.png (网格图)")
    print(f"  - digit*_sample*.png ({10 * n_samples_per_digit} 张单独图像)")
    print(f"  - config_{timestamp}.txt (配置信息)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()