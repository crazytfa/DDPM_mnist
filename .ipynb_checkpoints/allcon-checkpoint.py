import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime

from src.diffusion.ddpm import DDPM
from src.diffusion.unet import UNet
from src.data.mnist import get_mnist_loader_and_transform

# 导入训练好的分类器
from train_mnist_classifier import ResNet18
from weakclassifier import VeryWeakClassifier


class MNISTClassifierGuidedGenerator:
    def __init__(self, ddpm_path, classifier_path, T=1000, 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 # 温度控制参数（默认不使用）
                 temperature=1.0,
                 grad_clip=None,
                 adaptive_guidance=False):
        self.device = device
        self.T = T
        
        # 控制参数
        self.temperature = temperature
        self.grad_clip = grad_clip
        self.adaptive_guidance = adaptive_guidance
        
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
        
        # 加载训练好的DDPM权重
        self.ddpm.load_state_dict(torch.load(ddpm_path, map_location=device))
        self.ddpm.to(device)
        self.ddpm.eval()
        
        # 2. 加载分类器
        print("Loading MNIST Classifier...")
        self.classifier = ResNet18(num_classes=10).to(device)
        # self.classifier = VeryWeakClassifier(num_classes=10).to(device)
        
        checkpoint = torch.load(classifier_path, map_location=device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        print(f"✓ Classifier loaded with accuracy: {checkpoint['val_acc']:.2f}%")
        
        # 显示控制参数
        if temperature != 1.0 or grad_clip is not None or adaptive_guidance:
            print("\n温度控制参数:")
            print(f"  Temperature: {self.temperature}")
            print(f"  Gradient Clip: {self.grad_clip}")
            print(f"  Adaptive Guidance: {self.adaptive_guidance}")
        else:
            print("  (使用默认设置，不启用温度控制)")
        
        print("\n模型初始化完成!")
    
    def compute_classifier_gradient(self, x, target_class):
        """
        计算分类器引导的梯度
        支持温度缩放和梯度裁剪
        """
        x = x.requires_grad_(True)
        
        # 通过分类器
        logits = self.classifier(x)
        
        # 温度缩放
        scaled_logits = logits / self.temperature
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        target_log_prob = log_probs[:, target_class].sum()
        
        # 计算梯度
        grad = torch.autograd.grad(target_log_prob, x)[0]
        
        # 梯度裁剪
        if self.grad_clip is not None:
            grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
        
        return grad
    
    def guided_sampling_step(self, x_t, t, target_class, guidance_scale=1.0):
        """
        执行带分类器引导的采样步骤
        """
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
            
            # 应用分类器引导到预测的x0上
            if t <= self.guidance_start_step and guidance_scale > 0:
                pred_x0_guided = pred_x0.detach().requires_grad_(True)
                
                # 自适应引导（如果启用）
                current_guidance = guidance_scale * (t / self.guidance_start_step)
                
                if self.adaptive_guidance:
                    with torch.no_grad():
                        logits = self.classifier(pred_x0_guided)
                        probs = F.softmax(logits, dim=-1)
                        confidence = probs[0, target_class].item()
                    
                    if confidence > 0.8:
                        current_guidance *= 0.3
                    elif confidence > 0.5:
                        current_guidance *= 0.6
                
                # 计算分类器梯度
                grad = self.compute_classifier_gradient(pred_x0_guided, target_class)
                
                # 应用梯度到预测的x0
                pred_x0 = pred_x0.detach() + current_guidance * grad
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
    
    def generate(self, target_digit, n_samples=1, timesteps=1000, 
                 guidance_scale=10.0, show_progress=True):
        """
        生成指定数字的图像
        """
        # 初始化噪声
        shape = (n_samples, 1, 28, 28)
        x_t = torch.randn(shape, device=self.device)
        
        # 扩散采样循环（反向过程）
        iterator = tqdm(range(timesteps, 0, -1), desc=f"Digit {target_digit}") if show_progress else range(timesteps, 0, -1)
        
        for t in iterator:
            x_t, pred_x0 = self.guided_sampling_step(
                x_t, t, target_digit, guidance_scale
            )
        
        # 最终图像
        final_images = []
        for i in range(n_samples):
            img = self.tensor_to_image(x_t[i].cpu())
            final_images.append(img)
        
        # 最终分类（检查第一个样本）
        with torch.no_grad():
            logits = self.classifier(x_t[:1])
            probs = F.softmax(logits, dim=-1)
            predicted = probs.argmax(dim=-1).item()
            confidence = probs[0, target_digit].item()
        
        return final_images, predicted, confidence
    
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
    ddpm_path = "model.pth"
    classifier_path = "checkpoints/resnet18_mnist_classifier.pth"
    # classifier_path = "checkpoints/weak_classifier.pth"
    
    guidance_scale = 50  # 引导强度
    n_samples_per_digit = 6  # 每个数字生成6个样本
    
    # 温度控制参数（默认不使用）
    temperature = 1.0  # 默认1.0（不使用温度缩放）
    grad_clip = None   # 默认None（不使用梯度裁剪）
    adaptive_guidance = False  # 默认False（不使用自适应引导）
    
    # 创建输出目录
    output_dir = "resnet引导5+tem0.1"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("批量生成 MNIST 数字 (0-9)")
    print("="*70)
    print(f"输出目录: {output_dir}/")
    print(f"每个数字样本数: {n_samples_per_digit}")
    print(f"总共图像数: {10 * n_samples_per_digit}")
    print(f"引导强度: {guidance_scale}")
    print(f"温度控制: {'启用' if temperature != 1.0 else '未启用'}")
    print("="*70 + "\n")
    
    # ==================== 创建生成器 ====================
    generator = MNISTClassifierGuidedGenerator(
        ddpm_path=ddpm_path,
        classifier_path=classifier_path,
        T=1000,
        temperature=temperature,
        grad_clip=grad_clip,
        adaptive_guidance=adaptive_guidance
    )
    
    # ==================== 生成所有数字 ====================
    print("\n开始生成图像...")
    print("="*70)
    
    all_images = []  # 存储所有生成的图像
    results = []  # 存储生成结果
    
    for digit in range(10):
        print(f"\n生成数字 {digit} ({n_samples_per_digit} 个样本)...")
        
        # 生成该数字的多个样本
        digit_images, predicted, confidence = generator.generate(
            target_digit=digit,
            n_samples=n_samples_per_digit,
            timesteps=1000,
            guidance_scale=guidance_scale,
            show_progress=True
        )
        
        all_images.append(digit_images)
        
        # 记录结果
        results.append({
            'digit': digit,
            'predicted': predicted,
            'confidence': confidence,
            'correct': predicted == digit
        })
        
        status = "✓" if predicted == digit else "✗"
        print(f"  {status} 数字 {digit}: 预测={predicted}, 置信度={confidence:.3f}")
    
    # ==================== 创建网格图 ====================
    print("\n" + "="*70)
    print("创建网格图...")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_path = os.path.join(output_dir, f"all_digits_grid_{timestamp}.png")
    
    create_grid_image(all_images, grid_path, n_digits=10, n_samples=n_samples_per_digit)
    
    # ==================== 保存单独的图像（可选） ====================
    print("\n保存单独的图像...")
    for digit in range(10):
        for sample_idx, img in enumerate(all_images[digit]):
            img_path = os.path.join(output_dir, f"digit{digit}_sample{sample_idx + 1}.png")
            img.save(img_path)
    print(f"✓ 保存了 {10 * n_samples_per_digit} 张单独的图像")
    
    # ==================== 生成统计报告 ====================
    print("\n" + "="*70)
    print("生成统计")
    print("="*70)
    
    print(f"\n{'数字':<6} {'预测':<6} {'置信度':<10} {'状态'}")
    print("-"*40)
    
    for r in results:
        status = "✓ 正确" if r['correct'] else "✗ 错误"
        print(f"{r['digit']:<6} {r['predicted']:<6} {r['confidence']:<10.3f} {status}")
    
    # 计算准确率
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = 100.0 * correct_count / len(results)
    
    print("-"*40)
    print(f"总体准确率: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    
    # ==================== 保存详细报告 ====================
    report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MNIST 分类器引导生成报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"引导强度: {guidance_scale}\n")
        f.write(f"每个数字样本数: {n_samples_per_digit}\n")
        f.write(f"总图像数: {10 * n_samples_per_digit}\n")
        f.write(f"分类器: {classifier_path}\n")
        f.write(f"温度: {temperature}\n")
        f.write(f"梯度裁剪: {grad_clip}\n")
        f.write(f"自适应引导: {adaptive_guidance}\n\n")
        
        f.write("生成结果:\n")
        f.write("-"*70 + "\n")
        for r in results:
            status = "正确" if r['correct'] else "错误"
            f.write(f"数字 {r['digit']}: 预测={r['predicted']}, "
                   f"置信度={r['confidence']:.3f}, 状态={status}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"总体准确率: {correct_count}/{len(results)} ({accuracy:.1f}%)\n")
    
    print(f"\n✓ 详细报告已保存: {report_path}")
    
    # ==================== 完成 ====================
    print("\n" + "="*70)
    print("生成完成!")
    print("="*70)
    print(f"输出目录: {output_dir}/")
    print(f"文件:")
    print(f"  - all_digits_grid_{timestamp}.png (网格图)")
    print(f"  - digit*_sample*.png ({10 * n_samples_per_digit} 张单独图像)")
    print(f"  - report_{timestamp}.txt (详细报告)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()