import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import clip
import numpy as np

def test_clip_mnist():
    """独立测试CLIP对MNIST的理解"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载CLIP
    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # CLIP归一化
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    # 加载MNIST
    print("Loading MNIST dataset...")
    mnist = datasets.MNIST('./datasets', train=False, download=True)
    
    def mnist_to_clip(img_tensor):
        """转换MNIST到CLIP格式"""
        # 从[-1, 1]转换到[0, 1]
        images = (img_tensor + 1) / 2
        # 灰度转RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        # 上采样到224x224
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        # CLIP标准化
        return normalize(images)
    
    print("\n" + "="*60)
    print("Testing CLIP's understanding of MNIST digits")
    print("="*60)
    
    # 测试多个样本
    n_samples_per_digit = 1000
    accuracy = 0
    total = 0
    
    for digit in range(10):
        print(f"\n--- Digit {digit} ---")
        # 找多个该数字的样本
        indices = (mnist.targets == digit).nonzero()[:n_samples_per_digit].squeeze()
        
        digit_correct = 0
        
        for idx in indices:
            img = mnist[idx.item()][0]
            
            # 转换为tensor
            img_tensor = transforms.ToTensor()(img).unsqueeze(0) * 2 - 1
            img_tensor = img_tensor.to(device)
            
            # 测试CLIP识别
            clip_input = mnist_to_clip(img_tensor)
            with torch.no_grad():
                image_features = clip_model.encode_image(clip_input)
            
            # 与不同文本比较
            texts = [f"the number {i}" for i in range(10)]
            text_inputs = clip.tokenize(texts).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_inputs)
            
            # 计算相似度
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()
            
            predicted = probs.argmax()
            if predicted == digit:
                digit_correct += 1
            
            total += 1
        
        accuracy += digit_correct
        digit_accuracy = digit_correct / n_samples_per_digit * 100
        print(f"Accuracy: {digit_correct}/{n_samples_per_digit} ({digit_accuracy:.1f}%)")
    
    overall_accuracy = accuracy / total * 100
    print("\n" + "="*60)
    print(f"Overall Accuracy: {accuracy}/{total} ({overall_accuracy:.1f}%)")
    print("="*60)
    
    # 测试不同的文本描述
    print("\n\nTesting different text prompts for digit 1:")
    prompts = [
        "the number 1",
        "digit one",
        "a handwritten digit one",
        "the number one written in black ink",
        "a vertical line representing number 1",
        "number one",
        "1",
    ]
    
    # 取一个数字1的样本
    idx = (mnist.targets == 1).nonzero()[0].item()
    img = mnist[idx][0]
    img_tensor = transforms.ToTensor()(img).unsqueeze(0) * 2 - 1
    img_tensor = img_tensor.to(device)
    clip_input = mnist_to_clip(img_tensor)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(clip_input)
    
    print("\nSimilarity scores for different prompts:")
    for prompt in prompts:
        text_input = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_input)
        
        similarity = (image_features @ text_features.T).item()
        print(f"  '{prompt}': {similarity:.4f}")


if __name__ == "__main__":
    test_clip_mnist()