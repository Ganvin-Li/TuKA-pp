#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过曝光成像物理模型处理器 - 增强版
基于公式: I(x) = CRF(clip(G*T*L(x) + N_shot(x) + N_read(x), 0, S_sat))

处理R2R数据集，生成与原目录同级的R2R_{处理方式}目录
使用更精确的物理成像模型和高级过曝光效果
"""

import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json
from datetime import datetime


class EnhancedOverexposureImagingModel:
    """
    增强版过曝光成像物理模型
    基于 overexposure_noise_model.py 的实现
    """
    
    def __init__(self,
                 exposure: float = 2.0,           # 曝光倍数 (相对正常曝光)
                 gain: float = 1.0,              # 传感器增益 (ISO)
                 full_well: float = 1.0,         # 传感器饱和值（线性域最大值）
                 sigma_read: float = 0.01,       # 读出噪声标准差
                 gamma: float = 2.2,             # 伽马校正值
                 overexposure_type: str = "moderate",  # 过曝光类型
                 wavelength_effect: bool = True,  # 是否考虑波长相关效应
                 bloom_strength: float = 0.3,    # 光晕强度
                 highlight_rolloff: bool = True,  # 是否应用高光溢出
                 random_params: bool = False):    # 是否使用随机参数
        
        self.exposure = exposure
        self.gain = gain
        self.full_well = full_well
        self.sigma_read = sigma_read
        self.gamma = gamma
        self.overexposure_type = overexposure_type
        self.wavelength_effect = wavelength_effect
        self.bloom_strength = bloom_strength
        self.highlight_rolloff = highlight_rolloff
        self.random_params = random_params
        
        # 预计算CRF相关参数
        self.gamma_inv = 1.0 / gamma
        
        # 不同过曝光类型的参数配置
        self.overexposure_params = {
            'normal': {
                'exposure_multiplier': 1.0,
                'noise_multiplier': 1.0,
                'saturation_point': 1.0,
                'color_shift': np.array([1.0, 1.0, 1.0])
            },
            'slight': {
                'exposure_multiplier': 1.2,
                'noise_multiplier': 1.1,
                'saturation_point': 0.95,
                'color_shift': np.array([1.0, 0.98, 0.96])
            },
            'light_moderate': {  # 新增：介于slight和moderate之间
                'exposure_multiplier': 1.35,
                'noise_multiplier': 1.2,
                'saturation_point': 0.925,
                'color_shift': np.array([1.0, 0.97, 0.94])
            },
            'moderate': {
                'exposure_multiplier': 1.5,
                'noise_multiplier': 1.3,
                'saturation_point': 0.9,
                'color_shift': np.array([1.0, 0.96, 0.92])
            },
            'severe': {
                'exposure_multiplier': 2.0,
                'noise_multiplier': 1.5,
                'saturation_point': 0.8,
                'color_shift': np.array([1.0, 0.94, 0.88])
            },
            'extreme': {
                'exposure_multiplier': 3.0,
                'noise_multiplier': 2.0,
                'saturation_point': 0.7,
                'color_shift': np.array([1.0, 0.92, 0.84])
            }
        }
        
        self.current_params = self.overexposure_params.get(
            overexposure_type, self.overexposure_params['moderate']
        )
        
        # 计算有效的曝光值
        self.effective_exposure = self.exposure * self.current_params['exposure_multiplier']
        
        # 波长相关效应（不同颜色通道的响应差异）
        if wavelength_effect:
            # RGB对应的波长响应差异
            # 红色通道通常更容易过曝
            self.wavelength_response = np.array([1.1, 1.0, 0.9])
        else:
            self.wavelength_response = np.array([1.0, 1.0, 1.0])
    
    def generate_random_params(self):
        """生成随机参数"""
        if self.random_params:
            # 随机曝光倍数
            self.exposure = np.random.uniform(1.5, 6.0)
            
            # 随机增益
            self.gain = np.random.uniform(1.0, 3.0)
            
            # 随机饱和值
            self.full_well = np.random.uniform(0.7, 1.0)
            
            # 随机读出噪声
            self.sigma_read = np.random.uniform(0.005, 0.03)
            
            # 随机gamma值
            self.gamma = np.random.uniform(1.8, 2.6)
            self.gamma_inv = 1.0 / self.gamma
            
            # 随机过曝光类型
            overexposure_types = ['slight', 'light_moderate', 'moderate', 'severe', 'extreme']
            self.overexposure_type = np.random.choice(overexposure_types)
            self.current_params = self.overexposure_params[self.overexposure_type]
            
            # 重新计算有效曝光值
            self.effective_exposure = self.exposure * self.current_params['exposure_multiplier']
            
            # 随机光晕强度
            self.bloom_strength = np.random.uniform(0.1, 0.6)
    
    def _add_shot_noise(self, signal: np.ndarray) -> np.ndarray:
        """添加散粒噪声 (shot noise)"""
        # 散粒噪声遵循泊松分布，标准差 = sqrt(信号强度)
        # 为了数值稳定性，限制最小噪声水平
        noise_variance = np.maximum(signal, 1e-6)
        
        # 使用高斯近似泊松噪声
        shot_noise = np.zeros_like(signal)
        for i in range(3):  # RGB通道
            std_dev = np.sqrt(noise_variance[:, :, i] * 0.05)  # 调整噪声强度
            shot_noise[:, :, i] = np.random.normal(0, std_dev)
        
        return shot_noise
    
    def _apply_highlight_rolloff(self, image: np.ndarray) -> np.ndarray:
        """应用高光溢出效果 - 模拟过曝光的软削波"""
        # 使用S曲线来模拟胶片/数字传感器的高光溢出
        
        # 计算亮度
        luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        # 对高亮区域应用溢出效果
        bright_threshold = 0.7
        bright_mask = luminance > bright_threshold
        
        if np.any(bright_mask):
            # 软削波函数
            k = (self.effective_exposure - 1.0) * 0.5
            if k > 0:
                rolloff_factor = 1.0 / (1.0 + k * np.maximum(luminance - bright_threshold, 0))
                
                # 只对高亮区域应用
                for i in range(3):
                    image[:, :, i] = np.where(
                        bright_mask,
                        image[:, :, i] * rolloff_factor,
                        image[:, :, i]
                    )
        
        return image
    
    def _apply_bloom_effect(self, image: np.ndarray, bloom_strength: float = 0.3) -> np.ndarray:
        """应用光晕效果 (bloom effect) - 高光扩散"""
        # 提取高亮区域
        bright_threshold = 0.8
        bright_areas = np.maximum(image - bright_threshold, 0) / (1.0 - bright_threshold + 1e-8)
        
        # 对高亮区域应用多层高斯模糊创建光晕
        bloom = np.zeros_like(bright_areas)
        kernel_sizes = [7, 15, 31]
        weights = [0.5, 0.3, 0.2]
        
        for kernel_size, weight in zip(kernel_sizes, weights):
            blurred = cv2.GaussianBlur(bright_areas, (kernel_size, kernel_size), 0)
            bloom += blurred * weight
        
        # 将光晕效果叠加到原图像
        result = image + bloom * bloom_strength
        return np.clip(result, 0, 1)
    
    def simulate_overexposure_imaging(self, input_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        模拟过曝光成像过程 - 增强版实现
        基于公式: I(x) = CRF(clip(G*T*L(x) + N_shot(x) + N_read(x), 0, S_sat))
        
        Args:
            input_image: 输入的正常曝光sRGB图像
        
        Returns:
            过曝光图像, 中间结果字典
        """
        # 如果使用随机参数，每次都重新生成
        if self.random_params:
            self.generate_random_params()
        
        # 确保输入是float格式 [0,1]
        if input_image.dtype == np.uint8:
            img_float = input_image.astype(np.float32) / 255.0
            input_uint8 = True
        else:
            img_float = input_image.astype(np.float32)
            input_uint8 = False
        
        h, w, c = img_float.shape
        
        # 处理RGBA图像：只对RGB通道应用过曝光，保持Alpha通道不变
        has_alpha = (c == 4)
        if has_alpha:
            rgb_image = img_float[:, :, :3]  # 提取RGB通道
            alpha_channel = img_float[:, :, 3:4]  # 保存Alpha通道
        else:
            rgb_image = img_float
        
        # Step 1: 逆CRF - 将sRGB转换到线性域 L(x)
        # L(x) ≈ (I(x))^γ
        linear_radiance = np.power(rgb_image + 1e-8, self.gamma)
        
        # Step 2: 应用曝光时间和增益 S = G*T*L(x)
        # 考虑波长响应差异
        exposed_signal = np.zeros_like(linear_radiance)
        for i in range(3):  # RGB三个通道
            channel_exposure = self.effective_exposure * self.gain * self.wavelength_response[i]
            exposed_signal[:, :, i] = channel_exposure * linear_radiance[:, :, i]
        
        # Step 3: 添加散粒噪声 N_shot(x) - 泊松噪声
        # 散粒噪声强度与信号强度成正比: σ_shot = sqrt(S)
        shot_noise = self._add_shot_noise(exposed_signal)
        noisy_signal = exposed_signal + shot_noise
        
        # Step 4: 添加读出噪声 N_read(x) - 高斯噪声
        noise_multiplier = self.current_params['noise_multiplier']
        read_noise = np.random.normal(0, self.sigma_read * noise_multiplier, (h, w, 3))
        noisy_signal += read_noise
        
        # Step 5: 传感器饱和截断 - clip(S_noisy, 0, S_sat)
        saturation_point = self.full_well * self.current_params['saturation_point']
        saturated_signal = np.clip(noisy_signal, 0, saturation_point)
        
        # Step 6: 应用CRF和ISP处理
        # CRF: I(x) = (S_sat(x) / S_sat)^(1/γ)
        output_image = np.power(saturated_signal / saturation_point, self.gamma_inv)
        
        # Step 7: 应用颜色偏移（模拟不同通道的饱和特性）
        color_shift = self.current_params['color_shift']
        for i in range(3):
            output_image[:, :, i] *= color_shift[i]
        
        # Step 8: 应用高光溢出效果（可选）
        if self.highlight_rolloff:
            output_image = self._apply_highlight_rolloff(output_image)
        
        # Step 9: 应用光晕效果（可选）
        if self.bloom_strength > 0:
            output_image = self._apply_bloom_effect(output_image, self.bloom_strength)
        
        # 确保输出在有效范围内
        output_image = np.clip(output_image, 0, 1)
        
        # 重新组合RGBA图像
        if has_alpha:
            output_image = np.concatenate([output_image, alpha_channel], axis=2)
        
        # 保存中间结果
        intermediate_results = {
            'scene_radiance': linear_radiance,
            'exposed_signal': exposed_signal,
            'noisy_signal': noisy_signal,
            'saturated_signal': saturated_signal
        }
        
        # 转换回输入格式
        if input_uint8:
            output_image = (output_image * 255).astype(np.uint8)
            # 转换中间结果用于可视化
            for key in intermediate_results:
                result = intermediate_results[key]
                if key == 'scene_radiance':
                    # 场景辐照度已经在0-1范围内
                    intermediate_results[key] = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    # 其他信号需要根据饱和值归一化
                    normalized = np.clip(result / saturation_point, 0.0, 1.0)
                    intermediate_results[key] = (normalized * 255).astype(np.uint8)
        
        return output_image, intermediate_results


class R2ROverexposureProcessor:
    """R2R数据集过曝光成像批处理器 - 增强版"""
    
    def __init__(self, r2r_root: str = "./R2R"):
        self.r2r_root = Path(r2r_root)
        self.images_path = self.r2r_root / "images"
        
        # 增强版预设参数配置
        self.presets = {
            "normal": {
                "exposure": 1.0,
                "gain": 1.0,
                "full_well": 1.0,
                "sigma_read": 0.005,
                "gamma": 2.2,
                "overexposure_type": "normal",
                "wavelength_effect": False,
                "bloom_strength": 0.0,
                "highlight_rolloff": False,
                "random_params": False,
                "description": "正常曝光 - 无过曝效果"
            },
            "slight_overexpose": {
                "exposure": 1.5,
                "gain": 1.2,
                "full_well": 0.95,
                "sigma_read": 0.01,
                "gamma": 2.2,
                "overexposure_type": "slight",
                "wavelength_effect": True,
                "bloom_strength": 0.1,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "轻微过曝 - 保留大部分细节"
            },
            "light_moderate_overexpose": {  # 新增：介于slight和moderate之间
                "exposure": 2.0,
                "gain": 1.35,
                "full_well": 0.925,
                "sigma_read": 0.0125,
                "gamma": 2.1,
                "overexposure_type": "light_moderate",
                "wavelength_effect": True,
                "bloom_strength": 0.2,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "轻中度过曝 - 介于轻微和中等过曝之间，适度的高光溢出"
            },
            "moderate_overexpose": {
                "exposure": 2.5,
                "gain": 1.5,
                "full_well": 0.9,
                "sigma_read": 0.015,
                "gamma": 2.0,
                "overexposure_type": "moderate",
                "wavelength_effect": True,
                "bloom_strength": 0.3,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "中度过曝 - 明显曝光过度"
            },
            "heavy_overexpose": {
                "exposure": 4.0,
                "gain": 2.0,
                "full_well": 0.8,
                "sigma_read": 0.025,
                "gamma": 1.8,
                "overexposure_type": "severe",
                "wavelength_effect": True,
                "bloom_strength": 0.5,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "重度过曝 - 大面积白色区域"
            },
            "extreme_overexpose": {
                "exposure": 8.0,
                "gain": 3.0,
                "full_well": 0.7,
                "sigma_read": 0.04,
                "gamma": 1.5,
                "overexposure_type": "extreme",
                "wavelength_effect": True,
                "bloom_strength": 0.7,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "极度过曝 - 严重信息丢失"
            },
            "backlight_strong": {
                "exposure": 3.0,
                "gain": 1.8,
                "full_well": 0.85,
                "sigma_read": 0.02,
                "gamma": 2.0,
                "overexposure_type": "moderate",
                "wavelength_effect": True,
                "bloom_strength": 0.4,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "强逆光 - 模拟逆光拍摄"
            },
            "bright_sunny": {
                "exposure": 2.2,
                "gain": 1.4,
                "full_well": 0.88,
                "sigma_read": 0.012,
                "gamma": 2.3,
                "overexposure_type": "moderate",
                "wavelength_effect": True,
                "bloom_strength": 0.25,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "强阳光 - 户外强光环境"
            },
            "flash_overpower": {
                "exposure": 2.8,
                "gain": 2.5,
                "full_well": 0.82,
                "sigma_read": 0.03,
                "gamma": 2.0,
                "overexposure_type": "severe",
                "wavelength_effect": True,
                "bloom_strength": 0.6,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "闪光过强 - 闪光灯过曝"
            },
            "high_iso": {
                "exposure": 1.8,
                "gain": 2.5,
                "full_well": 0.92,
                "sigma_read": 0.03,
                "gamma": 2.2,
                "overexposure_type": "slight",
                "wavelength_effect": True,
                "bloom_strength": 0.15,
                "highlight_rolloff": True,
                "random_params": False,
                "description": "高ISO - 高感光度拍摄"
            },
            "random_overexpose": {
                "exposure": 3.0,  # 会被随机覆盖
                "gain": 2.0,      # 会被随机覆盖
                "full_well": 0.8,  # 会被随机覆盖
                "sigma_read": 0.02,  # 会被随机覆盖
                "gamma": 2.2,     # 会被随机覆盖
                "overexposure_type": "moderate",  # 会被随机覆盖
                "wavelength_effect": True,
                "bloom_strength": 0.3,  # 会被随机覆盖
                "highlight_rolloff": True,
                "random_params": True,
                "description": "随机过曝参数 - 每张图片使用不同参数"
            }
        }
    
    def check_r2r_structure(self) -> bool:
        """检查R2R数据集结构"""
        if not self.r2r_root.exists():
            print(f"❌ R2R根目录不存在: {self.r2r_root.absolute()}")
            return False
        
        if not self.images_path.exists():
            print(f"❌ images目录不存在: {self.images_path.absolute()}")
            return False
        
        return True
    
    def get_scene_folders(self) -> List[Path]:
        """获取所有场景文件夹"""
        scene_folders = []
        if self.images_path.exists():
            for item in self.images_path.iterdir():
                if item.is_dir():
                    rgb_dir = item / "rgb"
                    if rgb_dir.exists() and any(rgb_dir.glob("*.jpg")):
                        scene_folders.append(item)
        
        return sorted(scene_folders)
    
    def get_scene_images(self, scene_path: Path) -> List[Path]:
        """获取场景中的所有图片"""
        rgb_dir = scene_path / "rgb"
        image_files = []
        
        if rgb_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(rgb_dir.glob(ext))
        
        return sorted(image_files)
    
    def create_output_structure(self, output_root: Path, scene_name: str) -> Path:
        """创建输出目录结构"""
        output_scene_path = output_root / "images" / scene_name / "rgb"
        output_scene_path.mkdir(parents=True, exist_ok=True)
        return output_scene_path
    
    def process_single_image(self,
                           input_path: Path,
                           output_path: Path,
                           overexposure_model: EnhancedOverexposureImagingModel,
                           save_intermediates: bool = False) -> bool:
        """处理单张图片"""
        try:
            # 读取图像
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"❌ 无法读取图像: {input_path}")
                return False
            
            # BGR转RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用过曝光成像模拟
            overexposed_img, intermediate_results = overexposure_model.simulate_overexposure_imaging(image_rgb)
            
            # 保存过曝光图像 (RGB转BGR)
            if overexposed_img.dtype == np.uint8:
                overexposed_bgr = cv2.cvtColor(overexposed_img, cv2.COLOR_RGB2BGR)
            else:
                overexposed_bgr = cv2.cvtColor((overexposed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(output_path), overexposed_bgr)
            
            # 保存中间结果
            if save_intermediates:
                base_name = output_path.stem
                output_dir = output_path.parent
                
                # 保存场景辐照度
                if 'scene_radiance' in intermediate_results:
                    radiance_path = output_dir / f"{base_name}_radiance.png"
                    radiance_bgr = cv2.cvtColor(intermediate_results['scene_radiance'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(radiance_path), radiance_bgr)
                
                # 保存曝光信号
                if 'exposed_signal' in intermediate_results:
                    exposed_path = output_dir / f"{base_name}_exposed.png"
                    exposed_bgr = cv2.cvtColor(intermediate_results['exposed_signal'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(exposed_path), exposed_bgr)
                
                # 保存饱和信号
                if 'saturated_signal' in intermediate_results:
                    saturated_path = output_dir / f"{base_name}_saturated.png"
                    saturated_bgr = cv2.cvtColor(intermediate_results['saturated_signal'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(saturated_path), saturated_bgr)
            
            return True
            
        except Exception as e:
            print(f"❌ 处理失败 {input_path.name}: {e}")
            return False
    
    def save_processing_info(self, output_root: Path, preset: str, params: Dict[str, Any]):
        """保存处理信息"""
        info = {
            "processing_time": datetime.now().isoformat(),
            "preset": preset,
            "parameters": params,
            "model": "Enhanced Overexposure Imaging Physical Model",
            "formula": "I(x) = CRF(clip(G*T*L(x) + N_shot(x) + N_read(x), 0, S_sat))",
            "features": [
                "Physical-based imaging simulation",
                "Wavelength-dependent response",
                "Shot noise and read noise",
                "Highlight rolloff",
                "Bloom effects",
                "Color channel saturation modeling"
            ]
        }
        
        info_path = output_root / "processing_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def process_dataset(self,
                       preset: str = "moderate_overexpose",
                       max_scenes: Optional[int] = None,
                       max_images_per_scene: Optional[int] = None,
                       save_intermediates: bool = False,
                       skip_existing: bool = False) -> None:
        """批量处理整个数据集"""
        
        # 检查数据集结构
        if not self.check_r2r_structure():
            return
        
        # 验证预设
        if preset not in self.presets:
            print(f"❌ 未知预设: {preset}")
            print(f"可用预设: {list(self.presets.keys())}")
            return
        
        # 获取预设参数
        preset_params = self.presets[preset]
        
        # 创建增强版过曝光成像模型
        overexposure_model = EnhancedOverexposureImagingModel(
            exposure=preset_params["exposure"],
            gain=preset_params["gain"],
            full_well=preset_params["full_well"],
            sigma_read=preset_params["sigma_read"],
            gamma=preset_params["gamma"],
            overexposure_type=preset_params["overexposure_type"],
            wavelength_effect=preset_params["wavelength_effect"],
            bloom_strength=preset_params["bloom_strength"],
            highlight_rolloff=preset_params["highlight_rolloff"],
            random_params=preset_params["random_params"]
        )
        
        # 设置输出目录
        output_root = self.r2r_root.parent / f"R2R_{preset}"
        
        # 获取所有场景
        scene_folders = self.get_scene_folders()
        if max_scenes:
            scene_folders = scene_folders[:max_scenes]
        
        if not scene_folders:
            print("❌ 没有找到任何场景文件夹")
            return
        
        # 打印处理信息
        print(f"\n☀️ 增强版过曝光成像物理模型处理")
        print(f"📂 源目录: {self.r2r_root.absolute()}")
        print(f"📂 输出目录: {output_root.absolute()}")
        print(f"🎨 预设: {preset} - {preset_params['description']}")
        print(f"📊 参数: T={preset_params['exposure']:.1f}, G={preset_params['gain']:.1f}")
        print(f"🔬 饱和值: S_sat={preset_params['full_well']:.2f}, 读噪声={preset_params['sigma_read']:.3f}")
        print(f"💡 伽马: γ={preset_params['gamma']:.1f}")
        print(f"🌈 波长效应: {'✓' if preset_params['wavelength_effect'] else '✗'}")
        print(f"✨ 光晕效果: {preset_params['bloom_strength']:.1f}")
        print(f"🎯 高光溢出: {'✓' if preset_params['highlight_rolloff'] else '✗'}")
        print(f"🎲 随机参数: {'✓' if preset_params['random_params'] else '✗'}")
        print(f"📋 场景数量: {len(scene_folders)}")
        print("=" * 80)
        
        # 统计信息
        total_processed = 0
        total_skipped = 0
        total_failed = 0
        
        # 处理每个场景
        for scene_folder in tqdm(scene_folders, desc="🗺️ 处理场景", position=0):
            scene_name = scene_folder.name
            
            # 获取场景图像
            scene_images = self.get_scene_images(scene_folder)
            if max_images_per_scene:
                scene_images = scene_images[:max_images_per_scene]
            
            if not scene_images:
                print(f"⚠️ 场景 {scene_name} 中没有图像")
                continue
            
            # 创建输出目录
            output_scene_dir = self.create_output_structure(output_root, scene_name)
            
            # 场景统计
            scene_processed = 0
            scene_skipped = 0
            scene_failed = 0
            
            # 处理场景中的每张图像
            for img_path in tqdm(scene_images, 
                               desc=f"  📸 {scene_name}", 
                               position=1, 
                               leave=False):
                
                output_path = output_scene_dir / img_path.name
                
                # 检查是否跳过已存在文件
                if skip_existing and output_path.exists():
                    scene_skipped += 1
                    continue
                
                # 处理图像
                if self.process_single_image(
                    img_path, output_path, overexposure_model, save_intermediates
                ):
                    scene_processed += 1
                else:
                    scene_failed += 1
            
            # 更新总统计
            total_processed += scene_processed
            total_skipped += scene_skipped
            total_failed += scene_failed
            
            # 打印场景结果
            if scene_processed > 0 or scene_failed > 0:
                status_msg = f"✅ {scene_name}: 处理 {scene_processed}"
                if scene_skipped > 0:
                    status_msg += f", 跳过 {scene_skipped}"
                if scene_failed > 0:
                    status_msg += f", 失败 {scene_failed}"
                print(status_msg)
        
        # 保存处理信息
        self.save_processing_info(output_root, preset, preset_params)
        
        # 打印最终结果
        print("=" * 80)
        print(f"🎉 增强版过曝光成像处理完成!")
        print(f"✅ 成功处理: {total_processed} 张图片")
        if total_skipped > 0:
            print(f"⭐ 跳过已存在: {total_skipped} 张图片")
        if total_failed > 0:
            print(f"❌ 处理失败: {total_failed} 张图片")
        print(f"📁 输出位置: {output_root.absolute()}")
        print(f"📄 处理信息: {output_root / 'processing_info.json'}")
    
    def list_scenes(self) -> None:
        """列出所有场景"""
        if not self.check_r2r_structure():
            return
        
        scene_folders = self.get_scene_folders()
        
        print(f"\n📋 R2R数据集场景列表 (总计 {len(scene_folders)} 个场景)")
        print("-" * 60)
        
        for i, scene_folder in enumerate(scene_folders, 1):
            scene_images = self.get_scene_images(scene_folder)
            print(f"{i:3d}. {scene_folder.name} ({len(scene_images)} 张图片)")
    
    def show_presets(self) -> None:
        """显示所有预设"""
        print("\n☀️ 可用的增强版过曝光成像预设:")
        print("=" * 80)
        
        for name, params in self.presets.items():
            print(f"📋 {name}")
            print(f"   描述: {params['description']}")
            print(f"   曝光倍数(T): {params['exposure']:.1f}")
            print(f"   增益(G): {params['gain']:.1f}")
            print(f"   饱和值(S_sat): {params['full_well']:.3f}")
            print(f"   读出噪声: {params['sigma_read']:.4f}")
            print(f"   Gamma(γ): {params['gamma']:.1f}")
            print(f"   过曝类型: {params['overexposure_type']}")
            print(f"   波长效应: {'✓' if params['wavelength_effect'] else '✗'}")
            print(f"   光晕强度: {params['bloom_strength']:.1f}")
            print(f"   高光溢出: {'✓' if params['highlight_rolloff'] else '✗'}")
            print(f"   随机参数: {'✓' if params['random_params'] else '✗'}")
            print()


def create_enhanced_overexposure_demo():
    """创建增强版过曝光成像演示"""
    # 创建测试图像 (正常曝光的场景)
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # 创建一个中等亮度的场景
    # 天空 (中等蓝色)
    cv2.rectangle(test_image, (0, 0), (400, 100), (70, 130, 180), -1)
    # 建筑物 (中等灰色)
    cv2.rectangle(test_image, (50, 100), (150, 200), (120, 120, 120), -1)
    cv2.rectangle(test_image, (200, 90), (350, 210), (100, 100, 140), -1)
    # 树木 (中等绿色)
    cv2.rectangle(test_image, (320, 120), (330, 200), (101, 67, 33), -1)  # 树干
    cv2.circle(test_image, (325, 115), 20, (34, 139, 34), -1)  # 树冠
    # 前景地面 (中等亮度)
    cv2.rectangle(test_image, (0, 200), (400, 300), (60, 120, 60), -1)
    
    # 添加一些亮点（容易过曝的区域）
    cv2.circle(test_image, (80, 50), 15, (200, 200, 150), -1)  # 光源
    cv2.rectangle(test_image, (70, 130), (80, 140), (180, 180, 180), -1)  # 窗户
    cv2.rectangle(test_image, (220, 120), (230, 130), (180, 180, 180), -1)  # 窗户
    
    # 预设效果演示 - 包含新的预设
    presets_demo = ["slight_overexpose", "light_moderate_overexpose", "moderate_overexpose", 
                   "heavy_overexpose", "backlight_strong", "bright_sunny"]
    processor = R2ROverexposureProcessor()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, preset_name in enumerate(presets_demo):
        params = processor.presets[preset_name]
        
        # 创建增强版过曝光成像模型
        model = EnhancedOverexposureImagingModel(
            exposure=params["exposure"],
            gain=params["gain"],
            full_well=params["full_well"],
            sigma_read=params["sigma_read"],
            gamma=params["gamma"],
            overexposure_type=params["overexposure_type"],
            wavelength_effect=params["wavelength_effect"],
            bloom_strength=params["bloom_strength"],
            highlight_rolloff=params["highlight_rolloff"],
            random_params=False
        )
        
        # 应用过曝光效果
        overexposed_img, _ = model.simulate_overexposure_imaging(test_image)
        
        # 显示结果
        axes[i].imshow(overexposed_img)
        title = f'{preset_name}\n'
        title += f'T={params["exposure"]:.1f}, G={params["gain"]:.1f}\n'
        title += f'S_sat={params["full_well"]:.3f}, γ={params["gamma"]:.1f}\n'
        title += f'Bloom={params["bloom_strength"]:.1f}'
        if params["wavelength_effect"]:
            title += ', WL✓'
        if params["highlight_rolloff"]:
            title += ', HR✓'
        
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('增强版过曝光成像物理模型 - 效果预览\nI(x) = CRF(clip(G*T*L(x) + N_shot + N_read, 0, S_sat))\n包含新增的轻中度过曝预设', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('enhanced_overexposure_imaging_presets_with_custom.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 增强版过曝光成像预设效果演示图已保存: enhanced_overexposure_imaging_presets_with_custom.png")


def create_comparison_demo():
    """创建原版vs增强版效果对比演示"""
    # 创建测试图像
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # 创建具有不同亮度区域的测试场景
    # 暗区域
    cv2.rectangle(test_image, (0, 0), (100, 200), (30, 30, 30), -1)
    # 中等亮度区域
    cv2.rectangle(test_image, (100, 0), (200, 200), (120, 120, 120), -1)
    # 高亮区域
    cv2.rectangle(test_image, (200, 0), (300, 200), (200, 200, 200), -1)
    
    # 添加一些细节
    cv2.circle(test_image, (50, 100), 20, (80, 80, 80), -1)
    cv2.circle(test_image, (150, 100), 20, (180, 180, 180), -1)
    cv2.circle(test_image, (250, 100), 20, (240, 240, 240), -1)
    
    # 比较不同预设的效果，重点展示新预设
    processor = R2ROverexposureProcessor()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 显示原图
    axes[0, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原图', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('原图', fontsize=12)
    axes[1, 0].axis('off')
    
    # 测试不同预设的效果
    test_presets = [
        'slight_overexpose',
        'light_moderate_overexpose',  # 新预设
        'moderate_overexpose'
    ]
    
    for i, preset_name in enumerate(test_presets):
        params = processor.presets[preset_name]
        
        # 上排：正常设置
        model1 = EnhancedOverexposureImagingModel(
            exposure=params["exposure"],
            gain=params["gain"],
            full_well=params["full_well"],
            sigma_read=params["sigma_read"],
            gamma=params["gamma"],
            overexposure_type=params["overexposure_type"],
            wavelength_effect=params["wavelength_effect"],
            bloom_strength=params["bloom_strength"],
            highlight_rolloff=params["highlight_rolloff"]
        )
        
        result1, _ = model1.simulate_overexposure_imaging(test_image)
        axes[0, i+1].imshow(result1)
        axes[0, i+1].set_title(f"{preset_name}\nT={params['exposure']:.1f}", fontsize=10)
        axes[0, i+1].axis('off')
        
        # 下排：增强设置
        model2 = EnhancedOverexposureImagingModel(
            exposure=params["exposure"] * 1.3,
            gain=params["gain"] * 1.2,
            full_well=params["full_well"] * 0.9,
            sigma_read=params["sigma_read"] * 1.5,
            gamma=params["gamma"],
            overexposure_type=params["overexposure_type"],
            wavelength_effect=params["wavelength_effect"],
            bloom_strength=params["bloom_strength"] * 1.5,
            highlight_rolloff=params["highlight_rolloff"]
        )
        
        result2, _ = model2.simulate_overexposure_imaging(test_image)
        axes[1, i+1].imshow(result2)
        axes[1, i+1].set_title(f"{preset_name} (增强)\nT={params['exposure']*1.3:.1f}", fontsize=10)
        axes[1, i+1].axis('off')
    
    plt.suptitle('过曝光预设对比 - 新增轻中度过曝预设', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('overexposure_presets_comparison_with_custom.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 过曝光预设对比图已保存: overexposure_presets_comparison_with_custom.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强版过曝光成像物理模型 - R2R数据集批处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
☀️ 基于物理成像模型: I(x) = CRF(clip(G*T*L(x) + N_shot(x) + N_read(x), 0, S_sat))
   增强特性:
   - 🌈 波长相关效应 (不同RGB通道的响应差异)
   - ✨ 光晕效果 (bloom effect) - 高光扩散
   - 🎯 高光溢出 (highlight rolloff) - 软削波
   - 🔬 物理噪声模型 (散粒噪声 + 读出噪声)
   - 🎨 颜色通道饱和建模
   - 🎲 随机参数生成

💡 使用示例:
   # 处理所有场景 - 新的轻中度过曝效果
   python script.py --process-all --preset light_moderate_overexpose
   
   # 中度过曝 + 保存中间结果
   python script.py --process-all --preset moderate_overexpose --save-intermediates
   
   # 轻微过曝效果
   python script.py --process-all --preset slight_overexpose
   
   # 随机参数效果 (每张图片不同参数)
   python script.py --process-all --preset random_overexpose
   
   # 限制处理数量 (测试用)
   python script.py --process-all --preset light_moderate_overexpose --max-scenes 2 --max-images 5
   
   # 查看所有预设
   python script.py --show-presets
   
   # 创建效果演示
   python script.py --demo
   
   # 创建对比演示
   python script.py --comparison
        """
    )
    
    # 主要操作参数
    parser.add_argument("--process-all", action="store_true", 
                       help="处理所有场景")
    parser.add_argument("--preset", "-p", type=str, default="light_moderate_overexpose",
                       choices=["normal", "slight_overexpose", "light_moderate_overexpose", "moderate_overexpose", 
                               "heavy_overexpose", "extreme_overexpose", "backlight_strong", "bright_sunny",
                               "flash_overpower", "high_iso", "random_overexpose"],
                       help="增强版过曝光成像预设")
    
    # 限制参数
    parser.add_argument("--max-scenes", type=int, 
                       help="最大处理场景数 (用于测试)")
    parser.add_argument("--max-images", type=int,
                       help="每个场景最大图片数 (用于测试)")
    
    # 输出控制
    parser.add_argument("--save-intermediates", action="store_true",
                       help="保存中间结果 (辐照度图、曝光信号图、饱和信号图)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="跳过已存在的文件")
    
    # 信息查看
    parser.add_argument("--list-scenes", action="store_true",
                       help="列出所有场景")
    parser.add_argument("--show-presets", action="store_true",
                       help="显示所有预设参数")
    parser.add_argument("--demo", action="store_true",
                       help="创建效果演示图")
    parser.add_argument("--comparison", action="store_true",
                       help="创建效果对比图")
    
    # R2R路径
    parser.add_argument("--r2r-path", type=str, default="./R2R_ULsKaCPVFJR",
                       help="R2R数据集路径 (默认: ./R2R)")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = R2ROverexposureProcessor(args.r2r_path)
    
    # 执行相应操作
    if args.demo:
        create_enhanced_overexposure_demo()
    elif args.comparison:
        create_comparison_demo()
    elif args.list_scenes:
        processor.list_scenes()
    elif args.show_presets:
        processor.show_presets()
    elif args.process_all:
        processor.process_dataset(
            preset=args.preset,
            max_scenes=args.max_scenes,
            max_images_per_scene=args.max_images,
            save_intermediates=args.save_intermediates,
            skip_existing=args.skip_existing
        )
    else:
        print("请指定操作模式。使用 --help 查看详细帮助。")
        processor.show_presets()


if __name__ == "__main__":
    main()