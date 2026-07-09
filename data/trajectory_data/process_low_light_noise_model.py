#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低光成像物理模型处理器 - 增强版
基于公式: I(x) = CRF(G * T * L(x) + N(x))
其中 N(x) = N_shot(x) + N_read(x)

处理R2R数据集，生成与原目录同级的R2R_{处理方式}目录
使用精确的物理成像模型和高级低光效果
"""

import numpy as np
import cv2
import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json
from datetime import datetime
from scipy import ndimage


class EnhancedLowLightImagingModel:
    """
    增强版低光成像物理模型
    基于 low_light_noise_model.py 的实现
    """
    
    def __init__(self,
                 brightness_factor: float = 0.3,           # 亮度因子 (0.01-1.0)
                 noise_level: Union[str, float] = "low",   # 噪声级别或强度
                 exposure: float = 0.3,                    # 曝光倍数
                 gain: float = 4.0,                        # 传感器增益/ISO (1.0-32.0)
                 shot_noise_factor: float = 0.3,           # 散粒噪声因子
                 read_noise_sigma: float = 2.0,            # 读出噪声标准差
                 denoise_strength: float = 0.7,            # 降噪强度
                 preserve_details: float = 0.8,            # 细节保留度
                 gamma: float = 2.2,                       # 伽马校正值
                 contrast_factor: float = 0.85,            # 对比度因子
                 saturation_factor: float = 0.8,           # 饱和度因子
                 color_temperature: Optional[List[float]] = None,  # 色温调整
                 scene_type: str = "night",                # 场景类型
                 enable_noise: bool = True,                # 是否启用噪声
                 enable_blur: bool = False,                # 是否启用模糊
                 vignette_strength: float = 0.0,           # 暗角强度
                 depth_aware: bool = False,                # 是否考虑深度
                 random_params: bool = False):             # 是否使用随机参数
        
        # 基础参数设置
        self.brightness_factor = np.clip(brightness_factor, 0.01, 1.0)
        self.exposure = max(0.01, exposure)
        self.gain = np.clip(gain, 1.0, 32.0)
        
        # 处理噪声级别参数
        if isinstance(noise_level, str):
            noise_multipliers = {
                "minimal": 0.1,
                "low": 0.3,
                "medium": 0.6,
                "high": 1.0
            }
            self.noise_level = noise_level
            self.noise_multiplier = noise_multipliers.get(noise_level, 0.3)
        else:
            self.noise_level = "custom"
            self.noise_multiplier = np.clip(float(noise_level), 0.0, 1.0)
        
        # 噪声参数
        self.shot_noise_factor = shot_noise_factor * self.noise_multiplier
        self.read_noise_sigma = read_noise_sigma * self.noise_multiplier
        
        # 图像处理参数
        self.denoise_strength = np.clip(denoise_strength, 0.0, 1.0)
        self.preserve_details = np.clip(preserve_details, 0.0, 1.0)
        
        # 色彩参数
        self.gamma = gamma
        self.gamma_inv = 1.0 / gamma
        self.contrast_factor = np.clip(contrast_factor, 0.5, 1.0)
        self.saturation_factor = np.clip(saturation_factor, 0.0, 1.0)
        
        # 场景类型参数配置
        self.scene_params = {
            'neutral': {
                'color_temperature': np.array([1.0, 1.0, 1.0]),
                'noise_boost': 1.0,
                'detail_preservation': 1.0,
                'atmospheric_color': np.array([0.1, 0.1, 0.1])
            },
            'indoor': {
                'color_temperature': np.array([1.05, 1.0, 0.95]),  # 暖色调
                'noise_boost': 1.1,
                'detail_preservation': 0.9,
                'atmospheric_color': np.array([0.15, 0.12, 0.08])
            },
            'street': {
                'color_temperature': np.array([1.1, 1.05, 0.9]),   # 街灯黄光
                'noise_boost': 1.0,
                'detail_preservation': 0.85,
                'atmospheric_color': np.array([0.2, 0.15, 0.05])
            },
            'night': {
                'color_temperature': np.array([0.95, 1.0, 1.05]),  # 冷色调
                'noise_boost': 1.2,
                'detail_preservation': 0.8,
                'atmospheric_color': np.array([0.05, 0.05, 0.1])
            },
            'moonlight': {
                'color_temperature': np.array([0.85, 0.9, 1.1]),   # 月光蓝调
                'noise_boost': 1.3,
                'detail_preservation': 0.75,
                'atmospheric_color': np.array([0.05, 0.07, 0.15])
            },
            'extreme': {
                'color_temperature': np.array([0.8, 0.85, 1.15]),  # 极端低光
                'noise_boost': 1.5,
                'detail_preservation': 0.6,
                'atmospheric_color': np.array([0.02, 0.02, 0.05])
            }
        }
        
        self.scene_type = scene_type
        self.current_params = self.scene_params.get(
            scene_type, self.scene_params['night']
        )
        
        # 处理色温参数
        if color_temperature is not None:
            if isinstance(color_temperature, (list, tuple, np.ndarray)):
                self.color_temperature = np.array(color_temperature).flatten()
                if len(self.color_temperature) != 3:
                    self.color_temperature = np.ones(3)
            else:
                self.color_temperature = np.ones(3) * float(color_temperature)
        else:
            self.color_temperature = self.current_params['color_temperature']
        
        # 其他效果参数
        self.enable_noise = enable_noise
        self.enable_blur = enable_blur
        self.vignette_strength = np.clip(vignette_strength, 0.0, 1.0)
        self.depth_aware = depth_aware
        self.random_params = random_params
    
    def generate_random_params(self):
        """生成随机参数"""
        if self.random_params:
            # 随机亮度因子
            self.brightness_factor = np.random.uniform(0.01, 0.5)
            
            # 随机曝光和增益
            self.exposure = np.random.uniform(0.01, 0.5)
            self.gain = np.random.uniform(2.0, 16.0)
            
            # 随机噪声水平
            self.noise_multiplier = np.random.uniform(0.2, 1.0)
            self.shot_noise_factor = np.random.uniform(0.2, 0.6) * self.noise_multiplier
            self.read_noise_sigma = np.random.uniform(1.0, 5.0) * self.noise_multiplier
            
            # 随机降噪参数
            self.denoise_strength = np.random.uniform(0.3, 0.9)
            self.preserve_details = np.random.uniform(0.5, 0.9)
            
            # 随机色彩参数
            self.contrast_factor = np.random.uniform(0.6, 0.95)
            self.saturation_factor = np.random.uniform(0.3, 0.9)
            
            # 随机场景类型
            scene_types = ['indoor', 'street', 'night', 'moonlight', 'extreme']
            self.scene_type = np.random.choice(scene_types)
            self.current_params = self.scene_params[self.scene_type]
            
            # 随机暗角强度
            self.vignette_strength = np.random.uniform(0.0, 0.5)
    
    def _inverse_crf(self, srgb_image: np.ndarray) -> np.ndarray:
        """
        逆相机响应函数：将sRGB图像转换为线性辐照度
        使用标准sRGB逆变换
        """
        # 避免数值问题
        srgb_safe = np.maximum(srgb_image, 1e-8)
        
        # sRGB到线性的标准转换
        linear = np.where(
            srgb_safe <= 0.04045,
            srgb_safe / 12.92,
            np.power((srgb_safe + 0.055) / 1.055, 2.4)
        )
        
        return linear
    
    def _apply_crf(self, linear_signal: np.ndarray) -> np.ndarray:
        """
        应用相机响应函数：将线性信号转换为sRGB
        """
        # 避免数值问题
        linear_safe = np.maximum(linear_signal, 1e-8)
        
        # 使用伽马校正
        return np.power(linear_safe, self.gamma_inv)
    
    def _apply_darkness_curve(self, signal: np.ndarray) -> np.ndarray:
        """
        应用低光暗化曲线
        保留暗部细节，压缩亮部
        """
        # 使用S形曲线变换
        # 这样可以保留一些暗部细节，同时整体降低亮度
        
        # 幂函数变换
        power = 1.8 - 0.8 * self.brightness_factor
        darkened = np.power(signal, power) * self.brightness_factor
        
        # 应用大气散射效果（模拟低光下的雾化）
        atmospheric = self.current_params['atmospheric_color'].reshape(1, 1, 3)
        darkened = darkened * 0.95 + atmospheric * 0.05
        
        return darkened
    
    def _add_realistic_noise(self, image: np.ndarray) -> np.ndarray:
        """
        添加真实的低光噪声（控制强度）
        """
        h, w, c = image.shape
        
        # 获取噪声增强因子
        noise_boost = self.current_params['noise_boost']
        
        # 1. 读出噪声（高斯噪声）
        read_noise_std = (self.read_noise_sigma / 255.0) * noise_boost
        read_noise = np.random.normal(0, read_noise_std, (h, w, c))
        
        # 2. 散粒噪声（泊松噪声的高斯近似）
        # 标准差与信号平方根成正比，但强度要低
        signal_std = np.sqrt(np.maximum(image, 0.001)) * self.shot_noise_factor * 0.01 * noise_boost
        shot_noise = np.random.normal(0, signal_std)
        
        # 3. 低频噪声（传感器不均匀性）
        low_freq_noise = self._generate_low_frequency_noise(h, w, c)
        low_freq_noise *= 0.003 * noise_boost
        
        # 组合噪声（降低总体强度）
        total_noise = (read_noise + shot_noise + low_freq_noise) * 0.3
        
        return image + total_noise
    
    def _generate_low_frequency_noise(self, h: int, w: int, c: int) -> np.ndarray:
        """
        生成低频噪声模式（模拟传感器不均匀性）
        """
        # 生成小尺寸噪声
        small_h, small_w = max(1, h // 8), max(1, w // 8)
        small_noise = np.random.randn(small_h, small_w, c)
        
        # 上采样到原始尺寸
        low_freq_noise = np.zeros((h, w, c))
        for i in range(c):
            low_freq_noise[:, :, i] = cv2.resize(
                small_noise[:, :, i],
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )
        
        return low_freq_noise
    
    def _apply_intelligent_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        智能降噪处理
        保留细节的同时减少噪声
        """
        if self.denoise_strength <= 0:
            return image
        
        # 使用双边滤波进行保边降噪
        d = int(3 + 2 * self.denoise_strength)  # 滤波器直径
        sigma_color = 0.05 + 0.05 * self.denoise_strength
        sigma_space = 5 + 10 * self.denoise_strength
        
        # 转换到0-255范围进行滤波
        image_255 = (image * 255).astype(np.float32)
        
        # 应用双边滤波
        denoised_255 = cv2.bilateralFilter(
            image_255,
            d=d,
            sigmaColor=sigma_color * 255,
            sigmaSpace=sigma_space
        )
        
        denoised = denoised_255 / 255.0
        
        # 混合原图和降噪图（保留细节）
        detail_preservation = self.current_params['detail_preservation']
        alpha = self.preserve_details * detail_preservation
        result = alpha * image + (1 - alpha) * denoised
        
        return result
    
    def _apply_post_processing(self, image: np.ndarray) -> np.ndarray:
        """
        应用后处理效果
        包括色温、对比度、饱和度调整等
        """
        processed = image.copy()
        
        # 1. 应用色温调整
        processed = processed * self.color_temperature.reshape(1, 1, 3)
        
        # 2. 调整对比度
        mean = np.mean(processed)
        contrast_adjusted = self.contrast_factor + (1 - self.contrast_factor) * self.brightness_factor * 0.5
        processed = mean + (processed - mean) * contrast_adjusted
        
        # 3. 调整饱和度
        gray = np.mean(processed, axis=2, keepdims=True)
        saturation_adjusted = self.saturation_factor + (1 - self.saturation_factor) * self.brightness_factor * 0.3
        processed = gray + (processed - gray) * saturation_adjusted
        
        # 4. 添加暗角效果（如果启用）
        if self.vignette_strength > 0:
            processed = self._add_vignette(processed)
        
        # 5. 添加轻微模糊（如果启用）
        if self.enable_blur:
            processed = self._add_motion_blur(processed)
        
        # 6. 确保最低亮度（避免完全黑暗）
        min_brightness = 0.01
        current_mean = np.mean(processed)
        if current_mean < min_brightness:
            processed = processed * (min_brightness / (current_mean + 1e-8))
        
        return processed
    
    def _add_vignette(self, image: np.ndarray) -> np.ndarray:
        """
        添加暗角效果
        """
        h, w = image.shape[:2]
        
        # 创建径向渐变
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # 渐变蒙版
        vignette = 1 - (dist_from_center / max_dist) ** 2 * self.vignette_strength
        vignette = np.expand_dims(vignette, axis=2)
        
        return image * vignette
    
    def _add_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """
        添加轻微的运动模糊（模拟手持相机抖动）
        """
        # 创建小的运动模糊核
        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        
        # 应用到每个通道
        blurred = np.zeros_like(image)
        for i in range(image.shape[2]):
            blurred[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
        
        # 轻微混合
        return image * 0.9 + blurred * 0.1
    
    def simulate_low_light_imaging(self, input_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        模拟低光成像过程 - 增强版实现
        基于公式: I(x) = CRF(G * T * L(x) + N(x))
        
        Args:
            input_image: 输入的正常曝光sRGB图像
        
        Returns:
            低光图像, 中间结果字典
        """
        # 如果使用随机参数，每次都重新生成
        if self.random_params:
            self.generate_random_params()
        
        # 确保输入格式正确
        if input_image.dtype == np.uint8:
            img_float = input_image.astype(np.float32) / 255.0
            input_uint8 = True
        else:
            img_float = input_image.astype(np.float32)
            input_uint8 = False
        
        h, w, c = img_float.shape
        
        # 处理RGBA图像：只对RGB通道应用低光，保持Alpha通道不变
        has_alpha = (c == 4)
        if has_alpha:
            rgb_image = img_float[:, :, :3]  # 提取RGB通道
            alpha_channel = img_float[:, :, 3:4]  # 保存Alpha通道
        else:
            rgb_image = img_float
        
        # Step 1: 线性化 - 逆CRF获得线性辐照度 L(x)
        linear_radiance = self._inverse_crf(rgb_image)
        
        # Step 2: 应用低曝光和增益
        # S(x) = G * T * L(x)
        signal = self.gain * self.exposure * linear_radiance
        
        # Step 3: 降低整体亮度（主要低光效果）
        darkened = self._apply_darkness_curve(signal)
        
        # Step 4: 添加噪声（如果启用）
        if self.enable_noise:
            noisy = self._add_realistic_noise(darkened)
        else:
            noisy = darkened
        
        # Step 5: 应用降噪处理（模拟相机内部降噪）
        if self.enable_noise and self.denoise_strength > 0:
            denoised = self._apply_intelligent_denoising(noisy)
        else:
            denoised = noisy
        
        # Step 6: 应用CRF（相机响应函数）
        output = self._apply_crf(denoised)
        
        # Step 7: 后处理效果
        output = self._apply_post_processing(output)
        
        # 重新组合RGBA图像
        if has_alpha:
            output = np.concatenate([output, alpha_channel], axis=2)
        
        # 确保输出在有效范围内
        output = np.clip(output, 0, 1)
        
        # 保存中间结果
        intermediate_results = {
            'linear_radiance': linear_radiance,
            'exposed_signal': signal,
            'darkened_signal': darkened,
            'noisy_signal': noisy if self.enable_noise else darkened,
            'denoised_signal': denoised if (self.enable_noise and self.denoise_strength > 0) else noisy
        }
        
        # 转换回输入格式
        if input_uint8:
            output_image = (output * 255).astype(np.uint8)
            # 转换中间结果用于可视化
            for key in intermediate_results:
                result = intermediate_results[key]
                intermediate_results[key] = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            output_image = output
        
        return output_image, intermediate_results


class R2RLowLightProcessor:
    """R2R数据集低光成像批处理器 - 增强版"""
    
    def __init__(self, r2r_root: str = "./R2R"):
        self.r2r_root = Path(r2r_root)
        self.images_path = self.r2r_root / "images"
        
        # 增强版预设参数配置
        self.presets = {
            "normal": {
                "brightness_factor": 1.0,
                "noise_level": "minimal",
                "exposure": 1.0,
                "gain": 1.0,
                "shot_noise_factor": 0.1,
                "read_noise_sigma": 0.5,
                "denoise_strength": 0.0,
                "preserve_details": 1.0,
                "gamma": 2.2,
                "contrast_factor": 1.0,
                "saturation_factor": 1.0,
                "scene_type": "neutral",
                "enable_noise": False,
                "enable_blur": False,
                "vignette_strength": 0.0,
                "random_params": False,
                "description": "正常光照 - 无低光效果"
            },
            "dusk": {
                "brightness_factor": 0.6,
                "noise_level": "minimal",
                "exposure": 0.8,
                "gain": 1.5,
                "shot_noise_factor": 0.2,
                "read_noise_sigma": 1.0,
                "denoise_strength": 0.3,
                "preserve_details": 0.9,
                "gamma": 2.2,
                "contrast_factor": 0.95,
                "saturation_factor": 0.9,
                "color_temperature": [1.1, 1.0, 0.9],
                "scene_type": "neutral",
                "enable_noise": True,
                "enable_blur": False,
                "vignette_strength": 0.0,
                "random_params": False,
                "description": "黄昏 - 轻微降低亮度"
            },
            "dim_indoor": {
                "brightness_factor": 0.4,
                "noise_level": "low",
                "exposure": 0.5,
                "gain": 2.0,
                "shot_noise_factor": 0.25,
                "read_noise_sigma": 1.5,
                "denoise_strength": 0.5,
                "preserve_details": 0.85,
                "gamma": 2.2,
                "contrast_factor": 0.9,
                "saturation_factor": 0.85,
                "scene_type": "indoor",
                "enable_noise": True,
                "enable_blur": False,
                "vignette_strength": 0.0,
                "random_params": False,
                "description": "昏暗室内 - 温暖但昏暗的室内光"
            },
            "night_street": {
                "brightness_factor": 0.3,
                "noise_level": "low",
                "exposure": 0.3,
                "gain": 4.0,
                "shot_noise_factor": 0.3,
                "read_noise_sigma": 2.0,
                "denoise_strength": 0.6,
                "preserve_details": 0.8,
                "gamma": 2.2,
                "contrast_factor": 0.85,
                "saturation_factor": 0.75,
                "scene_type": "street",
                "enable_noise": True,
                "enable_blur": False,
                "vignette_strength": 0.0,
                "random_params": False,
                "description": "夜晚街道 - 街灯照明环境"
            },
            "moonlight": {
                "brightness_factor": 0.2,
                "noise_level": "low",
                "exposure": 0.25,
                "gain": 6.0,
                "shot_noise_factor": 0.35,
                "read_noise_sigma": 2.5,
                "denoise_strength": 0.7,
                "preserve_details": 0.75,
                "gamma": 2.2,
                "contrast_factor": 0.8,
                "saturation_factor": 0.6,
                "scene_type": "moonlight",
                "enable_noise": True,
                "enable_blur": False,
                "vignette_strength": 0.3,
                "random_params": False,
                "description": "月光 - 自然月光照明"
            },
            "deep_night": {
                "brightness_factor": 0.15,
                "noise_level": "medium",
                "exposure": 0.15,
                "gain": 8.0,
                "shot_noise_factor": 0.4,
                "read_noise_sigma": 3.0,
                "denoise_strength": 0.75,
                "preserve_details": 0.7,
                "gamma": 2.2,
                "contrast_factor": 0.75,
                "saturation_factor": 0.5,
                "scene_type": "night",
                "enable_noise": True,
                "enable_blur": False,
                "vignette_strength": 0.4,
                "random_params": False,
                "description": "深夜 - 极少光源"
            },
            "very_dark": {
                "brightness_factor": 0.08,
                "noise_level": "medium",
                "exposure": 0.1,
                "gain": 12.0,
                "shot_noise_factor": 0.5,
                "read_noise_sigma": 4.0,
                "denoise_strength": 0.8,
                "preserve_details": 0.6,
                "gamma": 2.2,
                "contrast_factor": 0.7,
                "saturation_factor": 0.4,
                "scene_type": "night",
                "enable_noise": True,
                "enable_blur": True,
                "vignette_strength": 0.5,
                "random_params": False,
                "description": "极暗 - 几乎无光"
            },
            "pitch_black": {
                "brightness_factor": 0.03,
                "noise_level": "high",
                "exposure": 0.05,
                "gain": 16.0,
                "shot_noise_factor": 0.6,
                "read_noise_sigma": 5.0,
                "denoise_strength": 0.85,
                "preserve_details": 0.5,
                "gamma": 2.2,
                "contrast_factor": 0.6,
                "saturation_factor": 0.3,
                "scene_type": "extreme",
                "enable_noise": True,
                "enable_blur": True,
                "vignette_strength": 0.6,
                "random_params": False,
                "description": "漆黑 - 极端低光条件"
            },
            "emergency_light": {
                "brightness_factor": 0.35,
                "noise_level": "low",
                "exposure": 0.4,
                "gain": 3.5,
                "shot_noise_factor": 0.28,
                "read_noise_sigma": 1.8,
                "denoise_strength": 0.55,
                "preserve_details": 0.82,
                "gamma": 2.1,
                "contrast_factor": 0.88,
                "saturation_factor": 0.7,
                "color_temperature": [1.0, 0.95, 0.9],
                "scene_type": "indoor",
                "enable_noise": True,
                "enable_blur": False,
                "vignette_strength": 0.15,
                "random_params": False,
                "description": "应急照明 - 应急灯光环境"
            },
            "candlelight": {
                "brightness_factor": 0.25,
                "noise_level": "low",
                "exposure": 0.3,
                "gain": 5.0,
                "shot_noise_factor": 0.32,
                "read_noise_sigma": 2.2,
                "denoise_strength": 0.65,
                "preserve_details": 0.78,
                "gamma": 2.3,
                "contrast_factor": 0.82,
                "saturation_factor": 0.65,
                "color_temperature": [1.15, 1.0, 0.85],
                "scene_type": "indoor",
                "enable_noise": True,
                "enable_blur": False,
                "vignette_strength": 0.25,
                "random_params": False,
                "description": "烛光 - 温暖的烛光照明"
            },
            "random_low_light": {
                "brightness_factor": 0.3,  # 会被随机覆盖
                "noise_level": "medium",   # 会被随机覆盖
                "exposure": 0.3,           # 会被随机覆盖
                "gain": 4.0,              # 会被随机覆盖
                "shot_noise_factor": 0.3, # 会被随机覆盖
                "read_noise_sigma": 2.0,  # 会被随机覆盖
                "denoise_strength": 0.7,  # 会被随机覆盖
                "preserve_details": 0.8,  # 会被随机覆盖
                "gamma": 2.2,             # 会被随机覆盖
                "contrast_factor": 0.85,  # 会被随机覆盖
                "saturation_factor": 0.8, # 会被随机覆盖
                "scene_type": "night",    # 会被随机覆盖
                "enable_noise": True,
                "enable_blur": False,      # 会被随机覆盖
                "vignette_strength": 0.0, # 会被随机覆盖
                "random_params": True,
                "description": "随机低光参数 - 每张图片使用不同参数"
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
                           low_light_model: EnhancedLowLightImagingModel,
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
            
            # 应用低光成像模拟
            low_light_img, intermediate_results = low_light_model.simulate_low_light_imaging(image_rgb)
            
            # 保存低光图像 (RGB转BGR)
            if low_light_img.dtype == np.uint8:
                low_light_bgr = cv2.cvtColor(low_light_img, cv2.COLOR_RGB2BGR)
            else:
                low_light_bgr = cv2.cvtColor((low_light_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(output_path), low_light_bgr)
            
            # 保存中间结果
            if save_intermediates:
                base_name = output_path.stem
                output_dir = output_path.parent
                
                # 保存线性辐照度
                if 'linear_radiance' in intermediate_results:
                    radiance_path = output_dir / f"{base_name}_radiance.png"
                    radiance_bgr = cv2.cvtColor(intermediate_results['linear_radiance'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(radiance_path), radiance_bgr)
                
                # 保存暗化信号
                if 'darkened_signal' in intermediate_results:
                    darkened_path = output_dir / f"{base_name}_darkened.png"
                    darkened_bgr = cv2.cvtColor(intermediate_results['darkened_signal'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(darkened_path), darkened_bgr)
                
                # 保存噪声信号
                if 'noisy_signal' in intermediate_results:
                    noisy_path = output_dir / f"{base_name}_noisy.png"
                    noisy_bgr = cv2.cvtColor(intermediate_results['noisy_signal'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(noisy_path), noisy_bgr)
            
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
            "model": "Enhanced Low Light Imaging Physical Model",
            "formula": "I(x) = CRF(G * T * L(x) + N_shot(x) + N_read(x))",
            "features": [
                "Physical-based imaging simulation",
                "Shot noise and read noise modeling",
                "Intelligent denoising",
                "Scene-specific color temperature",
                "Vignetting effects",
                "Detail preservation"
            ]
        }
        
        info_path = output_root / "processing_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def process_dataset(self,
                       preset: str = "night_street",
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
        
        # 创建增强版低光成像模型
        low_light_model = EnhancedLowLightImagingModel(
            brightness_factor=preset_params["brightness_factor"],
            noise_level=preset_params["noise_level"],
            exposure=preset_params["exposure"],
            gain=preset_params["gain"],
            shot_noise_factor=preset_params["shot_noise_factor"],
            read_noise_sigma=preset_params["read_noise_sigma"],
            denoise_strength=preset_params["denoise_strength"],
            preserve_details=preset_params["preserve_details"],
            gamma=preset_params["gamma"],
            contrast_factor=preset_params["contrast_factor"],
            saturation_factor=preset_params["saturation_factor"],
            color_temperature=preset_params.get("color_temperature"),
            scene_type=preset_params["scene_type"],
            enable_noise=preset_params["enable_noise"],
            enable_blur=preset_params["enable_blur"],
            vignette_strength=preset_params["vignette_strength"],
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
        print(f"\n🌙 增强版低光成像物理模型处理")
        print(f"📂 源目录: {self.r2r_root.absolute()}")
        print(f"📂 输出目录: {output_root.absolute()}")
        print(f"🎨 预设: {preset} - {preset_params['description']}")
        print(f"💡 亮度: {preset_params['brightness_factor']:.2f}")
        print(f"📊 参数: T={preset_params['exposure']:.2f}, G={preset_params['gain']:.1f}")
        print(f"🔊 噪声: {preset_params['noise_level']}, 降噪={preset_params['denoise_strength']:.1f}")
        print(f"🎯 场景: {preset_params['scene_type']}")
        print(f"✨ 暗角: {preset_params['vignette_strength']:.1f}")
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
                    img_path, output_path, low_light_model, save_intermediates
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
        print(f"🎉 增强版低光成像处理完成!")
        print(f"✅ 成功处理: {total_processed} 张图片")
        if total_skipped > 0:
            print(f"⭕ 跳过已存在: {total_skipped} 张图片")
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
        print("\n🌙 可用的增强版低光成像预设:")
        print("=" * 80)
        
        for name, params in self.presets.items():
            print(f"📋 {name}")
            print(f"   描述: {params['description']}")
            print(f"   亮度因子: {params['brightness_factor']:.2f}")
            print(f"   曝光倍数(T): {params['exposure']:.2f}")
            print(f"   增益(G): {params['gain']:.1f}")
            print(f"   噪声级别: {params['noise_level']}")
            print(f"   降噪强度: {params['denoise_strength']:.1f}")
            print(f"   细节保留: {params['preserve_details']:.1f}")
            print(f"   场景类型: {params['scene_type']}")
            print(f"   对比度: {params['contrast_factor']:.2f}")
            print(f"   饱和度: {params['saturation_factor']:.2f}")
            print(f"   噪声: {'✓' if params['enable_noise'] else '✗'}")
            print(f"   模糊: {'✓' if params['enable_blur'] else '✗'}")
            print(f"   暗角: {params['vignette_strength']:.1f}")
            print(f"   随机参数: {'✓' if params['random_params'] else '✗'}")
            print()


def create_enhanced_low_light_demo():
    """创建增强版低光成像演示"""
    # 创建测试图像 (正常曝光的场景)
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # 创建一个正常亮度的场景
    # 天空 (明亮蓝色)
    cv2.rectangle(test_image, (0, 0), (400, 100), (135, 206, 235), -1)
    # 建筑物 (亮灰色)
    cv2.rectangle(test_image, (50, 100), (150, 200), (180, 180, 180), -1)
    cv2.rectangle(test_image, (200, 90), (350, 210), (160, 160, 200), -1)
    # 树木 (绿色)
    cv2.rectangle(test_image, (320, 120), (330, 200), (101, 67, 33), -1)  # 树干
    cv2.circle(test_image, (325, 115), 20, (34, 139, 34), -1)  # 树冠
    # 前景地面 (亮绿色)
    cv2.rectangle(test_image, (0, 200), (400, 300), (90, 180, 90), -1)
    
    # 添加一些亮点（将在低光下成为光源）
    cv2.circle(test_image, (80, 50), 15, (255, 255, 200), -1)  # 太阳/月亮
    cv2.rectangle(test_image, (70, 130), (80, 140), (255, 255, 200), -1)  # 窗户灯光
    cv2.rectangle(test_image, (220, 120), (230, 130), (255, 255, 200), -1)  # 窗户灯光
    
    # 预设效果演示
    presets_demo = ["dusk", "dim_indoor", "night_street", 
                   "moonlight", "deep_night", "very_dark"]
    processor = R2RLowLightProcessor()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, preset_name in enumerate(presets_demo):
        params = processor.presets[preset_name]
        
        # 创建增强版低光成像模型
        model = EnhancedLowLightImagingModel(
            brightness_factor=params["brightness_factor"],
            noise_level=params["noise_level"],
            exposure=params["exposure"],
            gain=params["gain"],
            shot_noise_factor=params["shot_noise_factor"],
            read_noise_sigma=params["read_noise_sigma"],
            denoise_strength=params["denoise_strength"],
            preserve_details=params["preserve_details"],
            gamma=params["gamma"],
            contrast_factor=params["contrast_factor"],
            saturation_factor=params["saturation_factor"],
            color_temperature=params.get("color_temperature"),
            scene_type=params["scene_type"],
            enable_noise=params["enable_noise"],
            enable_blur=params["enable_blur"],
            vignette_strength=params["vignette_strength"],
            random_params=False
        )
        
        # 应用低光效果
        low_light_img, _ = model.simulate_low_light_imaging(test_image)
        
        # 显示结果
        axes[i].imshow(low_light_img)
        title = f'{preset_name}\n'
        title += f'亮度={params["brightness_factor"]:.2f}, T={params["exposure"]:.2f}, G={params["gain"]:.1f}\n'
        title += f'噪声={params["noise_level"]}, 降噪={params["denoise_strength"]:.1f}'
        if params["enable_noise"]:
            title += ', N✓'
        if params["vignette_strength"] > 0:
            title += f', V={params["vignette_strength"]:.1f}'
        
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('增强版低光成像物理模型 - 效果预览\nI(x) = CRF(G*T*L(x) + N_shot + N_read)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('enhanced_low_light_imaging_presets.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 增强版低光成像预设效果演示图已保存: enhanced_low_light_imaging_presets.png")


def create_comparison_demo():
    """创建不同低光级别对比演示"""
    # 创建测试图像
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # 创建具有不同亮度区域的测试场景
    # 暗区域
    cv2.rectangle(test_image, (0, 0), (100, 200), (60, 60, 60), -1)
    # 中等亮度区域
    cv2.rectangle(test_image, (100, 0), (200, 200), (150, 150, 150), -1)
    # 高亮区域
    cv2.rectangle(test_image, (200, 0), (300, 200), (220, 220, 220), -1)
    
    # 添加一些细节
    cv2.circle(test_image, (50, 100), 20, (100, 100, 100), -1)
    cv2.circle(test_image, (150, 100), 20, (200, 200, 200), -1)
    cv2.circle(test_image, (250, 100), 20, (255, 255, 255), -1)
    
    # 比较不同预设的效果
    processor = R2RLowLightProcessor()
    
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
        'dusk',
        'night_street',
        'deep_night'
    ]
    
    for i, preset_name in enumerate(test_presets):
        params = processor.presets[preset_name]
        
        # 上排：正常设置
        model1 = EnhancedLowLightImagingModel(
            brightness_factor=params["brightness_factor"],
            noise_level=params["noise_level"],
            exposure=params["exposure"],
            gain=params["gain"],
            shot_noise_factor=params["shot_noise_factor"],
            read_noise_sigma=params["read_noise_sigma"],
            denoise_strength=params["denoise_strength"],
            preserve_details=params["preserve_details"],
            gamma=params["gamma"],
            contrast_factor=params["contrast_factor"],
            saturation_factor=params["saturation_factor"],
            scene_type=params["scene_type"],
            enable_noise=params["enable_noise"],
            enable_blur=params["enable_blur"],
            vignette_strength=params["vignette_strength"]
        )
        
        result1, _ = model1.simulate_low_light_imaging(test_image)
        axes[0, i+1].imshow(result1)
        axes[0, i+1].set_title(f"{preset_name}\n亮度={params['brightness_factor']:.2f}", fontsize=10)
        axes[0, i+1].axis('off')
        
        # 下排：增强噪声设置
        model2 = EnhancedLowLightImagingModel(
            brightness_factor=params["brightness_factor"] * 0.7,
            noise_level="high",
            exposure=params["exposure"] * 0.8,
            gain=params["gain"] * 1.5,
            shot_noise_factor=params["shot_noise_factor"] * 1.5,
            read_noise_sigma=params["read_noise_sigma"] * 1.5,
            denoise_strength=params["denoise_strength"] * 0.8,
            preserve_details=params["preserve_details"] * 0.9,
            gamma=params["gamma"],
            contrast_factor=params["contrast_factor"] * 0.9,
            saturation_factor=params["saturation_factor"] * 0.8,
            scene_type=params["scene_type"],
            enable_noise=True,
            enable_blur=params["enable_blur"],
            vignette_strength=params["vignette_strength"] * 1.3
        )
        
        result2, _ = model2.simulate_low_light_imaging(test_image)
        axes[1, i+1].imshow(result2)
        axes[1, i+1].set_title(f"{preset_name} (增强)\n亮度={params['brightness_factor']*0.7:.2f}", fontsize=10)
        axes[1, i+1].axis('off')
    
    plt.suptitle('低光预设对比 - 正常 vs 增强效果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('low_light_presets_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 低光预设对比图已保存: low_light_presets_comparison.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强版低光成像物理模型 - R2R数据集批处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌙 基于物理成像模型: I(x) = CRF(G * T * L(x) + N_shot(x) + N_read(x))
   增强特性:
   - 📊 物理噪声模型 (散粒噪声 + 读出噪声)
   - 🎨 场景自适应色温
   - ✨ 智能降噪处理
   - 🎯 细节保留算法
   - 🌈 暗角效果
   - 🎲 随机参数生成

💡 使用示例:
   # 处理所有场景 - 夜晚街道效果
   python script.py --process-all --preset night_street
   
   # 月光效果 + 保存中间结果
   python script.py --process-all --preset moonlight --save-intermediates
   
   # 深夜效果
   python script.py --process-all --preset deep_night
   
   # 随机参数效果 (每张图片不同参数)
   python script.py --process-all --preset random_low_light
   
   # 限制处理数量 (测试用)
   python script.py --process-all --preset night_street --max-scenes 2 --max-images 5
   
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
    parser.add_argument("--preset", "-p", type=str, default="deep_night",
                       choices=["normal", "dusk", "dim_indoor", "night_street", "moonlight",
                               "deep_night", "very_dark", "pitch_black", "emergency_light",
                               "candlelight", "random_low_light"],
                       help="增强版低光成像预设")
    
    # 限制参数
    parser.add_argument("--max-scenes", type=int, 
                       help="最大处理场景数 (用于测试)")
    parser.add_argument("--max-images", type=int,
                       help="每个场景最大图片数 (用于测试)")
    
    # 输出控制
    parser.add_argument("--save-intermediates", action="store_true",
                       help="保存中间结果 (辐照度图、暗化信号图、噪声信号图)")
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
    processor = R2RLowLightProcessor(args.r2r_path)
    
    # 执行相应操作
    if args.demo:
        create_enhanced_low_light_demo()
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