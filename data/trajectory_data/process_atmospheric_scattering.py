#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大气散射物理模型处理器 - 增强版
基于公式: I(x) = J(x) * t(x) + A * (1 - t(x))
其中 t(x) = e^(-β*d(x))

处理R2R数据集，生成与原目录同级的R2R_{处理方式}目录
使用精确的物理散射模型和高级大气效果
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


class EnhancedAtmosphericScatteringModel:
    """
    增强版大气散射物理模型
    基于 atmospheric_scattering_noise_model.py 的实现
    """
    
    def __init__(self,
                 beta: float = 0.1,                    # 大气散射系数 (0.01-2.0 推荐范围)
                 atmospheric_light: Union[float, List[float]] = 0.8,  # 大气光强度或RGB值
                 max_distance: float = 50.0,           # 最大有效距离
                 scattering_type: str = "haze",        # 散射类型
                 wavelength_effect: bool = True,       # 波长相关散射
                 particle_size: float = 1.0,           # 颗粒尺寸
                 depth_estimation_method: str = "simple",  # 深度估计方法
                 use_real_depth: bool = False,         # 是否使用真实深度
                 random_params: bool = False):         # 是否使用随机参数
        
        self.beta = beta
        self.max_distance = max_distance
        self.scattering_type = scattering_type
        self.wavelength_effect = wavelength_effect
        self.particle_size = particle_size
        self.depth_estimation_method = depth_estimation_method
        self.use_real_depth = use_real_depth
        self.random_params = random_params
        
        # 处理大气光参数 - 可以是标量或RGB向量
        if isinstance(atmospheric_light, (list, tuple, np.ndarray)):
            self.atmospheric_light = np.array(atmospheric_light).flatten()
            if len(self.atmospheric_light) != 3:
                self.atmospheric_light = np.ones(3) * self.atmospheric_light[0]
        else:
            self.atmospheric_light = np.ones(3) * atmospheric_light
        
        # 确保大气光在合理范围内
        self.atmospheric_light = np.clip(self.atmospheric_light, 0, 1)
        
        # 不同散射类型的参数配置
        self.scattering_params = {
            'haze': {
                'color_shift': np.array([0.9, 0.9, 0.9]),
                'atmospheric_color': np.array([0.8, 0.8, 0.8]),
                'rayleigh_factor': 0.3,
                'mie_factor': 0.7,
                'beta_multiplier': 1.0
            },
            'fog': {
                'color_shift': np.array([0.95, 0.95, 0.95]),
                'atmospheric_color': np.array([0.9, 0.9, 0.9]),
                'rayleigh_factor': 0.2,
                'mie_factor': 0.8,
                'beta_multiplier': 1.2
            },
            'smog': {
                'color_shift': np.array([1.0, 0.8, 0.6]),
                'atmospheric_color': np.array([0.8, 0.7, 0.5]),
                'rayleigh_factor': 0.1,
                'mie_factor': 0.9,
                'beta_multiplier': 1.1
            },
            'dust': {
                'color_shift': np.array([1.0, 0.9, 0.7]),
                'atmospheric_color': np.array([0.9, 0.8, 0.6]),
                'rayleigh_factor': 0.05,
                'mie_factor': 0.95,
                'beta_multiplier': 1.5
            }
        }
        
        self.current_params = self.scattering_params.get(
            scattering_type, self.scattering_params['haze']
        )
        
        # 计算有效的beta值（考虑散射类型）
        self.effective_beta = self.beta * self.current_params['beta_multiplier']
        
        # 波长相关散射系数计算
        if wavelength_effect:
            # RGB对应的波长 (nm)
            wavelengths = np.array([700, 550, 450])  
            # 瑞利散射 ∝ λ^-4
            rayleigh_scatter = (wavelengths / 550) ** (-4)  
            # 米氏散射 ∝ λ^-1 到 λ^-2 (取决于颗粒大小)
            mie_power = -1 - 0.5 * np.log10(particle_size)
            mie_scatter = (wavelengths / 550) ** mie_power
            
            # 组合散射效应
            self.wavelength_scatter = (
                rayleigh_scatter * self.current_params['rayleigh_factor'] +
                mie_scatter * self.current_params['mie_factor']
            )
            # 归一化
            self.wavelength_scatter = self.wavelength_scatter / np.mean(self.wavelength_scatter)
        else:
            self.wavelength_scatter = np.array([1.0, 1.0, 1.0])
    
    def generate_random_params(self):
        """生成随机参数"""
        if self.random_params:
            # 随机散射系数
            self.beta = np.random.uniform(0.05, 0.5)
            
            # 随机大气光强度
            self.atmospheric_light = np.random.uniform(0.6, 0.95, 3)
            
            # 随机最大距离
            self.max_distance = np.random.uniform(20, 100)
            
            # 随机散射类型
            scattering_types = ['haze', 'fog', 'smog', 'dust']
            self.scattering_type = np.random.choice(scattering_types)
            self.current_params = self.scattering_params[self.scattering_type]
            
            # 随机颗粒大小
            self.particle_size = np.random.uniform(0.5, 5.0)
            
            # 重新计算有效beta值
            self.effective_beta = self.beta * self.current_params['beta_multiplier']
            
            # 重新计算波长散射
            if self.wavelength_effect:
                wavelengths = np.array([700, 550, 450])
                rayleigh_scatter = (wavelengths / 550) ** (-4)
                mie_power = -1 - 0.5 * np.log10(self.particle_size)
                mie_scatter = (wavelengths / 550) ** mie_power
                self.wavelength_scatter = (
                    rayleigh_scatter * self.current_params['rayleigh_factor'] +
                    mie_scatter * self.current_params['mie_factor']
                )
                self.wavelength_scatter = self.wavelength_scatter / np.mean(self.wavelength_scatter)
    
    def _estimate_depth_from_image(self, image: np.ndarray) -> np.ndarray:
        """从图像内容估算深度信息"""
        if self.depth_estimation_method == "simple":
            return self._simple_depth_estimation(image)
        elif self.depth_estimation_method == "gradient":
            return self._gradient_based_depth_estimation(image)
        else:
            return self._simple_depth_estimation(image)
    
    def _simple_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """简单的深度估算方法（基于亮度和位置）"""
        # 处理RGBA图像，只使用RGB通道进行深度估计
        if len(image.shape) == 3 and image.shape[2] > 3:
            rgb_image = image[:, :, :3]
        else:
            rgb_image = image
            
        gray = np.mean(rgb_image, axis=2)
        h, w = gray.shape
        
        # 基于亮度的深度估计（暗的区域通常较远）
        brightness_depth = (1.0 - gray) * self.max_distance * 0.4
        
        # 基于位置的深度估计（图像上方通常较远）
        y_coords = np.arange(h).reshape(-1, 1)
        position_depth = (y_coords / h) * self.max_distance * 0.6
        position_depth = np.repeat(position_depth, w, axis=1)
        
        # 添加一些随机变化，使效果更自然
        noise = np.random.normal(0, 0.05, (h, w)) * self.max_distance * 0.1
        
        # 组合深度估计
        estimated_depth = brightness_depth + position_depth + noise
        
        # 使用高斯滤波平滑深度图
        estimated_depth = cv2.GaussianBlur(estimated_depth, (5, 5), 1.0)
        
        # 限制在合理范围内
        return np.clip(estimated_depth, 0.1, self.max_distance)
    
    def _gradient_based_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """基于梯度的深度估算（更复杂的方法）"""
        if len(image.shape) == 3 and image.shape[2] > 3:
            rgb_image = image[:, :, :3]
        else:
            rgb_image = image
            
        gray = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 边缘通常表示深度变化
        depth_from_edges = (1 - gradient_magnitude) * self.max_distance
        
        # 平滑处理
        depth_from_edges = cv2.GaussianBlur(depth_from_edges, (7, 7), 2.0)
        
        return np.clip(depth_from_edges, 0.1, self.max_distance)
    
    def _get_atmospheric_light_color(self) -> np.ndarray:
        """获取大气光颜色 A"""
        # 基础大气光颜色（用户设置）
        base_atmospheric = self.atmospheric_light.copy()
        
        # 应用散射类型的颜色偏移
        color_shift = self.current_params['color_shift']
        atmospheric_color = base_atmospheric * color_shift
        
        # 混合散射类型的特征颜色
        type_color = self.current_params['atmospheric_color']
        atmospheric_color = 0.7 * atmospheric_color + 0.3 * type_color
        
        return np.clip(atmospheric_color, 0, 1)
    
    def simulate_atmospheric_scattering(self, input_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        模拟大气散射效果 - 增强版实现
        基于公式: I(x) = J(x) * t(x) + A * (1 - t(x))
        
        Args:
            input_image: 输入的清晰RGB图像
        
        Returns:
            散射效果图像, 中间结果字典
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
        
        # 处理RGBA图像：只对RGB通道应用散射，保持Alpha通道不变
        has_alpha = (c == 4)
        if has_alpha:
            rgb_image = img_float[:, :, :3]  # 提取RGB通道
            alpha_channel = img_float[:, :, 3:4]  # 保存Alpha通道
        else:
            rgb_image = img_float
        
        # 步骤1: 估算或获取深度信息 d(x)
        if self.use_real_depth:
            # 如果有真实深度信息，应该从其他传感器获取
            # 这里使用估算作为后备
            depth = self._estimate_depth_from_image(rgb_image)
        else:
            depth = self._estimate_depth_from_image(rgb_image)
        
        # 步骤2: 计算透射率 t(x) = e^(-β*d(x))
        # 使用有效beta值（已经考虑了散射类型）
        transmission_base = np.exp(-self.effective_beta * depth)
        transmission_base = np.clip(transmission_base, 0, 1)
        
        # 步骤3: 应用波长相关的散射（对不同颜色通道使用不同的透射率）
        if self.wavelength_effect:
            transmission = np.zeros((h, w, 3))
            for i in range(3):  # RGB三个通道
                # 每个通道的透射率略有不同，模拟波长相关散射
                channel_beta = self.effective_beta * self.wavelength_scatter[i]
                transmission[:, :, i] = np.exp(-channel_beta * depth)
        else:
            # 所有通道使用相同的透射率
            transmission = np.expand_dims(transmission_base, axis=2)
            transmission = np.repeat(transmission, 3, axis=2)
        
        # 确保透射率在合理范围内
        transmission = np.clip(transmission, 0, 1)
        
        # 步骤4: 获取大气光 A（考虑颜色偏移）
        atmospheric_color = self._get_atmospheric_light_color()
        
        # 步骤5: 应用散射模型 I(x) = J(x) * t(x) + A * (1 - t(x))
        # J(x) 是原始图像，t(x) 是透射率，A 是大气光
        scattered_rgb = (
            rgb_image * transmission + 
            atmospheric_color.reshape(1, 1, 3) * (1 - transmission)
        )
        
        # 确保输出在有效范围内
        scattered_rgb = np.clip(scattered_rgb, 0, 1)
        
        # 重新组合RGBA图像
        if has_alpha:
            scattered_image = np.concatenate([scattered_rgb, alpha_channel], axis=2)
        else:
            scattered_image = scattered_rgb
        
        # 保存中间结果
        intermediate_results = {
            'depth_map': depth,
            'transmission_map': transmission,
            'atmospheric_light': atmospheric_color,
            'clear_image': rgb_image
        }
        
        # 转换回输入格式
        if input_uint8:
            output_image = (scattered_image * 255).astype(np.uint8)
            # 转换中间结果用于可视化
            intermediate_results['depth_map'] = (
                (depth / self.max_distance * 255).astype(np.uint8)
            )
            intermediate_results['transmission_map'] = (
                (transmission * 255).astype(np.uint8)
            )
            intermediate_results['atmospheric_light'] = (
                (atmospheric_color * 255).astype(np.uint8)
            )
            intermediate_results['clear_image'] = (
                (rgb_image * 255).astype(np.uint8)
            )
        else:
            output_image = scattered_image
            
        return output_image, intermediate_results


class R2RAtmosphericScatteringProcessor:
    """R2R数据集大气散射批处理器 - 增强版"""
    
    def __init__(self, r2r_root: str = "./R2R"):
        self.r2r_root = Path(r2r_root)
        self.images_path = self.r2r_root / "images"
        
        # 增强版预设参数配置
        self.presets = {
            "clear": {
                "beta": 0.01,
                "atmospheric_light": [0.95, 0.95, 1.0],
                "max_distance": 200.0,
                "scattering_type": "haze",
                "wavelength_effect": False,
                "particle_size": 0.1,
                "depth_estimation_method": "simple",
                "random_params": False,
                "description": "清晰天空 - 几乎无散射效果"
            },
            "light_haze": {
                "beta": 0.05,
                "atmospheric_light": [0.85, 0.85, 0.9],
                "max_distance": 100.0,
                "scattering_type": "haze",
                "wavelength_effect": True,
                "particle_size": 1.0,
                "depth_estimation_method": "simple",
                "random_params": False,
                "description": "轻微雾霾 - 远处略微模糊"
            },
            "moderate_haze": {
                "beta": 0.1,
                "atmospheric_light": [0.8, 0.8, 0.85],
                "max_distance": 80.0,
                "scattering_type": "haze",
                "wavelength_effect": True,
                "particle_size": 1.5,
                "depth_estimation_method": "gradient",
                "random_params": False,
                "description": "中度雾霾 - 明显的大气散射"
            },
            "heavy_haze": {
                "beta": 0.2,
                "atmospheric_light": [0.75, 0.75, 0.8],
                "max_distance": 50.0,
                "scattering_type": "haze",
                "wavelength_effect": True,
                "particle_size": 2.0,
                "depth_estimation_method": "gradient",
                "random_params": False,
                "description": "重度雾霾 - 能见度显著降低"
            },
            "thick_fog": {
                "beta": 0.3,
                "atmospheric_light": [0.9, 0.9, 0.9],
                "max_distance": 30.0,
                "scattering_type": "fog",
                "wavelength_effect": True,
                "particle_size": 5.0,
                "depth_estimation_method": "simple",
                "random_params": False,
                "description": "浓雾 - 近距离才能看清"
            },
            "urban_smog": {
                "beta": 0.15,
                "atmospheric_light": [0.7, 0.65, 0.6],
                "max_distance": 60.0,
                "scattering_type": "smog",
                "wavelength_effect": True,
                "particle_size": 2.5,
                "depth_estimation_method": "gradient",
                "random_params": False,
                "description": "城市雾霾 - 带黄色调的污染"
            },
            "dust_storm": {
                "beta": 0.4,
                "atmospheric_light": [0.9, 0.8, 0.6],
                "max_distance": 20.0,
                "scattering_type": "dust",
                "wavelength_effect": True,
                "particle_size": 10.0,
                "depth_estimation_method": "simple",
                "random_params": False,
                "description": "沙尘暴 - 黄色沙尘遮蔽"
            },
            "severe_pollution": {
                "beta": 0.5,
                "atmospheric_light": [0.6, 0.55, 0.5],
                "max_distance": 15.0,
                "scattering_type": "smog",
                "wavelength_effect": True,
                "particle_size": 3.0,
                "depth_estimation_method": "gradient",
                "random_params": False,
                "description": "严重污染 - 极低能见度"
            },
            "morning_mist": {
                "beta": 0.08,
                "atmospheric_light": [0.9, 0.88, 0.85],
                "max_distance": 70.0,
                "scattering_type": "fog",
                "wavelength_effect": True,
                "particle_size": 3.0,
                "depth_estimation_method": "simple",
                "random_params": False,
                "description": "晨雾 - 温和的雾气效果"
            },
            "evening_haze": {
                "beta": 0.06,
                "atmospheric_light": [0.85, 0.75, 0.65],
                "max_distance": 90.0,
                "scattering_type": "haze",
                "wavelength_effect": True,
                "particle_size": 1.2,
                "depth_estimation_method": "gradient",
                "random_params": False,
                "description": "傍晚霾 - 带暖色调的散射"
            },
            "random_scattering": {
                "beta": 0.1,  # 会被随机覆盖
                "atmospheric_light": [0.8, 0.8, 0.8],  # 会被随机覆盖
                "max_distance": 50.0,  # 会被随机覆盖
                "scattering_type": "haze",  # 会被随机覆盖
                "wavelength_effect": True,
                "particle_size": 1.0,  # 会被随机覆盖
                "depth_estimation_method": "simple",
                "random_params": True,
                "description": "随机散射参数 - 每张图片使用不同参数"
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
                           scattering_model: EnhancedAtmosphericScatteringModel,
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
            
            # 应用大气散射模拟
            scattered_img, intermediate_results = scattering_model.simulate_atmospheric_scattering(image_rgb)
            
            # 保存散射效果图像 (RGB转BGR)
            if scattered_img.dtype == np.uint8:
                scattered_bgr = cv2.cvtColor(scattered_img, cv2.COLOR_RGB2BGR)
            else:
                scattered_bgr = cv2.cvtColor((scattered_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(output_path), scattered_bgr)
            
            # 保存中间结果
            if save_intermediates:
                base_name = output_path.stem
                output_dir = output_path.parent
                
                # 保存深度图
                if 'depth_map' in intermediate_results:
                    depth_path = output_dir / f"{base_name}_depth.png"
                    depth_map = intermediate_results['depth_map']
                    if len(depth_map.shape) == 2:
                        # 单通道深度图，使用颜色映射
                        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                        cv2.imwrite(str(depth_path), depth_colored)
                    else:
                        cv2.imwrite(str(depth_path), depth_map)
                
                # 保存透射率图
                if 'transmission_map' in intermediate_results:
                    trans_path = output_dir / f"{base_name}_transmission.png"
                    trans_map = intermediate_results['transmission_map']
                    if trans_map.shape[2] == 3:
                        trans_bgr = cv2.cvtColor(trans_map, cv2.COLOR_RGB2BGR)
                    else:
                        trans_bgr = trans_map
                    cv2.imwrite(str(trans_path), trans_bgr)
            
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
            "model": "Enhanced Atmospheric Scattering Physical Model",
            "formula": "I(x) = J(x) * t(x) + A * (1 - t(x)), where t(x) = e^(-β*d(x))",
            "features": [
                "Physical-based scattering simulation",
                "Wavelength-dependent scattering (Rayleigh + Mie)",
                "Multiple scattering types (haze, fog, smog, dust)",
                "Depth estimation from image",
                "Color shift modeling",
                "Atmospheric light simulation"
            ]
        }
        
        info_path = output_root / "processing_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def process_dataset(self,
                       preset: str = "moderate_haze",
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
        
        # 创建增强版大气散射模型
        scattering_model = EnhancedAtmosphericScatteringModel(
            beta=preset_params["beta"],
            atmospheric_light=preset_params["atmospheric_light"],
            max_distance=preset_params["max_distance"],
            scattering_type=preset_params["scattering_type"],
            wavelength_effect=preset_params["wavelength_effect"],
            particle_size=preset_params["particle_size"],
            depth_estimation_method=preset_params["depth_estimation_method"],
            use_real_depth=False,
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
        print(f"\n🌫️ 增强版大气散射物理模型处理")
        print(f"📂 源目录: {self.r2r_root.absolute()}")
        print(f"📂 输出目录: {output_root.absolute()}")
        print(f"🎨 预设: {preset} - {preset_params['description']}")
        print(f"📊 参数: β={preset_params['beta']:.2f}, 最大距离={preset_params['max_distance']:.0f}m")
        print(f"🌈 散射类型: {preset_params['scattering_type']}")
        print(f"💡 大气光: {preset_params['atmospheric_light']}")
        print(f"🔬 波长效应: {'✓' if preset_params['wavelength_effect'] else '✗'}")
        print(f"📏 深度估计: {preset_params['depth_estimation_method']}")
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
                    img_path, output_path, scattering_model, save_intermediates
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
        print(f"🎉 增强版大气散射处理完成!")
        print(f"✅ 成功处理: {total_processed} 张图片")
        if total_skipped > 0:
            print(f"⭐ 跳过已存在: {total_skipped} 张图片")
        if total_failed > 0:
            print(f"❌ 处理失败: {total_failed} 张图片")
        print(f"📍 输出位置: {output_root.absolute()}")
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
        print("\n🌫️ 可用的增强版大气散射预设:")
        print("=" * 80)
        
        for name, params in self.presets.items():
            print(f"📋 {name}")
            print(f"   描述: {params['description']}")
            print(f"   散射系数(β): {params['beta']:.3f}")
            print(f"   大气光: {params['atmospheric_light']}")
            print(f"   最大距离: {params['max_distance']:.0f}m")
            print(f"   散射类型: {params['scattering_type']}")
            print(f"   颗粒大小: {params['particle_size']:.1f}")
            print(f"   波长效应: {'✓' if params['wavelength_effect'] else '✗'}")
            print(f"   深度估计: {params['depth_estimation_method']}")
            print(f"   随机参数: {'✓' if params['random_params'] else '✗'}")
            print()


def create_atmospheric_scattering_demo():
    """创建增强版大气散射效果演示"""
    # 创建测试图像 (清晰的场景)
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # 创建一个具有深度层次的场景
    # 天空 (明亮蓝色)
    cv2.rectangle(test_image, (0, 0), (400, 150), (135, 206, 235), -1)
    
    # 远山 (深色)
    points = np.array([[0, 150], [100, 120], [200, 130], [300, 110], [400, 140], [400, 150], [0, 150]])
    cv2.fillPoly(test_image, [points], (100, 100, 120))
    
    # 中景建筑物
    cv2.rectangle(test_image, (50, 140), (120, 200), (150, 150, 150), -1)
    cv2.rectangle(test_image, (180, 130), (280, 210), (160, 160, 180), -1)
    
    # 近景树木
    cv2.rectangle(test_image, (320, 160), (335, 220), (101, 67, 33), -1)  # 树干
    cv2.circle(test_image, (327, 150), 25, (34, 139, 34), -1)  # 树冠
    
    # 前景地面
    cv2.rectangle(test_image, (0, 200), (400, 300), (90, 150, 90), -1)
    
    # 添加一些细节
    # 窗户
    for i in range(3):
        for j in range(2):
            cv2.rectangle(test_image, (60 + i*20, 150 + j*20), (70 + i*20, 160 + j*20), (200, 200, 200), -1)
            cv2.rectangle(test_image, (200 + i*25, 145 + j*20), (215 + i*25, 155 + j*20), (200, 200, 200), -1)
    
    # 预设效果演示
    presets_demo = ["light_haze", "moderate_haze", "heavy_haze", 
                   "thick_fog", "urban_smog", "dust_storm"]
    processor = R2RAtmosphericScatteringProcessor()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, preset_name in enumerate(presets_demo):
        params = processor.presets[preset_name]
        
        # 创建增强版大气散射模型
        model = EnhancedAtmosphericScatteringModel(
            beta=params["beta"],
            atmospheric_light=params["atmospheric_light"],
            max_distance=params["max_distance"],
            scattering_type=params["scattering_type"],
            wavelength_effect=params["wavelength_effect"],
            particle_size=params["particle_size"],
            depth_estimation_method=params["depth_estimation_method"],
            use_real_depth=False,
            random_params=False
        )
        
        # 应用散射效果
        scattered_img, _ = model.simulate_atmospheric_scattering(test_image)
        
        # 显示结果
        axes[i].imshow(scattered_img)
        title = f'{preset_name}\n'
        title += f'β={params["beta"]:.2f}, d_max={params["max_distance"]:.0f}m\n'
        title += f'Type: {params["scattering_type"]}'
        if params["wavelength_effect"]:
            title += ', WL✓'
        
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('增强版大气散射物理模型 - 效果预览\nI(x) = J(x) * t(x) + A * (1 - t(x)), t(x) = e^(-β*d(x))', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('atmospheric_scattering_presets.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 增强版大气散射预设效果演示图已保存: atmospheric_scattering_presets.png")


def create_visibility_comparison():
    """创建不同能见度对比演示"""
    # 创建测试图像
    test_image = np.zeros((200, 800, 3), dtype=np.uint8)
    
    # 创建渐变场景（从近到远）
    for x in range(800):
        depth = x / 800.0
        color = int(255 * (1 - depth * 0.3))
        cv2.line(test_image, (x, 0), (x, 200), (color, color, color), 1)
    
    # 添加深度标记
    for i in range(5):
        x = i * 200
        cv2.rectangle(test_image, (x - 20, 80), (x + 20, 120), (200, 100, 100), -1)
        cv2.circle(test_image, (x, 100), 30, (100, 200, 100), -1)
    
    # 不同能见度的对比
    visibility_distances = [200, 100, 50, 30, 20, 10]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    
    for i, visibility in enumerate(visibility_distances):
        # 根据能见度计算beta (能见度 ≈ 3.912/beta)
        beta = 3.912 / visibility
        
        # 创建模型
        model = EnhancedAtmosphericScatteringModel(
            beta=beta,
            atmospheric_light=[0.8, 0.8, 0.85],
            max_distance=visibility * 2,
            scattering_type="haze",
            wavelength_effect=True,
            particle_size=1.5,
            depth_estimation_method="simple"
        )
        
        # 应用散射效果
        scattered_img, _ = model.simulate_atmospheric_scattering(test_image)
        
        # 显示结果
        axes[i].imshow(scattered_img)
        axes[i].set_title(f'能见度: {visibility}m\nβ={beta:.3f}', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('大气散射模型 - 不同能见度对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visibility_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 能见度对比图已保存: visibility_comparison.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="增强版大气散射物理模型 - R2R数据集批处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌫️ 基于物理散射模型: I(x) = J(x) * t(x) + A * (1 - t(x)), t(x) = e^(-β*d(x))
   增强特性:
   - 🌈 波长相关散射 (瑞利散射 + 米氏散射)
   - 🎨 多种散射类型 (雾霾、浓雾、烟尘、沙尘)
   - 📏 深度估计算法 (简单/梯度)
   - 💡 大气光建模
   - 🎲 随机参数生成

💡 使用示例:
   # 处理所有场景 - 中度雾霾效果
   python script.py --process-all --preset moderate_haze
   
   # 重度雾霾 + 保存中间结果
   python script.py --process-all --preset heavy_haze --save-intermediates
   
   # 城市雾霾效果
   python script.py --process-all --preset urban_smog
   
   # 沙尘暴效果
   python script.py --process-all --preset dust_storm
   
   # 随机参数效果 (每张图片不同参数)
   python script.py --process-all --preset random_scattering
   
   # 限制处理数量 (测试用)
   python script.py --process-all --preset moderate_haze --max-scenes 2 --max-images 5
   
   # 查看所有预设
   python script.py --show-presets
   
   # 创建效果演示
   python script.py --demo
   
   # 创建能见度对比
   python script.py --visibility-comparison
        """
    )
    
    # 主要操作参数
    parser.add_argument("--process-all", action="store_true", 
                       help="处理所有场景")
    parser.add_argument("--preset", "-p", type=str, default="clear",
                       choices=["clear", "light_haze", "moderate_haze", "heavy_haze", 
                               "thick_fog", "urban_smog", "dust_storm", "severe_pollution",
                               "morning_mist", "evening_haze", "random_scattering"],
                       help="增强版大气散射预设")
    
    # 限制参数
    parser.add_argument("--max-scenes", type=int, 
                       help="最大处理场景数 (用于测试)")
    parser.add_argument("--max-images", type=int,
                       help="每个场景最大图片数 (用于测试)")
    
    # 输出控制
    parser.add_argument("--save-intermediates", action="store_true",
                       help="保存中间结果 (深度图、透射率图)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="跳过已存在的文件")
    
    # 信息查看
    parser.add_argument("--list-scenes", action="store_true",
                       help="列出所有场景")
    parser.add_argument("--show-presets", action="store_true",
                       help="显示所有预设参数")
    parser.add_argument("--demo", action="store_true",
                       help="创建效果演示图")
    parser.add_argument("--visibility-comparison", action="store_true",
                       help="创建能见度对比图")
    
    # R2R路径
    parser.add_argument("--r2r-path", type=str, default="./R2R_ULsKaCPVFJR",
                       help="R2R数据集路径 (默认: ./R2R)")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = R2RAtmosphericScatteringProcessor(args.r2r_path)
    
    # 执行相应操作
    if args.demo:
        create_atmospheric_scattering_demo()
    elif args.visibility_comparison:
        create_visibility_comparison()
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