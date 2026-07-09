#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
照片合成GIF脚本
将指定目录下的照片按001, 002...顺序合成GIF动画
"""

import os
import glob
from PIL import Image
import argparse
import sys

def create_gif_from_photos(input_dir, output_path, duration=500, resize_width=None, loop=0):
    """
    将目录下的照片合成GIF
    
    参数:
    input_dir: 输入照片目录路径
    output_path: 输出GIF文件路径
    duration: 每张照片显示时间(毫秒)
    resize_width: 调整图片宽度(可选，保持宽高比)
    loop: 循环次数(0表示无限循环)
    """
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"错误: 在目录 '{input_dir}' 中没有找到图片文件")
        return False
    
    # 按文件名排序（这样001, 002等会按正确顺序排列）
    image_files.sort()
    
    print(f"找到 {len(image_files)} 张图片")
    print("图片顺序:")
    for i, file in enumerate(image_files, 1):
        print(f"  {i:03d}: {os.path.basename(file)}")
    
    try:
        # 加载所有图片
        images = []
        for i, img_path in enumerate(image_files):
            print(f"正在处理第 {i+1}/{len(image_files)} 张图片: {os.path.basename(img_path)}")
            
            img = Image.open(img_path)
            
            # 转换为RGB模式（GIF需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 可选：调整图片大小
            if resize_width and img.width > resize_width:
                # 按比例缩放
                ratio = resize_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((resize_width, new_height), Image.Resampling.LANCZOS)
            
            images.append(img)
        
        if not images:
            print("错误: 没有成功加载任何图片")
            return False
        
        # 保存为GIF
        print(f"\n正在生成GIF: {output_path}")
        print(f"帧数: {len(images)}")
        print(f"每帧间隔: {duration}ms")
        print(f"循环: {'无限循环' if loop == 0 else f'{loop}次'}")
        
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )
        
        print(f"\n✅ GIF生成成功: {output_path}")
        
        # 显示文件大小
        file_size = os.path.getsize(output_path)
        if file_size > 1024 * 1024:
            print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")
        else:
            print(f"文件大小: {file_size / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='将目录下的照片按顺序合成GIF动画',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python photo_to_gif.py /path/to/photos output.gif
  python photo_to_gif.py /path/to/photos output.gif --duration 1000
  python photo_to_gif.py /path/to/photos output.gif --duration 500 --width 800
  python photo_to_gif.py /path/to/photos output.gif --duration 300 --loop 5

注意:
  - 图片将按文件名排序，建议使用001.jpg, 002.jpg等命名格式
  - duration单位为毫秒，默认500ms
  - 支持jpg, png, bmp, gif, tiff, webp等格式
        """
    )
    
    parser.add_argument('--input_dir', '-i', type=str, 
                    default="./trajectory_data/R2R_light_moderate_overexpose/images/5LpN3gDmAk7_r2r_000006/rgb",
                    help='包含照片的输入目录')
    parser.add_argument('--output_gif', '-o', type=str, 
                    default="./R2R_light_moderate_overexpose.gif",
                    help='输出GIF文件路径')
    parser.add_argument('--duration', '-d', type=int, default=200, 
                       help='每张照片显示时间(毫秒), 默认50ms')
    parser.add_argument('--width', '-w', type=int, 
                       help='调整图片宽度(像素), 保持宽高比')
    parser.add_argument('--loop', '-l', type=int, default=0,
                       help='循环次数, 0表示无限循环(默认)')
    
    args = parser.parse_args()
    
    # # 检查输入目录
    # if not os.path.exists(args.input_dir):
    #     print(f"错误: 输入目录不存在: {args.input_dir}")
    #     sys.exit(1)
    
    # if not os.path.isdir(args.input_dir):
    #     print(f"错误: 输入路径不是目录: {args.input_dir}")
    #     sys.exit(1)
    
    # 检查duration参数
    if args.duration <= 0:
        print("错误: duration必须大于0")
        sys.exit(1)
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output_gif)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成GIF
    success = create_gif_from_photos(
        args.input_dir, 
        args.output_gif, 
        args.duration,
        args.width,
        args.loop
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()