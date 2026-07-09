import numpy as np
from PIL import Image, ImageFilter
import trimesh
import os
from pathlib import Path
import json

class MP3DTextureDebugger:
    """用于调试MP3D GLB文件纹理提取的工具"""
    
    def __init__(self):
        pass
    
    def inspect_glb_structure(self, glb_path):
        """详细检查GLB文件的结构"""
        print(f"正在检查GLB文件: {glb_path}")
        print("="*60)
        
        try:
            scene = trimesh.load(str(glb_path))
            
            # 1. 基本信息
            print("1. 基本信息:")
            print(f"   类型: {type(scene)}")
            print(f"   是否为场景: {hasattr(scene, 'geometry')}")
            
            # 2. 几何体信息
            if hasattr(scene, 'geometry'):
                print(f"   几何体数量: {len(scene.geometry)}")
                geometries = list(scene.geometry.items())
            else:
                print("   单一几何体")
                geometries = [('single', scene)] if hasattr(scene, 'visual') else []
            
            print("\n2. 几何体详情:")
            for i, (name, geometry) in enumerate(geometries):
                print(f"   几何体 {i}: {name}")
                print(f"      类型: {type(geometry)}")
                print(f"      有视觉属性: {hasattr(geometry, 'visual')}")
                
                if hasattr(geometry, 'visual'):
                    visual = geometry.visual
                    print(f"      视觉类型: {type(visual)}")
                    print(f"      有材质: {hasattr(visual, 'material')}")
                    
                    if hasattr(visual, 'material'):
                        material = visual.material
                        print(f"      材质类型: {type(material)}")
                        
                        # 检查材质属性
                        print("      材质属性:")
                        for attr in dir(material):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(material, attr)
                                    if not callable(value):
                                        print(f"        {attr}: {type(value)} = {value}")
                                except:
                                    print(f"        {attr}: (无法访问)")
            
            # 3. 检查场景级别的纹理和材质
            print("\n3. 场景级别检查:")
            if hasattr(scene, 'materials'):
                print(f"   场景材质数量: {len(scene.materials) if scene.materials else 0}")
                if scene.materials:
                    for i, material in enumerate(scene.materials):
                        print(f"   材质 {i}: {type(material)}")
            
            if hasattr(scene, 'textures'):
                print(f"   场景纹理数量: {len(scene.textures) if scene.textures else 0}")
                if scene.textures:
                    for i, texture in enumerate(scene.textures):
                        print(f"   纹理 {i}: {type(texture)}")
            
            # 4. 尝试不同的纹理提取方法
            print("\n4. 尝试提取纹理:")
            self.try_extract_textures_multiple_ways(scene)
            
        except Exception as e:
            print(f"检查失败: {e}")
    
    def try_extract_textures_multiple_ways(self, scene):
        """尝试多种方法提取纹理"""
        
        methods = [
            self.method_1_base_color_texture,
            self.method_2_material_textures,
            self.method_3_visual_textures,
            self.method_4_gltf_materials,
            self.method_5_image_data
        ]
        
        for i, method in enumerate(methods, 1):
            print(f"\n   方法 {i}: {method.__name__}")
            try:
                textures = method(scene)
                print(f"      找到 {len(textures)} 个纹理")
                for j, texture_info in enumerate(textures[:3]):  # 只显示前3个
                    print(f"      纹理 {j}: {texture_info}")
            except Exception as e:
                print(f"      失败: {e}")
    
    def method_1_base_color_texture(self, scene):
        """原始方法：baseColorTexture"""
        textures = []
        if hasattr(scene, 'geometry'):
            geometries = scene.geometry.values()
        else:
            geometries = [scene] if hasattr(scene, 'visual') else []
        
        for geometry in geometries:
            if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                material = geometry.visual.material
                if hasattr(material, 'baseColorTexture') and material.baseColorTexture:
                    texture = material.baseColorTexture
                    if hasattr(texture, 'data') and texture.data is not None:
                        textures.append({
                            'source': 'baseColorTexture',
                            'shape': texture.data.shape,
                            'dtype': texture.data.dtype
                        })
        return textures
    
    def method_2_material_textures(self, scene):
        """方法2：检查材质的所有纹理属性"""
        textures = []
        if hasattr(scene, 'geometry'):
            geometries = scene.geometry.values()
        else:
            geometries = [scene] if hasattr(scene, 'visual') else []
        
        for geometry in geometries:
            if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                material = geometry.visual.material
                
                # 检查常见的纹理属性
                texture_attrs = ['baseColorTexture', 'diffuseTexture', 'texture', 
                               'normalTexture', 'metallicRoughnessTexture', 'emissiveTexture']
                
                for attr in texture_attrs:
                    if hasattr(material, attr):
                        texture = getattr(material, attr)
                        if texture and hasattr(texture, 'data') and texture.data is not None:
                            textures.append({
                                'source': attr,
                                'shape': texture.data.shape,
                                'dtype': texture.data.dtype
                            })
        return textures
    
    def method_3_visual_textures(self, scene):
        """方法3：直接从visual对象获取纹理"""
        textures = []
        if hasattr(scene, 'geometry'):
            geometries = scene.geometry.values()
        else:
            geometries = [scene] if hasattr(scene, 'visual') else []
        
        for geometry in geometries:
            if hasattr(geometry, 'visual'):
                visual = geometry.visual
                
                # 检查visual对象的各种属性
                if hasattr(visual, 'uv') and visual.uv is not None:
                    textures.append({
                        'source': 'visual.uv',
                        'shape': visual.uv.shape,
                        'type': 'UV坐标'
                    })
                
                if hasattr(visual, 'diffuse') and visual.diffuse is not None:
                    textures.append({
                        'source': 'visual.diffuse',
                        'data': visual.diffuse,
                        'type': '漫反射颜色'
                    })
        return textures
    
    def method_4_gltf_materials(self, scene):
        """方法4：检查GLTF格式的材质"""
        textures = []
        
        # 检查场景级别的材质
        if hasattr(scene, 'materials') and scene.materials:
            for material in scene.materials:
                if hasattr(material, 'pbrMetallicRoughness'):
                    pbr = material.pbrMetallicRoughness
                    if hasattr(pbr, 'baseColorTexture'):
                        textures.append({
                            'source': 'GLTF PBR baseColorTexture',
                            'material': str(material)
                        })
        
        return textures
    
    def method_5_image_data(self, scene):
        """方法5：查找图像数据"""
        textures = []
        
        # 递归搜索所有属性中的图像数据
        def find_image_data(obj, path=""):
            if isinstance(obj, np.ndarray):
                if len(obj.shape) >= 2:  # 可能是图像
                    textures.append({
                        'source': f'numpy_array at {path}',
                        'shape': obj.shape,
                        'dtype': obj.dtype
                    })
            elif hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    if not attr_name.startswith('_'):
                        find_image_data(attr_value, f"{path}.{attr_name}")
        
        find_image_data(scene, "scene")
        return textures
    
    def extract_and_save_found_textures(self, glb_path, output_dir):
        """提取并保存找到的纹理"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n尝试从 {glb_path} 提取并保存纹理...")
        
        try:
            scene = trimesh.load(str(glb_path))
            
            # 尝试所有方法
            all_textures = []
            
            # 方法1: 标准方式
            if hasattr(scene, 'geometry'):
                geometries = list(scene.geometry.items())
            else:
                geometries = [('single', scene)] if hasattr(scene, 'visual') else []
            
            for geom_name, geometry in geometries:
                if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                    material = geometry.visual.material
                    
                    # 尝试多种纹理属性
                    texture_attrs = ['baseColorTexture', 'diffuseTexture', 'texture',
                                   'normalTexture', 'metallicRoughnessTexture', 'emissiveTexture']
                    
                    for attr_name in texture_attrs:
                        if hasattr(material, attr_name):
                            texture = getattr(material, attr_name)
                            if texture and hasattr(texture, 'data') and texture.data is not None:
                                try:
                                    # 保存纹理
                                    if texture.data.dtype != np.uint8:
                                        texture_data = np.clip(texture.data * 255, 0, 255).astype(np.uint8)
                                    else:
                                        texture_data = texture.data
                                    
                                    img = Image.fromarray(texture_data)
                                    filename = f"{geom_name}_{attr_name}.png"
                                    img.save(output_dir / filename)
                                    
                                    all_textures.append({
                                        'geometry': geom_name,
                                        'attribute': attr_name,
                                        'filename': filename,
                                        'shape': texture.data.shape
                                    })
                                    
                                    print(f"保存纹理: {filename}, 尺寸: {texture.data.shape}")
                                    
                                except Exception as e:
                                    print(f"保存纹理失败 {attr_name}: {e}")
            
            if not all_textures:
                print("未找到任何可保存的纹理数据")
            else:
                print(f"总共保存了 {len(all_textures)} 个纹理")
                
            return all_textures
            
        except Exception as e:
            print(f"提取纹理失败: {e}")
            return []

def debug_mp3d_scene(scene_name, mp3d_dir="mp3d"):
    """调试特定的MP3D场景"""
    debugger = MP3DTextureDebugger()
    
    scene_dir = Path(mp3d_dir) / scene_name
    glb_files = list(scene_dir.glob("*.glb"))
    
    if not glb_files:
        print(f"在 {scene_dir} 中未找到GLB文件")
        return
    
    glb_file = glb_files[0]
    print(f"调试场景: {scene_name}")
    print(f"GLB文件: {glb_file}")
    
    # 检查结构
    debugger.inspect_glb_structure(glb_file)
    
    # 尝试提取纹理
    output_dir = f"debug_textures_{scene_name}"
    textures = debugger.extract_and_save_found_textures(glb_file, output_dir)
    
    return textures

# 使用示例
if __name__ == "__main__":
    # 调试第一个场景
    scene_name = "1LXtFkjw3qL"  # 根据你的目录结构调整
    debug_mp3d_scene(scene_name)