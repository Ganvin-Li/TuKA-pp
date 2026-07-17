import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import habitat-sim first, then the custom noise models
import habitat_sim
from habitat_extensions import AtmosphericScatteringNoiseModel, AtmosphericScatteringPresets
from habitat_extensions import LowLightNoiseModel, LowLightPresets
from habitat_extensions import OverexposureNoiseModel, OverexposurePresets

# Verify the registrations and register the noise models manually
try:
    if "AtmosphericScatteringNoiseModel" not in habitat_sim.registry._get_noise_model_registry():
        habitat_sim.registry.register_noise_model(AtmosphericScatteringNoiseModel)
    if "LowLightNoiseModel" not in habitat_sim.registry._get_noise_model_registry():
        habitat_sim.registry.register_noise_model(LowLightNoiseModel)
    if "OverexposureNoiseModel" not in habitat_sim.registry._get_noise_model_registry():
        habitat_sim.registry.register_noise_model(OverexposureNoiseModel)
except:
    pass

import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
# from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN

import torch.nn as nn
from pathlib import Path

class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        tokenizer: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            # self.config.habitat.task.measurements.success.success_distance=3.0
            self.config.habitat.dataset.split = self.split
            # Optional --episodes_path override: per-task eval scripts point
            # this at data/task/Task_<k>/val_seen.json.gz so habitat ignores
            # the yaml's data_path entirely. Skip the {split} placeholder
            # in that case -- the path is already concrete.
            _eps = getattr(args, "episodes_path", None)
            if _eps:
                self.config.habitat.dataset.data_path = _eps
                print(f"[eval] --episodes_path override: {_eps}")
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        # Fog-of-war off: with visibility_dist=5 m, scans the
                        # agent only briefly visits (e.g. Tasks 2/6/15/18/22
                        # whose scans appear once in the schedule) end up
                        # with most of the map hidden, giving the impression
                        # the floor plan "did not finish loading". Drawing
                        # the full plan + the agent's path (still green)
                        # produces the expected look for every task.
                        fog_of_war=FogOfWarConfig(
                            draw=False,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

        # TuKA++ (5D Tucker) hard routing over the (scene, env, instr) triple.
        # At inference the factor indices are resolved to the U3/U4/U5 rows that
        # were remembered during training (paper Sec. 4.4 / Algorithm 2).
        self.scene_idx = getattr(args, 'scene_idx', None)
        self.env_idx = getattr(args, 'env_idx', None)
        self.instr_idx = getattr(args, 'instr_idx', None)
        if getattr(model, 'is_tucker_5d', False):
            self.set_model_hard_route_5d(
                model,
                self.scene_idx if self.scene_idx is not None else 0,
                self.env_idx if self.env_idx is not None else 0,
                self.instr_idx if self.instr_idx is not None else 0,
            )

    def set_model_hard_route_5d(self, model, scene_idx, env_idx, instr_idx):
        """
        Resolve original (scene_idx, env_idx, instr_idx) ids to the row
        indices remembered by the TaskExpansionManager during training, then
        broadcast those rows to every Tucker5DLoRALinear adapter.
        """
        mgr_state = getattr(model, 'tucker_5d_manager_state', {}) or {}
        scene_map = {int(k): v for k, v in mgr_state.get('scene_row', {}).items()}
        env_map = {int(k): v for k, v in mgr_state.get('env_row', {}).items()}
        instr_map = {int(k): v for k, v in mgr_state.get('instr_row', {}).items()}

        # Zero-shot / unseen-category fallback. Inference-only tasks (24-30)
        # may reference a scan/env/instr that was NEVER trained -> it has no
        # U3/U4/U5 row. The old code did scene_map.get(scene_idx, scene_idx),
        # which passed the raw id through as a row index and then indexed a
        # non-existent row (IndexError) or, worse, silently applied the WRONG
        # task's adapter. Instead, if any of the three categories is unseen we
        # route to None on every 5D layer, so forward() returns base_out
        # (delta = 0) -- the model cleanly falls back to the untuned base
        # network for this task. This is the agreed behaviour for e.g. Task 25
        # (scan kEZ7cmS4wCh, never trained).
        unseen = []
        if scene_idx not in scene_map:
            unseen.append(f"scene={scene_idx}")
        if env_idx not in env_map:
            unseen.append(f"env={env_idx}")
        if instr_idx not in instr_map:
            unseen.append(f"instr={instr_idx}")

        if unseen:
            if hasattr(model, 'set_tucker5d_route'):
                model.set_tucker5d_route(None, None, None)
            else:
                from streamvln.model.tucker_5d_lora_layers import set_active_route_all
                set_active_route_all(model, None, None, None)
            print(f"[TuKA-5D] UNSEEN category ({', '.join(unseen)}) -> routing to "
                  f"BASE model for this task (adapter delta = 0, no 5D adaptation)")
            return

        s_row = scene_map[scene_idx]
        e_row = env_map[env_idx]
        p_row = instr_map[instr_idx]

        if hasattr(model, 'set_tucker5d_route'):
            model.set_tucker5d_route(s_row, e_row, p_row)
        else:
            from streamvln.model.tucker_5d_lora_layers import set_active_route_all
            set_active_route_all(model, s_row, e_row, p_row)

        print(f"[TuKA-5D] Hard routing set: scene={scene_idx}(row={s_row})  "
              f"env={env_idx}(row={e_row})  instr={instr_idx}(row={p_row})")

    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        target_height = self.image_processor.crop_size['height']  # 384
        target_width  = self.image_processor.crop_size['width']  # 384
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
    
        return img, (target_width, target_height)
    
    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        # ma = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def _compose_vis_frame(self, rgb, info, panel: int = 512, pad_value: int = 255):
        """Build ONE demo video frame with a fixed, uniform layout:
            [ first-person RGB | full top-down map ]
        Both panels are letterboxed (fit-preserving-aspect, never cropped) into
        equal `panel x panel` squares on a fixed `panel x 2*panel` canvas, so
        every saved frame is identical in size/proportion and the ENTIRE map is
        always visible with padding on the sides.

        Replaces habitat's observations_to_image(), which fit the map to the RGB
        HEIGHT and h-concatenated -> variable frame width per episode + the map
        getting visually cropped/inconsistent across tasks.

        pad_value: 255 -> white borders, 0 -> black borders.
        """
        import cv2

        def _fit_square(img):
            img = np.ascontiguousarray(img).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            h, w = img.shape[:2]
            s = min(panel / float(h), panel / float(w))
            nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.full((panel, panel, 3), pad_value, dtype=np.uint8)
            y0, x0 = (panel - nh) // 2, (panel - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
            return canvas

        left = _fit_square(rgb)

        td = info.get("top_down_map", None)
        if td is not None:
            # Colorize the FULL navigable map (fog-of-war off -> whole floor),
            # draw the agent triangle, then letterbox the complete map.
            top_down_map = maps.colorize_topdown_map(td["map"], td.get("fog_of_war_mask"))
            agent_radius = max(4, min(top_down_map.shape[:2]) // 32)
            top_down_map = maps.draw_agent(
                image=top_down_map,
                agent_center_coord=td["agent_map_coord"],
                agent_rotation=td["agent_angle"],
                agent_radius_px=agent_radius,
            )
            right = _fit_square(top_down_map)
        else:
            right = np.full((panel, panel, 3), pad_value, dtype=np.uint8)

        return np.concatenate([left, right], axis=1)

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        res = json.loads(line)
                    except json.JSONDecodeError:
                        # tolerate stray malformed lines from earlier interrupted runs
                        continue
                    # Per-episode records have "scene_id" + "episode_id".
                    # The end-of-run aggregate line ({"sucs_all": ..., "length": ...})
                    # does NOT and used to crash this resume path with KeyError.
                    # Skip any line that isn't a per-episode record.
                    if "scene_id" not in res or "episode_id" not in res:
                        continue
                    done_res.append([res["scene_id"], res["episode_id"], res.get("episode_instruction", "")])
                    if get_rank() == 0:
                        # Use .get with safe defaults in case earlier code wrote
                        # partial records during a crash.
                        sucs.append(res.get('success', 0.0))
                        spls.append(res.get('spl', 0.0))
                        oss.append(res.get('os', 0.0))
                        ones.append(res.get('ne', 0.0))
                        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()
                
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []
                step_id = 0
                
                # Pre-create the per-episode video directory so the .mp4
                # lands INSIDE it instead of sitting alongside an empty
                # placeholder dir (the previous behaviour produced both
                # vis_<ep>/scene_ep/ AND vis_<ep>/scene_ep.mp4 -- confusing).
                _video_dir = os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}')
                if self.save_video:
                    os.makedirs(_video_dir, exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                depth_list = []
                depth_images_list = []
                pose_list = []
                intrinsic_list = []
                time_ids = []
                action_seq = []
                past_key_values = None
                output_ids = None
                while not env.episode_over:
                    self.model.eval()
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height # Habitat GPS makes west negative, so flip y
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                    
                    rotation = agent_state.rotation
                    translation = agent_state.position
                    rotation_matrix = quaternion.as_rotation_matrix(rotation)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = translation
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    image_size = image.size
                    # image = self.image_processor.preprocess(images=image, do_rescale=True, do_normalize=True, return_tensors='pt')['pixel_values'][0]
                    image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                    depth_image, resize_shape = self.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)
                    
                    intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                    intrinsic = torch.from_numpy(intrinsic).float()
    
                    rgb_list.append(image)
                    depth_list.append(torch.from_numpy(depth_image).float())
                    pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
                    intrinsic_list.append(intrinsic)
                    
                    info = env.get_metrics()
                    if info.get('top_down_map') is not None:
                        # Fixed-layout demo frame: [ FPV | full top-down map ],
                        # uniform size + complete map + side padding.
                        frame = self._compose_vis_frame(observations['rgb'], info)
                        vis_frames.append(frame)
                    # import ipdb; ipdb.set_trace()
                    if len(action_seq) == 0:
                        if output_ids is None:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            if step_id != 0 :
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)
                            add_system = True
                            print(step_id, sources[0]["value"])
                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                            
                        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)

                        images = rgb_list[-1:]
                        depths = depth_list[-1:]
                        poses = pose_list[-1:]
                        intrinsics = intrinsic_list[-1:]
                        # import ipdb; ipdb.set_trace()
                        if step_id != 0 and step_id % self.num_frames == 0:
                            if self.num_history is None:
                                history_ids = slice(0, time_ids[0], self.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], (time_ids[0] // self.num_history))
                            images = rgb_list[history_ids] + images
                            depths = depth_list[history_ids] + depths
                            poses = pose_list[history_ids] + poses
                            intrinsics = intrinsic_list[history_ids] + intrinsics
                                
                        input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':torch.stack(depths).unsqueeze(0), \
                                        'poses':torch.stack(poses).unsqueeze(0), 'intrinsics':torch.stack(intrinsics).unsqueeze(0), 'inputs':input_ids, 'env_id':idx, 'time_ids':[time_ids],'task_type':[0]}
                            
                        input_dict = dict_to_cuda(input_dict, self.device)
                        
                        for key, value in input_dict.items():
                            if key in ['images', 'depths', 'poses', 'intrinsics']:
                                input_dict[key] = input_dict[key].to(torch.bfloat16)
                        
                        try:
                            outputs = self.model.generate(**input_dict, do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=past_key_values)
                            output_ids = outputs.sequences
                            past_key_values = outputs.past_key_values
                            llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                            print(llm_outputs, flush=True)
                            action_seq = self.parse_actions(llm_outputs)
                            print('actions', action_seq, flush=True)
                            if len(action_seq) == 0: ## if generated llm without Specific values
                                action_seq = [0]
                        except Exception as _gen_err:
                            # A single generation failure (e.g. memory/KV-cache
                            # desync on a runaway episode) must NOT crash the whole
                            # rank -- that kills every remaining episode of the task
                            # and NCCL-times-out the sibling ranks. Force a STOP so
                            # this episode terminates and is scored as a failure,
                            # then carry on to the next episode.
                            import traceback
                            print(f"[eval] generate() failed at step {step_id} "
                                  f"ep {scene_id}_{episode_id}: "
                                  f"{type(_gen_err).__name__}: {_gen_err}; forcing STOP",
                                  flush=True)
                            traceback.print_exc()
                            action_seq = [0]
                            output_ids = None
                            past_key_values = None
                    action = action_seq.pop(0)
                    
                    observations = env.step(action)
                    step_id += 1
                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                        
                process_bar.update(1)
                # episode_id += 1
                metrics = env.get_metrics()
                if self.save_video:
                    # Write the .mp4 INSIDE the per-episode dir so file +
                    # folder are not siblings in the same vis_<epoch>/ dir.
                    images_to_video(
                        vis_frames, _video_dir, 'video', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])

                # Backfill oracle metrics if the measure is unavailable
                if 'oracle_success' not in metrics:
                    metrics['oracle_success'] = metrics.get('success', 0)
                if 'oracle_navigation_error' not in metrics:
                    metrics['oracle_navigation_error'] = 0.0
                    
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)



    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.",add_system: bool = False):
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}
        # import ipdb; ipdb.set_trace()
        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            # import ipdb; ipdb.set_trace()
            # New version, use apply chat template
            # Build system message for each sentence
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            

            # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids,  conversations # tensor(bs x seq_len)

def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
   
def load_model_and_tokenizer(args):
    """Load the StreamVLN backbone and, if present, the TuKA++ (5D Tucker) adapter.

    TuKA++ is detected via a training snapshot (tucker5d_latest.pt). If no
    snapshot is found, the frozen base model is loaded as-is.
    """
    import json

    # TuKA++ is detected via its training snapshot file.
    tucker5d_snapshot = getattr(args, 'tucker_5d_snapshot', None)
    if tucker5d_snapshot is None:
        # Auto-probe: <model_path>/tucker_5d/tucker5d_latest.pt
        cand = os.path.join(args.model_path, 'tucker_5d', 'tucker5d_latest.pt')
        if os.path.exists(cand):
            tucker5d_snapshot = cand
    is_tucker_5d = tucker5d_snapshot is not None and os.path.exists(tucker5d_snapshot)

    print(f"Model path: {args.model_path}")
    print(f"Base model path: {args.base_model_path if hasattr(args, 'base_model_path') else 'Not specified'}")
    print(f"Is TuKA++ (5D Tucker) model: {is_tucker_5d}  (snapshot={tucker5d_snapshot})")

    if is_tucker_5d:
        print("=" * 50)
        print("LOADING TuKA++ (5D TUCKER) MODEL")
        print("=" * 50)

        base_model_path = getattr(args, 'base_model_path', None) or args.model_path
        if not base_model_path:
            raise ValueError("Base model path not specified for 5D Tucker-LoRA model.")

        print(f"Loading base model from: {base_model_path}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            model_max_length=getattr(args, 'model_max_length', 2048),
            padding_side="right",
        )
        config = transformers.AutoConfig.from_pretrained(base_model_path)
        model = StreamVLNForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        model = load_tucker5d_model(model, tucker5d_snapshot)
        model.eval()
        return model, tokenizer, config, True

    # No TuKA++ snapshot found -> load the frozen base model as-is.
    print("=" * 50)
    print("LOADING BASE MODEL")
    print("=" * 50)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=getattr(args, 'model_max_length', 2048),
        padding_side="right"
    )

    config = transformers.AutoConfig.from_pretrained(args.model_path)

    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
        local_files_only=True
    )

    print("Base model loaded successfully!")
    model.eval()

    return model, tokenizer, config, False

def load_tucker5d_model(model, snapshot_path: str, target_modules=None):
    """
    Load a 5D Tucker-LoRA snapshot (produced by streamvln_train.py's
    train_with_continual_learning_5d) into the given model. Replaces target
    Linear modules with Tucker5DLoRALinear adapters, grows factors to the
    saved ranks, and loads the saved factor tensors.
    """
    from streamvln.model.tucker_5d_lora_layers import Tucker5DLoRALinear

    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj']

    ckpt = torch.load(snapshot_path, map_location='cpu')
    saved_layers = ckpt["layers"]
    manager_state = ckpt.get("manager_state", {})
    model_dtype = next(model.parameters()).dtype

    # CRITICAL: the layer must be constructed with the INITIAL ranks (matching
    # training-time __init__) so r1_initial ends up = the training-time value.
    # If we constructed with the FINAL saved ranks, r1_initial would be ~52
    # instead of 16, and scaling = lora_alpha / r1_initial would be wrong by
    # ~3x, garbling the LM output and producing step=1 / unparseable text on
    # the most-trained tasks.
    #
    # We prefer r1_initial saved in the snapshot (added by the latest training
    # code); for older snapshots without it we fall back to the conventional
    # initial ranks (16,16,8,8,4) which has always been the StreamVLN default.
    any_key = next(iter(saved_layers))
    first = saved_layers[any_key]
    init_r1 = first.get("r1_initial", 16)
    init_ranks = (init_r1, init_r1, 8, 8, 4)

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        if name not in saved_layers:
            continue

        *parent_path, layer_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Step 1: Create adapter at INITIAL ranks (NOT saved ranks). This
        # sets r1_initial = the training-time initial r1, so scaling matches.
        s = saved_layers[name]
        adapter = Tucker5DLoRALinear(
            module, ranks=init_ranks, lora_alpha=32, lora_dropout=0.0, init_std=0.0
        )
        # Step 2: expand to saved (final) ranks so the parameter shapes match.
        delta_r1 = s["r1"] - adapter.lora_layer.r1
        delta_r2 = s["r2"] - adapter.lora_layer.r2
        delta_r3 = s["r3"] - adapter.lora_layer.r3
        delta_r4 = s["r4"] - adapter.lora_layer.r4
        delta_r5 = s["r5"] - adapter.lora_layer.r5
        if delta_r1 > 0 or delta_r2 > 0 or delta_r3 > 0 or delta_r4 > 0 or delta_r5 > 0:
            adapter.lora_layer.expand(
                max(delta_r1, 0), max(delta_r2, 0),
                max(delta_r3, 0), max(delta_r4, 0), max(delta_r5, 0),
            )
        # Append empty rows to U3/U4/U5 to match saved shapes
        for axis, key in (("U3", "U3"), ("U4", "U4"), ("U5", "U5")):
            want = s[key].shape[0]
            while getattr(adapter.lora_layer, axis).shape[0] < want:
                adapter.lora_layer.append_category_row(axis, init="zeros")

        with torch.no_grad():
            adapter.lora_layer.U1.data = s["U1"].to(model_dtype)
            adapter.lora_layer.U2.data = s["U2"].to(model_dtype)
            adapter.lora_layer.U3.data = s["U3"].to(model_dtype)
            adapter.lora_layer.U4.data = s["U4"].to(model_dtype)
            adapter.lora_layer.U5.data = s["U5"].to(model_dtype)
            adapter.lora_layer.G.data = s["G"].to(model_dtype)
            adapter.lora_layer.frozen_r1.fill_(s["frozen_r1"])
            adapter.lora_layer.frozen_r2.fill_(s["frozen_r2"])
            adapter.lora_layer.frozen_r3.fill_(s["frozen_r3"])
            adapter.lora_layer.frozen_r4.fill_(s["frozen_r4"])
            adapter.lora_layer.frozen_r5.fill_(s["frozen_r5"])
            adapter.lora_layer.frozen_rows_U3.fill_(s["frozen_rows_U3"])
            adapter.lora_layer.frozen_rows_U4.fill_(s["frozen_rows_U4"])
            adapter.lora_layer.frozen_rows_U5.fill_(s["frozen_rows_U5"])

        setattr(parent, layer_name, adapter)
        replaced += 1

    model.is_tucker_5d = True
    model.tucker_5d_manager_state = manager_state
    print(f"[TuKA-5D] Loaded {replaced} 5D Tucker-LoRA adapters from {snapshot_path}")
    print(f"[TuKA-5D] Manager state -> scenes={list(manager_state.get('scene_row', {}).keys())}, "
          f"envs={list(manager_state.get('env_row', {}).keys())}, "
          f"instrs={list(manager_state.get('instr_row', {}).keys())}")
    return model


def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_seen')
    parser.add_argument("--episodes_path", type=str, default=None,
                        help="Optional absolute/relative path to a .json.gz "
                             "VLN-CE episodes file. When set, overrides "
                             "habitat.dataset.data_path entirely (used by "
                             "per-task eval to point at "
                             "data/task/Task_<k>/val_seen.json.gz instead of "
                             "the full data/datasets/<ds>/val_seen split).")
    parser.add_argument("--output_path", type=str, default='./results/val_seen/streamvln')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help="Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Path to the frozen StreamVLN backbone')

    # TuKA++ (5D Tucker) hard-routing over the (scene, env, instruction) triple.
    parser.add_argument('--tucker_5d_snapshot', type=str, default=None,
                        help='Path to the TuKA++ (5D Tucker) snapshot (.pt). '
                             'If omitted, looks for <model_path>/tucker_5d/tucker5d_latest.pt')
    parser.add_argument('--scene_idx', type=int, default=None,
                        help='Scene factor index s for hard routing')
    parser.add_argument('--env_idx', type=int, default=None,
                        help='Environment factor index e (0=Normal, 1=Low-light, 2=Scattering, 3=Overexposure)')
    parser.add_argument('--instr_idx', type=int, default=None,
                        help='Instruction-style factor index p (0=VLN, 1=OLN, 2=DUN)')

    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    # Load the backbone and, if a snapshot exists, the TuKA++ adapter
    model, tokenizer, config, is_lora_model = load_model_and_tokenizer(args)

    # Move to device
    model.to('cuda')
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(local_rank)

    # Report adapter status
    if is_lora_model:
        print("Verifying TuKA++ (5D Tucker) model...")
        print(f"Model type: {type(model)}")

        if getattr(model, 'is_tucker_5d', False):
            print("TuKA++ (5D Tucker) model confirmed")
            # Count TuKA++ adapter parameters (U1/U2 shared, U3/U4/U5 factors, G core)
            tucker_params = 0
            for name, module in model.named_modules():
                if hasattr(module, 'lora_layer'):
                    layer = module.lora_layer
                    for tag in ('U1', 'U2', 'U3', 'U4', 'U5', 'G'):
                        if hasattr(layer, tag):
                            tucker_params += getattr(layer, tag).numel()
            print(f"Total TuKA++ parameters: {tucker_params:,}")

        print("=" * 50)

    evaluate(model, tokenizer, args)


def evaluate(model, tokenizer, args):
    model.eval()
    
    world_size = get_world_size()
    model.reset(world_size)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()