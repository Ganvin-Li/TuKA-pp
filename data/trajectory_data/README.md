---
license: cc-by-sa-4.0
language:
- en
pretty_name: StreamVLN
extra_gated_prompt: >-
  ### StreamVLN COMMUNITY LICENSE AGREEMENT StreamVLN Release Date: Augest 20,
  2025 All the data and code within this repo are under [CC BY-NC-SA
  4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
extra_gated_fields:
  First Name: text
  Last Name: text
  Email: text
  Country: country
  Affiliation: text
  Phone: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI researcher
    - AI developer/engineer
    - Reporter
    - Other
  Research interest: text
  geo: ip_location
  By clicking Submit, you accept the license terms and acknowledge that the information you provide will be collected stored and processed for academic communication and progress tracking of this research: checkbox
extra_gated_description: >-
  By clicking Submit, you accept the license terms and acknowledge that the
  information you provide will be collected, stored, and processed for academic
  communication and progress tracking of this research.
extra_gated_button_content: Submit
tags:
- navigation
size_categories:
- n>1T
---
This repo contains the data for the paper **"StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling."**

[![arxiv](https://img.shields.io/badge/arXiv_2507.05240-red?logo=arxiv)](http://arxiv.org/abs/2507.05240)
[![project](https://img.shields.io/badge/Project_Page-0065D3?logo=rocket&logoColor=white)](https://streamvln.github.io/)
[![hf](https://img.shields.io/badge/Hugging_Face-FF9D00?logo=huggingface&logoColor=white)](https://huggingface.co/papers/2507.05240/)
[![video-en](https://img.shields.io/badge/YouTube-D33846?logo=youtube)](https://www.youtube.com/watch?v=gG3mpefOBjc)


## News
[2025/09/30] For R2R, we have now removed all v1 version data and only retained the v1-3 data.

[2025/08/20] We have updated the R2R to a new version, which now includes both the v1 and v1-3 datasets. Additionally. And we have fixed the episode ID issue in RxR to ensure compatibility with the currently available RxR download links.

## Overview

The dataset consists of visual observations and annotations collected in the Matterport3D (MP3D) environment using the Habitat simulator. It combines data from several open-source Vision-and-Language Navigation (VLN) datasets.

## Data Collection

Data collected in this repo are from the following open-source datasets:
- [R2R-VLNCE](https://drive.google.com/file/d/18DCrNcpxESnps1IbXVjXSbGLDzcSOqzD/view)
- [RxR-VLNCE](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view)
- [R2R-EnvDrop](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view)

To get actions and observations, we enable a `ShortestPathFollower` agent in the Habitat simulator to follow the subgoals and collect rgb observations along the path. The data is collected across the Matterport3D (MP3D) scenes.

## Dataset Description

### Dataset Structure

After **extracting `images.tar.gz`**, the dataset has the following structure:

```shell
StreamVLN-Trajectory-Data/
├── R2R/
│   ├── images/
│   │   ├── 1LXtFkjw3qL_r2r_000087/
│   │   │   └── rgb/
│   │   │       ├── 000.jpg
│   │   │       ├── 001.jpg
│   │   │       └── ...
│   │   ├── 1LXtFkjw3qL_r2r_000099/
│   │   ├── 1LXtFkjw3qL_r2r_000129/
│   │   └── ...
│   └── annotations.json
├── RxR/
│   ├── images/
│   └── annotations.json
├── EnvDrop/
│   └── annotations.json
└── ScaleVLN/
    ├── annotations.json
    └── scalevln_subset_150k.json.gz

```

### Contents

`images/`: The folder contains the rgb observations collected from Habitat simulator. 

`annotations.json`: The file contain the navigation instructions and discrete actions sequence from Habitat Simulator for each dataset. The structure of annotation for each episode is as follows:

```python
{
    "id": (int) Identifier for the episode,
    "video": (str) Video ID to identify the relative path to the directory which contains the episode, format: "images/{scene}_{dataset_source}_{id}",
    "instruction": (list[str]) Navigation instructions,
    "actions": (list[int]) Discrete actions sequence in Habitat simulator, 
                # 1 = MoveForward (25cm)
                # 2 = TurnLeft (15°)
                # 3 = TurnRight (15°)
                # -1 = Dummy
                # 0 = Stop (omitted in annotations)
}
```

Each episode in the `annotations.json` file corresponds to a folder in the `images/` directory, where the folder name is included in the `video` ID. The rgb images are stored in the `rgb/` subdirectory of each episode folder. Length of the `actions` list corresponds to the number of rgb images in the episode to ensure observation-action data pairs.


## EnvDrop & ScaleVLN Dataset Note
For **EnvDrop** and **ScaleVLN**, only **navigation annotations** are provided due to the large number of episodes. 

Considering the discrete setting of the original **ScaleVLN** dataset, we provide the converted episodes in continuous environment settings in `StreamVLN-Trajectory-Data/ScaleVLN/scalevln_subset_150k.json.gz`. These episodes correspond to the annotations we have provided. The format of these episodes is consistent with R2R-CE. You can load these episodes in the same way as R2R/EnvDrop.

To obtain RGB observations, you can replay the annotated actions using the Habitat simulator. **Below is a example demonstrating how to replay stored actions and get EnvDrop RGB frames:**

1. Before proceeding, you need to modify the configuration file to specify the path to the **EnvDrop** episodes. Please overwrite the `habitat.dataset.data_path` in **config/vln_r2r.yaml**:

    ```yaml
    habitat:
        ...
        dataset:
            ...
            data_path: data/datasets/envdrop/envdrop.json.gz
    ```

    
2. Run the code below to save RGB images. 

    ```python
    import os
    import json
    import habitat

    from habitat_baselines.config.default import get_config
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from streamvln.habitat_extensions import measures

    CONFIG_PATH = "config/vln_r2r.yaml"  # Path to the Habitat config file
    ANNOT_PATH = "data/trajectory_data/EnvDrop/annotations.json"  # Path to the annotations file
    GOAL_RADIUS = 0.25  # Radius for the goal in meters. not used if get actions from annotations

    env = habitat.Env(config=get_config(CONFIG_PATH))
    annotations = json.load(open(ANNOT_PATH, "r"))

    for episode in env.episodes:
        env.current_episode = episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
        observation = env.reset()

        annotation = next(annot for annot in annotations if annot["id"] == int(episode.episode_id))  # Get annotation for current episode
        reference_actions = annotation["actions"][1:] + [0]  # Pop the dummy action at the beginning and add stop action at the end
        step_id = 0  # Initialize step ID

        while not env.episode_over:
            rgb = observation["rgb"]  # Get the current rgb observation
            
            # TODO: Save RGB frame (customize as needed)
            # --------------------------------------------------------
            import PIL.Image as Image
            video_id = annotation["video"]  # Get the video ID from the annotation
            rgb_dir = f"data/trajectory_data/EnvDrop/{video_id}/rgb"
            os.makedirs(rgb_dir, exist_ok=True)
            Image.fromarray(rgb).convert("RGB").save(os.path.join(rgb_dir, f"{step_id:03d}.jpg"))
            # --------------------------------------------------------

            action = reference_actions.pop(0)  # Get next action from our annotation
            observation = env.step(action)  # Update observation
            step_id += 1
    
    env.close()
    
    ```