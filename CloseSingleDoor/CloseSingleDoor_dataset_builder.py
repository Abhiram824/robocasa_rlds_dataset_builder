from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py
import json

class CloseSingleDoor(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'robot0_agentview_left_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main left camera RGB observation.',
                        ),
                        'robot0_agentview_right_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main right camera RGB observation.',
                        ),
                        'robot0_eye_in_hand_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'robot0_eef_pos': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float64,
                            doc='End effector position.',
                        ),
                        'robot0_eef_quat': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float64,
                            doc='End effector quaternion.',
                        ),
                        'robot0_gripper_qpos': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float64,
                            doc='Gripper joint positions.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(12,),
                        dtype=np.float64,
                        doc='Robot action, 12 dimensional.',
                    ),
                    'action_abs': tfds.features.Tensor(
                        shape=(12,),
                        dtype=np.float64,
                        doc='Absolute robot action, 12 dimensional.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'mujoco_state': tfds.features.Tensor(
                        shape=(None,),
                        dtype=np.float64,
                        doc='Mujoco state.'
                    ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'xml_string': tfds.features.Text(  
                        doc='mujoco XML string.'
                    ),
                'ep_meta': tfds.features.Text(
                    doc='Episode metadata.'
                ),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/home/abhim/robocasa/robocasa-dev/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(demo, num):

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            ep_len = demo["actions"].shape[0]
            for i in range(ep_len):
                # get the ep_meta attribute from data/demo

                ep_meta = json.loads(demo.attrs['ep_meta'])
                lang = ep_meta['lang']
                # compute Kona language embedding
                language_embedding =  np.zeros(512) #self._embed([lang])[0].numpy()

                obs = demo['obs']
                episode.append({
                    'observation': {
                        'robot0_agentview_left_image': obs['robot0_agentview_left_image'][i],
                        'robot0_agentview_right_image': obs['robot0_agentview_right_image'][i],
                        'robot0_eye_in_hand_image': obs['robot0_eye_in_hand_image'][i],
                        'robot0_eef_pos': np.array(obs['robot0_eef_pos'][i]),
                        'robot0_eef_quat': np.array(obs['robot0_eef_quat'][i]),
                        'robot0_gripper_qpos': np.array(obs['robot0_gripper_qpos'][i]),
                    },
                    'action': demo['actions'][i],
                    'action_abs': demo['actions_abs'][i],
                    'discount': 1.0,
                    'reward': demo['rewards'][i],
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': demo['dones'][i],
                    'language_instruction': lang,
                    'mujoco_state': demo['states'][i],
                })

            # create output data sample
            sample = {
                'steps': episode,
                'xml_string': demo.attrs['model_file'],
                'ep_meta': demo.attrs['ep_meta'],
            }

            # if you want to skip an example for whatever reason, simply return None
            return f"demo_{num}", sample

        # create list of all examples
        f = h5py.File(path, 'r')
        data = f['data']
        demos = sorted(list(f["data"].keys()))
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        # for smallish datasets, use single-thread parsing
        for i, demo in enumerate(demos):
            yield _parse_example(data[demo], i)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

