# v0.3.0b: Minari is ready for beta testing

For this beta release Minari has experienced considerable changes from its past `v0.2.2` version. As a major refactor, the C source code and Cython dependency have been removed in favor of a pure Python API in order to reduce code complexity. If we require a more efficient API in the future we will explore the use of C.

Apart from the API changes and new features we are excited to include the first official Minari datasets which have been re-created from the D4RL project.

The documentation page at https://minari.farama.org/, has also been updated with the latest changes.

We are constantly developing this library. Please don't hesitate to open a GitHub issue or reach out to us directly. Your ideas and contributions are highly appreciated and will help shape the future of this library. Thank you for using our library!

## New Features and Improvements

### Dataset File Format

We are keeping the [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file format to store the Minari datasets. However, the internal structure of the datasets has been modified. The data is now stored in a per episode basis. Each Minari dataset has a minimum of one HDF5 file (:page_facing_up:, `main_data.hdf5`). In the dataset file,  the collected transitions are separated by episodes groups (:file_folder:) that contain 5 required datasets(:floppy_disk:) : `observations`, `actions`, `terminations`, `truncations`, and `rewards`. Other optional group and dataset collections can be inlcuded in each episode; such is the case of the `infos` step return.

![image](https://user-images.githubusercontent.com/60633730/231363871-9d6aa7a6-9256-4cca-a9dc-af74ae6095cb.png)

### MinariDataset



### Dataset Sampling (https://github.com/Farama-Foundation/Minari/pull/34)

For this release Minari doesn't provide replay buffers. The user will be in charge of creating their own replay buffers
`sample_episodes`
`iterate_episodes`
`filter_episodes`

### Dataset Creation (https://github.com/Farama-Foundation/Minari/pull/31)

We are facilitating the logging of environment data by providing a Gymnasium environment wrapper,  [`DataCollectorV0`](https://minari.farama.org/main/api/data_collector/#minari-datacollectorv0). This wrapper buffers the parameters from a Gymnasium [step](https://gymnasium.farama.org/api/env/#gymnasium.Env.step) transition. The [`DataCollectorV0`](https://minari.farama.org/main/api/data_collector/#minari-datacollectorv0) is also memory efficient by providing a step/episode scheduler to cache the recorded data. In addition, this wrapper can be initialized with two custom callbacks:

- [`StepDataCallback`](https://minari.farama.org/main/api/data_collector_callbacks/step_data_callback/) - This callback automatically flattens Dictionary or Tuple observation/action spaces.  (the observation/action spaces of the environment must be kept)

- [`EpisodeMetadataCallback`](https://minari.farama.org/main/api/data_collector_callbacks/episode_metadata_callback/) - . For now automatic metadata will be added to the rewards dataset of each episode.

To save the Minari dataset in disk two functions are provided depending on the way data was collected. If to collect the data the environment was wrapped with a [`DataCollectorV0`](https://minari.farama.org/main/api/data_collector/#minari-datacollectorv0), use [`create_dataset_from_collector_env`](https://minari.farama.org/main/api/minari_functions/#minari.create_dataset_from_collector_env). Otherwise you can collect the episode trajectories and use `create_dataset_from_buffers`

We provide a curated tutorial in the documentation on how to use these dataset creation tools: https://minari.farama.org/main/tutorials/dataset_creation/point_maze_dataset/#sphx-glr-tutorials-dataset-creation-point-maze-dataset-py



### CLI

To improve accessibility to the remote public datasets, we are also including a [CLI](https://minari.farama.org/main/content/minari_cli/) tool with commands to list, download, and upload Minari datasets.

### New Public Datasets

Bellow is a list of new available dataset ids from different Gymnasium environments. These datasets have been re-created from the original [D4RL](https://sites.google.com/view/d4rl-anonymous/) project.

#### [AdroitHandDoor-v1](https://robotics.farama.org/envs/adroit_hand/adroit_door/)

| Dataset ID |
| ---------- |
| [door-human-v0](https://minari.farama.org/main/datasets/door/human/) |
| [door-expert-v0](https://minari.farama.org/main/datasets/door/expert/) |
| [door-cloned-v0](https://minari.farama.org/main/datasets/door/cloned/) |

#### [AdroitHandHammer-v1](https://robotics.farama.org/envs/adroit_hand/adroit_hammer/)

| Dataset ID |
| ---------- |
| [hammer-human-v0](https://minari.farama.org/main/datasets/hammer/human/) |
| [hammer-expert-v0](https://minari.farama.org/main/datasets/hammer/expert/) |
| [hammer-cloned-v0](https://minari.farama.org/main/datasets/hammer/cloned/) |

#### [AdroitHandPen-v1](https://robotics.farama.org/envs/adroit_hand/adroit_pen/)

| Dataset ID |
| ---------- |
| [pen-human-v0](https://minari.farama.org/main/datasets/pen/human/) |
| [pen-expert-v0](https://minari.farama.org/main/datasets/pen/expert/) |
| [pen-cloned-v0](https://minari.farama.org/main/datasets/pen/cloned/) |

#### [AdroitHandRelocate-v1](https://robotics.farama.org/envs/adroit_hand/adroit_relocate/)

| Dataset ID |
| ---------- |
| [relocate-human-v0](https://minari.farama.org/main/datasets/relocate/human/) |
| [relocate-expert-v0](https://minari.farama.org/main/datasets/relocate/expert/) |
| [relocate-cloned-v0](https://minari.farama.org/main/datasets/relocate/cloned/) |

#### [PointMaze](https://robotics.farama.org/envs/maze/point_maze/)

| Dataset ID |
| ---------- |
| [pointmaze-umaze-v0](https://minari.farama.org/main/datasets/pointmaze/umaze/) |
| [pointmaze-umaze-dense-v0](https://minari.farama.org/main/datasets/pointmaze/umaze-dense/) |
| [pointmaze-open-v0](https://minari.farama.org/main/datasets/pointmaze/open/) |
| [pointmaze-open-dense-v0](https://minari.farama.org/main/datasets/pointmaze/open-dense/) |
| [pointmaze-medium-v0](https://minari.farama.org/main/datasets/pointmaze/medium/) |
| [pointmaze-medium-dense-v0](https://minari.farama.org/main/datasets/pointmaze/medium-dense/) |
| [pointmaze-large-v0](https://minari.farama.org/main/datasets/pointmaze/large/) |
| [pointmaze-large-dense-v0](https://minari.farama.org/main/datasets/pointmaze/large-dense/) |

#### [FrankaKitchen-v1](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/)

| Dataset ID |
| ---------- |
| [door-human-v0](https://minari.farama.org/main/datasets/door/human/) |
| [door-expert-v0](https://minari.farama.org/main/datasets/door/expert/) |
| [door-cloned-v0](https://minari.farama.org/main/datasets/door/cloned/) |