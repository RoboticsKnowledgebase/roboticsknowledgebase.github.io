---
title: Traffic Modelling Datasets
published: true
---

Traffic modelling is a hot topic in the field of autonomous cars currently. Here you will find a list of open datasets which can be used as source for building a traffic model. The list will include data captured from a variety of sources as listed below:
- UAV/Drones
- Traffic Camera 
- Autonomous Cars

> This is not a list of datasets for learning-to-drive. This list is more focused towards dataset which provide global perspective of the traffic scenarios, rather than ego-vehicle perspective. Though few ego-vehicle datasets can be used for traffic modelling as well.

Each dataset is scored based upon several parameters crucial for learning behavior and interactions between vehicles in a recorded scenario.

| S.No. | Data Source | Road                     | Lane Boundary            | Vehicle | Pedestrian | Traffic Light                             |
|-------|-------------|--------------------------|--------------------------|---------|------------|-------------------------------------------|
| 1     | Argoverse   | Yes                      | Yes                      | Yes     | Yes        | No GT, can be extracted from front camera |
|       |             |                          |                          |         |            |                                           |
| 2     | Interaction | Yes                      | Yes                      | Yes     | No         | No                                        |
|       |             |                          |                          |         |            |                                           |
| 3     | In-D        | Yes                      | Yes                      | Yes     | Yes        | No                                        |
|       |             |                          |                          |         |            |                                           |
| 4     | NGSIM       | NoNeeds to be hardcoded  | NoNeeds to be hardcoded  | Yes     | No         | Yes                                       |
|       |             |                          |                          |         |            |                                           |
| 5     | Stanford    | NoNeeds to be hard-coded | NoNeeds to be hard-coded | Yes     | Yes        | No                                        |
|       |             |                          |                          |         |            |                                           |
| 6     | NuScenes    | Yes                      | Yes                      | Yes     | Yes        | No(position available from Map layers)    |
|       |             |                          |                          |         |            |                                           |
| 7     | Apollo      | No                       | No                       | Yes     | Yes        | No                                        |
