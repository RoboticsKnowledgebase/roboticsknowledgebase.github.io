---
date: 2026-04-27
title: Running SLAM in Production - A Practitioner's Guide
author: Shreenabh Agrawal
---

## Introduction

You are about to deploy SLAM on a robot, an AR/VR headset, or anything else that needs 6DoF pose estimation in the wild. The thing that breaks production deployments is rarely the algorithm itself. It is the choice of algorithm, and the failure modes nobody warned you about. The picking question narrows to one or two viable systems for a given sensor and compute envelope. Once you have one, the actual deployment risk is whatever the system does silently when you push it past its design envelope. This guide is for the most common production case, monocular feature-based SLAM via ORB-SLAM3. The smaller section at the end is on running evaluation loops locally if defaults are not enough.

## Choosing the right SLAM stack

Visual SLAM splits into a few algorithmic families. The default for most production deployments is feature-based: detect keypoints, match across frames, run bundle adjustment over the resulting graph. ORB-SLAM3 sits in this family. Direct methods like DSO minimize photometric error on raw pixel intensities. They recover more pose information in low-texture scenes than feature-based methods can, but they fall apart under exposure changes. Learned methods (DROID-SLAM, DPVO) are newer and handle some failure cases classical SLAM cannot, at the cost of a GPU at inference time. On top of that, you choose a sensor combination: monocular (one camera, no metric scale), stereo (metric scale, more compute), or visual-inertial (camera plus IMU, hardest to calibrate but most robust to fast motion).

Among these, [ORB-SLAM3](https://arxiv.org/abs/2007.11898) is the right pick for most monocular production cases. ORB-SLAM3 is the first system able to perform visual, visual-inertial and multi-map SLAM with monocular, stereo and RGB-D cameras, using pin-hole and fisheye lens models. The second main novelty is a multiple map system that relies on a new place recognition method with improved recall. Thanks to it, ORB-SLAM3 is able to survive to long periods of poor visual information. The [UZ-SLAMLab repo](https://github.com/UZ-SLAMLab/ORB_SLAM3) is the upstream this guide runs against, and it ships the ORB Vocabulary file and the EuRoC/KITTI example launchers used here unmodified.

Pick monocular ORB-SLAM3 when you have a single camera, feature-rich indoor or urban scenes, and you can tolerate scale ambiguity. Switch to a stereo or visual-inertial alternative (VINS-Fusion, OKVIS, ORB-SLAM3 stereo-inertial) when you need metric scale out of the box, expect aggressive motion, or face feature-sparse outdoors. Save direct or learned methods (DSO, DROID-SLAM, DPVO) for the cases where you have GPU compute at runtime and an appetite for the bleeding edge.

## Failure modes you will hit in production

Where does ORB-SLAM3 actually break in the field, and what do you do at deployment time to keep that from happening? The numbers in this section come from a controlled study of monocular ORB-SLAM3 across EuRoC indoor and KITTI outdoor sequences, corrupted with noise families calibrated to [Hendrycks & Dietterich's ImageNet-C](https://arxiv.org/abs/1903.12261). Their first benchmark, ImageNet-C, standardizes and expands the corruption robustness topic, while showing which classifiers are preferable in safety-critical applications. Unlike recent robustness research, this benchmark evaluates performance on common corruptions and perturbations not worst-case adversarial perturbations.

Evaluate with [evo](https://github.com/MichaelGrupp/evo) under Sim(3) alignment with scale correction. Monocular has no metric scale, and under heavy corruption ATE on a surviving prefix can deceive unless tracking success and trajectory completion are logged alongside it.

### How it looks in the wild

![Per-sequence tracking success across all 16 conditions. EuRoC top, KITTI bottom.](/assets/images/state-estimation/orbslam3-tracking-heatmap.png)

The strongest signal in the data is environmental, not algorithmic. EuRoC indoor sequences average 80.6% tracking success at severity 2 and 63.6% at severity 3. KITTI driving sequences collapse to 35.8% and 21.2% at the same conditions, and when both track, median KITTI ATE is roughly 100x larger. Indoors you have walls and clutter a meter away. On KITTI you are tracking a building three blocks down, so a noise level the indoor scene shrugs off is enough to push the outdoor scene below ORB's initialization threshold.

![Trajectory reconstructions, 5 trials per panel. Top: KITTI 00. Bottom: EuRoC MH01.](/assets/images/state-estimation/orbslam3-trajectory-degradation.png)

Clean KITTI loops stack into one path, blur S1 stays close, blur S3 drifts visibly, and Gaussian S2 mostly fails to initialize. The EuRoC row tells the opposite story even at severity 5.

![Per-noise-type degradation curves on KITTI. Gaussian and shot noise collapse fastest, while motion blur degrades gracefully.](/assets/images/state-estimation/orbslam3-noise-comparison-kitti.png)

Gaussian and shot noise collapse together. Motion blur takes 2-3 severity levels longer to fail at the same operating point.

### Why ORB-SLAM3 breaks the way it does

Most of the heatmap pattern is just ORB's front end leaking through. ORB descriptors are not lighting invariant - could probably explain the degradation under shot noise. Motion blur is anisotropic, so corner detection can still fire along the perp. axis. BRIEF descriptor tests are random pairs, so as long as the affected pixels aren't touched, the matching may still survive. Gaussian blur kills high-frequency signal -> corner detection breaks -> slam collapses. Shot noise can flip bits from BRIEF's intensity ordering, essentially flipping the order, collapsing slam.

### How to prioritize this in production

A few engineering responses fall out of these mechanisms.

The biggest lever is sensor noise, not shutter time. Additive pixel noise (Gaussian, shot) hurts ORB much more than blur does, so specify a sensor with a low read-noise floor and an exposure budget that prefers a clean image over a sharp one.

Log tracking-success and trajectory completion at runtime, not just ATE. ATE on a short surviving prefix scores well on the easy part of the run, and completion is what catches that.

The cheapest fix, and the one to put first if you are starting over, is to validate the clean baseline against a published number on every fork of ORB-SLAM3 you use. KITTI silently saved only keyframes (~60% completion) and this went undetected during the first run and push, so I had to run KITTI experiments again. Fixed by patching relevant source code.

## Tuning at scale on a laptop

You will need this only if you are debugging a SLAM regression, sweeping parameters, or running ablations on a new sensor. Two pieces of plumbing make laptop-scale evaluation tractable: a containerized runtime, and a single-dispatcher batch runner.

I ran ~1800 full ORB-SLAM3 runs (11+11=22 sequences) x (1 baseline + 3 noise types x 5 severity levels = 16 conditions) x (5 trials) = 1760 runs which took considerable amount of setting up, orchestration and compute.

Because I did this on my local systems (predominantly just a MacBook), I had to come up with efficient docker containerization, monitoring and scheduling to ensure I was on track (in part because of various cmake and build compatibility issues with ORB-SLAM). A working monocular EuRoC invocation, once the image is built, looks like:

```sh
docker run --rm \
    -v $PWD/EuRoC:/data -v $PWD/out:/out \
    orbslam3:v0.6 xvfb-run -a \
    /ORB_SLAM3/Examples/Monocular/mono_euroc \
    Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml \
    /data/MH01 Examples/Monocular/EuRoC_TimeStamps/MH01.txt /out/MH01
```

Pin Pangolin v0.6 inside the image and run Xvfb on a virtual display. [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/) are the right primitive here: ORB-SLAM3's build closure (Pangolin + OpenCV + g2o + Eigen + DBoW2) is heavier than its runtime closure. The Docker docs: Multi-stage builds are useful to anyone who has struggled to optimize Dockerfiles while keeping them easy to read and maintain. You can selectively copy artifacts from one stage to another, leaving behind everything you don't want in the final image. The end result is a tiny production image with nothing but the binary inside. None of the build tools required to build the application are included in the resulting image.

For the dispatcher, the first version capped at 5 and had 14 containers running. Eleven concurrent loops each polled docker ps and launched whenever they saw a free slot. ORB-SLAM3 resource usage patterns were typical: mostly low memory and CPU usage followed by clear spike towards the end. This was exploited to create dispatcher and monitors that maximized resource usage while preventing OOM errors. The fix: a flat 880-item queue, one dispatcher, hard cap of 6. CPU holds at 50-80%, no OOMs.
