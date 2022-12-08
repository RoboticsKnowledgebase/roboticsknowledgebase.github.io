# MapViz for Map Based Visualization in ROS2 

Mapviz is a highly customizable ROS-based visualization tool focused on large-scale 2D data, with a plugin system for extreme extensibility. Mapviz is similar to Rviz, but is specifically designed for 2D, top-down viewing of outdoor robots, especially in overlaying data on an external map from OpenStreetMaps or Google maps.

This tutorial will explain how to setup Mapviz for ROS2 along with Google Maps satellite view.

## Setting up Mapviz
A setup guide is provided in the official [website](https://swri-robotics.github.io/mapviz/).  This assumes you already have a version of ROS2 installed along with a colcon workspace.

MapViz can be installed directly from apt using the following commands:
```bash
sudo apt-get install ros-$ROS_DISTRO-mapviz \
                       ros-$ROS_DISTRO-mapviz-plugins \
                       ros-$ROS_DISTRO-tile-map \
                       ros-$ROS_DISTRO-multires-image
```

In case, its not available or you need to build it from source, you can do so with the following steps:

1. Clone the latest version of the repository using the most recent branch into your `src` folder inside your workspace. At the time of writing the latest branch was `ros2-devel`.
```bash
git clone -b ros2-devel https://github.com/swri-robotics/mapviz.git
```
2. Build the workspace
```bash
colcon build --symlink-install --packages-select mapviz_interfaces mapviz mapviz_plugins tile_map multires_image
```

## Setting up Google Maps Satellite 
This part of the tutorial uses the following repo [GitHub - danielsnider/MapViz-Tile-Map-Google-Maps-Satellite: ROS Offline Google Maps for MapViz](https://github.com/danielsnider/MapViz-Tile-Map-Google-Maps-Satellite)  to proxy Google Maps satellite view into a  WMTS tile service so that it can be viewed on Mapviz.

The following are the steps to set it up, such that this service autostart on boot.

1. Install Docker 
    ```bash
    sudo apt install docker.io
    sudo systemctl enable --now docker
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```
    After running these commands log out and log back into your user account.

2. Setup the proxy
    ```bash
    sudo docker run -p 8080:8080 -d -t --restart=always danielsnider/mapproxy
    ```

    **Note:**  
    1. The ‘—restart=always’ argument will make the container always run in the background even after reboots
    2. This will bind to port 80 which might be needed for other applications especially during web development

3. Setup Custom Source

    Launch Mapviz
    ```bash
    ros2 launch mapviz mapviz.launch.py
    ```

   1. You can then add a new layer to the map by clicking on the add button on the bottom left corner of the map. 
   2. Add a new `map_tile` display component
   3. In the `Source` dropdown select `Custom WMTS source`
   4. Set the `Base URL` to `http://localhost:8080/wmts/gm_layer/gm_grid/{level}/{x}/{y}.png`
   5. Set the 'Max Zoom' to 19
   6. Click `Save`

   The map should now load up with Google Maps satellite view. This may take some time initally.

## Advanced Setup

1. Create a custom launch file
You can create a custom launch file to load Mapviz with a custom configuration and initalize to a custom origin by default.

    ```python
    import launch
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
    from launch_ros.actions import Node
    from launch_ros.substitutions import FindPackageShare


    def generate_launch_description():
        current_pkg = FindPackageShare('your_package_name')

        return launch.LaunchDescription(
            [
                DeclareLaunchArgument(
                    'mapviz_config',
                    default_value=PathJoinSubstitution([current_pkg, 'mapviz', 'mapviz.mvc']),
                    description='Full path to the Mapviz config file to use',
                ),
                Node(
                    package='mapviz',
                    executable='mapviz',
                    name='mapviz',
                    output={'both': 'log'},
                    parameters=[
                        {'config': LaunchConfiguration('mapviz_config'), 'autosave': False}
                    ],
                ),
                Node(
                    package='mapviz',
                    executable='initialize_origin.py',
                    name='initialize_origin',
                    parameters=[
                        {'local_xy_frame': 'map'},
                        {'local_xy_navsatfix_topic': 'gps/fix/origin'},
                        {'local_xy_origin': 'auto'},
                        {
                            'local_xy_origins': """[
                        {'name': 'pitt',
                            'latitude': 40.438889608527084,
                            'longitude': -79.95833630855975,
                            'altitude': 273.1324935602024,
                            'heading': 0.0}
                    ]"""
                        },
                    ],
                ),
            ]
        )
    ```

    This will find the share directory of your package. This generally where all configs are stored for ROS2 packages.
    
    ```python
    current_pkg = FindPackageShare('your_package_name')
    ```

    Using this we can load the custom Mapviz config. This line assumes by default the config file is stored in the `mapviz` folder of your package and is named `mapviz.mvc`.
    ```python
    DeclareLaunchArgument(
        'mapviz_config',
        default_value=PathJoinSubstitution([current_pkg, 'mapviz', 'mapviz.mvc']),
        description='Full path to the Mapviz config file to use',
    ),
    ```

    This will load the Mapviz node with the custom config file and ensure that autosave is disabled.
    ```python
    Node(
        package='mapviz',
        executable='mapviz',
        name='mapviz',
        output={'both': 'log'},
        parameters=[
            {'config': LaunchConfiguration('mapviz_config'), 'autosave': False}
        ],
    ),
    ```

    This will load the `initialize_origin.py` node which will initialize the origin of the map to the specified location. This is useful if you want to start the map at a specific location or using your current GPS location.

        By setting local_xy_origin to `auto` it will use the current GPS location as the origin. For this to work you need to have a GPS sensor publishing the origin GPS coordinate to the topic `gps/fix/origin` with the message type `sensor_msgs/msg/NavSatFix`.

        Incase you want to set the origin to a specific location you can set the `local_xy_origin` to the name of the origin you want to use. This name should match the name of the origin in the `local_xy_origins` parameter.   
        For this example we will set the origin to the name `pitt` which is the name of the origin in the `local_xy_origins` parameter. This sets it to a specific location in Pittsburgh, PA.

    ```python
    Node(
        package='mapviz',
        executable='initialize_origin.py',
        name='initialize_origin',
        parameters=[
            {'local_xy_frame': 'map'},
            {'local_xy_navsatfix_topic': 'gps/fix/origin'},
            {'local_xy_origin': 'auto'},
            {
                'local_xy_origins': """[
            {'name': 'pitt',
                'latitude': 40.438889608527084,
                'longitude': -79.95833630855975,
                'altitude': 273.1324935602024,
                'heading': 0.0}
        ]"""
            },
        ],
    )
    ```

2. Setting the origin to the current GPS location

    The following script subscribes the current GPS location and re-publishes the first GPS coordinate it recieves as the origin on the topic `gps/fix/origin`.

    Incase you are using the `robot_localization` package to fuse GPS, it also calls the `SetDatum` service offered by the `robot_localization` package to set the datum of the robot_localization node.
    This is necessary to ensure that the robot_localization node is using the same origin as the one used by Mapviz. 

    You will need to run this script before running Mapviz. This can be done by adding it to the `launch` file of your package or by running it manually.

    ```python
    #!/usr/bin/env python3

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        qos_profile_sensor_data,
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
    )

    from robot_localization.srv import SetDatum
    from sensor_msgs.msg import NavSatFix, NavSatStatus


    class GpsDatum(Node):
        """
        Republishes the first valid gps fix and sets datum in robot_localization.

        Subscribes and stores the first valid gps fix, then republishes it as the
        origin. Also calls SetDatum service offered by robot_localization.

        """

        def __init__(self):
            super().__init__('gps_datum')

            self.gps_datm_msg_ = None
            self.rl_datum_future_ = None
            self.rl_datum_set_ = False

            self.sub_gps_ = self.create_subscription(
                NavSatFix, 'gps/fix', self.sub_gps_cb, qos_profile_sensor_data
            )

            self.pub_gps_datum_ = self.create_publisher(
                NavSatFix,
                'gps/fix/origin',
                QoSProfile(
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                ),
            )

            # Need to use a timer since latching behaviour doesn't behave like ROS1
            # https://github.com/ros2/ros2/issues/464
            timer_period_ = 1.0
            self.timer_ = self.create_timer(timer_period_, self.timer_callback)

            self.rl_datum_client = self.create_client(SetDatum, 'datum')
            self.get_logger().info('Waiting for robot_localization datum service')
            self.rl_datum_client.wait_for_service()
            self.get_logger().info('robot_localization datum service now available')

        def sub_gps_cb(self, msg):
            if msg.status.status == NavSatStatus.STATUS_NO_FIX:
                return
            self.gps_datm_msg_ = msg
            self.get_logger().info('Successfully set origin. Unsubscribing to gps fix')
            self.destroy_subscription(self.sub_gps_)

        def timer_callback(self):
            if self.gps_datm_msg_ is None:
                return
            self.pub_gps_datum_.publish(self.gps_datm_msg_)
            self.send_rl_request()

        def send_rl_request(self):
            if self.rl_datum_set_ or self.gps_datm_msg_ is None:
                return

            if self.rl_datum_future_ is None:
                req = SetDatum.Request()
                req.geo_pose.position.latitude = self.gps_datm_msg_.latitude
                req.geo_pose.position.longitude = self.gps_datm_msg_.longitude
                req.geo_pose.position.altitude = self.gps_datm_msg_.altitude
                req.geo_pose.orientation.w = 1.0
                self.get_logger().info(
                    'Sending request to SetDatum request to robot_localization'
                )
                self.rl_datum_future_ = self.rl_datum_client.call_async(req)
            else:
                if self.rl_datum_future_.done():
                    try:
                        self.rl_datum_future_.result()
                    except Exception as e:  # noqa: B902
                        self.get_logger().info(
                            'Call to SetDatum service in robot_localization failed %r'
                            % (e,)
                        )
                    else:
                        self.get_logger().info('Datum set in robot_localization')
                        self.rl_datum_set_ = True


    def main(args=None):
        rclpy.init(args=args)

        gps_datum = GpsDatum()

        rclpy.spin(gps_datum)

        gps_datum.destroy_node()
        rclpy.shutdown()


    if __name__ == '__main__':
        main()
    ```

3. Custom Configuration

    Below is an example configuration file mentioned above as `mapviz.mvc` for Mapviz. This loads the Google Maps Satellite layer and shows the GPS location published on the `/gps/fix` topic.

    ```
    capture_directory: "~"
    fixed_frame: map
    target_frame: <none>
    fix_orientation: false
    rotate_90: true
    enable_antialiasing: true
    show_displays: true
    show_status_bar: true
    show_capture_tools: true
    window_width: 1848
    window_height: 1016
    view_scale: 0.09229598
    offset_x: 0
    offset_y: 0
    use_latest_transforms: true
    background: "#a0a0a4"
    image_transport: raw
    displays:
    - type: mapviz_plugins/tile_map
        name: Map
        config:
        visible: true
        collapsed: true
        custom_sources:
            - base_url: http://localhost:8080/wmts/gm_layer/gm_grid/{level}/{x}/{y}.png
            max_zoom: 19
            name: GMaps
            type: wmts
            - base_url: https://tile.openstreetmap.org/{level}/{x}/{y}.png
            max_zoom: 19
            name: OSM
            type: wmts
        bing_api_key: ""
        source: GMaps
    - type: mapviz_plugins/navsat
        name: INS Location
        config:
        visible: true
        collapsed: true
        topic: /gps/fix
        color: "#fce94f"
        draw_style: points
        position_tolerance: 0.5
        buffer_size: 0
        show_covariance: true
        show_all_covariances: false
    ```