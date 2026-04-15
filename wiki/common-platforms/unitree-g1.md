---
date: 2026-03-12
title: Unitree G1
---
The Unitree G1 is a compact bipedal humanoid robot manufactured by Unitree Robotics. At a lower price point than most humanoid platforms, the G1 has seen growing adoption in university and research settings.

The official documentation covers hardware specifications and physical setup reasonably well, but the software side is significantly lacking. This article aims to address the most common questions and usage patterns when working with the G1 SDK and integrating it into a ROS 2 stack.

## SDK Overview
The Unitree SDK provides a way of programmatically talking to the robot. As the official documentation mentions, the SDK directly uses DDS messages (CycloneDDS, to be specific) as the message middleware, cutting out ROS (2) entirely when communicating with the robot.

The SDK exposes certain "clients" through which we can exercise our programmatic control. The two main clients in focus for this article are the `MotionSwitcherClient` and the `LocoClient`.

### High-Level Motion Control Service (`ai_sport`)
The robot ships with an internal "high-level" motion controller that supports certain actions, such as those the hand-held controller is capable of commanding. This controller is governed by a service called `ai_sport`. All "high-level" actions / commands, such as those exposed by the `LocoClient`, depend on this service being active to be executed.

**Warning**: It seems it is entirely possible to send low-level joint commands while the high-level controller is active. Multiple sources state that the robot may try to execute both commands (high-level and low-level) at once and perform potentially unsafe actions. It is best to not send both types of commands at the same time. It is an even better idea to make it impossible to do the same (perhaps via a command gate / multiplexer).
{: .notice--warning}

### `MotionSwitcherClient`
The SDK documentation states that users can use this client to "release the G1 motion control mode via RPC" (this disables `ai_sport`) and enter debug mode, wherein the user can directly command all joints of the robot via low-level PD control.

```cpp
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp> // Seems misleading, but is actually correct
#include <iostream>

int main() {
    unitree::robot::b2::MotionSwitcherClient msc{};
    msc.SetTimeout(10.0F);
    msc.Init();

    const std::int32_t ret = msc.ReleaseMode();
    if (ret == 0) {
        std::cout << "Entered debug mode" << std::endl;
    } else {
        std::cout << "Failed to enter debug mode" << std::endl;
    }
}
```

When the `ai_sport` service is disabled, the robot will go limp. It is recommended that it be supported by the gantry when performing the switch, or the robot will fall and sustain damage.

The `MotionSwitcherClient` also exposes `MotionSwitcherClient::CheckMode(std::string& form, std::string& name)` and `MotionSwitcherClient::SelectMode(const std::string& name)` methods. Both return `std::int32_t` error codes detailing the success / failure of the call, similar to `MotionSwitcherClient::ReleaseMode`.

The `form` returned as an out parameter of `MotionSwitcherClient::CheckMode(...)` is actually irrelevant. What is of relevance is the `name` string. If it is empty, the robot is in debug mode. If it is not empty, `ai_sport` is active.

`MotionSwitcherClient::SelectMode("ai")` activates the `ai_sport` service, allowing us to once again use high-level controls. If not already active, upon re-enabling, the G1 enters zero torque mode (goes limp), similar to boot behavior. Once again, it is recommended that the robot is supported by the gantry when performing the switch. 

**Note**: The `name` string to re-enable `ai_sport` is "ai" - no other string seems to have worked at the time of the writing of this article.
{: .notice--info}

### `LocoClient`
This allows us to use the high-level controls provided by the high-level motion control service. The documentation for this client from Unitree's side is complete in terms of the interface, but lacks some information about the requirements for some methods to take effect.

`LocoClient::Damp()` enters damping mode and serves as a requirement for calls to `LocoClient::StandUp()` and `LocoClient::ZeroTorque()`. 

**Note**: Calls to those methods might return 0 (meaning the call was successful), but will produce no effect on the robot if `LocoClient::Damp()` was not called before.
{: .notice--info}

Similarly, `LocoClient::StandUp()` must be called before calling `LocoClient::Start()`, which puts the robot into active locomotion mode, i.e., we can now send velocity commands (via `LocoClient::Move(...)` or `LocoClient::SetVelocity(...)`) or use the hand-held controller to command the robot to move.

Internally, the high-level motion controller service contains a FSM. Each mode is represented as a state in this FSM. `LocoClient::Start()` puts the motion control service into state `500`, which exhibits frequent intermittent jittering during motion. The hand-held controller's `Regular Mode` directly sets the motion control FSM to `501`, which does not exhibit jittering. Similarly, the hand-held controller's `Running Mode` directly sets the motion control FSM to `801`, which also does not exhibit jittering and seems to exhibit even better balance than `Regular Mode`. 

**Note**: It is recommended to swap out all calls to `LocoClient::Start()` with calls to `LocoClient::SetFsmId(501)` or `LocoClient::SetFsmId(801)`.
{: .notice--info}

**Note**: There are some caveats to the FSM modes. State `500` is actually `Regular Mode` for the version of the Unitree G1 with 1 DoF at the waist, while `501` is `Regular Mode` for the version of the G1 with 3 DoF at the waist. By locking the waist via the Unitree Explore app, the G1 effectively becomes the 1 DoF version. It is unknown why Unitree ships a broken locomotion policy for the 1 DoF G1, or if this is only a problem for 3 DoF G1s that are waist locked to 1 DoF. Starting in firmware version v1.4.9, it seems the 1 DoF G1 can no longer use state `501`, constraining it to the broken locomotion policy.
{: .notice--info}

### Arm Control
`rt/arm_sdk` allows controlling the upper body joints (both arms and the waist) via low-level PD commands. Commands only take effect after a call to `LocoClient::StandUp()` (which in turn needs a call to `LocoClient::Damp()` as discussed before).

The command format is exactly the same as for low-level commands, with the additional requirement of setting a weight parameter. This is accomplished by setting the `q` value of the 29th joint to a value between `0.0` and `1.0`. Once given a command, the robot will hold it (as long as the weight parameter is `1.0` or no velocity commands are issued).

```cpp
const float weight = 1.0F;

unitree_hg::msg::dds::LowCmd_ command{};

// The 29th joint doesn't exist, but it's q value represents the weight
command.motor_cmd().at(29).q() = weight;

// Construct the rest of the message
```

It is generally recommended to blend the commands coming from the high-level motion control service and your own commands by ramping weight up from `0.0` to `1.0`. Similarly, you can return control to the motion control service by ramping the weight down from `1.0` to `0.0`. Until the last issues command does not have `0.0` as the weight, the motion control service will not regain full control of the arms.

**Note**: In `Running Mode`, we cannot use arm control and set velocity commands (use `LocoClient::Move(...)`) at the same time. Enabling arm control (with weight `1.0`) will make the G1 not respond to velocity commands. Releasing arms (with weight `0.0`) will re-enable the usage of velocity commands. `Regular Mode` does not have this restriction.
{: .notice--info}

**Warning**: The high-level locomotion policy is not aware of how the arms are being controlled. If the arms are held outstretched in front, the policy will attempt to balance the robot by stepping forward and failing to regain balance (since the arms are still outstretched), continuing to shuffle forward in a loop indefinitely (until we rebalance the arms through control or let go of arm control).
{: .notice--warning}

### Debug Mode
Similar to arm control, once given a low-level command, the robot will hold it. No setting of a weight parameter is required.

**Note**: The mode machine parameter of the low-level command must match the mode machine value returned by `rt/lowstate` for any sent low-level commands to be obeyed.
{: .notice--info}

### `ChannelFactory` Init and DDS Configuration
Unitree examples may feature some code similar to this:

```cpp
unitree::robot::ChannelFactory::Instance()->Init(0);
```

This is fine.

One may be tempted to use the overload that directly takes in a network interface string as well. The behavior of these two is subtle but different: if a network interface string is passed, this function will attempt to **create** the DDS domain. If the domain already exists (perhaps due to a ROS node creating it first), this will error out and crash. It is best to not pass in a network interface string, which causes it to attempt to bind to the currently created domain.

## References
- [Unitree G1 Documentation](https://support.unitree.com/home/en/G1_developer/about_G1)
