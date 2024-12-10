---
date: 2024-12-07
title: Basic Arduino Components
---

Arduino is a widely used platform in robotics and electronics due to its ease of use, affordability, and the wide variety of compatible components. Whether you're building a simple circuit or a complex robot, understanding how to use basic components like LEDs, push buttons, potentiometers, buzzers, servos, and stepper motors is essential. This guide will cover these fundamental Arduino components, explain their functionality, and provide simple examples to get you started to where you can incorporate them in to future robotic projects.

## LEDs

An LED (Light Emitting Diode) is one of the simplest and most commonly used components in Arduino projects. LEDs provide visual feedback and are often used as indicators for status, power, or signals. LEDs work by emitting light when an electric current flows through them in the correct direction. They require a resistor in series to prevent excessive current, which could damage the LED. To wire an LED to an Arduino, connect the long leg (anode) of the LED to a digital pin on the Arduino. The short leg (cathode) should be connected to the ground through a resistor, typically 220 ohms, to limit the current.

A simple wiring setup for an LED can be seen below along with a code snippet. The code will cause the LED to blink on and off. 

```
// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
  digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on (HIGH is the voltage level)
  delay(1000);                      // wait for a second
  digitalWrite(LED_BUILTIN, LOW);   // turn the LED off by making the voltage LOW
  delay(1000);                      // wait for a second
}

```


## Push Button

A push button is a simple input device that completes an electrical circuit when pressed. It is commonly used to trigger events or control devices in Arduino projects. When the button is pressed, it allows current to flow, sending a signal to the Arduino. When released, the circuit is open, and no signal is sent. Pull-down or pull-up resistors are often used to ensure the circuit has a defined state (HIGH or LOW) when the button is not pressed. To connect a push button to an Arduino, attach one leg of the button to a digital input pin and the other to the ground. Additionally, use a pull-up resistor or enable the Arduino's internal pull-up resistor to stabilize the input state.

A simple wiring setup for a push button can be seen below along with a code snippet. The code will cause the serial monitor to print "Button Pressed" when the button is pressed, and print "Not Pressed" when the button is not pressed.

```
const int buttonPin = 2;  // the number of the pushbutton pin

// variables will change:
int buttonState = 0;  // variable for reading the pushbutton status

void setup() {
  // initialize the pushbutton pin as an input:
  pinMode(buttonPin, INPUT);
  Serial.begin(9600); // Start serial communication
}

void loop() {
  // read the state of the pushbutton value:
  buttonState = digitalRead(buttonPin);

  // check if the pushbutton is pressed. If it is, the buttonState is HIGH:
  if (buttonState == HIGH) {
	Serial.println("Button Pressed):
} else {
    Serial.println("Not Pressed):
  }
}
```

Note: When a push button is pressed, it may generate multiple rapid on/off signals due to the mechanical nature of the button. This is known as "bouncing" and can cause unexpected behavior in your circuit. To address this, you can use a small delay in your code or implement software debouncing techniques. You can read more about debouncing at the link in the "Further Reading" section below

## Potentiometers

A potentiometer is a variable resistor that allows you to adjust resistance by turning a knob or sliding a lever. It is commonly used as an input device for controlling brightness, volume, or speed in Arduino projects. A potentiometer has three terminals: two outer terminals connected to a fixed resistor and one middle terminal (the wiper) that moves along the resistor as you adjust the knob. By reading the voltage at the wiper, the Arduino can determine the position of the potentiometer.To wire a potentiometer, onnect one outer terminal to 5V on the Arduino and the other outer terminal to GND. Connect the middle terminal (wiper) to an analog input pin, such as A0. This setup allows the Arduino to read a voltage that corresponds to the potentiometer's position.

An image of a simple wiring for a potentiometer can be seen below along with a code snippet. To read a potentiometer you will need to use an analog pin and the analogRead() Arduino function. 

Note: A useful function when dealing with analog signals is the map() function. More can be found about this function in the "Further Reading section.

```
const int potPin = A0; // Pin connected to the potentiometer

void setup() {
  Serial.begin(9600); // Start serial communication
}

void loop() {
  int potValue = analogRead(potPin); // Read potentiometer value (0-1023)
  Serial.println(potValue);         // Print value to serial monitor
  delay(100);                       // Delay for readability
}
```

## Buzzers

A buzzer is a simple electronic component that generates sound when powered, often used in Arduino projects for audible feedback, alarms, or notifications. There are two main types of buzzers: active and passive. An active buzzer generates sound when supplied with power and does not require any signal control, making it straightforward to use. A passive buzzer, on the other hand, requires a signal (such as a PWM signal) to produce sound, allowing for the creation of different tones.

Buzzers work by converting electrical energy into sound through the vibration of a piezoelectric diaphragm. The frequency of the signal determines the tone of the sound produced. To wire a buzzer, connect the positive terminal to a digital output pin on the Arduino and the negative terminal to the ground (GND). For an active buzzer, you can simply turn it on and off using digital signals. For example, you can alternate between HIGH and LOW states to create a beep. A passive buzzer can generate tones of varying frequencies using the Arduino's tone() function.

Buzzers are versatile components with applications in security systems, timers, notification systems, and user interfaces where audible feedback is needed. They provide an effective way to communicate events, warnings, or statuses in a project.

The wiring for a simple piezo buzzer can be seen below alogn with a code snippet. The code will cause the buzzer to continuously start and stop buzzing.

```
const int buzzer = 9; //buzzer to arduino pin 9

void setup(){
  pinMode(buzzer, OUTPUT); // Set buzzer - pin 9 as an output
}

void loop(){
  tone(buzzer, 1000); // Send 1KHz sound signal...
  delay(1000);        // ...for 1 sec
  noTone(buzzer);     // Stop sound...
  delay(1000);        // ...for 1sec
}
```

## Servo Motors

A servo motor is a rotary actuator that allows precise control of angular position, speed, and torque. It is widely used in robotics and Arduino projects due to its compact size and ability to move to a specific position within a range. Servo motors are ideal for tasks that require controlled movements, such as steering mechanisms, robotic arms, or pan-tilt camera systems.

Servo motors typically have three wires: a power wire (usually red), a ground wire (usually black or brown), and a signal wire (often yellow, orange, or white). The power and ground wires connect to the Arduino’s 5V and GND pins, respectively, while the signal wire connects to a digital output pin. The signal wire receives pulse-width modulation (PWM) signals from the Arduino, which dictate the servo's angle of rotation. Most standard servo motors have a range of motion from 0 to 180 degrees.

Using the Arduino Servo library simplifies the control of servo motors. For example, you can set the angle of the motor by sending a specific value through the write() function.


This template acts as a tutorial on writing articles for the Robotics Knowledgebase. In it we will cover article structure, basic syntax, and other useful hints. Every tutorial and article should start with a proper introduction.

This goes above the first subheading. The first 100 words are used as an excerpt on the Wiki's Index. No images, HTML, or special formating should be used in this section as it won't be displayed properly.

If you're writing a tutorial, use this section to specify what the reader will be able to accomplish and the tools you will be using. If you're writing an article, this section should be used to encapsulate the topic covered. Use Wikipedia for inspiration on how to write a proper introduction to a topic.

In both cases, tell them what you're going to say, use the sections below to say it, then summarize at the end (with suggestions for further study).

## First subheading
Use this section to cover important terms and information useful to completing the tutorial or understanding the topic addressed. Don't be afraid to include to other wiki entries that would be useful for what you intend to cover. Notice that there are two \#'s used for subheadings; that's the minimum. Each additional sublevel will have an added \#. It's strongly recommended that you create and work from an outline.

This section covers the basic syntax and some rules of thumb for writing.

### Basic syntax
A line in between create a separate paragraph. *This is italicized.* **This is bold.** Here is [a link](/). If you want to display the URL, you can do it like this <http://ri.cmu.edu/>.

> This is a note. Use it to reinforce important points, especially potential show stoppers for your readers. It is also appropriate to use for long quotes from other texts.


#### Bullet points and numbered lists
Here are some hints on writing (in no particular order):
- Focus on application knowledge.
  - Write tutorials to achieve a specific outcome.
  - Relay theory in an intuitive way (especially if you initially struggled).
    - It is likely that others are confused in the same way you were. They will benefit from your perspective.
  - You do not need to be an expert to produce useful content.
  - Document procedures as you learn them. You or others may refine them later.
- Use a professional tone.
  - Be non-partisan.
    - Characterize technology and practices in a way that assists the reader to make intelligent decisions.
    - When in doubt, use the SVOR (Strengths, Vulnerabilities, Opportunities, and Risks) framework.
  - Personal opinions have no place in the Wiki. Do not use "I." Only use "we" when referring to the contributors and editors of the Robotics Knowledgebase. You may "you" when giving instructions in tutorials.
- Use American English (for now).
  - We made add support for other languages in the future.
- The Robotics Knowledgebase is still evolving. We are using Jekyll and GitHub Pages in and a novel way and are always looking for contributors' input.

Entries in the Wiki should follow this format:
1. Excerpt introducing the entry's contents.
  - Be sure to specify if it is a tutorial or an article.
  - Remember that the first 100 words get used else where. A well written excerpt ensures that your entry gets read.
2. The content of your entry.
3. Summary.
4. See Also Links (relevant articles in the Wiki).
5. Further Reading (relevant articles on other sites).
6. References.

#### Code snippets
There's also a lot of support for displaying code. You can do it inline like `this`. You should also use the inline code syntax for `filenames` and `ROS_node_names`.

Larger chunks of code should use this format:
```
def recover_msg(msg):

        // Good coders comment their code for others.

        pw = ProtocolWrapper()

        // Explanation.

        if rec_crc != calc_crc:
            return None
```
This would be a good spot further explain you code snippet. Break it down for the user so they understand what is going on.

#### LaTex Math Support
Here is an example MathJax inline rendering $ \phi(x\|y) $ (note the additional escape for using \|), and here is a block rendering:
$$ \frac{1}{n^{2}} $$

#### Images and Video
Images and embedded video are supported.

![Put a relevant caption here](assets/images/Hk47portrait-298x300.jpg)

{% include video id="8P9geWwi9e0" provider="youtube" %}

{% include video id="148982525" provider="vimeo" %}

The video id can be found at the end of the URL. In this case, the URLs were
`https://www.youtube.com/watch?v=8P9geWwi9e0`
& `https://vimeo.com/148982525`.

## Summary
Use this space to reinforce key points and to suggest next steps for your readers.

## See Also:
- Links to relevant material within the Robotics Knowledgebase go here.

## Further Reading
- Links to articles of interest outside the Wiki (that are not references) go here.

## References
- Links to References go here.
- References should be in alphabetical order.
- References should follow IEEE format.
- If you are referencing experimental results, include it in your published report and link to it here.


Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.
