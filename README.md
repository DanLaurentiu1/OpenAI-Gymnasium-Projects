

# Project Status
### 🟠 Paused

# ReinforcementLearningProjects
# Content:
  * [CartPole-v1](#cartpole-v1)
  * [MountainCar-v0](#mountaincar-v0)
  * [Taxi-v3](#taxi-v3)
  * [FrozenLake-v1](#frozenlake-v1)
  * [Blackjack](#blackjack-v1)
  * [CliffWalking](#cliffwalking-v0)
  * [ChromeDinoGame](#chromedinogame)

# CartPole-v1
A **pole** is attached by an un-actuated joint to a cart, which moves along a frictionless track. 

The pendulum is placed **upright** on the cart and the goal is to **balance** the pole by applying forces in the **left** and **right** direction on the cart.

### More information about the environment
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CartPole/resources/SpacesTypes.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CartPole/resources/ActionSpace.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CartPole/resources/ObservationSpace.png" width="400" /> 
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CartPole/resources/Rewards.png" width="400" /> 
</p>

### Learning curve and epsilon decay
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CartPole/resources/learning_curve.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CartPole/resources/epsilon_decay.png" width="400" /> 
</p>

### Demo
<p align="center"><img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CartPole/resources/CartPoleDemo.gif" width="400"/></p>

<br> <br> <br>

# MountainCar-v0

The Mountain Car environment is a deterministic environment that consists of a car placed stochastically at the **bottom** of a sinusoidal **valley**, with the only possible actions being the **accelerations** that can be applied to the car in **either direction**.

The goal is to strategically accelerate the car to reach the **goal** state on **top of the right hill**.



### More information about the environment
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/MountainCarDiscrete/resources/SpaceTypes.png" width="400" height="100"/>
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/MountainCarDiscrete/resources/Rewards.png" width="400" height="100"/> 
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/MountainCarDiscrete/resources/ActionSpace.png" width="400" height="120"/>
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/MountainCarDiscrete/resources/ObservationSpace.png" width="400" height="120"/> 
</p>

### Learning curve and epsilon decay
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/MountainCarDiscrete/resources/learning_curve.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/MountainCarDiscrete/resources/epsilon_decay.png" width="400" /> 
</p>

### Demo
<p align="center"><img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/MountainCarDiscrete/resources/MountainCarDiscreteDemo.gif" width="400"/></p>

<br> <br> <br>

# Taxi-v3

There are **four** designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the 5x5 grid world.

The taxi starts off at a **random square** and the passenger at one of the designated locations.

The goal is move the taxi to the passenger’s location, **pick up the passenger**, move to the passenger’s desired destination, and **drop off the passenger**.



### More information about the environment
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Taxi/resources/SpaceTypes.png" width="400"/>
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Taxi/resources/Rewards.png" width="400"/> 
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Taxi/resources/ActionSpace.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Taxi/resources/ObservationSpace.png" width="400" /> 
</p>

### Learning curve and epsilon decay
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Taxi/resources/learning_curve.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Taxi/resources/epsilon_decay.png" width="400" /> 
</p>

### Demo
<p align="center"><img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Taxi/resources/TaxiDemo.gif" width="400"/></p>

<br> <br> <br>

# FrozenLake-v1

The game starts with the player at **location [0,0]** of the frozen lake grid world with the goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

**Holes** in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated.

The player **makes moves** until they **reach the goal** or **fall** in a hole.

### More information about the environment
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/FrozenLake/resources/StateSpaces.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/FrozenLake/resources/ActionSpace.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/FrozenLake/resources/ObservationSpace.png" width="400" height="120"/> 
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/FrozenLake/resources/Rewards.png" width="400" height="120"/> 
</p>

### Learning curve and epsilon decay
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/FrozenLake/resources/learning_curve.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/FrozenLake/resources/epsilon_decay.png" width="400" /> 
</p>

<br> <br> <br>

### Demo
<p align="center"><img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/FrozenLake/resources/FrozenLakeDemo.gif" width="400"/></p>

# Blackjack-v1

The game starts with the **dealer** having **one face up** and **one face down** card, while the player has two face up cards. All cards are drawn from an infinite deck (i.e. with replacement).

The card values are:

-   Face cards (Jack, Queen, King) have a point value of 10.
    
-   Aces can either count as 11 (called a ‘usable ace’) or 1.
    
-   Numerical cards (2-9) have a value equal to their number.
    

The player has the sum of cards held. The player can request **additional cards (hit)** until they decide to **stop (stick)** or exceed 21 (bust, immediate **loss**).


### More information about the environment
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Blackjack/resources/SpacesTypes.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Blackjack/resources/ActionSpace.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Blackjack/resources/ObservationSpace.png" width="400" height="125"/> 
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Blackjack/resources/Rewards.png" width="400" height="125"/> 
</p>

### Learning curve and epsilon decay
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Blackjack/resources/losing_rate.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Blackjack/resources/epsilon_decay.png" width="400" /> 
</p>

### Demo
<p align="center"><img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/Blackjack/resources/BlackjackDemo.gif" width="400"/></p>

<br> <br> <br>

# CliffWalking-v0

The game starts with the player at **location [3, 0]** of the 4x12 grid world with the **goal** located at **[3, 11]**.

If the player reaches the goal the episode ends.

A cliff runs along [3, 1..10]. If the player **moves to a cliff** location it **returns** to the **start location**.
The player makes moves until they reach the goal.

### More information about the environment
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CliffWalking/resources/SpaceTypes.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CliffWalking/resources/Rewards.png" width="400" /> 
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CliffWalking/resources/ActionSpace.png" width="400" height="125"/>
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CliffWalking/resources/ObservationSpace.png" width="400" height="125"/> 
</p>

### Learning curve and epsilon decay
<p float="left">
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CliffWalking/resources/learning_curve.png" width="400" />
  <img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CliffWalking/resources/epsilon_decay.png" width="400" /> 
</p>

### Demo
<p align="center"><img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/CliffWalking/resources/CliffWalkingDemo.gif" width="400"/></p>

<br> <br> <br>

# ChromeDinoGame

The Chrome Dino game is an offline game built into the Google Chrome **web browser**.

The objective is to keep the T-Rex **running as long as possible by avoiding obstacles**, such as cacti and flying pterodactyls.

 The game **speeds up over time**, increasing the difficulty.

### More information about the environment
| Space | Type |
| ----------- | ----------- |
| Action Space | Discrete(3) |
| Observation Space | Discrete(2040000)|
###
| Action | Information |
| ----------- | ----------- |
| 0 | Jump |
| 1 | Duck |
| 2 | Idle, do nothing |
###
| Observation| Information |
| ----------- | ----------- |
| 80x100 array | each observation is a screen capture of the game, 100 pixels horizontal and 80 pixels vertical |
###
| Rewards | Information |
| ----------- | ----------- |
| +1 | if the dino jumps or ducks|
| +2 | if the dino does nothing |

### Demo
<p align="center"><img src="https://github.com/DanLaurentiu1/RLProjects/blob/main/ChromeDinoGame/ChromeDinoDemo.gif" width="700" heigth="500"/></p>
