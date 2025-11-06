# Dynamic Honeypot Placement as a Game-Theoretic Defense

## Overview
This project models **honeypot placement** in a network as a **two-player zero-sum game** between an *attacker* and a *defender*.  
- The **defender** strategically deploys honeypots (decoy systems) to protect critical assets.  
- The **attacker** chooses nodes to attack in an attempt to compromise those assets.  
- Payoffs depend on whether attacks are detected or succeed.  
- The simulation uses **game-theoretic learning (Multiplicative Weights Update)** to approximate the **Nash equilibrium**, revealing optimal defensive and offensive strategies.

The outcome is a visualization of equilibrium honeypot placement probabilities, attacker target probabilities, and convergence behavior over time.

---

## Key Concepts

| Concept | Description |
|----------|--------------|
| **Grid Network** | The network is represented as a 2D grid graph (`rows × cols`). Nodes are computers/systems connected to their adjacent neighbors. |
| **Assets** | High-value systems that the attacker targets and defender must protect. |
| **Honeypots** | Decoy systems placed by the defender each round to catch attackers. |
| **Detection Model** | Determines how honeypots influence nearby nodes using a probability parameter `α` and neighborhood radius `r`. |
| **Payoff Matrix** | Defines the defender’s payoff for every defender (honeypot set) and attacker (target node) pair. |
| **MWU Algorithm** | Both players update strategy weights using Multiplicative Weights Update until the average strategies converge to equilibrium. |

---

## Project Structure
TODO

## Installation

```
pip install -r requirements.txt
```
## Running a Simulation

```
python cli.py --config experiments/<experiment>.yaml
```

## Config Parameters
TODO
