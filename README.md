# Training-free Task-oriented Grasp Generation

This project integrates **Contact GraspNet** and **ManiSkill** to enable a robotic system to generate task-oriented grasps **without training**. The system consists of two main components:

1. **Contact GraspNet** - Generates grasp candidates based on object geometry.
2. **ManiSkill** - Controls the robotic arm to execute grasping actions based on task requirements.

---

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/your_username/Training-free-Task-oriented-Grasp-Generation.git
cd Training-free-Task-oriented-Grasp-Generation
```

---

## Setup Instructions

### Step 1: Setup Conda Environments
Ensure you have **Conda** installed. Then, activate the required environments:

#### Grasp Generation (Contact GraspNet)
```bash
conda activate cgn_env
```

#### Robot Control (ManiSkill)
```bash
conda activate mani_env2
```

---

### Step 2: Start the Contact GraspNet Server
In the **first terminal**, navigate to the Contact GraspNet directory and launch the **grasping server**:

```bash
cd ./third_party/contact_graspnet/contact_graspnet/
python socket_server.py
```
This starts a socket server that generates task-oriented grasping points.

---

### Step 3: Run the Robot Controller
In **another terminal**, activate the **ManiSkill** environment and execute the robot control script:

```bash
conda activate mani_env2
python move_robot.py --task task_1 --model gemini --task_idx 1 --visual_method gripper_one
```

#### Command Arguments:
| Argument | Description |
|----------|------------|
| `--task task_1` | Specifies the task (e.g., gripping toys). |
| `--model gemini` | Model used for grasp evaluation. |
| `--task_idx 1` | Index of the task instance. |
| `--visual_method gripper_one` | Defines the visualization method for grasp execution. |


---

##  Dependencies
Make sure the following dependencies are installed in their respective environments:

### cgn_env (for grasp generation)
```bash
conda env create -f cgn_environment.yml
```

### mani_env2 (for robot control)
```bash
conda env create -f mani_environment.yml
```


