# Training-free Task-oriented Grasp Generation

This project integrates **Contact GraspNet** and **ManiSkill** to enable a robotic system to generate task-oriented grasps **without training**. The system consists of two main components:

1. **Contact GraspNet** - Generates grasp candidates based on object geometry.
2. **ManiSkill** - Controls the robotic arm to execute grasping actions based on task requirements.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your_username/Training-free-Task-oriented-Grasp-Generation.git
cd Training-free-Task-oriented-Grasp-Generation
```

---

## 🛠 Setup Instructions

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
🔹 This starts a socket server that generates task-oriented grasping points.

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

## 📦 Project Structure
```
Training-free-Task-oriented-Grasp-Generation/
│── third_party/
│   ├── contact_graspnet/  # Contact GraspNet source
│   ├── mani_skill/        # ManiSkill environment
│── move_robot.py          # Robot movement script
│── README.md              # Project documentation
│── requirements.txt       # Dependencies
│── LICENSE                # License file
```

---

## 🔧 Dependencies
Make sure the following dependencies are installed in their respective environments:

### cgn_env (for grasp generation)
```bash
pip install torch numpy open3d transforms3d scipy
```

### mani_env2 (for robot control)
```bash
pip install mani_skill2 gym numpy scipy
```

Alternatively, install all dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🛠 Troubleshooting
### ❌ Issue: Contact GraspNet Server Not Responding
✔ Ensure `cgn_env` is activated before running `socket_server.py`.

### ❌ Issue: Robot Not Moving
✔ Check if `mani_env2` is activated before running `move_robot.py`.

### ❌ Issue: Environment Not Found
✔ Verify that `cgn_env` and `mani_env2` are installed with the correct dependencies.

---

## 👨‍💻 Contributors
- **Your Name** (Modify as needed)

---

## 📜 License
This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

---

## 🤝 Acknowledgements
- **Contact GraspNet** ([GitHub Repo](https://github.com/NVlabs/ContactGraspNet))
- **ManiSkill** ([GitHub Repo](https://github.com/haosulab/ManiSkill2))

---

## ✨ Future Work
- 🛠 Improve grasp selection with heuristic-based refinement.
- 🔄 Extend to multi-object grasping scenarios.
- 🧠 Explore reinforcement learning for fine-tuned grasp execution.
