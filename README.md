# TOG: Training-free Task-oriented Grasp Generation

This project integrates **Contact GraspNet** and **ManiSkill** for robotic grasping. The workflow involves running a **grasping server** and a **robot movement script** in separate environments.

---

## **Setup Instructions**

### **1️⃣ Start the Grasping Server**
In the first terminal, activate the **Contact GraspNet** environment and run the server:

```bash
conda activate cgn_env
cd ./third_party/contact_graspnet/contact_graspnet/
python socket_server.py
