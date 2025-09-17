import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import base64
import requests
import os

# ============================================================================== 
# 核心数据处理逻辑 (从 C++ 逻辑翻译而来)
# ============================================================================== 

def conv1d(input_data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    一维卷积层前向传播
    """
    input_rows, input_cols = input_data.shape
    kernel_rows, kernel_cols = kernel.shape
    stride = 1

    if input_cols != kernel_cols:
        raise ValueError(f"输入数据的列数({input_cols})与卷积核的列数({kernel_cols})不匹配")

    output_rows = (input_rows - kernel_rows) // stride + 1
    if output_rows <= 0:
        print("警告：卷积输出行数为0，可能是输入数据太少或卷积核太大")
        return np.array([])

    output = np.zeros((output_rows, input_cols))

    for r in range(output_rows):
        for c in range(input_cols):
            # 提取与卷积核对应区域的数据进行逐元素乘法和求和
            current_slice = input_data[r * stride : r * stride + kernel_rows, c]
            kernel_slice = kernel[:, c]
            output[r, c] = np.sum(current_slice * kernel_slice)
            
    return output

def relu(input_data: np.ndarray) -> np.ndarray:
    """
    ReLU激活函数
    """
    return np.maximum(0, input_data)

def max_pool1d(input_data: np.ndarray, pool_size: int) -> np.ndarray:
    """
    一维最大池化层前向传播
    """
    if input_data.size == 0:
        return np.array([])
        
    input_rows, input_cols = input_data.shape
    stride = pool_size

    output_rows = (input_rows - pool_size) // stride + 1
    if output_rows <= 0:
        print("警告：池化输出行数为0，可能是输入数据太少或池化大小太大")
        return np.array([])

    output = np.zeros((output_rows, input_cols))

    for r in range(output_rows):
        for c in range(input_cols):
            # 在池化窗口内找到最大值
            current_slice = input_data[r * stride : r * stride + pool_size, c]
            output[r, c] = np.max(current_slice)
            
    return output

def run_cnn_processing(sensor_data: np.ndarray) -> np.ndarray:
    """
    执行完整的CNN处理流程
    """
    if sensor_data.size == 0:
        messagebox.showerror("错误", "传感器数据为空，无法处理。")
        return np.array([])

    # 1. 定义卷积核
    num_sensors = sensor_data.shape[1]
    kernel_size = min(3, sensor_data.shape[0])
    
    # 创建一个简单的卷积核
    kernel = np.array([[0.5 - (0.1 * i) + (0.1 * j) for j in range(num_sensors)] for i in range(kernel_size)])
    print("使用的卷积核:\n", kernel)

    # 2. 卷积操作
    conv_output = conv1d(sensor_data, kernel)
    if conv_output.size == 0:
        messagebox.showwarning("警告", "卷积操作失败，无法继续。")
        return np.array([])
    print("卷积层输出:\n", conv_output)

    # 3. ReLU激活
    relu_output = relu(conv_output)
    print("ReLU激活后输出:\n", relu_output)

    # 4. 最大池化
    pool_size = min(2, relu_output.shape[0])
    if pool_size == 0: # 如果relu输出行数小于2，则无法池化
        final_output = relu_output
        print("池化操作跳过，使用ReLU激活后的输出作为最终特征向量")
    else:
        final_output = max_pool1d(relu_output, pool_size)
        print("最大池化后输出 (CNN特征向量):\n", final_output)
        
    return final_output

# ============================================================================== 
# GitHub 文件上传功能
# ============================================================================== 

def upload_to_github(token, repo_name, file_path, content):
    """
    将文件内容上传到指定的GitHub仓库
    """
    owner, repo = repo_name.split('/')
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    
    data = {
        "message": f"feat: Add processed data {os.path.basename(file_path)}",
        "content": encoded_content
    }
    
    # 检查文件是否存在，如果存在则需要提供SHA
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        if r.status_code == 200:
            data['sha'] = r.json()['sha']
            print(f"文件 {file_path} 已存在，将进行更新。")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code != 404: # 404表示文件不存在，是正常情况
            raise e

    response = requests.put(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


# ============================================================================== 
# GUI 应用主逻辑
# ============================================================================== 

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("传感器数据处理器")
        self.root.geometry("400x200")

        self.label = tk.Label(root, text="请选择一个Excel文件进行CNN处理", font=("Arial", 12))
        self.label.pack(pady=20)

        self.process_button = tk.Button(root, text="选择文件并开始处理", command=self.process_file)
        self.process_button.pack(pady=10)

        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=5)

    def process_file(self):
        file_path = filedialog.askopenfilename(
            title="选择传感器数据Excel文件",
            filetypes=(("Excel Files", "*.xlsx *.xls"), ("All files", "*.*"))
        )
        if not file_path:
            return

        try:
            # 1. 读取数据
            self.status_label.config(text="正在读取Excel文件...")
            df = pd.read_excel(file_path)
            sensor_data = df.to_numpy()
            print("从Excel读取的原始数据:\n", sensor_data)

            # 2. CNN处理
            self.status_label.config(text="正在进行CNN数据处理...")
            cnn_features = run_cnn_processing(sensor_data)
            
            if cnn_features.size == 0:
                self.status_label.config(text="处理失败，请查看控制台输出。")
                return

            # 3. 保存结果到CSV
            result_df = pd.DataFrame(cnn_features, columns=[f"Feature_{i+1}" for i in range(cnn_features.shape[1])])
            output_filename = "cnn_processed_result.csv"
            result_df.to_csv(output_filename, index=False)
            messagebox.showinfo("成功", f"处理完成！结果已保存到 {output_filename}")

            # 4. 上传到GitHub
            if messagebox.askyesno("上传", "是否要将结果文件上传到GitHub?"):
                self.upload_result(output_filename)

        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {e}")
            self.status_label.config(text="操作失败")

    def upload_result(self, file_to_upload):
        try:
            token = simpledialog.askstring("GitHub Token", "请输入您的GitHub Personal Access Token:", show='*')
            if not token:
                messagebox.showwarning("取消", "未提供Token，上传已取消。")
                return

            repo_name = "eureka137285/Eureka_Toybear"
            self.status_label.config(text="正在上传到GitHub...")

            with open(file_to_upload, 'r', encoding='utf-8') as f:
                content = f.read()

            upload_info = upload_to_github(token, repo_name, file_to_upload, content)
            
            messagebox.showinfo("上传成功", f"文件已成功上传/更新到 {repo_name}！\nCommit SHA: {upload_info['commit']['sha'][:7]}")
            self.status_label.config(text="上传完成！")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                messagebox.showerror("GitHub错误", "授权失败！请检查您的Token是否正确且具有repo权限。")
            else:
                messagebox.showerror("GitHub错误", f"上传失败: {e.response.text}")
            self.status_label.config(text="上传失败")
        except Exception as e:
            messagebox.showerror("错误", f"上传过程中发生未知错误: {e}")
            self.status_label.config(text="上传失败")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()