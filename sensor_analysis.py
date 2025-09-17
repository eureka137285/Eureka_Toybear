import numpy as np
import pandas as pd

def print_data(data: np.ndarray, title: str):
    """使用 pandas DataFrame 格式化打印数据。"""
    print(title)
    if data is None or data.size == 0:
        print("无数据\n")
        return
    
    # 为传感器创建列标题
    columns = [f"Sensor {i+1}" for i in range(data.shape[1])]
    # 为时间点创建索引
    index = [f"T_{i+1}" for i in range(data.shape[0])]
    
    df = pd.DataFrame(data, columns=columns, index=index)
    print(df.to_string(float_format="%.2f"))
    print("\n")

def input_data_manually() -> np.ndarray:
    """提示用户手动输入传感器数据。"""
    try:
        time_points = int(input("请输入时间点数量: "))
        sensors = int(input("请输入传感器数量: "))
        
        if time_points <= 0 or sensors <= 0:
            print("错误：时间点和传感器数量必须大于0。\n")
            return None
            
    except ValueError:
        print("错误：请输入有效的整数。\n")
        return None

    all_data = []
    print("请按时间顺序输入每个时间点的传感器数据 (用空格分隔):")
    for i in range(time_points):
        while True:
            row_input = input(f"时间点 {i+1}: ")
            try:
                # 按空格分割并转换为浮点数
                row_data = [float(val) for val in row_input.strip().split()]
                if len(row_data) == sensors:
                    all_data.append(row_data)
                    break
                else:
                    print(f"错误：您输入了 {len(row_data)} 个值，但需要 {sensors} 个传感器数据。请重试。")
            except ValueError:
                print("错误：输入包含非数字值。请重新输入该行。")
    
    print("\n数据输入完成。
")
    return np.array(all_data)

def conv1d(input_data: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """对输入数据执行一维卷积。"""
    input_rows, input_cols = input_data.shape
    kernel_rows, kernel_cols = kernel.shape

    if input_cols != kernel_cols:
        print(f"错误：输入列数 ({input_cols}) 和卷积核列数 ({kernel_cols}) 不匹配。")
        return None

    # 计算输出维度
    output_rows = (input_rows - kernel_rows) // stride + 1
    if output_rows <= 0:
        print("警告：卷积输出大小为0。可能是输入太小或卷积核太大。")
        return np.array([])
        
    output = np.zeros((output_rows, input_cols))

    # 应用卷积
    for r in range(output_rows):
        for c in range(input_cols):
            window = input_data[r*stride : r*stride + kernel_rows, c]
            kernel_col = kernel[:, c]
            output[r, c] = np.sum(window * kernel_col)
            
    return output

def relu(input_data: np.ndarray) -> np.ndarray:
    """应用 ReLU 激活函数。"""
    return np.maximum(0, input_data)

def max_pool1d(input_data: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    """执行一维最大池化。"""
    input_rows, input_cols = input_data.shape
    
    output_rows = (input_rows - pool_size) // stride + 1
    if output_rows <= 0:
        print("警告：最大池化输出大小为0。可能是输入太小或池化大小太大。")
        return np.array([])

    output = np.zeros((output_rows, input_cols))

    for r in range(output_rows):
        for c in range(input_cols):
            window = input_data[r*stride : r*stride + pool_size, c]
            output[r, c] = np.max(window)
            
    return output

def simple_moving_average(input_data: np.ndarray, period: int) -> np.ndarray:
    """计算简单移动平均 (SMA)。"""
    if period > len(input_data):
        print(f"错误：SMA周期 ({period}) 不能大于数据点数 ({len(input_data)})。")
        return None
        
    df = pd.DataFrame(input_data)
    sma = df.rolling(window=period).mean().dropna()
    return sma.values

def run_cnn(sensor_data: np.ndarray):
    """运行完整的CNN处理流程。"""
    print("--- CNN 处理结果 ---")
    
    kernel_size = min(3, sensor_data.shape[0])
    if kernel_size == 0:
        print("数据点不足，无法创建卷积核。")
        return
        
    # 创建一个简单的卷积核
    kernel = np.array([[0.5 - (0.1 * i) + (0.1 * j) for j in range(sensor_data.shape[1])] for i in range(kernel_size)])
    print_data(kernel, "使用的卷积核:")

    # 1. 卷积
    conv_output = conv1d(sensor_data, kernel)
    if conv_output is None or conv_output.size == 0:
        print("卷积失败，中止CNN处理。")
        return
    print_data(conv_output, "1. 卷积层输出:")

    # 2. ReLU 激活
    relu_output = relu(conv_output)
    print_data(relu_output, "2. ReLU 激活输出:")

    # 3. 最大池化
    pool_size = min(2, relu_output.shape[0])
    if pool_size == 0:
        print("数据点不足以进行池化，使用ReLU输出作为最终特征向量。")
        final_output = relu_output
    else:
        final_output = max_pool1d(relu_output, pool_size=pool_size, stride=pool_size)

    print_data(final_output, "3. 最大池化输出 (最终CNN特征向量):")

def run_sma(sensor_data: np.ndarray):
    """运行SMA处理流程。"""
    print("--- SMA 处理结果 ---")
    
    period = min(3, sensor_data.shape[0])
    if period == 0:
        print("数据点不足，无法计算SMA。")
        return
        
    print(f"使用 {period} 个时间点的简单移动平均。\n")
    sma_output = simple_moving_average(sensor_data, period)
    if sma_output is not None:
        print_data(sma_output, "SMA 输出:")

def main():
    """主函数，驱动程序运行。"""
    while True:
        sensor_data = input_data_manually()
        
        if sensor_data is not None:
            print_data(sensor_data, "原始传感器数据:")
            
            while True:
                choice = input("请选择要执行的分析 ('cnn', 'sma', 'both', 或 'none'): ").lower()
                if choice in ['cnn', 'sma', 'both', 'none']:
                    break
                else:
                    print("无效的选择，请输入 'cnn', 'sma', 'both', 或 'none'。")

            if choice == 'cnn':
                run_cnn(sensor_data)
            elif choice == 'sma':
                run_sma(sensor_data)
            elif choice == 'both':
                run_cnn(sensor_data)
                run_sma(sensor_data)

        another = input("是否要输入新数据并再次分析? (y/n): ").lower()
        if another != 'y':
            break
            
    print("程序已退出。")

if __name__ == "__main__":
    main()