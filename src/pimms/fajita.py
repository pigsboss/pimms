import numpy as np
from numba import njit
import cv2

# =====================================================================
# 1. 核心底层算子 (Numba JIT 编译，达到 C++ 级别的运行速度)
# =====================================================================

@njit
def _fajita_nearest_2d(f, mask):
    """
    零阶填充 (0-order): 对应原 MATLAB 代码中的 fajitanearest 和 adjaverage。
    通过热传导的方式，利用 8 邻域像素迭代向内填补空洞。
    """
    Y = np.copy(f)
    M = np.copy(mask)
    rows, cols = Y.shape
    
    # 模拟原版 MATLAB 的权重设计: 
    # 正交方向为 1.0，对角线方向为 sqrt(2) ≈ 1.414
    w_ortho = 1.0
    w_diag = 1.41421356 
    
    # 当还有缺失像素 (M 中存在 False) 时继续迭代
    while np.sum(~M) > 0:
        Y_new = np.copy(Y)
        M_new = np.copy(M)
        
        for i in range(rows):
            for j in range(cols):
                if not M[i, j]:  # 如果当前是空洞
                    val_sum = 0.0
                    w_sum = 0.0
                    
                    # 遍历 8 邻域
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0: continue
                            ni, nj = i + di, j + dj
                            
                            # 边界检查
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if M[ni, nj]: # 如果邻居是已知像素
                                    w = w_ortho if (di == 0 or dj == 0) else w_diag
                                    val_sum += Y[ni, nj] * w
                                    w_sum += w
                    
                    # 如果周围有已知像素，则计算加权平均并标记为已知
                    if w_sum > 0.7:  # 对应原代码中的 W > 0.7
                        Y_new[i, j] = val_sum / w_sum
                        M_new[i, j] = True
        
        Y = Y_new
        M = M_new
        
    return Y

@njit
def _laplacian(f):
    """
    计算离散拉普拉斯算子。使用 Neumann 边界条件 (边缘镜像) 防止能量在边界泄漏。
    """
    rows, cols = f.shape
    out = np.zeros_like(f)
    for i in range(rows):
        for j in range(cols):
            val = -4.0 * f[i, j]
            val += f[i-1, j] if i > 0 else f[i, j]
            val += f[i+1, j] if i < rows-1 else f[i, j]
            val += f[i, j-1] if j > 0 else f[i, j]
            val += f[i, j+1] if j < cols-1 else f[i, j]
            out[i, j] = val
    return out

@njit
def _dlt2(f, times):
    """
    对应原代码的 dlt2：连续应用指定次数的拉普拉斯算子
    """
    curr = np.copy(f)
    for _ in range(times):
        curr = _laplacian(curr)
    return curr

@njit
def _fajita_tv_loop(f, mask, order_times, maxloop, precision):
    """
    高阶 TV 最小化迭代，带有标志性的随机松弛因子。
    """
    Y = np.copy(f)
    m_hole = ~mask # True 表示缺失区域 (需要在这些地方做梯度更新)
    
    mean_f = np.mean(Y)
    delta = precision * mean_f
    
    for k in range(maxloop):
        l = _dlt2(Y, order_times)
        g = _dlt2(l, order_times)
        h = _dlt2(g, order_times)
        
        # 计算最速下降步长 gamma: gamma = sum(l.*h) / sum(h.^2)
        num = np.sum(l * h)
        den = np.sum(h * h)
        gamma = num / den if den != 0 else 0.0
        
        # 神来之笔：注入随机涨落因子的梯度下降
        # np.random.rand() 在 Numba JIT 中是完美支持的
        d = np.random.rand() * 2.0 * gamma * g * m_hole
        Y = Y - d
        
        # 收敛条件判定
        if np.max(np.abs(d)) <= delta:
            break
            
    return Y

# =====================================================================
# 2. Python 用户接口层 (处理数据类型、通道和参数逻辑)
# =====================================================================

def fajita(img, mask=None, order=0, precision=1e-3, maxloop=100):
    """
    FAJITA Image inpainting by Fast AdJacent InpainTing Algorithm.
    
    参数:
        img: 输入图像，可以是 2D (灰度) 或 3D (彩色) NumPy 数组。
        mask: 掩膜，布尔数组。True 表示有效像素，False 表示需要修补的空洞。
              如果为 None，则默认 img > 0 的区域为有效像素。
        order: 算子阶数。0 表示最近邻平滑；1及以上表示高阶 TV 迭代。
        precision: 收敛精度阈值。
        maxloop: 最大迭代次数。
    
    返回:
        修复后的图像，数据类型与输入保持一致。
    """
    # 强制转换为 float64 进行高精度物理运算，防止溢出
    f = img.astype(np.float64)
    
    # 解析掩膜
    if mask is None:
        # 对应 MATLAB: m_d = (f>0)
        # 对于彩色图，通常三通道只要有一个 >0 就认为像素有效
        if f.ndim == 3:
            mask = np.any(f > 0, axis=2)
        else:
            mask = (f > 0)
            
    # 解析 Order (支持原版的字符串输入)
    if isinstance(order, str):
        order_str = order.lower()
        if order_str in ['nearest']:
            order = 0
        elif order_str in ['linear', 'bilinear']:
            order = 1
        elif order_str in ['cubic', 'bicubic']:
            order = 3
            
    order_times = order * 2

    # 统一将输入转为 3D (H, W, Channels) 以便遍历处理
    is_3d = (f.ndim == 3)
    if not is_3d:
        f = f[..., np.newaxis]
        
    out = np.zeros_like(f)
    
    # 逐通道进行处理
    for c in range(f.shape[2]):
        f_c = f[:, :, c]
        
        # 1. 无论如何，先进行 0 阶填充作为最佳初值场 (对应 ff = fajitanearest(f,m))
        f_filled = _fajita_nearest_2d(f_c, mask)
        
        # 2. 如果 Order > 0，则进入 TV 最小化迭代
        if order > 0:
            # 注: 原 MATLAB 中比较了直接求梯度和0阶后求梯度的大小，
            # 这里为保证鲁棒性，直接使用 f_filled 作为 TV 的初始场是非常标准且稳定的做法。
            f_final = _fajita_tv_loop(f_filled, mask, order_times, maxloop, precision)
        else:
            f_final = f_filled
            
        out[:, :, c] = f_final
        
    # 还原维度和数据类型
    if not is_3d:
        out = out[:, :, 0]
        
    if img.dtype == np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
        
    return out

# =====================================================================
# 3. 测试与调用示例
# =====================================================================
if __name__ == "__main__":
    # 1. 创建一张渐变的测试图
    size = 200
    test_img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        test_img[:, i] = int(255 * (i / size))
        
    # 2. 模拟 99% 的像素缺失 (随机采样 1%)
    np.random.seed(42)
    mask = (np.random.rand(size, size) < 0.01)
    
    # 破损图像 (只保留 1% 的数据)
    corrupted_img = test_img * mask
    
    # 3. 运行 Fajita
    print("开始运行 Fajita (首次运行 Numba 会触发 JIT 编译，稍等一两秒)...")
    
    # 零阶填充
    result_order0 = fajita(corrupted_img, mask=mask, order=0)
    
    # 高阶 TV 迭代 (启用随机涨落)
    result_order1 = fajita(corrupted_img, mask=mask, order=1, maxloop=500)
    
    print("运行完成！")
    
    # 可视化
    cv2.imshow('1% Sampled Corrupted', corrupted_img)
    cv2.imshow('Fajita Order 0', result_order0)
    cv2.imshow('Fajita Order 1 (TV + Random)', result_order1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()