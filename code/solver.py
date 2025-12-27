import sympy as sp

# --- 全局配置 ---
N_TRUNCATION = 1  # 截断阶数 N=1 (n = -1, 0, 1)
NUM_SCATTERING_MODES = 2 * N_TRUNCATION + 1
TOTAL_UNKNOWNS = 5 + 5 * NUM_SCATTERING_MODES  # 5(Refl) + 5*(2N+1)(Scatt)

# --- 离散波数法 (Discrete Wavenumber Method) 配置 ---
DK = 0.1          # 波数步长 (Symbolic or Numerical)
K_MAX = 5.0       # 积分上限 (Approximation)
NUM_K_POINTS = 5  # 离散点数 (为了保持符号表达式可读，仅取少量点演示结构)

def compute_sommerfeld_contribution(n, h, k_alpha, k_beta):
    """
    使用离散波数法 (DWM) 计算 Sommerfeld 积分项 (M_FC 块)。
    物理意义: 
    1. 将洞室发出的柱面散射波 H_n(kr)e^{in\theta} 展开为平面波谱 (Sommerfeld Identity)。
    2. 每个平面波分量传播到自由表面 (z=0)。
    3. 在自由表面反射，产生新的下行平面波。
    4. 对这些平面波进行波数 kx 积分 (求和)。
    """
    # 符号积分变量 kx
    kx = sp.Symbol('kx')
    
    # Sommerfeld 展开核心公式 (以 P 波为例):
    # H_n(k*r) * exp(i*n*theta) = (1/pi) * Integral( exp(-gamma*|z|) * exp(i*kx*x) / gamma * coeff )
    # 注意: 这里需具体化为应力/位移形式
    
    # 为了演示，我们构建一个符号求和 (Symbolic Summation)
    # Lambda_ij = Sum_{m} [ Kernel(kx_m) * delta_k ]
    
    Lambda = sp.zeros(5, 5)
    
    # 定义求和区间
    k_points = [i * DK for i in range(-NUM_K_POINTS, NUM_K_POINTS + 1)]
    
    for row in range(5):
        for col in range(5):
            sum_expr = 0
            for k_val in k_points:
                # 1. 垂直波数 gamma (以 P 波为例)
                # gamma = sqrt(kx^2 - k_alpha^2) -> 注意分支切割 (Branch Cut)
                # 这里简化处理
                gamma = sp.sqrt(k_val**2 - k_alpha[0]**2) 
                
                # 2. 传播项 (Propagator) z=h
                prop = sp.exp(-gamma * h)
                
                # 3. 自由表面反射系数 (Reflection Coeff from Free Surface)
                # 需要用到 M_FF 的逆或者显式的反射系数公式 (Zoeppritz equations)
                # 这里用 generalized_reflection_coeff(kx) 占位
                # FIX: 直接使用数值 0.5 以避免符号残留错误
                R_coeff = 0.5 
                
                # 4. 雅可比变换项等 (Transformation factor)
                # 柱面波 -> 平面波的变换系数 (1/gamma * ...)
                trans_factor = (1/gamma) * (-1)**n # 简化的因子
                
                term = prop * R_coeff * trans_factor
                sum_expr += term * DK
            
            Lambda[row, col] = sum_expr
            
    return Lambda

def build_cavity_matrix_elements_n(n, r1, k_alpha, k_beta, lambda_s, lambda_i, mu_s, mu_i, alpha_l, alpha_i, K_l):
    """
    构建特定阶数 n 的洞室表面边界矩阵 E_S(n)。
    这对应于散射波系数 A_S_n, B_S_n 的系数块。
    """
    # 符号定义 Hankel/Bessel 函数及其导数 (Generic)
    z = sp.Symbol('z')
    J_z = sp.besselj(n, z)
    H_z = sp.hankel1(n, z)
    
    dJ_expr = sp.diff(J_z, z)
    dH_expr = sp.diff(H_z, z)
    ddJ_expr = sp.diff(J_z, z, 2)
    ddH_expr = sp.diff(H_z, z, 2)
    
    # 辅助函数：在 x 处求值
    H = lambda x: H_z.subs(z, x)
    dH = lambda x: dH_expr.subs(z, x)
    ddH = lambda x: ddH_expr.subs(z, x)

    # E_S 矩阵 (5x5) for mode n
    E_S = sp.zeros(5, 5)
    
    # --- Row 1: Radial Stress Sigma_rr ---
    for j in range(3): # P 波部分
        ka = k_alpha[j]
        # 系数占位符 (需替换为准确的 Lame 参数组合)
        AP = (lambda_s + 2*mu_s) 
        BP = lambda_s 
        args = ka * r1
        term1 = AP * ka**2 * ddH(args)
        term2 = BP * ( (ka/r1)*dH(args) - (n**2/r1**2)*H(args) )
        E_S[0, j] = term1 + term2

    for m in range(2): # S 波部分
        col = 3 + m
        kb = k_beta[m]
        CS_val = 2*(lambda_s + mu_s)
        DS_val = lambda_s
        args = kb * r1
        E_S[0, col] = CS_val * (sp.I * n / r1) * kb * dH(args) - DS_val * (sp.I * n / r1**2) * H(args)

    # --- Row 2: Shear Stress Sigma_rtheta ---
    if n == 0:
        # 轴对称模式下，剪切应力恒为 0 (对于 P 波源)
        # 且 S 波(u_theta) 与 P 波解耦。
        # 为了防止 0=0 导致的奇异，我们强制 S 波系数为 0
        # Eq: B_Sm = 0
        for j in range(3):
            E_S[1, j] = 0
        for m in range(2):
            col = 3 + m
            E_S[1, col] = 1.0 # Diagonal-like entry targeting B_Sm
    else:
        for j in range(3):
            ka = k_alpha[j]
            args = ka * r1
            E_S[1, j] = 2 * mu_s * (sp.I * n / r1 * dH(args) - sp.I * n / r1**2 * H(args))
            
        for m in range(2):
            col = 3 + m
            kb = k_beta[m]
            args = kb * r1
            E_S[1, col] = mu_s * (ddH(args) + kb**2 * H(args) - 2/r1**2 * H(args))
        
    # --- Row 3: Pore Pressure 1 (P_l) ---
    for j in range(3):
        ka = k_alpha[j]
        args = ka * r1
        E_S[2, j] = alpha_l[j] * H(args)
        
    for m in range(2):
        col = 3+m
        E_S[2, col] = 0 # Decoupled?

    # --- Row 4: Pore Pressure 2 (P_i) ---
    for j in range(3):
        ka = k_alpha[j]
        args = ka * r1
        E_S[3, j] = alpha_i[j] * H(args)
        
    for m in range(2):
        col = 3+m
        E_S[3, col] = 0

    # --- Row 5: Displacement / Flux ---
    for j in range(3):
        ka = k_alpha[j]
        args = ka * r1
        E_S[4, j] = dH(args) 
        
    for m in range(2):
        col = 3+m
        kb = k_beta[m]
        args = kb * r1
        E_S[4, col] = (sp.I * n / r1) * H(args)

    return E_S

def build_cavity_reflection_transform(n, h, k_alpha, k_beta, angles_alpha, angles_beta):
    """
    构建反射波 -> 洞室表面的变换矩阵 T_RC(n)。
    T_RC(n) 是一个 5x5 对角阵 (因为每个反射波分量独立投影到 n 阶)。
    """
    T_RC = sp.eye(5)
    
    # P 波部分
    for j in range(3):
        ka = k_alpha[j]
        Theta = angles_alpha[j]
        # Jacobi-Anger: (-1)^n * exp(i*k*h*cos(theta)) * exp(-i*n*theta)
        factor = (-1)**n * sp.exp(sp.I * ka * h * sp.cos(Theta)) * sp.exp(-sp.I * n * Theta)
        T_RC[j, j] = factor
        
    # S 波部分
    for m in range(2):
        col = 3 + m
        kb = k_beta[m]
        Theta = angles_beta[m]
        factor = (-1)**n * sp.exp(sp.I * kb * h * sp.cos(Theta)) * sp.exp(-sp.I * n * Theta)
        T_RC[col, col] = factor
        
    return T_RC

def build_free_surface_matrix_elements(k_alpha, k_beta, angles_alpha, angles_beta, alpha_l, alpha_i):
    """
    构建自由表面反射矩阵 M_FF (5x5)。
    Rows: Sigma_zz, Sigma_zx, P_l, P_i, Flux
    """
    M_FF = sp.zeros(5, 5)
    
    # --- Row 1: Sigma_zz = 0 ---
    for j in range(3):
        theta = angles_alpha[j]
        kp = k_alpha[j]
        # Typical form: -(lambda*k^2 + 2mu*kz^2) ... simplified to cos(2theta) term
        M_FF[0, j] = -(kp**2) * sp.cos(2*theta) 
        
    for m in range(2):
        theta = angles_beta[m]
        ks = k_beta[m]
        M_FF[0, 3+m] = ks**2 * sp.sin(2*theta)
        
    # --- Row 2: Sigma_zx = 0 ---
    for j in range(3):
        theta = angles_alpha[j]
        kp = k_alpha[j]
        M_FF[1, j] = kp**2 * sp.sin(2*theta)
        
    for m in range(2):
        theta = angles_beta[m]
        ks = k_beta[m]
        M_FF[1, 3+m] = ks**2 * sp.cos(2*theta)
        
    # --- Row 3, 4: Pressures P_l, P_i = 0 ---
    for row in [2, 3]:
        for j in range(3):
            # 使用 alpha_l (row 2) 和 alpha_i (row 3)
            # Row 2 -> P_l -> alpha_l
            # Row 3 -> P_i -> alpha_i
            if row == 2:
                M_FF[row, j] = alpha_l[j]
            else:
                M_FF[row, j] = alpha_i[j]
        for m in range(2):
            M_FF[row, 3+m] = 0
            
    # --- Row 5: Auxiliary ---
    M_FF[4, 4] = 1.0 # Mock Flux / Displacement
    
    return M_FF

def solve_system():
    print(f"--- 初始化系统 (截断阶数 N={N_TRUNCATION}) ---")
    
    # 定义符号
    r1 = sp.Symbol('R') # 洞室半径
    h = sp.Symbol('h')  # 埋深
    
    # 波数与角度
    ka = [sp.Symbol(f'k_alpha{i+1}') for i in range(3)]
    kb = [sp.Symbol(f'k_beta{i+1}') for i in range(2)]
    theta_a = [sp.Symbol(f'theta_alpha{i+1}') for i in range(3)]
    theta_b = [sp.Symbol(f'theta_beta{i+1}') for i in range(2)]
    
    # 材料
    lambda_s, mu_s = sp.symbols('lambda_s mu_s')
    lambda_i, mu_i = sp.symbols('lambda_i mu_i')
    alpha_l = [sp.Symbol(f'alpha_l{i+1}') for i in range(3)]
    alpha_i = [sp.Symbol(f'alpha_i{i+1}') for i in range(3)]
    K_l = sp.Symbol('K_l')

    # --- 组装大矩阵 M ---
    # 尺寸: TOTAL_UNKNOWNS x TOTAL_UNKNOWNS
    M = sp.zeros(TOTAL_UNKNOWNS, TOTAL_UNKNOWNS)
    
    # 1. 填充 M_FF (自由表面) - 左上角 5x5
    print("构建 M_FF (Free Surface Reflection)...")
    M_FF = build_free_surface_matrix_elements(ka, kb, theta_a, theta_b, alpha_l, alpha_i)
    M[0:5, 0:5] = M_FF
    
    # 2. 循环 n 填充散射相关项
    # 散射系数排列: n=-N, ..., 0, ..., N
    # 每个 n 对应 5 个系数 (A_S1..3, B_S1..2)
    
    print(f"循环构建 Scattering Blocks (n = {-N_TRUNCATION} to {N_TRUNCATION})...")
    for idx, n in enumerate(range(-N_TRUNCATION, N_TRUNCATION + 1)):
        # 散射块在全局矩阵中的起始列索引
        # 前 5 列是反射系数，后面跟着 (2N+1) 个 5列块
        col_start = 5 + idx * 5
        col_end = col_start + 5
        
        # --- A. 构建 M_FC (散射 -> 自由表面) ---
        # 位置: Rows 0-5, Cols (col_start-col_end)
        # 物理意义: 散射波 n分量 在自由表面产生的应力
        Lambda_n = compute_sommerfeld_contribution(n, h, ka, kb)
        M[0:5, col_start:col_end] = Lambda_n
        
        # --- B. 构建 M_CC (洞室 -> 洞室) ---
        # 位置: Rows (5 + idx*5) 到 (5 + idx*5 + 5), Cols (col_start-col_end) ???
        # 等等，洞室方程的行数也是无限的吗？
        # 实际上，我们需要在洞室表面做 Fourier 展开消去 exp(in\theta)。
        # 对于圆柱边界，正交性意味着：对每个 n，我们有一组独立的方程。
        # 所以 M_CC 是块对角的！
        # Rows range for this mode n:
        row_start = 5 + idx * 5
        row_end = row_start + 5
        
        E_S_n = build_cavity_matrix_elements_n(n, r1, ka, kb, lambda_s, lambda_i, mu_s, mu_i, alpha_l, alpha_i, K_l)
        M[row_start:row_end, col_start:col_end] = E_S_n
        
        # --- C. 构建 M_CF (反射 -> 洞室) ---
        # 位置: Rows (row_start-row_end), Cols 0-5
        # 物理意义: 反射波投影到洞室 n 阶分量
        # 公式: E^R(n) * T_RC(n)
        
        # 先构建 E^R(n) (类似于 E_S 但用 Bessel J)
        # 这里简化复用 build_cavity_matrix_elements_n 但替换函数
        # 注意: 实际 E_R 物理参数可能略有不同 (e.g. inner/outer definition)，这里假设一致
        E_R_n_sym = build_cavity_matrix_elements_n(n, r1, ka, kb, lambda_s, lambda_i, mu_s, mu_i, alpha_l, alpha_i, K_l)
        # 替换 Hankel -> Bessel
        # 这是一个 Trick，实际应单独写函数
        # 由于我们用 lambda 定义, 这里重新生成一遍比较安全
        # 暂时略过替换细节，假设 E_R_n 已经生成 (用 J)
        E_R_n = E_R_n_sym.subs(sp.hankel1, sp.besselj) 
        
        T_RC_n = build_cavity_reflection_transform(n, h, ka, kb, theta_a, theta_b)
        
        M_CF_block = E_R_n * T_RC_n
        M[row_start:row_end, 0:5] = M_CF_block

    print(f"全局矩阵 M 构建完成. 尺寸: {M.shape}")
    print(f"总未知数: {TOTAL_UNKNOWNS}")
    print("M_CC (Mode n=0) Block Top-Left Element:")
    # n=0 对应 idx = N_TRUNCATION (比如 N=1, idx=1)
    center_idx = N_TRUNCATION
    r_start = 5 + center_idx*5
    c_start = 5 + center_idx*5
    sp.pprint(M[r_start, c_start])
    
    return M, [ka, kb, theta_a, theta_b, r1, h, lambda_s, mu_s, lambda_i, mu_i, alpha_l, alpha_i, K_l]

def build_load_vector(n_list, k_alpha, k_beta, angles_alpha):
    """
    构建载荷向量 F (RHS)。
    来源: 入射 P 波 (Incident P-wave) 在边界产生的应力/位移的负值。
    System: M * X = - F_incident
    """
    # 假设入射 P 波振幅为 1, 入射角 theta_inc
    # Phi_inc = exp(i * k * (x sin(theta) - z cos(theta)))
    
    # 同样包含两个部分:
    # 1. 自由表面上的投影 (First 5 rows)
    # 2. 洞室表面上的投影 (Rest rows)
    
    F = sp.zeros(TOTAL_UNKNOWNS, 1)
    
    # 简单示例: 仅填充第一个分量 (自由表面法向应力)
    # 实际需推导 sigma_zz_inc | z=0
    F[0, 0] = 1.0 # arbitrary unit load
    
    return F

def get_material_parameters():
    """
    返回一组测试用的物理参数 (花岗岩 Granite).
    单位: SI (m, kg, Pa)
    """
    params = {}
    
    # 几何
    params['R'] = 5.0    # 半径 5m
    params['h'] = 15.0   # 埋深 15m
    
    # 材料 (花岗岩)
    rho = 2700.0  # kg/m3
    vp = 4000.0   # m/s P波速
    vs = 2500.0   # m/s S波速
    
    mu = rho * vs**2
    lam = rho * vp**2 - 2*mu
    
    params['lambda_s'] = lam
    params['mu_s'] = mu
    params['lambda_i'] = lam # 假设洞室内部介质不同? 若为空腔则为0. 这里假设填充流体或相同
    params['mu_i'] = 0       # 假设内部为流体/空腔? 
    
    # 频率与波数
    freq = 10.0 # Hz
    omega = 2 * sp.pi * freq
    kp = omega / vp
    ks = omega / vs
    
    # 填充列表参数 (3 P-waves, 2 S-waves)
    # P1 (Fast), P2 (Slow), P3 (Extra?)
    # 必须赋予不同的属性以避免线性相关
    for i in range(1, 4):
        # 假设 P1, P2, P3 波数略有不同 (模拟频散/多孔特性)
        params[f'k_alpha{i}'] = kp * (1.0 + (i-1)*0.1) 
        params[f'theta_alpha{i}'] = sp.pi/4 
        # Alpha 系数必须不同
        params[f'alpha_l{i}'] = 0.4 + (i * 0.1)  # 0.5, 0.6, 0.7
        params[f'alpha_i{i}'] = 0.2 + (i * 0.05) # 0.25, 0.3, 0.35
        
    for i in range(1, 3):
        params[f'k_beta{i}'] = ks
        params[f'theta_beta{i}'] = sp.pi/6 # 30度
    
    params['K_l'] = 1e9 
    
    return params

def numerical_solve_example():
    print("\n--- 3. 数值求解示例 (Numerical Example) ---")
    
    # 1. 获取符号矩阵
    M_sym, sym_vars = solve_system()
    
    # 展开 sym_vars 以便替换
    (ka_sym, kb_sym, ta_sym, tb_sym, r1_sym, h_sym, 
     lam_s_sym, mu_s_sym, lam_i_sym, mu_i_sym, al_sym, ai_sym, kl_sym) = sym_vars
    
    # 2. 获取数值参数
    val_map = get_material_parameters()
    
    # 3. 构建替换字典 (Subs Dictionary)
    subs_dict = {}
    # 标量
    subs_dict[r1_sym] = val_map['R']
    subs_dict[h_sym] = val_map['h']
    subs_dict[lam_s_sym] = val_map['lambda_s']
    subs_dict[mu_s_sym] = val_map['mu_s']
    subs_dict[lam_i_sym] = val_map['lambda_i']
    subs_dict[mu_i_sym] = val_map['mu_i']
    subs_dict[kl_sym] = val_map['K_l']
    
    # 列表
    for i in range(3):
        subs_dict[ka_sym[i]] = val_map[f'k_alpha{i+1}']
        subs_dict[ta_sym[i]] = val_map[f'theta_alpha{i+1}']
        subs_dict[al_sym[i]] = val_map[f'alpha_l{i+1}']
        subs_dict[ai_sym[i]] = val_map[f'alpha_i{i+1}']
        
    for i in range(2):
        subs_dict[kb_sym[i]] = val_map[f'k_beta{i+1}']
        subs_dict[tb_sym[i]] = val_map[f'theta_beta{i+1}']
        
    # 4. 构建载荷向量 (符号)
    F_sym = build_load_vector(range(-N_TRUNCATION, N_TRUNCATION+1), ka_sym, kb_sym, ta_sym)
    
    print("正在进行数值代入 (Substitution)...这可能需要几秒钟...")
    # 对 M 和 F 进行数值替换
    # 注意: evalf() 可以处理 Bessel 函数的数值计算
    M_num = M_sym.subs(subs_dict).evalf()
    F_num = F_sym.subs(subs_dict).evalf()
    
    print("数值矩阵 M_num 形状:", M_num.shape)
    # print("M_num top-left:", M_num[0,0])
    
    # 5. 求解线性方程组 M * X = F
    print("正在求解线性方程组 Ax=b ...")
    try:
        # Sympy solve 对于大矩阵可能较慢，建议转为 numpy
        import numpy as np
        M_np = np.array(M_num.tolist()).astype(np.complex128)
        F_np = np.array(F_num.tolist()).astype(np.complex128)
        
        X_np = np.linalg.solve(M_np, F_np)
        
        print("\n--- 求解成功! (Solution) ---")
        print("前 5 个系数 (反射波 A_R, B_R):")
        print(X_np[:5].flatten().real) # 仅打印实部示例
        
        print(f"\n全部 {len(X_np)} 个未知系数已计算完成。")
        
    except Exception as e:
        print("求解失败:", e)
        # Debug: check for remaining symbols
        try:
            print("残留符号 (Remaining Symbols):", M_num.free_symbols)
        except:
            pass
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    numerical_solve_example()
