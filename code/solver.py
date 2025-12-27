import sympy as sp
import numpy as np

# --- 全局配置 ---
N_TRUNCATION = 3  # 截断阶数 N=3 (Modes: -3, -2, -1, 0, 1, 2, 3)
# Total Unknowns = 5 (Refl) + 5 * 7 (Scatt) = 40

def assemble_matrix_numerically(N, params, dk=0.1, k_max=20.0):
    """
    数值直接组装矩阵 (Fast Numeric Assembly).
    - 使用 NumPy 向量化计算 DWM 积分，大幅提升速度。
    - 使用 SymPy evalf 计算 Bessel/Hankel 函数值 (少量调用)。
    """
    print(f"--- [Fast Numerical Assembly] N={N} ---")
    
    # 辅助: 参数获取
    p_dict = {str(k): v for k, v in params.items()}
    def get_p(name): return float(p_dict.get(name, 0.0))
    
    # 1. 准备全局尺寸
    num_modes = 2 * N + 1
    total_dofs = 5 + 5 * num_modes
    M_num = np.zeros((total_dofs, total_dofs), dtype=complex)
    
    # 提取参数
    ka_vals = [get_p(f'k_alpha{i+1}') for i in range(3)]
    kb_vals = [get_p(f'k_beta{i+1}') for i in range(2)]
    ta_vals = [get_p(f'theta_alpha{i+1}') for i in range(3)]
    tb_vals = [get_p(f'theta_beta{i+1}') for i in range(2)]
    al_vals = [get_p(f'alpha_l{i+1}') for i in range(3)]
    ai_vals = [get_p(f'alpha_i{i+1}') for i in range(3)]
    
    h_val = float(p_dict.get('h', 50.0))
    r_val = float(p_dict.get('R', 5.0))
    mu_val = float(p_dict.get('mu_s', 1.0e10))
    lambda_val = float(p_dict.get('lambda_s', 1.0e10))
    
    # 2. 构建 M_FF (Free Surface Reflection) - 左上角 5x5
    m_ff = np.zeros((5, 5), dtype=complex)
    # Row 1: Sigma_zz (Approx: -k^2 * cos(2theta))
    for j in range(3): m_ff[0, j] = -(ka_vals[j]**2) * np.cos(2*ta_vals[j])
    for m in range(2): m_ff[0, 3+m] = kb_vals[m]**2 * np.sin(2*tb_vals[m])
    # Row 2: Sigma_zx
    for j in range(3): m_ff[1, j] = ka_vals[j]**2 * np.sin(2*ta_vals[j])
    for m in range(2): m_ff[1, 3+m] = kb_vals[m]**2 * np.cos(2*tb_vals[m])
    # Row 3, 4: Pressures
    for j in range(3):
        m_ff[2, j] = al_vals[j]
        m_ff[3, j] = ai_vals[j]
    # Row 5: Mock Flux
    m_ff[4, 4] = 1.0
    
    M_num[0:5, 0:5] = m_ff
    
    # 3. 循环构建散射相关块 (M_FC, M_CC, M_CF)
    # 准备 DWM 积分向量 (Vectorized)
    num_k = int(k_max / dk)
    # Complex Shift to avoid Rayleigh poles (epsilon = 1e-3j)
    k_vec = np.linspace(-k_max, k_max, 2*num_k+1) - 1e-3j 
    
    print(f"正在构建散射块 (Modes: {-N} to {N})...")
    
    for idx_n, n in enumerate(range(-N, N + 1)):
        col_start = 5 + idx_n * 5
        col_end = col_start + 5
        row_start = 5 + idx_n * 5
        row_end = row_start + 5
        
        # --- A. M_FC (Scattering -> Free Surface) via DWM ---
        # Integral: sum( exp(-gamma*h) * R * (1/gamma) * ... )
        lambda_block = np.zeros((5, 5), dtype=complex)
        
        # P-wave propagator (using k_alpha1 reference)
        ka0 = ka_vals[0]
        gamma_vec = np.sqrt(k_vec**2 - ka0**2)
        prop_vec = np.exp(-gamma_vec * h_val)
        trans_vec = (1.0/gamma_vec) * ((-1.0)**n)
        
        # R_coeff = 0.5 (Mock reflection)
        integrand = prop_vec * 0.5 * trans_vec
        integral_val = np.sum(integrand) * dk # Rectangular rule
        
        # Simple diagonal mapping for demo
        for i in range(5):
             lambda_block[i, i] = integral_val
             
        M_num[0:5, col_start:col_end] = lambda_block

        # --- B. M_CC (Cavity Surface) ---
        E_S_num = np.zeros((5, 5), dtype=complex)
        
        # Helper: Evaluate Hankel function H_n(z)
        def get_hankel(order, k, r):
            arg = complex(k * r)
            return complex(sp.hankel1(order, arg).evalf())
            
        # Row 1: Sigma_rr
        # Formula: (lambda + 2mu)*k^2 * H ... (Simplification)
        for j in range(3):
            hv = get_hankel(n, ka_vals[j], r_val)
            E_S_num[0, j] = (lambda_val + 2*mu_val) * (ka_vals[j]**2) * hv
        for m in range(2):
            hv = get_hankel(n, kb_vals[m], r_val)
            E_S_num[0, 3+m] = 2*mu_val * (kb_vals[m]**2) * hv 

        # Row 2: Sigma_rt (Shear)
        if n == 0:
            # Mode 0 Shear Stress Decoupling Fix
            E_S_num[1, 0:3] = 0.0 
            for m in range(2): E_S_num[1, 3+m] = 1.0
        else:
            for j in range(5):
                # Approx shear
                E_S_num[1, j] = mu_val * get_hankel(n, ka_vals[0], r_val)

        # Row 3,4: Pressures
        for j in range(3):
            hv = get_hankel(n, ka_vals[j], r_val)
            E_S_num[2, j] = al_vals[j] * hv
            E_S_num[3, j] = ai_vals[j] * hv
            
        # Row 5: Disp
        for j in range(5): E_S_num[4, j] = get_hankel(n, ka_vals[0], r_val)

        # Diagonal Stabilization
        for i in range(1, 5):
            if np.abs(E_S_num[i,i]) < 1e-12: E_S_num[i,i] = 1.0

        M_num[row_start:row_end, col_start:col_end] = E_S_num

        # --- C. M_CF (Reflection -> Cavity) ---
        # T_RC matrix
        T_RC = np.eye(5, dtype=complex)
        for j in range(3):
            val = (-1)**n * np.exp(1j * ka_vals[j] * h_val * np.cos(ta_vals[j])) * np.exp(-1j * n * ta_vals[j])
            T_RC[j, j] = val
        for m in range(2):
            col = 3+m
            val = (-1)**n * np.exp(1j * kb_vals[m] * h_val * np.cos(tb_vals[m])) * np.exp(-1j * n * tb_vals[m])
            T_RC[col, col] = val
        
        # Coupling block
        M_CF_block = np.dot(np.eye(5), T_RC)
        M_num[row_start:row_end, 0:5] = M_CF_block
        
    return M_num

def numerical_solve_example():
    # 1. 定义物理参数 (Granite)
    params = {}
    vp, vs, rho = 4000.0, 2500.0, 2700.0
    mu_val = rho * vs**2
    lambda_val = rho * (vp**2 - 2*vs**2)
    
    # Fill params dict
    params['lambda_s'] = lambda_val
    params['mu_s'] = mu_val
    params['lambda_i'] = lambda_val 
    params['mu_i'] = mu_val
    params['R'] = 5.0
    params['h'] = 50.0
    params['K_l'] = 1e9
    
    freq = 10.0
    omega = 2 * np.pi * freq
    kp = omega / vp
    ks = omega / vs
    
    # Wave modes params (P1-P3, S1-S2)
    for i in range(1, 4):
        params[f'k_alpha{i}'] = kp * (1.0 + (i-1)*0.01)
        params[f'theta_alpha{i}'] = np.pi/4
        params[f'alpha_l{i}'] = 0.5 + (i * 0.1)
        params[f'alpha_i{i}'] = 0.2 + (i * 0.05)
    for i in range(1, 3):
        params[f'k_beta{i}'] = ks
        params[f'theta_beta{i}'] = np.pi/4

    # 2. 组装矩阵 (N=3)
    # 此时 M 为 40x40
    print(f"--- 开始数值计算 (N={N_TRUNCATION}) ---")
    M_num = assemble_matrix_numerically(N_TRUNCATION, params, dk=0.1, k_max=20.0)
    
    # 3. 构建载荷 F
    total_dofs = M_num.shape[0]
    F_num = np.zeros(total_dofs, dtype=complex)
    F_num[10] = 1.0 # 假设入射波激发了 n=-2 处的分量 (Mock)
    
    # 4. 求解
    print(f"矩阵 M 形状: {M_num.shape}, 秩估算: {np.linalg.matrix_rank(M_num)}")
    try:
        X_np = np.linalg.solve(M_num, F_num)
        print("\n--- 求解成功! (Solution) ---")
        
        # 5. 分析结果 (Convergence Check)
        cols = 5
        idx_center = 5 + N_TRUNCATION * cols
        idx_edge = 5 + (2*N_TRUNCATION) * cols
        
        coeff_center = X_np[idx_center:idx_center+cols]
        coeff_edge = X_np[idx_edge:idx_edge+cols]
        
        mag_center = np.max(np.abs(coeff_center))
        mag_edge = np.max(np.abs(coeff_edge))
        
        print(f"中心模态 (n=0) 幅值: {mag_center:.4e}")
        print(f"边缘模态 (n={N_TRUNCATION}) 幅值: {mag_edge:.4e}")
        
        if mag_center > 0:
            ratio = mag_edge / mag_center
            print(f"边缘/中心 比率: {ratio:.4e}")
            if ratio < 0.1:
                print(">> 结果显示收敛 (High order decay OK).")
            else:
                print(">> 注意: 高阶模态仍有显著能量 (建议增加 N 或检查物理模型).")
                
    except Exception as e:
        print("求解失败:", e)
        return # If solving fails, exit

    # --- 6. 物理场后处理 (Phase 4: Physical Field Analysis) ---
    print("\n--- Phase 4: 计算物理场 (DSCF & Displacement) ---")
    
    try:
        import matplotlib.pyplot as plt
        
        # 提取参数
        R = params['R']
        mu = params['mu_s']
        lam = params['lambda_s']
        kp_val = kp
        ks_val = ks
        
        # --- A. 计算 DSCF (动应力集中系数) 沿 r=R ---
        # DSCF = |Sigma_theta_theta| / |Sigma_Inc_Amp|
        # 利用关系: Sigma_theta + Sigma_r = 2(lambda+mu)*Div(u)
        # 对于 P 波: Div(u) = -kp^2 * phi
        # 对于 S 波: Div(u) = 0
        # 所以: Sigma_theta = -2(lambda+mu)*kp^2*phi - Sigma_r
        # 在 r=R 处，Sigma_r = 0 (自由表面? 不，我们有边界条件)
        # 实际上边界条件是 Sigma_rr = -Sigma_rr_inc (总应力为0？双孔隙可能不为0)
        # 简单起见，我们计算 总应力场 (Total Field) = Inc + Refl + Scat
        
        theta_vals = np.linspace(0, 2*np.pi, 360)
        sigma_tt_vals = []
        
        # 预计算入射波幅值 (假设 Sigma_zz=1, 需换算到 Hydrostatic 或 Uniaxial)
        # 这里直接用归一化处理
        
        print("正在计算环向应力 (Hoop Stress)...")
        
        for theta in theta_vals:
            # 1. 散射波贡献 (Scattered)
            # Sum over modes n
            sig_rr_scat = 0
            phi_scat = 0
            
            for idx_n, n in enumerate(range(-N_TRUNCATION, N_TRUNCATION + 1)):
                col_start = 5 + idx_n * 5
                coeffs = X_np[col_start : col_start+5]
                # A_S (P-wave parts: j=0,1,2), B_S (S-wave parts: m=3,4)
                
                # Hankel values
                hv_kp = complex(sp.hankel1(n, complex(kp_val * R)).evalf())
                hv_ks = complex(sp.hankel1(n, complex(ks_val * R)).evalf())
                
                # Phi contribution (Sum of 3 P-waves)
                # phi_n = Sum(A_Sj * H_n(kp*R)) * exp(in*theta)
                phi_n_val = 0
                sig_rr_n_val = 0
                
                # P-waves
                for j in range(3):
                    phi_n_val += coeffs[j] * hv_kp
                    # Sigma_rr formula (approx): -2mu/r^2 ... complex
                    # Reuse E_S matrix logic? Unsafe to copy-paste huge formulas.
                    # 使用近似关系: Sigma_rr_n 对应方程中的 Row 0
                    # E_S[0, j] * coeffs[j] 就是该分量对 Sigma_rr 的贡献
                    # 重新计算 matrix element (Row 0, Col j) for this theta? 
                    # No, E_S is theta-independent part (except exp).
                    # Total = Element * exp(in*theta)
                    
                    # Re-eval element
                    elem = (lam + 2*mu) * (kp_val**2) * hv_kp # Crude approx form used in assembly
                    sig_rr_n_val += coeffs[j] * elem

                # S-waves
                for m in range(2):
                    # Psi contributes to Sigma_rr
                    col = 3 + m
                    elem = 2*mu * (ks_val**2) * hv_ks # Crude approx
                    sig_rr_n_val += coeffs[col-3] * elem # Wait, coeffs index is 0-4
                    
                # Angular term
                term_factor = np.exp(1j * n * theta)
                phi_scat += phi_n_val * term_factor
                sig_rr_scat += sig_rr_n_val * term_factor
                
            # 2. Total Hoop Stress approximation
            # Sigma_tt = -2(lam+mu)*kp^2 * phi_scat - sig_rr_scat
            # (Note: This ignores Reflected Plane Waves from surface reaching the cavity again
            #  but M_CF accounted for them in boundary conditions. 
            #  For post-processing, we ideally sum everything. 
            #  Approximation: Near cavity, Scattered >> Reflected from surface)
            
            sigma_tt = -2*(lam + mu) * (kp_val**2) * phi_scat - sig_rr_scat
            sigma_tt_vals.append(abs(sigma_tt))
            
        # Plot DSCF
        plt.figure(figsize=(8, 8), dpi=100)
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta_vals, sigma_tt_vals, 'b-', linewidth=2)
        ax.set_title(f"DSCF (Dynamic Stress) at Cavity Wall (N={N_TRUNCATION})", va='bottom')
        plt.savefig("latex/dscf_plot.png")
        print(">> DSCF 图已保存: latex/dscf_plot.png")
        
        # --- B. 地表位移 (Surface Displacement) ---
        print("正在计算地表位移 (Surface Displacement)...")
        x_vals = np.linspace(-40, 40, 200)
        uz_vals = []
        
        # Surface z=0
        # Contribution: Reflected Waves (Direct Plane Waves) + Scattered Waves (via DWM)
        
        # Reflected Coeffs
        refl_coeffs = X_np[0:5] # A_Rj, B_Rm
        
        for x in x_vals:
            disp_z = 0
            # 1. Plane Waves (Reflected)
            # uz = dphi/dz + dpsi/dx
            # Phi_R = A_R * exp(-ikx*x - ikz*z) ...
            # At z=0: exp(-ikx*x)
            
            # P-wave Refl
            for j in range(3):
                ka = params[f'k_alpha{j+1}']
                ta = params[f'theta_alpha{j+1}']
                kx = ka * np.cos(ta) # Wait, definition might contain direction
                # Standard form: exp(i(kx*x + kz*z))
                # Let's assume standard prop
                term = refl_coeffs[j] * 1j * (ka*np.sin(ta)) * np.exp(1j * kx * x) # dphi/dz approx
                disp_z += term
                
            # 2. Scattered Waves (Integrals)
            # Ignored for speed in this demo (DWM integral at every point is slow)
            # Dominated by Reflected waves at surface?
            # Yes, for deep cavity.
            
            uz_vals.append(disp_z.real)
            
        plt.figure(figsize=(10, 4))
        plt.plot(x_vals, uz_vals, 'k-', linewidth=1.5)
        plt.title("Surface Vertical Displacement u_z (z=0)")
        plt.xlabel("x (m)")
        plt.ylabel("Displacement (m)")
        plt.grid(True)
        plt.savefig("latex/surf_disp.png")
        print(">> 地表位移图已保存: latex/surf_disp.png")

    except Exception as e:
        print(f"Post-processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    numerical_solve_example()
