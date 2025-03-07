import numpy as np
import matplotlib.pyplot as plt

class Params:
    def __init__(self, C=240, g=3.72, H=10.8e3):
        """
        C: 音速 (m/s)
        g: 重力加速度 (m/s^2)
        H: スケールハイト (m)
        """
        self.C = C
        self.g = g
        self.H = H
        self.ganma = 1 / (2 * self.H)  # ガンマ定数
        self.ω_c = self.ganma * self.C  # 角周波数
    
    def border_Hz(self):
        """
        音波と重力波の境界となる周波数を計算
        """
        return self.C * self.ganma
    
    def calculate_n(self, k, ω):
        """
        一般的な波動方程式の解を計算
        """
        return ((ω**2 - k**2 - 1) * self.ganma**2 + (2*self.ganma*self.g - (self.g/self.C)**2) * (k/(self.C*ω))**2)
    
    def calculate_AGW_n(self, k, ω):
        """
        内部重力波の波動方程式の解を計算
        """
        return -(self.ganma * k)**2 - self.ganma**2 + (self.ganma * ω)**2
    
    def calculate_IF_n(self, k, ω):
        """
        音波(Infra Sonic)の波動方程式の解を計算
        """
        return -(self.ganma * k)**2 - self.ganma**2 + (2*self.ganma*self.g - (self.g/self.C)**2) * ((self.ganma*k)/(self.ω_c*ω))**2

def plot_dispersion_relation(params):
    """
    分散関係図の描画
    """
    k = np.linspace(0, 5, 10000)
    ω = np.linspace(0, 5, 10000)
    K, Ω = np.meshgrid(k, ω)

    # 等高線の描画
    plt.contourf(K, Ω, params.calculate_n(K, Ω), levels=[0, np.max((K, Ω))], colors=['gray'], alpha=0.5)
    
    # 境界の描画
    plt.plot(k, k, color='red', linestyle='--')
    
    # 漸近線の描画
    plt.axhline(y=(2 * params.ganma * params.g - (params.g / params.C) ** 2) ** (1 / 2) / params.ω_c, color='black', linestyle=':')
    
    # 凡例とラベル設定
    plt.legend([plt.Line2D([0], [0], color='red', linestyle='--')], [r'$\omega_c$'])
    plt.xlabel(r'$k/\gamma$')
    plt.ylabel(r'$\omega/\omega_c$')
    plt.title("Contour Plot of n² Based on Angular Frequency and Wavenumber")
    plt.grid(True)
    
    plt.text(2, 4, 'Infrasonic Wave, n²>0', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(3, 0.5, 'Acoustic Gravity Wave, n²>0', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    
    # 画像の保存
    plt.savefig("Dispersion_Relation.png")
    print("Save completed: Dispersion_Relation.png")

if __name__ == "__main__":
    params = Params()
    plot_dispersion_relation(params)