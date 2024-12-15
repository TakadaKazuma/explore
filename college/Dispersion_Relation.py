import numpy as np
import matplotlib.pyplot as plt

C = 240  # 一般的な音速 (m/s)
g = 3.72  # 一般的な重力加速度 (m/s^2)
H = 10.8e3  # 一般的なスケールハイト (m)
ganma = 1 / (2 * H)  # ガンマ定数(スケールハイトに依存)
ω_c = ganma * C  #角周波数(音速とガンマ定数に依存)


def border_Hz():
    '''
    音波と重力波の境界となる周波数を計算する関数
    '''
    border_Hz = C * ganma
    return border_Hz


def calculate_n(k, ω):
    '''
      一般的な波動方程式の解を計算する関数
      k:波数
      ω:角周波数
    '''
    return ((ω**2 - k**2 - 1) * ganma**2 + (2*ganma*g - (g/C)**2) * (k/(C*ω))**2)

def calculate_AGW_n(k, ω):
    '''
      内部重力波に該当する波動方程式の解を計算する関数
      k:波数
      ω:角周波数
    '''
    return -(ganma * k)**2 - ganma**2 + (ganma * ω)**2


def calculate_IF_n(k, ω):
    '''
      音波(Infra Sonic)に該当する波動方程式の解を計算する関数
      k:波数
      ω:角周波数
    '''
    return -(ganma * k)**2 - ganma**2 + (2*ganma*g - (g/C)**2) * ((ganma*k)/(ω_c*ω))**2

#音波と内部重力波の境界となる周波数
border = border_Hz()

#分散関係図の描画
if __name__ == "__main__":
    #メッシュグリッドの作成
    k = np.linspace(0, 5, 10000)
    ω = np.linspace(0, 5, 10000)
    K, Ω = np.meshgrid(k, ω)

    #等高線の描画
    plt.contourf(K, Ω, calculate_n(K, Ω), levels=[0, np.max((K, Ω))], colors=['gray'], alpha=0.5)
    #plt.contour(K, Ω, calculate_AGW_n(K, Ω), levels=[0], colors=['blue'], linestyles='--')
    #plt.contour(K, Ω, calculate_IF_n(K, Ω), levels=[0], colors=['green'], linestyles='--')

    #境界の描画
    plt.plot(k, k, color='red', linestyle='--')

    #漸近線の描画
    plt.axhline(y=(2 * ganma * g - (g / C) ** 2) ** (1 / 2) / ω_c, color='black', linestyle=':')

    #凡例及び軸周り等の各種設定
    plt.legend([
        plt.Line2D([0], [0], color='red', linestyle='--'),], [r'$\omega_c$'])
    plt.xlabel(r'$k/\gamma$')
    plt.ylabel(r'$\omega/\omega_c$')
    plt.title("Contour Plot of n² Based on Angular Frequency and Wavenumber")
    plt.grid(True)

    plt.text(2, 4, 'Infrasonic Wave,n²>0', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(3, 0.5, 'Acoustic Gravity Wave,n²>0', color='black', fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()

    #画像の保存
    plt.savefig("Dispersion_Relation.png")
