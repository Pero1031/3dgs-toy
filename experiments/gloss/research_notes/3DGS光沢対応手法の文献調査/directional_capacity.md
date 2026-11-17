# 3DGS で視点依存色を SH 以外に置き換える手法（各プリミティブの方向表現力を上げる系統）

調査日: 2026-11-17。対象は主に 2023〜2026 年の 3DGS 系。反射方向パラメータ化＋共有環境（Ref-NeRF / GaussianShader / 3DGS-DR）、相互反射、逆レンダリング、半透明は別の担当なので、比較の基準として数値を借りる程度にとどめた。

用語の注意: 「Shiny」（NeX の前向き実写 8 シーン）と「Shiny Blender」（Ref-NeRF の合成 6 物体。論文によって "Ref-NeRF" "Shiny Synthetic" とも呼ぶ）は別のデータセット。下記の 6DGS と SpecGaussian-latent は前者の Shiny で、Shiny Blender ではない。

---

## Q1. どんな SH 代替があるか。表現、パラメータ数、学習の工夫

### Takeaway
SH 代替は大きく 2 系統に分かれる。(A) 点ごとに独立な解析的球面関数・高次元カーネル（SG、球面ベータ、球面ボロノイ、NASG/NASGabor、6DGS/N-DG/UBS、視点依存不透明度）と、(B) 点ごとの潜在特徴を共有 MLP / CNN でデコードする方式（Spec-Gaussian、SpecGaussian-latent、Scaffold-GS 系、Hahlbohm らの小型ニューラル外観）である。代表例の Spec-Gaussian は「ASG に置き換えた」と紹介されることが多いが、実際は (B) に属する。ASG のパラメータを点ごとの特徴から共有 MLP で予測し、最終色もさらに共有 MLP で出している。

### Cited Findings

**(A) 点ごとに独立（共有ネットワークなし）**

- **SG-Splatting**（Wang, Chen, Yi。arXiv 2501.00342、2024-12-31。会議は未確認）: 3 次 SH の代わりに球面ガウス（SG）で視点依存色を表し、低次 SH と混ぜて高周波・低周波の両方を受け持たせる。色表現のパラメータ数を大きく減らし、描画の高速化を主な目的とする。— [arXiv abs](https://arxiv.org/abs/2501.00342)
  - 「直交する 3 つの SG ＋ 適応的な低次 SH、約 10 パラメータ（SH は 48）、3 ローブは実験で選んだもので理論的根拠はない」という記述は検索要約（二次情報）でしか確認できていない。本文 PDF のテキストは取り出せなかった。— [pith.science 要約](https://pith.science/paper/2501.00342)
- **Deformable Beta Splatting（DBS）の球面ベータ（Spherical Beta, SB）**（Liu ら）: c(V̂) = c₀ + Σ_m ℬ(1 − R̂_m·V̂; b_m) c_m。c₀ が拡散色で、各ローブは方向 R̂_m・色 c_m・鋭さ（shininess）b_m を持つ。既定は 2 ローブで、パラメータは 3+6M = 15（SH 3 次の 48 に対して 31%）。共有ネットワークはなく完全に陽的。ただしローブ 1 つの要素を数えると 7 値になり、6M という式と食い違う。— [arXiv HTML 2501.18630](https://arxiv.org/html/2501.18630)
  - PlayCanvas への実装 PR は「2 ローブで 12 float、SH の 1〜3 バンドは 45 float」と書いており、数え方が論文と違う。— [playcanvas/engine PR #9729](https://github.com/playcanvas/engine/pull/9729)
  - アブレーション（Tanks&Temples）では SH(3 次) 24.57 dB、SB(2 ローブ) 24.79 dB。— [arXiv HTML 2501.18630](https://arxiv.org/html/2501.18630)
- **Spherical Voronoi（SV）**（Di Sario, Rebain, Verbin, Grangetto, Tagliasacchi。CVPR 2026）: 方向の定義域を学習可能な分割で表す。f_SV(ω) = Σ_k w_k(ω) c_k、w_k = exp(τ_k ŝ_k·ω) / Σ exp(τ_k' ŝ_k'·ω)。温度 τ_k はサイトベクトルのノルムで表す。視点依存色には 1 ガウシアンあたり K = 8 サイト ×（位置 3＋色 3）= 48 パラメータを使い、SH 3 次と同数にそろえている。高速化のため、キューブマップでサイト候補を引き、softmax を打ち切る。— [arXiv HTML 2512.14180](https://arxiv.org/html/2512.14180)、会議名は [CVPR 2026 supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Di_Sario_Spherical_Voronoi_Directional_CVPR_2026_supplemental.pdf)
- **NASG / NASGabor**（Miazga, Condor, Didyk。"Beyond Spherical Harmonics"、EGSR 2026 / Computer Graphics Forum 45(4)）: 球面関数を体系的に比べた研究。比べたのは SH、等方系（SG、Spherical Cauchy、SB）、リング状・二峰系（Spherical Logistic、Fisher-Bingham FB4/6/8）、異方系（ASG、NASG、LTC）。そのうえで NASGabor（正規化異方性球面ガウスに余弦キャリアと周波数パラメータを足したもの）を提案した。パラメータ数は 1/2/4 ローブで 12/21/39（SH は 48）。すべて点ごとの関数で、共有ネットワークはない。— [arXiv HTML 2606.09794](https://arxiv.org/html/2606.09794)、[検索結果（CGF/EGSR 表記）](https://arxiv.org/abs/2606.09794)
  - NASG はもともとニューラルパスガイディング用の関数（Huang ら、ACM TOG 43(3), 2024）で、Miazga らがプリミティブ型の再構成に転用した。— [Hahlbohm ら arXiv HTML 2609.05255](https://arxiv.org/html/2609.05255)
- **N-DG**（Diolatzis, Zirr, Kuznetsov, Kopanas, Kaplanyan。SIGGRAPH 2024）: MLP の代わりに N 次元ガウス混合で高次元関数を当てはめる。6D の応用例では、ガウシアンを「位置＋視線方向」でパラメータ化する。LSH に着想を得た高次元カリングと、損失に応じた密度制御を導入した。— [arXiv 2405.20067](https://arxiv.org/abs/2405.20067)、[GitHub intel/ndg-fitting](https://github.com/intel/ndg-fitting)
- **6DGS**（Gao ら。ICLR 2025）: 位置 μp と方向 μd を持つ 6D ガウシアン（6×6 共分散 Σ = LLᵀ）を、視線方向 d で条件付けて 3D ガウシアンに切り出す。μ_cond = μp + Σpd Σd⁻¹(d − μd) で中心が方向とともに動き、Σ_cond は d に依存しない。不透明度は α_cond = α·exp(−λ_opa (d−μd)ᵀ Σd⁻¹ (d−μd)) で方向に応じて変わり、λ_opa は 0.35 か学習。**色は SH 3 次のまま**（N-DG の学習可能 RGB を置き換えた）。切り出した後は 3DGS と同じラスタライザで描く。— [arXiv HTML 2410.04974](https://arxiv.org/html/2410.04974)、会議名は [プロジェクトページ](https://gaozhongpai.github.io/6dgs/)
  - 後続の 7DGS（ICCV 2025）は時間軸を加えた。— [6DGS プロジェクトページ](https://gaozhongpai.github.io/6dgs/)
- **Universal Beta Splatting（UBS）**（Liu ら。arXiv 2510.03312、会議は未確認）: N 次元ベータカーネルを視線方向（6D）や時間＋視線（7D）で条件付ける。各プリミティブは RGB だけを持ち、SH も補助ネットワークも使わない。UBS-6D は 1 プリミティブ 35 パラメータ（幾何 32＋色 3）。— [arXiv HTML 2510.03312](https://arxiv.org/html/2510.03312)
- **VoD-3DGS（視点依存不透明度）**（Nowak, Jarosz, Chin。arXiv 2501.17978、会議は未確認）: 不透明度を α̂ = σ(γ + ωᵀŜω) とする。Ŝ は学習可能な 3×3 対称行列（SGGX の射影面積の 2 乗に相当）で、1 ガウシアンあたり 6 パラメータ増える。色は SH のまま。ある視点の組で不透明度が一致するよう求める「視点一貫性損失」を加え、組は LightGlue のマッチ数で選ぶ。— [arXiv HTML 2501.17978v2](https://arxiv.org/html/2501.17978v2)

**(B) 点ごとの特徴＋共有デコーダ**

- **Spec-Gaussian**（Yang ら。NeurIPS 2024）: c = c_d + c_s。c_d は低次 SH。c_s は ASG(ν) = ξ·max(ν·z,0)·exp(−λ(ν·x)² − μ(ν·y)²) を **N = 32 個**使って求める。
  - ASG の軸は半球上に一様に**あらかじめ配置**しておく。λ, μ, ξ は、**点ごとの特徴 f ∈ R²⁴ を共有のコンパクト MLP Θ に通して**予測する。
  - ASG に入れる方向は反射方向 ω_r = 2(ω_o·n)n − ω_o。法線にはガウシアンの最短軸を使い、著者自身「物理的に正確ではない」と書いている。
  - 32 個の ASG 出力をまとめた κ と、γ(d)（2 次の位置符号化）、⟨n, −d⟩ を、**共有の小型 MLP Ψ（3 層 64 ユニット）**に入れて c_s を出す。
  - アブレーションでは、ASG の出力をそのまま色にすると鏡面・異方性成分を表現できなかった。
  - アンカー版もある（Scaffold-GS 方式、アンカー特徴 f_v ∈ R³²）。
  - 学習の工夫は 2 つ。高密度化の勾配を符号付きの和ではなく L1 ノルムで累積する（閾値 0.0005〜0.0006）。実写シーンでは 1/4 解像度から 5k ステップかけて解像度を上げていく（粗→細）。
  - MLP のパラメータ数は書かれていない。
  - — [arXiv HTML 2402.15870](https://arxiv.org/html/2402.15870)、会議名は [GitHub README](https://github.com/ingra14m/Spec-Gaussian) と [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/708e0d691a22212e1e373dc8779cbe53-Abstract.html)
- **SpecGaussian with Latent Features**（Wang ら。ACM MM 2024）: SH を 16 次元の潜在特徴（拡散 8＋鏡面 8）に置き換える。特徴をアルファ合成して画面上の特徴マップにし、Diffuse-UNet と Specular-CNN（画素ごとのレイ方向を位置符号化して連結）でデコードする。c = 拡散 + 鏡面 × マスクで、マスクは法線と視線から小型 MLP で作る。デコーダは画面空間で全点が共有する。— [arXiv HTML 2409.05868](https://arxiv.org/html/2409.05868)
- **Scaffold-GS**（Lu ら。CVPR 2024 Highlight）: 各アンカーが局所的なガウシアン群を受け持ち、その属性（色・不透明度など）を「アンカー特徴、カメラとアンカーの相対的な視線方向、距離」から MLP でデコードする。— [プロジェクトページ](https://city-super.github.io/scaffold-gs/)、[arXiv abs](https://arxiv.org/abs/2312.00109)
- **Compact Neural Appearance Models**（Hahlbohm, Condor, Franke, Eisemann, Magnor。arXiv 2609.05255、2026-09、査読前）: 1 ガウシアンあたり潜在 8 次元＋基本色 3 を持たせ、共有 MLP（816 パラメータ）でデコードする。fp16 で 1 ガウシアン 28 バイト（SH 3 次は 192 バイト）、最適化は 1.3 倍速い。— [arXiv abs](https://arxiv.org/abs/2609.05255)、[arXiv HTML](https://arxiv.org/html/2609.05255)

**(C) 参考: 共有照明場で点の間を結ぶもの（別担当の領域なので最小限）**

- **3iGS**（Tang & Cham。ECCV 2024）: 3DGS の SH 係数の枠を BRDF 特徴に使い直す。各ガウシアンの中心で、共有の TensoRF 型照明場（150³）から入射照明特徴 L_i を引く。小型ネットワーク F(L_i, ρ_i, ω_o) で c_s を出し、c = c_d + c_s とする。— [arXiv HTML 2408.03753](https://arxiv.org/html/2408.03753)、会議名は [ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02170.pdf)
- **SV の反射モード**: 反射を扱うときは、2DGS の法線で遅延シェーディングする。近景には共有の学習可能ライトプローブ（合成シーン 128 個、実写 1024 個。各プローブが SV 関数を持つ）を使い、画素ごとに近い k 個を逆距離で重み付けする。遠景には共有の環境キューブマップ（256²×6）を使う。点ごとのパラメータは拡散色と粗さだけ。— [arXiv HTML 2512.14180](https://arxiv.org/html/2512.14180)

### Inferences
- 「SH 代替」という見出しの手法でも、Spec-Gaussian、SpecGaussian-latent、Scaffold-GS、Hahlbohm らのニューラル版は**デコーダを共有**しており、点ごとに完全に独立ではない。ただし共有されるのは「関数の形（基底・事前分布）」であって、「放射の中身（何が映り込むか）」ではない。各点の映り込みの内容は、その点の特徴ベクトルに保存しなければならない。この意味では、照明場やプローブを共有する (C) 系統とは性質が違う。
- 点ごとに独立な (A) の手法は、どれも 12〜48 パラメータ程度の予算に収まるよう設計されている（SB 15、NASGabor 12〜39、SV 48、SH 48）。関心の中心は「同じ予算でより鋭いローブ」と効率であり、ほぼ鏡面を学習視点の間で補間できるかは直接の主題になっていない。
- 6DGS は方向表現力を色ではなく「中心の移動」と「不透明度の減衰」に入れている。色が SH 3 次のままなので、鏡面ハイライトの鋭さは、方向ごとに切り替わる多数のガウシアンを重ねることで間接的に作るしかない、と読める。

### Gaps
- SG-Splatting のローブ数とパラメータ数は、本文 PDF から取り出せず二次情報でしか確認できなかった。
- Spec-Gaussian の MLP Θ のサイズと総パラメータ数は本文に記載がない。
- Compact 3D Gaussian（Lee ら。ハッシュグリッド＋MLP で色を出す共有場）なども (B) の候補だが、この調査では一次情報を確認していない。

---

## Q2. 光沢データでどの程度改善したか。何が直り、何が直らないか

### Takeaway
点ごとに表現力を上げる手法（A・B）の多くは、Shiny Blender / Glossy Synthetic / Ref-Real で評価されていない。評価の中心は Mip-NeRF360・NeRF Synthetic・自作の異方性データで、SH 3 次からの上積みはおおむね 0.3〜1 dB である。例外は異方性ハイライト（Spec-Gaussian の自作データで +3.9 dB）。光沢ベンチマークで大きく伸びた SV（Shiny Blender 36.09 dB）は、反射モードでは共有プローブと共有キューブマップを使っており、「点ごとの方向表現力」だけの成果ではない。直るのは粗めの鏡面・異方性ハイライトで、ほぼ鏡面の映り込みは、著者たち自身が「直らない」と書いている。

### Cited Findings

**Spec-Gaussian（自論文、PSNR 平均）**
- 自作の Anisotropic Synthetic（8 シーン）: 3DGS 33.82 → Spec-Gaussian 37.70（Scaffold-GS 35.34、アンカー版 36.76）。シーン別では teapot 27.24 → 35.24、plane 26.80 → 30.95。— [arXiv HTML 2402.15870](https://arxiv.org/html/2402.15870)
- NeRF Synthetic: 33.32 → 34.19（GS-Shader 33.38）。materials シーンは 30.00 → 30.98。NSVF: 37.07 → 38.40。Mip-NeRF360: 27.79 → 28.18（ただし同じ論文の表 11 では 3DGS が 27.47 で、数値が食い違う）。FPS は NeRF Synthetic で 315 → 121。— [arXiv HTML 2402.15870](https://arxiv.org/html/2402.15870)
- 自論文では Shiny Blender、Glossy Synthetic、Ref-Real の数値を報告していない（Ref-NeRF とは定性比較のみ）。— [arXiv HTML 2402.15870](https://arxiv.org/html/2402.15870)
- 限界節: 映り込み（reflection）にはまだ課題がある。鏡面・異方性は材質に由来するが、映り込みは環境と幾何に依存する。3DGS には陽な幾何がないので、Ref-NeRF や NeRO のような法線制約で映り込みを材質から分けられない、と著者は書いている。— [arXiv HTML 2402.15870](https://arxiv.org/html/2402.15870)

**Spec-Gaussian の第三者評価**
- SpecGaussian-latent の論文が、NeX Shiny（実写・前向き、8 枚ごとに 1 枚をテスト）でアンカー版 Spec-Gaussian を再実行した。PSNR/SSIM/LPIPS は 3DGS 25.58/0.874/0.118、Spec-Gaussian 26.43/0.883/0.099、SpecGaussian-latent 27.23/0.884/0.109、Ref-NeRF 26.50/0.724/0.283。— [arXiv HTML 2409.05868](https://arxiv.org/html/2409.05868)
- VoD-3DGS の論文の再計算（†）では、Spec-Gaussian は Mip-NeRF360 27.97、T&T 24.53、DB 30.33 で、いずれも VoD-3DGS より PSNR が高い。一方 Mip-NeRF360 での FPS は 22 と遅い。— [arXiv HTML 2501.17978v2](https://arxiv.org/html/2501.17978v2)

**6DGS / UBS**
- NeX Shiny（1008×567 または 1008×756）: 3DGS 22.44/0.719 → 6DGS 23.01/0.724（点数は 198 万 → 114 万）。pasta、tools など一部のシーンでは 3DGS より低い。Deep Blending でも 29.48 → 28.59 と下がる。— [arXiv HTML 2410.04974](https://arxiv.org/html/2410.04974)
- 自作の 6DGS-PBR（雲・煙・爆発・ガラスの suzanne・CT など体積/透過系。学習 100、テスト 50）: 27.53 → 37.61。NeRF Synthetic では 33.25 → 33.26 でほぼ差がない。— [arXiv HTML 2410.04974](https://arxiv.org/html/2410.04974)
- 注意: NeX Shiny での 3DGS の値が、6DGS 論文（22.44）と SpecGaussian-latent 論文（25.58）で大きく違う。解像度や評価手順の違いと思われ、論文をまたいで数値を比べることはできない。— [6DGS](https://arxiv.org/html/2410.04974)、[SpecGaussian-latent](https://arxiv.org/html/2409.05868)
- UBS は 6DGS-PBR で 3DGS 27.40 / 6DGS 36.68 / UBS-6D 40.10。光沢ベンチマーク（Shiny Blender 等）での評価はない。— [arXiv HTML 2510.03312](https://arxiv.org/html/2510.03312)

**DBS（球面ベータ）**
- 光沢専用のベンチマークはない。Mip-NeRF360 で 28.75（3DGS 27.20、3DGS-MCMC 28.29）。counter シーンでは 3DGS-MCMC より約 1.2 dB 高い。ハイライトについては定性比較だけ。— [arXiv HTML 2501.18630](https://arxiv.org/html/2501.18630)
- 限界: 球面ベータは「鏡のような映り込みと異方性の鏡面ハイライトを表現しにくい」と著者自身が書いている。— [arXiv HTML 2501.18630](https://arxiv.org/html/2501.18630)

**SV（Spherical Voronoi）**
- 視点依存色だけを置き換えた比較（点ごとに独立、Beta Splatting 基盤、PSNR）:

  | データセット | SH | SG | SB | SV |
  |---|---|---|---|---|
  | NeRF-Synthetic | 34.15 | 34.26 | 34.10 | 34.58 |
  | Mip-NeRF360 | 28.09 | 28.18 | 28.12 | 28.71 |
  | DeepBlending | 29.80 | 29.67 | 29.56 | 30.63 |

  — [arXiv HTML 2512.14180](https://arxiv.org/html/2512.14180)
- 反射モード（共有プローブ＋共有キューブマップ。PSNR/SSIM/LPIPS）:

  | 手法 | Ref-NeRF（Shiny Blender） | Glossy Synthetic | Ref-Real |
  |---|---|---|---|
  | 3DGS | 30.37/0.947/0.083 | 26.50/0.917/0.092 | 23.85/0.660/0.230 |
  | GShader† | 30.91 | 27.54 | 23.32 |
  | 3iGS† | 30.91 | 26.86 | 23.67 |
  | 3DGS-DR† | 34.13 | 30.36 | 23.83 |
  | Ref-GS† | 35.57 | 31.27 | 23.81 |
  | SV | 36.09/0.976/0.050 | 31.30/0.962/0.046 | 23.91/0.659/0.244 |

  †は著者による再学習。Spec-Gaussian はこの表にない。— [arXiv HTML 2512.14180](https://arxiv.org/html/2512.14180)
- プローブの中身の表現だけを変えたアブレーション（Ref-NeRF、PSNR）: SG 34.19、SB 34.07、キューブマップ 34.48、SV 36.09。— [arXiv HTML 2512.14180](https://arxiv.org/html/2512.14180)
- 著者は「Ref-Real では PSNR/SSIM/LPIPS がどの手法でもほぼ同じ」と書いている。反射を考慮したモデルが見た目で優れていても、指標には差が出にくいということ。速度は 3DGS 比で描画 0.45 倍、学習 2.43 倍。— [arXiv HTML 2512.14180](https://arxiv.org/html/2512.14180)

**3iGS（共有照明場）**
- Shiny Blender: 3iGS 30.77/0.949/0.075、3DGS 30.37/0.947/0.083、GaussianShader 31.94（報告値）/ 30.41（再現値）。著者は GaussianShader の方がやや良いと認め、その理由として Shiny Blender が単一物体で、直接照明が支配的なことを挙げている。— [arXiv HTML 2408.03753](https://arxiv.org/html/2408.03753)

**比較基準（反射方向＋共有環境の系統。詳細は別担当）**
- Spec-Gloss Surfels（Kouros ら、WACV 2026 supplemental あり）の表（PSNR）:

  | 手法 | Shiny Synthetic | Glossy Synthetic | Shiny Real |
  |---|---|---|---|
  | 3DGS | 30.36 | 26.50 | 23.85 |
  | GaussianShader | 31.97 | 27.54 | 23.46 |
  | 3DGS-DR | 34.09 | 30.22 | 23.99 |
  | Spec-Gloss Surfels | 35.50 | 31.22 | 24.35 |

  — [arXiv HTML 2510.02069v2](https://arxiv.org/html/2510.02069v2)
- Ref-Gaussian（Reflective Gaussian Splatting）の付録には、Shiny Blender の平均 PSNR として Ref-Gaussian 35.04、NDE 35.48 が載っている（検索スニペットで確認。本文の表は直接読んでいない）。— [arXiv 2412.19282](https://arxiv.org/pdf/2412.19282)

**標準データセットでの横並び比較（第三者による同一条件の比較）**
- Hahlbohm ら（PSNR、5 回の平均）:

  | データセット | なし（基本色のみ） | SH | SV | NASG | NASGabor | ニューラル |
  |---|---|---|---|---|---|---|
  | NeRF Synthetic | 31.46 | 34.01 | 33.45 | 32.69 | 32.95 | 34.04 |
  | Mip-NeRF360 | 27.73 | 28.36 | 28.64 | 28.50 | 28.54 | 28.55 |

  著者は「差の大半は試行間のばらつきの範囲内」と書いている。— [arXiv HTML 2609.05255](https://arxiv.org/html/2609.05255)
- Miazga ら（Mip-NeRF360）: Beta Splatting+SH 28.00、+SB 28.09、SV(48) 28.46、NASGabor 1 ローブ 28.40 / 4 ローブ 28.46。光沢専用データセットは使っていない。— [arXiv HTML 2606.09794](https://arxiv.org/html/2606.09794)

**何が直り、何が直らないか（著者・第三者の記述）**
- 直るもの: 異方性の鏡面ハイライト（Spec-Gaussian）。帯域制限のある SH ではぼけるハイライト・光沢・きらめき（glint）（Miazga ら、Hahlbohm ら）。SH は鋭い映り込みを、細長い白いガウシアンで「偽装」しがちで、これも減る（Hahlbohm ら）。— [Spec-Gaussian](https://arxiv.org/html/2402.15870)、[Miazga ら](https://arxiv.org/html/2606.09794)、[Hahlbohm ら](https://arxiv.org/html/2609.05255)
- 直らないもの:
  - Hahlbohm らは限界節で、どの球面色関数も、二次レイを陽に追跡しない限り鏡面反射を正確には表現できないと述べている。— [arXiv HTML 2609.05255](https://arxiv.org/html/2609.05255)
  - DBS は鏡のような映り込みが苦手と自ら書いている。— [arXiv HTML 2501.18630](https://arxiv.org/html/2501.18630)
  - Spec-Gaussian は映り込みを限界として挙げている。— [arXiv HTML 2402.15870](https://arxiv.org/html/2402.15870)

### Inferences
- ユーザーの金（GGX α 0.01、ほぼ鏡面）のケースは、上の手法群が「直らない」とする領域そのものにあたる。点ごとの表現を SG・ASG・SB・NASGabor・SV に替えて報告されている改善は、標準データで 0.3〜0.7 dB、自作の異方性ハイライトで数 dB。ほぼ鏡面での改善を示す一次データは見つからなかった。
- 光沢ベンチマークで大きく伸びているのは、共有環境（3DGS-DR、Ref-GS）か共有プローブ（SV 反射モード）を持つ手法だけで、Shiny Blender では 30.4 → 34〜36 dB。SV のアブレーションでは、プローブ内の表現を SG から SV に替えるだけで +1.9 dB 伸びた。つまり方向表現力の向上は、共有の照明表現と組み合わせて初めて大きく効く、という読み方ができる。
- Ref-Real では、どの系統もほぼ 23.3〜24.4 dB に集まっている。実写の光沢では、共有環境を使ってもテスト PSNR はほとんど上がらない。

### Gaps
- Spec-Gaussian、6DGS、DBS、SG-Splatting、VoD-3DGS の Shiny Blender / Glossy Synthetic / Ref-Real / NeRF-Casting shiny での数値は、一次情報でも第三者の表でも見つからなかった（2 回検索した）。
- Ref-Gaussian の付録にある NDE の Shiny Blender 値は、検索スニペット経由でしか確認していない。
- Miazga らの表 4（Spec-Gaussian、Glossy-Gaussian との比較）は詳細を読めていない。「Glossy-Gaussian」がどの手法を指すかも未確認。

---

## Q3. 「点ごとに独立」か「点の間で共有」か。新規視点への汎化について何が言われているか

### Takeaway
新規視点への汎化を、学習視点とテスト視点の差という形で数値で示した論文は、光沢データについては見つからなかった。ただし、第三者による 2 本の比較研究が質的に同じ傾向を報告している。「表現力の高い点ごとの関数やニューラルデコーダほど学習視点によく当てはまり、視点が疎だと過学習しやすい。SH は最も過学習しにくい」。これは「1 点の表現力を上げるだけでは、観測の粗さで決まる天井に当たる」というユーザーの仮説と整合する。

### Cited Findings

**共有の有無による分類（各論文の記述に基づく）**

| 手法 | 点ごとに持つもの | 共有するもの | 出典 |
|---|---|---|---|
| SG-Splatting | SG ローブ＋低次 SH | なし | [arXiv](https://arxiv.org/abs/2501.00342) |
| DBS（SB） | 基本色＋ローブ（方向・色・鋭さ） | なし | [arXiv](https://arxiv.org/html/2501.18630) |
| SV（色モード） | 8 サイト（方向・温度・色） | なし | [arXiv](https://arxiv.org/html/2512.14180) |
| NASGabor | ローブ＋拡散色 | なし | [arXiv](https://arxiv.org/html/2606.09794) |
| 6DGS / N-DG / UBS | 6D〜7D のカーネルと色 | なし | [6DGS](https://arxiv.org/html/2410.04974)、[UBS](https://arxiv.org/html/2510.03312) |
| VoD-3DGS | 不透明度用の対称行列 | なし（一貫性損失で視点の組を結ぶだけ） | [arXiv](https://arxiv.org/html/2501.17978v2) |
| Spec-Gaussian | 特徴 f ∈ R²⁴（アンカー版は f_v ∈ R³²） | ASG パラメータ予測 MLP Θ、デコーダ MLP Ψ | [arXiv](https://arxiv.org/html/2402.15870) |
| SpecGaussian-latent | 16 次元の潜在特徴 | 画面空間の UNet / CNN、マスク MLP | [arXiv](https://arxiv.org/html/2409.05868) |
| Scaffold-GS | アンカー特徴 | 属性をデコードする MLP | [プロジェクト](https://city-super.github.io/scaffold-gs/) |
| Hahlbohm らのニューラル版 | 潜在 8＋基本色 3 | 816 パラメータの MLP | [arXiv](https://arxiv.org/html/2609.05255) |
| 3iGS | BRDF 特徴・拡散色 | TensoRF 照明場、デコーダ | [arXiv](https://arxiv.org/html/2408.03753) |
| SV（反射モード） | 拡散色・粗さ | ライトプローブ群、環境キューブマップ | [arXiv](https://arxiv.org/html/2512.14180) |

**汎化・過学習についての記述**
- Miazga ら（EGSR 2026）:
  - 視点が密な室内シーンで、異方性関数の利得が最も大きい。
  - 視点が疎な屋外シーンでは、表現力の高い関数が過学習して汎化しにくく、SH の方が滑らかに補間する（ただし PSNR の差は約 0.2 dB）。
  - 限界節にも「学習視点が疎だと表現力の高いプリミティブは過学習しうる」とある。
  - 学習視点とテスト視点の差の数値は報告していない。
  - — [arXiv HTML 2606.09794](https://arxiv.org/html/2606.09794)
- Hahlbohm ら（2026-09、査読前）の表 3（各モデルの特徴まとめ）:
  - SH は「最も過学習しにくい（SSIM/LPIPS が強い）。鋭い映り込みは幾何で補うしかない」。
  - ニューラル（共有 MLP）は「明示的な球面関数よりはるかによく学習画像に当てはまる。正則化で容量を減らすと視点依存の弱いシーンは良くなるが、視点依存の強いシーンは悪くなる」。シーンごとに正則化を調整するのが難しい、とも注記している。
  - NASG(abor) は「鋭いローブがよく汎化し、撮影時のアーティファクトも説明してしまう」。
  - — [arXiv HTML 2609.05255](https://arxiv.org/html/2609.05255)
- Hahlbohm らはさらに、表現力の高い外観モデルは、動く物体・照明変化・シーンの変化といった不整合を視点依存効果として吸収できる、と指摘する。新規視点での指標が上がっても、それは放射のモデル化が良くなったのではなく、静的シーンという仮定の破れに強くなっただけかもしれない。この 2 つを分けて評価すべきだと提案している。— [arXiv HTML 2609.05255](https://arxiv.org/html/2609.05255)
- Hahlbohm らによると、色の表現力が低いと幾何で補おうとして幾何のアーティファクトが増える。逆に表現力を上げると、プリミティブがまとまり、ブレンドの回数が減る。— [arXiv HTML 2609.05255](https://arxiv.org/html/2609.05255)
- 3iGS の著者は、3DGS ではガウシアンごとの最適化が「共有されたシーン照明から生じるはずの相互依存」をモデル化しないまま進む、と批判している。そして共有照明場を、正確な視点依存効果のための中核要素と位置づけている。汎化を数値で検証してはいない。— [arXiv HTML 2408.03753](https://arxiv.org/html/2408.03753)
- Spec-Gaussian は、実写シーンで過学習すると新規視点に浮遊物（floater）が出るとし、粗→細の解像度スケジュールでそれを減らしたと述べる。学習とテストの差は分析していない。— [arXiv HTML 2402.15870](https://arxiv.org/html/2402.15870)
- SpecGaussian-latent が視点を減らした条件で試したのは、Stump シーンで 10% の視点に絞った定性比較だけ。— [arXiv HTML 2409.05868](https://arxiv.org/html/2409.05868)
- 評価方法の問題:
  - SV の著者は、Beta Splatting が最良のチェックポイントをテストセットで選んでいる点を「テストセットによる選択」として批判し、自分たちは固定の反復回数で評価した。— [arXiv HTML 2512.14180](https://arxiv.org/html/2512.14180)
  - UBS も、比較相手の手法は 500 反復ごとにテストし最良値を報告していると書いている。— [arXiv HTML 2510.03312](https://arxiv.org/html/2510.03312)

### Inferences
- ユーザーの (b) の観察は、これら 2 本の比較研究の傾向と同じ方向を向いている。(b) とは、表面の点ごとに SH を当てはめると、次数を上げても学習視点を覚えるだけでテスト視点は 24〜27 dB で頭打ちになる、というものだった。比較研究が示しているのは、表現力を上げると学習側によく当てはまり、疎な視点では汎化が伸びない（あるいは下がる）という傾向である。90 視点で 1 点あたり約 26 回、12°×20° 間隔という観測は、GGX α 0.01 の鋭いローブに対しては「疎」の側にあると考えられる。
- 共有デコーダ（Spec-Gaussian、ニューラル版）は「方向の関数として滑らかであれ」という事前分布を全点で共有するが、映り込みの内容を点から点へ運ぶ仕組みは持たない。Hahlbohm らが「ニューラル版ははるかによく学習画像に当てはまる」と書いていることからも、共有デコーダだけで観測の隙間が埋まるとは期待しにくい。点の間で情報が実際に共有されるのは、同じ反射方向を見た別の点の観測が、共有の照明表現（環境、プローブ、照明場）を通して効く構造（C 系統）である。光沢ベンチマークで大きな利得が出ているのもこの系統だけ（Q2）。
- 6DGS と VoD-3DGS は、方向依存を「中心の移動」や「不透明度の変化」で表すぶん、学習視点ごとに別のガウシアンを当てる自由度が増える。疎な観測では過学習の方向に働くはずだが、それを検証したデータは見つからなかった。

### Gaps
- 学習視点 PSNR とテスト PSNR を並べた表、または学習視点数を減らしたときの劣化曲線を、光沢データ（Shiny Blender 等）について示した 3DGS 系の論文は見つからなかった。Miazga らと Hahlbohm らも、学習視点の数値は報告していない。
- 6DGS と UBS の 6DGS-PBR は学習 100 / テスト 50 だが、視点数を変えた実験はない。

---

## Q4. NeRF 側で最低限おさえるべき源流（方向符号化）

### Takeaway
3DGS 系の理解に最低限必要な源流は次の 3 つ。(1) N-DG（位置＋方向の 6D ガウス混合）。6DGS と UBS の直接の祖先。(2) NDE（CVPR 2024）。方向の定義域に特徴グリッドを置き、さらに空間側の特徴をコーントレースして「空間的に変化する方向符号化」を作った。点の間で情報を共有して相互反射まで扱う方向性の代表例。(3) NeRF-Casting の問題提起。方向の関数を巨大なネットワークで表す既存手法は高コストで、近くの物体の映り込みを一貫して合成できない。反射方向パラメータ化（Ref-NeRF の IDE）自体は別担当。

### Cited Findings
- **NDE**（Wu, Bi, Xu, Luan, Zhang, Georgiev, Sunkavalli, Ramamoorthi。CVPR 2024）:
  - 空間で使われてきた特徴グリッド符号化を方向（角度）の定義域に持ち込み、高周波の角度信号を扱えるようにした。
  - 空間側の特徴をコーントレースして空間的に変化する方向符号化を作り、相互反射を扱う。
  - 既存の符号化関数は角度だけを入力にしている、と指摘している。
  - 小さなネットワークで実時間推論ができる。
  - — [arXiv abs 2405.14847](https://arxiv.org/abs/2405.14847)、会議名は [GitHub README（BibTeX）](https://github.com/lwwu2/nde)
- **NeRF-Casting**（arXiv 2405.14871）の要旨: 既存手法は遠景照明の映り込みの細部は改善したが、近くの物体の映り込みを一貫して合成できず、出射放射を大きく計算コストの高いネットワークで表している。— [arXiv abs](https://arxiv.org/abs/2405.14871)
- **N-DG**（SIGGRAPH 2024）: MLP の代わりに N 次元ガウス混合を使い、6D（位置＋視線方向）の視点依存シーンを数分で最適化できる。6DGS はこの学習可能 RGB を SH に置き換え、切り出しで 3DGS の描画系に乗せた。— [arXiv 2405.20067](https://arxiv.org/abs/2405.20067)、[6DGS arXiv HTML](https://arxiv.org/html/2410.04974)
- NASG の出自はパスガイディング（Huang ら、TOG 2024）。レンダリング分野の球面関数が再構成に転用されている例として挙げておく。— [Hahlbohm ら](https://arxiv.org/html/2609.05255)

### Inferences
- NDE が空間側の特徴をコーントレースする仕組みは、「ある点の方向符号化が、周囲の空間の特徴に依存する」構造であり、点の間の共有にあたる。点ごとの方向関数を強くする (A) とは逆の発想で、ユーザーの仮説で言えば「天井を越えうる側」の源流と位置づけられる。
- NeRF-Casting の問題提起（大きなネットワーク、近景の映り込みの一貫性）は、方向の関数をネットワークや点ごとの関数に覚えさせる方式全般への批判として読める。

### Gaps
- NeRF-Casting の採録会議は、取得したページに記載がなく未確認（SIGGRAPH Asia 2024 と記憶しているが一次情報で確かめていない）。導入節にある「点ごとに映り込みを記憶する」という趣旨の議論も本文では確認していない。
- ASG の原典（Xu ら 2013）、NeRF の方向位置符号化、Plenoxels / PlenOctrees の SH 採用は、今回一次情報を開いていない。

---

## Q5. コード公開と依存

### Takeaway
主要手法のうち、コードを確認できたのは Spec-Gaussian、SpecGaussian-latent、Scaffold-GS、3iGS、N-DG、NDE（tiny-cuda-nn 依存）、DBS（gsplat ベース、プロジェクトページ経由）。6DGS、VoD-3DGS、SG-Splatting はリポジトリを見つけられなかった。SV、UBS、NASGabor、Hahlbohm らはプロジェクトページまでしか確認していない。

### Cited Findings
- **Spec-Gaussian**: https://github.com/ingra14m/Spec-Gaussian
  - 依存は Python 3.7、PyTorch 1.13.1+cu116、torch-scatter、`--recursive` で取得するサブモジュール（名前は README に書かれていない）。
  - アンカー版の学習・描画スクリプト（train_anchor.py ほか）が同梱されている。README は非アンカー版が高品質、アンカー版が高速と説明している。
  - README には「2024 年 2 月の arXiv 版は古く、現行版はかなり高品質」とある。NeRF-synthetic を train+val で学習した条件で 35.89 dB（Rip-NeRF 35.44）。
  - データセットとして Shiny Blender への言及はない。
  - — [GitHub README](https://github.com/ingra14m/Spec-Gaussian)
- **SpecGaussian-latent**: https://github.com/MarcWangzhiru/SpeclatentGS — [arXiv HTML 2409.05868](https://arxiv.org/html/2409.05868)
- **Scaffold-GS**: https://github.com/city-super/Scaffold-GS — [プロジェクトページ](https://city-super.github.io/scaffold-gs/)
- **6DGS**: プロジェクトページ（https://gaozhongpai.github.io/6dgs/）にはデータへのリンクだけで、コードへのリンクはない。本文によると、切り出し処理は CUDA 実装されておらず、326.3 FPS という値は FlashGS と統合したときのもの。— [プロジェクトページ](https://gaozhongpai.github.io/6dgs/)、[arXiv HTML](https://arxiv.org/html/2410.04974)
- **N-DG**: https://github.com/intel/ndg-fitting — [検索結果](https://github.com/intel/ndg-fitting)
- **DBS（球面ベータ）**: 本文に「ソースコードはプロジェクトページで公開」とある（https://rongliu-leo.github.io/beta-splatting/）。gsplat ベース。PlayCanvas のエンジンとエディタ SuperSplat に、SB 表示の PR が出ている。— [arXiv HTML 2501.18630](https://arxiv.org/html/2501.18630)、[playcanvas/supersplat PR #1036](https://github.com/playcanvas/supersplat/pull/1036)
  - 採録先は ACM DL の DOI（10.1145/3721238.3730716）から SIGGRAPH 2025 と判断した。ただし arXiv HTML のヘッダは「TOG」表記で、一部未確認。— [ACM DL](https://dl.acm.org/doi/10.1145/3721238.3730716)
- **UBS**: プロジェクトページ https://rongliu-leo.github.io/universal-beta-splatting/ のみ確認。— [arXiv HTML](https://arxiv.org/html/2510.03312)
- **SV**: プロジェクトページ sphericalvoronoi.github.io のみ確認。— [arXiv HTML](https://arxiv.org/html/2512.14180)
- **NASGabor（Miazga ら）**: https://arcanous98.github.io/projectPages/beyondSH.html — [arXiv HTML](https://arxiv.org/html/2606.09794)
- **Hahlbohm ら**: 本文に「実装は公開済み」とある。プロジェクトページは https://fhahlbohm.github.io/efficient-gaussian-appearance。ニューラルデコードが安いのは CUDA の Tensor Core のおかげで、Tensor Core のない WebGL では解析的モデルの方が速いとも書いている。— [arXiv HTML](https://arxiv.org/html/2609.05255)
- **3iGS**: https://github.com/TangZJ/3iGS — [arXiv HTML](https://arxiv.org/html/2408.03753)
- **NDE**: https://github.com/lwwu2/nde
  - 依存は Python 3.8、CUDA 11.7、PyTorch 2.0.1、pytorch-lightning 2.0.8、nerfacc、tinycudann。tinycudann は安定した最適化のため fp32 でビルドする（TCNN_HALF_PRECISION=0）。
  - 対応データセットは NeRF-Synthetic（materials）、RefNeRF-Synthetic、NeRO-GlossyReal。
  - — [GitHub README](https://github.com/lwwu2/nde)
- **VoD-3DGS**、**SG-Splatting**: 取得したページにはコードの URL がなかった。— [VoD arXiv HTML](https://arxiv.org/html/2501.17978v2)、[SG-Splatting abs](https://arxiv.org/abs/2501.00342)

### Inferences
- ユーザーの自前パイプライン（3DGS トイ実験）に組み込みやすいのは、点ごとに独立で CUDA 拡張を必要としないもの（SB、SG、SV の色モード）。これらは 3DGS の SH 評価部分を差し替えるだけで試せる。Spec-Gaussian 型は小型 MLP が 2 つ加わる。NDE は tiny-cuda-nn と NeRF の枠組みに依存し、3DGS への移植は自明ではない。

### Gaps
- Spec-Gaussian のサブモジュールの中身（専用のラスタライザかどうか）は README に書かれておらず、確認できなかった。
- SV、UBS、NASGabor のリポジトリが実在するかどうかは、プロジェクトページを開いていないため未確認。
