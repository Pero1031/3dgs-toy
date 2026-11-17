# 光沢面の再構成における粗さ・微細形状の扱いと、光沢対応手法の評価データセット・ベンチマーク

調査日: 2026-11-17。本文は取得ツール（要約付き）経由で読んだもので、PDF の逐語読みではない。数値は各論文の表からの転記で、「筆者計算」と書いたものは各シーンの値から調査者が平均を出したもの。

---

## (A) 画素より細かい形状・粗さの空間変化を、NeRF / Gaussian Splatting はどう扱っているか（IDE、split-sum と事前フィルタ環境マップ、アンチエイリアス、粗さ推定、法線の解像度）。微細形状による「実効的な粗さ」を限界として明示した論文はあるか

### Takeaway
主要手法はどれも「点ごと（ガウシアンごと）の粗さ ρ」でぼかし幅を決める。その仕組みは三つで、方向符号化を減衰させるもの（Ref-NeRF の IDE）、粗さで mip レベルを引くもの（Ref-GS の Sph-Mip、NDE、split-sum 系）、反射コーンを広げるもの（NeRF-Casting）である。画素の中の法線のばらつきを実効的な粗さに換算する仕組みを明示的に入れた NeRF/3DGS 論文は見つからなかった。最も近いのは NeRF-Casting で、反射コーンの幅を「画素半径＋粗さ」としている。微細形状が表現できないことを限界として書いた論文は、NeRO、NDE、TextureSplat などいくつかある。ただし、それが共有照明の当てはめを妨げると論じたものは見当たらなかった。

### Cited Findings
**方向符号化のぼかし（Ref-NeRF の IDE）**
- Ref-NeRF の IDE では、方向 MLP に反射ベクトル ω_r を中心とする vMF 分布（集中度 κ = 1/ρ）を入れる。期待値は E[Y_ℓ^m] = A_ℓ(κ)·Y_ℓ^m(ω_r)、A_ℓ(κ) ≈ exp(−ℓ(ℓ+1)/(2κ)) で、粗さが大きいほど高次の球面調和を減衰させる。ρ は空間 MLP が softplus で出す点ごとの量である — [Ref-NeRF (arXiv HTML)](https://arxiv.org/html/2112.03907v1)
- Ref-NeRF は、IDE が mip-NeRF の integrated positional encoding に着想を得たものだと書いている。また、方向 MLP が表す反射放射輝度は事前フィルタした環境マップに似ているが、材質ごとに事前フィルタしたマップを持つ必要がない点が違うと述べる。ただし、画素のフットプリントについては論じていない — [Ref-NeRF](https://arxiv.org/html/2112.03907v1)
- 前提として、mip-NeRF は画素を円錐で表し、その断面の分散を画素フットプリントの分散に合わせることで、位置符号化側のアンチエイリアスを行っている（幾何側の話で、鏡面の法線分布の話ではない） — [mip-NeRF (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.pdf)
- Ref-NeRF のアブレーション（Shiny Blender の平均）では、"no roughness" が PSNR 33.39 / MAE 25.96°、完全版が 35.96 / 18.38° だった。粗さの入力を外すと画質も法線誤差も悪化する — [Ref-NeRF Table 1](https://arxiv.org/html/2112.03907v1)
- SpecNeRF は、Ref-NeRF に倣い「粗さで鏡面色の最大周波数を制御する」。そのために各ガウス基底のスケール σ_i に ρ を掛ける。初期化段階では、入力画像を複数の σ でぼかした擬似教師を作り、粗さのレベルごとに使う — [SpecNeRF](https://arxiv.org/html/2312.13102v3)

**事前フィルタ環境マップ・mip（split-sum 系）**
- NeRO は第 1 段で split-sum 近似と Ref-NeRF 型の IDE（vMF）を使い、第 2 段で形状を固定して GGX の重点サンプリング（拡散 512・鏡面 256 サンプル）に切り替える。第 1 段は「非常に滑らかな材質」を推定しがちで、著者はそれが鏡面色の当てはめに役立つと述べる — [NeRO](https://arxiv.org/html/2305.17398)
- Ref-GS（CVPR 2025）は粗さ ρ∈[0,1] をガウシアンごとに持たせ、遅延シェーディングの G バッファにラスタライズする。照明側の Sph-Mip グリッドは、方向 (θ, φ) と粗さを軸にした多段の球面特徴グリッド（基底 512×1024×16、9 レベル）で、三線形補間で引く。フットプリントに応じた mip 選択やアンチエイリアス、法線解像度については記述がない — [Ref-GS](https://arxiv.org/html/2412.00905)
- Ref-GS のアブレーション（Table 3。データセットの明記なし、完全版 PSNR 34.00 / MAE 2.21°）では、"w/o Sph-Mip" が 29.95 / 3.61°、"w/o mipmap" が 30.12 / 5.12° だった。粗さに応じた多段化を外すと約 4 dB 落ち、法線誤差も増える — [Ref-GS Table 3](https://arxiv.org/html/2412.00905)
- NDE は、遠方の特徴を GGX カーネルで事前フィルタしたキューブマップ（粗さレベルを [0,1] に配置して線形補間）から引く。近接の特徴は三面グリッドをコーントレースして集め、mip レベル λ_i = log₂(2r_i)、コーン半径 r_i = √3·ρ²·‖x − x′_i‖（GGX ローブの約 75% を覆う）とする。これは反射コーンのフットプリントであり、画面の画素フットプリントではない — [NDE](https://arxiv.org/html/2405.14847)
- IBL-NeRF（Pacific Graphics 2023）は、粗さの違う鏡面反射を、テクスチャの mipmap のように適切な事前フィルタ段の放射場から引く（検索要約で確認） — [IBL-NeRF](https://arxiv.org/pdf/2210.08202)
- 2026 年のプレプリント SSR-GS は、環境マップの mip レベル ℓ を表面粗さ r から決める（検索要約のみ、未読） — [SSR-GS](https://arxiv.org/pdf/2603.05152v1)

**粗さを持たない手法**
- 3DGS-DR（SIGGRAPH 2024）には物理ベースの粗さがなく、ぼけた反射は基底の SH 色に残る。粗さの導入は今後の課題とされる。環境マップは双線形補間で引くだけで、事前フィルタや mip の記述はない。法線は各ガウシアンの最短軸である — [3DGS-DR](https://arxiv.org/html/2404.18454v2)
- EnvGS（CVPR 2025）は鏡面反射方向 d_ref = d_cam − 2(d_cam·n)n だけを使い、粗さパラメータを持たない。基底色と反射色をガウシアンごとの重み β で混ぜる — [EnvGS](https://arxiv.org/html/2412.15215)
- Spec-Gaussian は、ガウシアンごとに 32 個の異方性球面ガウス（ASG）で鏡面色を表し、法線は最短軸を使う（物理的に正確ではないと著者自身が書く）。粗さは扱わない。限界として、明示的な形状がないため法線制約で反射と材質を分けられないと述べる — [Spec-Gaussian](https://arxiv.org/html/2402.15870)

**反射コーンに画素フットプリントを足す手法（最も近い例）**
- NeRF-Casting は反射を vMF コーンとして扱い、幅を κ̄⁻¹ = ṙ + ρ̄（画素半径＋合成された粗さ、Eq. 6）とする。そのコーンを 5 本の反射レイ（unscented directional sampling）で近似する。Zip-NeRF 型の特徴の重み下げは、3D ではなく 2D の方向領域で行う。それ以上に画素より細かい形状を論じてはいない — [NeRF-Casting](https://arxiv.org/html/2405.14871)

**マイクロファセットの導入**
- Neural Microfacet Fields（ICCV 2023）は、体積の各サンプルを（不透明とは限らない）マイクロファセット面として扱う。GGX の D、Smith G1、Schlick Fresnel を使い、α・アルベド・F0 は TensoRF 特徴から空間的に変化させる。体積で集約することが実効的な粗さになるという議論や、画素より細かい法線のばらつき・エイリアスの議論はない。限界として遠方照明を仮定しており、Shiny Blender の coffee（近接光源あり）で性能が落ちると明記している — [NMF](https://ar5iv.labs.arxiv.org/html/2303.17806), [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Mai_Neural_Microfacet_Fields_for_Inverse_Rendering_ICCV_2023_paper.html)

**アンチエイリアス（幾何側）**
- Mip-Splatting（CVPR 2024）は 3D 平滑化フィルタで各ガウシアンの最高周波数を学習視点の最大サンプリングレートに制限し、2D の膨張を箱フィルタ近似の 2D Mip フィルタに置き換える。対象は幾何と色のエイリアスで、鏡面の法線分布ではない — [Mip-Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html)

**法線の解像度不足**
- TextureSplat（Inria、arXiv v2 2025-12、会議名は未確認）の動機は「プリミティブごとに属性を一つしか持たない解像度の限界」である。反射面には「プリミティブごとの単一法線では捉えにくい高周波の法線変化」があるとして、プリミティブごとに法線・粗さ・金属度・アルベドのテクスチャを持たせる。評価は 2×2 テクスチャ・双線形補間で、mipmap は使わない（2DGS がアンチエイリアスされていないため統合が難しい）。限界として、縮小時のアーティファクトを解消しきれないと書く — [TextureSplat](https://arxiv.org/html/2506.13348)
- 同論文のアブレーション（Shiny Blender）は、Ref-Gaussian 34.89 → 法線マッピング追加 35.33 → 完全版 35.90 dB。法線 MAE は Ref-Gaussian 2.078° に対して 1.78° — [TextureSplat Table 2/4](https://arxiv.org/html/2506.13348)

**粗さの空間変化を推定する別系統**
- GS-2M は、粗さを神経的な事前分布ではなく視点間の測光の不一致から推定し、不一致を反射面のしるしとして扱う（検索要約のみ） — [GS-2M](https://arxiv.org/html/2509.22276v2)

**グラフィックス側の先行概念（NeRF/3DGS 以外）**
- Kaplanyan ら（HPG 2016）は、画素フットプリントをマイクロファセットの定義域である勾配（slope）空間に写し、その範囲で NDF をフィルタする（検索要約で確認）。これが「実効的な粗さ」を明示的に扱う実時間描画の手法である — [Filtering Distributions of Normals for Shading Antialiasing](https://research.nvidia.com/sites/default/files/pubs/2016-06_Filtering-Distributions-of/NDFFiltering.pdf)

**限界として微細形状・法線を明示した記述**
- NeRO: 神経 SDF は「急な法線の変化を作りにくい」ため細部が欠けたり誤ったりする（Fig. 19）。Glossy-Real の Maneki で細部が欠ける — [NeRO](https://arxiv.org/html/2305.17398)
- NDE: 表面法線の質に敏感で、そのためハッシュグリッドのような速い幾何表現を使えない — [NDE](https://arxiv.org/html/2405.14847)
- SpecNeRF: 数百個のガウス基底では鏡面に比べて低周波で、失敗例では「ぼけた反射しか学べない」 — [SpecNeRF](https://arxiv.org/html/2312.13102v3)
- 3DGS-DR: 凹形状では法線の伝播が遅い。bell は収束するが、伝播が終わるのは約 72k ステップ — [3DGS-DR](https://arxiv.org/html/2404.18454v2)
- Ref-NeRF: 相互反射と非遠方照明を明示的にモデル化しないため、そうした場面では改善幅が小さい。微細形状はむしろ強みとして挙げている — [Ref-NeRF](https://arxiv.org/html/2112.03907v1)

### Inferences
- **点ごとの ρ は実効的な粗さを吸収できる。ただし吸収できるのは「平均反射方向を中心とする対称なローブ」の形の場合だけ。** IDE、split-sum、Sph-Mip、NeRF-Casting のコーンは、いずれも一点に一つの代表法線と、その周りの等方なローブを仮定する。画素の中で法線が稜線や襞をまたいで二峰性になる場合や、異方的に分布する場合は、単一の vMF/GGX ローブでは表せない。そのとき ρ を大きくしても色は合わず、照明側（共有の環境マップ）が歪められて残差を吸収すると考えられる。ユーザーの 90% 点 28° のような画素は、この領域にある可能性が高い。
- **大まかな換算（調査者の試算）。** 法線の角度ばらつき 8° を勾配の標準偏差 tan 8° ≈ 0.14 と見なすと、Beckmann 換算の α は √2·0.14 ≈ 0.2。28° では tan 28° ≈ 0.53 で α ≈ 0.75 になる。Toksvig や LEAN のように分散を足す近似（α_eff² ≈ α² + 2σ²）では、材質の α = 0.01 は無視でき、実効的な α は材質値の 20〜75 倍になる。「ばらつき」が範囲なのか標準偏差なのかで係数は変わるが、一桁以上違うという結論は変わらない。（追記 2026-11-18: ばらつきは acos|n̄| なので α ≈ ばらつきの角度（ラジアン）で 0.14〜0.48。experiments/gloss/README.md の追加の検証 3 を参照。）
- **3DGS-DR と EnvGS は、実効的な粗さを反射成分で表せない。** ぼけた反射は基底色（SH）か β の低下に逃げる。ユーザーの実験で共有照明に当てはまらなかった成分は、この種の手法では「拡散色として焼き込まれる」形で現れると予想できる。
- **フットプリント依存の部分は、点ごとの ρ では原理的に扱えない。** 点ごとの ρ は 3D の場に固定されているが、画素フットプリントは視距離と解像度で変わる。学習とテストが同程度の距離・解像度（ユーザーの 90/10 の周回）なら問題は小さく、マルチスケールの評価では顕在化するはずである。この点を足し算で扱っているのは NeRF-Casting の ṙ だけである。ṙ は画素の角半径であり、画素の中の法線分散そのものではない。
- **TextureSplat の 2×2 法線テクスチャは「法線の解像度を上げる」方向の対策で、「画素の中の分布を集約する」方向（NDF フィルタ）の対策ではない。** 後者を 3DGS に入れた例は見つからなかった。

### Gaps
- 画素の中の法線分散（または footprint 内 NDF）を、実効的な粗さとして明示的にモデル化した NeRF/3DGS 論文は、今回の検索（3 回）では見つからなかった。Kaplanyan 2016 や Toksvig/LEAN 系を引用する NeRF/3DGS 論文を被引用検索すれば見つかる可能性があり、未実施である。
- 「微細形状による実効的な粗さが共有照明の当てはめを妨げる」ことを限界として明示した論文は見つからなかった。
- Ref-Gaussian と GaussianShader の粗さの扱い（split-sum、事前フィルタの段数）は、この担当では本文を確認していない（他担当の範囲）。
- Ref-GS の Table 3 のアブレーションがどのデータセットかは、本文から読み取れなかった。

---

## (B-1) 各データセットの構成（物体、凹凸・相互反射、視点数、解像度、照明、正解の有無）

### Takeaway
標準の 3 セットはいずれも環境マップや自然光で照らした「中央の一物体を周回撮影する」構成である。Shiny Blender は 6 物体・遠方の環境マップ（coffee は例外の可能性）で、形状は比較的単純（ball、teapot）。Glossy Synthetic は 8 物体で、像のような凹凸の多い形（angel、horse、luyu）を含む。ただし、3DGS 系の論文の一部は angel と horse を除いた 6 物体で評価している。Ref-Real は正解形状のない実写 3 シーンである。近接光源と相互反射が強いのは、NeRF-Casting の実写 4 シーンなどの新しいデータで、こちらにも正解形状はない。

### Cited Findings
**比較表（データセットの構成）**

| データセット | 種別 | 物体・シーン | 視点数 | 解像度 | 照明 | 正解 | 出典 |
|---|---|---|---|---|---|---|---|
| Shiny Blender（Ref-NeRF） | 合成（Blender） | 6: ball, car, coffee, helmet, teapot, toaster。coffee・toaster・car・helmet は BlendSwap のモデル | 学習 100 / テスト 200 | 本文では未確認（「NeRF のデータと同様の条件」とある） | 環境マップ、遠方光（SpecNeRF、EnvGS）。ただし NMF は coffee に近接光源ありと書く | 法線 MAE が評価されている（法線正解の形式は本文で未確認）。Blender のソースが公開されている | [Ref-NeRF](https://arxiv.org/html/2112.03907v1), [Ref-Gaussian](https://arxiv.org/html/2412.19282), [プロジェクト](https://dorverbin.github.io/refnerf/) |
| Glossy Synthetic / Glossy-Blender（NeRO） | 合成（Blender Cycles） | 8: angel, bell, cat, horse, luyu, potion, tbell, teapot | 128 枚/物体（学習/テスト分割は本文に記載なし） | 800×800 | PolyHaven の屋内 HDR 環境マップ | 正解形状（Chamfer）、再照明参照（新しい HDR 3 枚×16 視点）。法線・材質の正解は記載なし | [NeRO](https://arxiv.org/html/2305.17398) |
| Glossy Real（NeRO） | 実写（スマートフォン） | 5: coral, maneki, bear, bunny, vase | 約 100〜130 枚 | 1024×768 | 実環境（静的と仮定、撮影者が映り込む） | 非反射塗料を塗ったうえで構造化光スキャナ（EinScan Pro 2X）で形状を取得。再照明の正解なし | [NeRO](https://arxiv.org/html/2305.17398) |
| Ref-Real（Ref-NeRF の実写） | 実写・非有界 | 3: sedan（著者撮影）、gardenspheres と toycar（SNeRG のデータ） | 未確認 | 未確認（EnvGS は 1/4 解像度で評価） | 自然光 | 正解形状なし | [Ref-NeRF](https://arxiv.org/html/2112.03907v1), [EnvGS](https://arxiv.org/html/2412.15215) |
| NeRF-Casting の光沢シーン | 実写 | 4: toaster 348 枚（テスト 44）、hatchback 308（38）、grinder 231（28）、compact 332（41） | 8 枚ごとにテスト | 3504×2336 を 1/4 に縮小 | 自然光、近接の映り込みあり、撮影者が映り込む | 正解形状なし | [NeRF-Casting](https://arxiv.org/html/2405.14871) |
| Anisotropic Synthetic（Spec-Gaussian） | 合成 | 8: Teapot, Plane, Record, Ashtray, Dishes, Headphone, Jupyter, Lock | 記載なし | 記載なし | 記載なし | 記載なし | [Spec-Gaussian](https://arxiv.org/html/2402.15870) |
| SpecNeRF が使ったデータ | 実写と合成 | Eyeful Tower の反射 9 シーン（854×1280 に縮小、評価 50〜70 視点）、FIPT 合成（法線 MAE あり）、NISR+Inria | — | — | 室内の近接照明 | FIPT は正解形状あり | [SpecNeRF](https://arxiv.org/html/2312.13102v3) |
| EnvGS の自前撮影 | 実写 | 2: Audi, Dog | 未確認 | 未確認 | 自然光 | なし | [EnvGS](https://arxiv.org/html/2412.15215) |
| RefRef（2025-05） | 合成 | 50 物体（単一材質の凸形状から多材質まで）、屈折と反射 | 未確認 | 未確認 | 未確認 | 未確認 | [RefRef（ResearchGate、要約のみ）](https://www.researchgate.net/publication/391658553_RefRef_A_Synthetic_Dataset_and_Benchmark_for_Reconstructing_Refractive_and_Reflective_Objects) |

- Stanford-ORB などの逆レンダリング評価は既存調査で扱い済み（14 物体・2,795 枚の HDR・走査メッシュ付き）なので、ここでは繰り返さない — [既存調査（ローカル）](D:\GoLD_rayt\out\studies\nerf_survey\reports\NeRF質感再構成の文献調査.md)
- 評価で使う部分集合に違いがある。3DGS-DR の Table 1 の Glossy Synthetic は bell, cat, luyu, potion, tbell, teapot の 6 物体で、angel と horse を含まない — [3DGS-DR](https://arxiv.org/html/2404.18454v2)。Ref-Gaussian は 8 物体すべてを評価している — [Ref-Gaussian](https://arxiv.org/html/2412.19282)
- 物体ごとの難しさにも差がある。Shiny Blender の Ref-NeRF は ball 47.46 dB / MAE 1.548°、teapot 47.90 / 9.234° と高い一方、toaster は 25.70 / 42.870° で、法線誤差は mip-NeRF（42.787°）よりわずかに悪い — [Ref-NeRF 補遺 Table S6/S8/S9](https://arxiv.org/html/2112.03907v1)
- NMF の補足として、NeRF の Blender の drums・ship（屈折）、lego（影とアルベドの分離）も失敗例に挙がっている — [NMF](https://ar5iv.labs.arxiv.org/html/2303.17806)

### Inferences
- Shiny Blender は「NeRF のデータと同様の条件」で描画されているので、800×800 である可能性が高い。NDE は FPS を 800×800 で測り、Ref-GS の序論も 800×800 での実時間性を主張している（[NDE](https://arxiv.org/html/2405.14847), [Ref-GS](https://arxiv.org/html/2412.00905)）。ただし Ref-NeRF の本文では確認できなかった。
- Glossy Synthetic は Cycles で描画されているので、相互反射を含むはずである。像のように凹凸の多い angel・horse・luyu は、ユーザーの Erato に最も近い既存物体である。NeRO の Table 1 では、angel の Chamfer は Ref-NeRF が 0.0291 で全物体中最悪、NeRO は 0.0034 であり、形状の難所になっている。それを除いた 6 物体の平均（3DGS-DR、Ref-GS の表）は、像形状に対する性能を反映しない。
- Shiny Blender の中で、凹形状・相互反射・近接光の要素を持つのは toaster と coffee（と car）だと読める。その他はほぼ凸で滑らかである。

### Gaps
- Shiny Blender の解像度・環境マップ名・法線正解の形式、Ref-Real の枚数と解像度は、一次情報で確認できなかった（データセットの README か zip の中身を見る必要がある）。
- 各データセットの材質の粗さ（特に Glossy Synthetic の金属の α）は、どの論文にも記載が見つからなかった。Shiny Blender は Blender のソースが公開されているので確認できる（[shiny_blender_source.zip](https://dorverbin.github.io/refnerf/data/shiny_blender_source.zip)）。
- 画素の中の法線分散のような「微細形状の量」を報告したデータセットはなかった。
- 既存調査の注記にあった「Shiny Blender は 66 物体」は、今回の取得でも同じ誤りが出た（Ref-Real も「33 シーン」と出た）。数字の二重化による取得側の誤りとみられ、各シーンの表と Ref-Gaussian の記述から 6 物体・3 シーンと判断した。

---

## (B-2) 主要手法の報告値（Shiny Blender / Glossy Synthetic / Ref-Real、PSNR/SSIM/LPIPS、法線誤差）

### Takeaway
同じ手法でも論文によって値が大きく違う。Shiny Blender の Ref-NeRF は 32.32〜35.96 dB で、3.6 dB の幅がある。そのため、比較は「同じ論文の同じ表の中」に限るべきである。主要な横並びの表は Ref-GS（CVPR 2025、公開コードで再学習）と 3DGS-DR（SIGGRAPH 2024）で、Ref-GS の Glossy Synthetic と Ref-Real の値の大半は 3DGS-DR の表と一致する。GS 系の最高値は Shiny Blender で 35.0 dB 前後（Ref-Gaussian 35.04、Ref-GS 34.80）。NeRF 系では NeRF-Casting 37.39 と NDE 37.19 が上回るが、表が異なる。

### Cited Findings
**表 1: Shiny Blender、6 シーン平均（Ref-GS Table 1、公開コード・設定で再学習と記載、V100）** — [Ref-GS](https://arxiv.org/html/2412.00905)

| 手法 | PSNR | SSIM | LPIPS | 法線 MAE°（Table 5） |
|---|---|---|---|---|
| Ref-NeRF | 32.32 | 0.956 | 0.110 | 18.38 |
| NeRO | 29.84 | 0.962 | 0.072 | — |
| ENVIDR | 32.88 | 0.969 | 0.072 | 4.61 |
| 3DGS | 30.37 | 0.947 | 0.083 | — |
| GaussianShader | 30.42 | 0.951 | 0.082 | 10.93 |
| 3iGS | 30.64 | 0.948 | 0.077 | 15.47 |
| 3DGS-DR | 33.94 | 0.971 | 0.059 | 2.43 |
| Ref-GS | 34.80 | 0.973 | 0.056 | 2.21 |

同じ表の法線 MAE: NVDiffRec 17.05、GS-IR 23.99、RelightGS 19.81、GS-ROR 7.23。Ref-NeRF の 18.38 は Ref-NeRF の自己報告値と同じである — [Ref-GS Table 5](https://arxiv.org/html/2412.00905)

**表 2: Shiny Blender / Glossy Synthetic（6 物体）/ Ref-Real（3DGS-DR Table 1、RTX 4090。論文には平均がないため、以下は各シーンの値からの筆者計算）** — [3DGS-DR](https://arxiv.org/html/2404.18454v2)

| 手法 | Shiny Blender PSNR / SSIM / LPIPS | Glossy（6 物体）PSNR / SSIM / LPIPS | Ref-Real PSNR（garden / sedan / toycar） |
|---|---|---|---|
| Ref-NeRF | 33.13 / 0.961 / 0.080 | 27.50 / 0.927 / 0.100 | 22.01 / 25.21 / 23.65 |
| NPC | — | — | 21.01 / 24.77 / 22.84 |
| 3DGS | 30.36 / 0.947 / 0.084 | 26.50 / 0.917 / 0.092 | 21.75 / 26.03 / 23.78 |
| GaussianShader | 31.97 / 0.958 / 0.067 | 27.54 / 0.922 / 0.087 | 21.74 / 24.89 / 23.76 |
| ENVIDR | 33.46 / 0.968 / 0.046 | 29.58 / 0.952 / 0.057 | 21.47 / 24.61 / 22.92 |
| 3DGS-DR（遅延） | 34.09 / 0.971 / 0.057 | 30.14 / 0.953 / 0.058 | 21.82 / 26.32 / 23.83 |

3DGS-DR の Table 2（Shiny Blender の法線 MAE°）: GaussianShader 22.31、NVDiffRec 17.02、ENVIDR 4.618、3DGS-DR 4.871。学習時間と FPS（Shiny Blender）: Ref-NeRF 19 h / 0.06、ENVIDR 3.2 h / 3、3DGS 6 分 / 277、GaussianShader 60 分 / 51、3DGS-DR 16 分 / 251 — [3DGS-DR Table 2/3](https://arxiv.org/html/2404.18454v2)

**表 3: Glossy Synthetic の平均（Ref-GS Table 2）** — [Ref-GS](https://arxiv.org/html/2412.00905)
Ref-NeRF 27.50 / 0.927 / 0.100、ENVIDR 29.58 / 0.952 / 0.057、3DGS 26.50 / 0.917 / 0.092、GaussianShader 27.54 / 0.922 / 0.087、3iGS 26.39 / 0.913 / 0.089、3DGS-DR 29.78 / 0.954 / 0.057、GS-ROR 29.70 / 0.956 / —、Ref-GS 30.59 / 0.957 / 0.058。

**表 4: Ref-Gaussian（ICLR 2025、A6000）** — [Ref-Gaussian](https://arxiv.org/html/2412.19282), [ICLR 2025 論文集](https://proceedings.iclr.cc/paper_files/paper/2025/file/abf3682c9cf9245a0294a4bebe4544ff-Paper-Conference.pdf)
- Shiny Blender（Table 1 の各シーンから筆者計算。Table 6 の値と一致）: Ref-Gaussian 35.04 / 0.973 / 0.056。同じ表の 3DGS-DR は筆者計算で 33.95。
- Glossy Synthetic（8 物体）: Ref-Gaussian の筆者計算 30.37 / 0.958 / 0.049（Table 4・5 は 30.33）。6 物体に限ると筆者計算で 30.92。Table 5 は NeuS 27.86 / 0.878 / 0.375、NeRO 29.73 / 0.904 / 0.324、Ref-Gaussian 30.33 / 0.958 / 0.049。
- Ref-Real: garden 22.97 / 0.617 / 0.256、sedan 26.60 / 0.777 / 0.245、toycar 24.27 / 0.660 / 0.256（筆者計算の平均 24.61 / 0.685 / 0.252）。
- 法線 MAE（Table 2、「データセット平均」と記載）: ENVIDR 2.74、GaussianShader 7.00、3DGS-DR 2.62、Ref-Gaussian 2.15。学習時間と FPS（Shiny Blender）: GaussianShader 0.48 h / 28、3DGS-DR 0.35 h / 251、Ref-Gaussian 0.58 h / 122。
- Table 6 の比較は NDE 35.48 / 0.976 / 0.027 / 66 FPS 対 Ref-Gaussian 35.04 / 0.973 / 0.056 / 122 FPS。

**表 5: NeRF 系の表**
- NeRF-Casting Table 5（Shiny Blender 6 シーン平均）: Ref-NeRF 35.96 / 0.967 / 0.059、ENVIDR 35.85 / 0.983 / 0.036、Zip-NeRF 30.33 / 0.949 / 0.086、UniSDF 36.82 / 0.976 / 0.044、NeRF-Casting 37.39 / 0.977 / 0.041 — [NeRF-Casting](https://arxiv.org/html/2405.14871)
- NeRF-Casting Table 4（PSNR、Ref-Real）: NeRF-Casting は sedan 26.77、toycar 24.20、spheres 23.04、Zip-NeRF は 25.85 / 23.41 / 21.77。Table 1（マスク付き指標、Ref-Real 3 シーンと新規 4 シーンの平均）: UniSDF 33.38 / 0.961 / 0.042、Zip-NeRF 33.09 / 0.960 / 0.041、Ref-NeRF* 32.59 / 0.957 / 0.042、NeRF-Casting 33.91 / 0.965 / 0.036、近接反射なしの版 33.30 — [NeRF-Casting](https://arxiv.org/html/2405.14871)
- NDE Table 1（「合成シーンの平均」= Shiny Blender ＋ NeRF の Materials）: NeRO 30.61 / 0.936 / 0.109、ENVIDR 34.95 / 0.982 / 0.035、Ref-NeRF 35.88 / 0.969 / 0.053、NDE 37.19 / 0.982 / 0.022、NDE-RT 35.48 / 0.976 / 0.027（66 FPS） — [NDE](https://arxiv.org/html/2405.14847)
- NMF Table 1（Shiny Blender）: PhySG 26.21 / .921 / .121 / MAE 8.46、NVDiffRec 29.90 / .945 / .114 / 31.89、NVDiffRecMC 28.20 / .902 / .175 / 28.68、Ref-NeRF 35.96 / .967 / .059 / 18.38、NMF 34.56 / .963 / .053 / 17.50。再照明（Shiny Blender）は NMF 25.17 dB — [NMF](https://ar5iv.labs.arxiv.org/html/2303.17806)
- SpecNeRF Table 4（Shiny Blender、自前で再学習）: SpecNeRF 34.65 / 0.9615 / 0.0515、Ref-NeRF 34.69 / 0.9619 / 0.0508 — [SpecNeRF](https://arxiv.org/html/2312.13102v3)
- Ref-NeRF の自己報告（Table 1、Shiny Blender）: 35.96 / 0.967 / 0.058 / MAE 18.38。mip-NeRF 29.76 / 0.942 / 0.092 / 60.38、PhySG 26.21 / 0.921 / 0.121 / 8.46。Ref-Real（Table S1）: sedan 25.65 / 0.720 / 0.119、toycar 24.25 / 0.674 / 0.168、garden 23.46 / 0.601 / 0.138 — [Ref-NeRF](https://arxiv.org/html/2112.03907v1)

**表 6: 実写（EnvGS Table 1、1/4 解像度、RTX 4090）** — [EnvGS](https://arxiv.org/html/2412.15215)

| 手法 | Ref-Real PSNR / SSIM / LPIPS | NeRF-Casting の光沢シーン PSNR / SSIM / LPIPS | FPS | 学習時間 |
|---|---|---|---|---|
| Ref-NeRF* | 23.087 / 0.625 / 0.261 | 30.583 / 0.890 / 0.124 | <0.1 | 47 h |
| UniSDF | 23.700 / 0.635 / 0.266 | 30.838 / 0.889 / 0.130 | <0.1 | >47 h |
| Zip-NeRF | 23.677 / 0.635 / 0.247 | 31.740 / 0.904 / 0.105 | <0.1 | >47 h |
| NeRF-Casting | 24.670 / 0.659 / 0.246 | 31.023 / 0.889 / 0.128 | <0.1 | >47 h |
| 3DGS | 23.700 / 0.641 / 0.262 | 28.860 / 0.877 / 0.159 | 182 | 0.6 h |
| 2DGS | 23.804 / 0.654 / 0.281 | 28.276 / 0.862 / 0.193 | 159 | 0.7 h |
| GaussianShader | 22.875 / 0.622 / 0.314 | 26.412 / 0.835 / 0.216 | 28 | 1.6 h |
| 3DGS-DR | 23.522 / 0.640 / 0.274 | 28.487 / 0.858 / 0.197 | 134 | 1.0 h |
| EnvGS | 24.617 / 0.671 / 0.241 | 30.444 / 0.886 / 0.148 | 26 | 2.5 h |

- EnvGS の補遺 Table 8（Shiny Blender、各シーンの値。平均は取得ツールの計算）: Ref-NeRF 35.96、3DGS-DR 33.96、EnvGS 33.83。Shiny Blender では EnvGS が 3DGS-DR を下回る。Table 5（物体向け手法を実写で）: ENVIDR 15.890、NDE 19.399、EnvGS 27.947 dB — [EnvGS](https://arxiv.org/html/2412.15215)

**その他**
- 形状（Glossy-Blender の Chamfer、NeRO Table 1）: NeRO 0.0042、NeuS 0.0233、Ref-NeRF 0.0241、NDR* 0.0329、NDRMC* 0.0253。Glossy-Real: NeRO 0.0019、NeuS 0.0061、Ref-NeRF 0.0078 — [NeRO](https://arxiv.org/html/2305.17398)
- Spec-Gaussian は Shiny Blender の表を持たない。Anisotropic Synthetic の平均は 3DGS 33.81 / 0.966 / 0.062、Spec-Gaussian 37.70 / 0.980 / 0.042 — [Spec-Gaussian](https://arxiv.org/html/2402.15870)
- IRGS（CVPR 2025）は Shiny Blender、Glossy、Ref-Real を使っていない。評価は Synthetic4Relight（新規視点 35.48、再照明 34.90 dB）と TensoIR のデータ（法線 MAE 4.112°、再照明 29.907 dB）で、Stanford-ORB は定性比較のみ — [IRGS](https://arxiv.org/html/2412.15867)
- TextureSplat の再学習値（Ref-Gaussian の ball は 36.07。原論文は 37.01）、Shiny Blender の法線 MAE（GaussianShader 22.31、3DGS-DR 4.871、Ref-Gaussian 2.078、TextureSplat 1.78） — [TextureSplat](https://arxiv.org/html/2506.13348)
- 既存調査では、GaussianShader の自己報告値を Shiny Blender 31.94、学習 0.58 h としている — [既存調査（ローカル）](D:\GoLD_rayt\out\studies\nerf_survey\reports\NeRF質感再構成の文献調査.md)

### Inferences
- **Ref-GS の Glossy Synthetic と Ref-Real の値は、3DGS-DR の表と一致する。** Ref-GS の Table 2 の Ref-NeRF・3DGS・GaussianShader・ENVIDR の Glossy 平均（27.50 / 26.50 / 27.54 / 29.58）は、3DGS-DR の 6 物体の値から計算した平均と SSIM・LPIPS まで一致した。Ref-Real のシーンごとの値も同じである。したがって Ref-GS の Glossy は 6 物体（angel・horse なし）で、これらの値は 3DGS-DR から引き継いだものと推定される。本文の「公開コードで学習」との関係は不明である。3DGS-DR 自身の値は 29.78 対 30.14 で一致しない（Ref-GS が 3DGS-DR を再学習したとみられる）。
- **論文間のばらつきの例。** Shiny Blender の Ref-NeRF は 35.96（自己報告、NeRF-Casting、NMF、EnvGS）、35.88（NDE、Materials を含む）、34.69（SpecNeRF の再学習）、33.13（3DGS-DR の筆者計算）、32.32（Ref-GS）。ENVIDR は 35.85（NeRF-Casting）から 32.88（Ref-GS）まで。法線 MAE は、3DGS-DR が 4.871（自己報告）対 2.43（Ref-GS）、GaussianShader が 22.31（3DGS-DR）対 10.93（Ref-GS）対 7.00（Ref-Gaussian、複数データの平均）。比較表を作るなら「出典の表」を列に持たせ、異なる表の値を同じ列に混ぜないこと。
- **Ref-Gaussian の Table 6 の NDE（35.48）は NDE-RT（実時間版）の値である。** 完全版の NDE は 37.19（Materials を含む平均）。
- **LPIPS の水準も表によって違う。** Ref-Real の LPIPS は、Ref-NeRF 自身の表が 0.119〜0.168、3DGS-DR や Ref-GS が 0.23〜0.26、Ref-Gaussian が 0.245〜0.256。解像度、マスクの有無、LPIPS のバックボーンの違いが考えられるが、どの論文も明記していない。
- 実写の Ref-Real は、どの手法も 22〜27 dB に収まり、差は 1〜1.5 dB 程度と小さい。一方、近接反射を持つ NeRF-Casting のシーンでは、Zip-NeRF（31.74）が反射専用の手法（NeRF-Casting 31.02、EnvGS 30.44）を上回る。PSNR の全体平均では、反射のモデル化の差が見えにくい。

### Gaps
- Ref-GS がどの値を再学習し、どれを引き継いだかは表ごとには書かれていない。
- 3DGS-DR、Ref-Gaussian、Ref-GS の評価解像度は本文で確認できなかった（800×800 と推定）。
- ENVIDR、GaussianShader、NeRO の Shiny Blender の自己報告値は、この担当では一次確認していない（GaussianShader は既存調査の値）。
- Glossy Synthetic の法線 MAE を報告した表は見つからなかった（正解法線を配布していない可能性がある）。

---

## (B-3) 既存ベンチマークの弱点（滑らかで凸、照明が遠い、視点が多いなど）を指摘した論文はあるか

### Takeaway
「Shiny Blender は遠方の環境光だけ」という指摘は複数ある（SpecNeRF、EnvGS、NMF、Ref-NeRF 自身）。新しいデータ（NeRF-Casting の 4 シーン、SpecNeRF の室内シーン、EnvGS の自前撮影）は、近接の反射を入れるために作られた。一方、「物体が滑らかで凸」「視点が多い」「画素より細かい形状が少ない」ことをベンチマークの弱点として正面から論じた一次資料は見つからなかった。

### Cited Findings
- SpecNeRF の補遺: Shiny Blender の「すべてのシーンは完全な 2D（遠方）環境光で照らされている」。本文でも Shiny の実写データを遠方照明の場面と呼ぶ。NISR+Inria での改善が小さいのは「反射面が少ないデータだから」とする — [SpecNeRF](https://arxiv.org/html/2312.13102v3)
- NMF は遠方照明を仮定しており、近接光源のある coffee で性能が落ちると明記する — [NMF](https://ar5iv.labs.arxiv.org/html/2303.17806)。これは SpecNeRF の「すべて遠方」という記述と食い違う。
- EnvGS は Shiny Blender を「遠方照明の環境マップで描画した合成シーン」と説明する。近接反射を扱う EnvGS は、Shiny Blender では 3DGS-DR を下回る（33.83 対 33.96） — [EnvGS](https://arxiv.org/html/2412.15215)
- Ref-NeRF 自身も、相互反射や非遠方照明では改善が小さくなると書く — [Ref-NeRF](https://arxiv.org/html/2112.03907v1)
- NeRF-Casting は、Ref-NeRF のパラメータ化は遠方照明で最もよく働くと述べ、新しい実写 4 シーンを撮影した。Wu ら（NDE）については、近接と遠方が明確に分かれた合成シーンに焦点を当てていると書いている — [NeRF-Casting](https://arxiv.org/html/2405.14871)
- TextureSplat によれば、Shiny Blender は「物体が中央にあり、周回する視点」で構成されている。そのため、テクスチャの解像度を上げても容量と品質の釣り合いが悪かった。他の種類のシーンでは高解像度が有益な可能性があるとも書く — [TextureSplat](https://arxiv.org/html/2506.13348)
- 3DGS-DR は凹形状での法線伝播の遅さを限界に挙げる（Glossy の bell）— [3DGS-DR](https://arxiv.org/html/2404.18454v2)。NeRO は、神経 SDF では急な法線変化（細部）が出にくいことを挙げる — [NeRO](https://arxiv.org/html/2305.17398)
- 2026 年のプレプリント Ref-DGS は、従来手法が自己反射や形状による相互反射といった近接の鏡面反射を捉えられないと述べ、レイトレーシングで対処すると学習速度が大きく落ちると指摘する（検索要約のみ） — [Ref-DGS](https://arxiv.org/pdf/2603.07664)
- RefRef（2025-05）は、単一材質の凸形状から多材質の物体まで、複雑さを段階づけた 50 物体の合成ベンチマークである（屈折中心、検索要約のみ） — [RefRef](https://www.researchgate.net/publication/391658553_RefRef_A_Synthetic_Dataset_and_Benchmark_for_Reconstructing_Refractive_and_Reflective_Objects)
- アブレーションに見える近接・相互反射の寄与は小さい。Ref-Gaussian の「相互反射なし」は 30.33 → 30.14（Table 4、データセット不明。値から Glossy と推定）。NeRF-Casting の「近接反射なし」は 33.91 → 33.30（実写 7 シーン、マスク付き） — [Ref-Gaussian](https://arxiv.org/html/2412.19282), [NeRF-Casting](https://arxiv.org/html/2405.14871)

### Inferences
- **既存の標準ベンチマークは「近接・相互反射のモデル化」を PSNR の平均で評価しにくい。** 相互反射を除いても 0.2〜0.6 dB しか落ちず、近接反射を扱う手法が Shiny Blender で不利にさえなる。これは、データにそうした成分が少ないか、全画素平均では薄まるためだと推定される。
- **視点数と解像度の点では、既存データはユーザーの設定より有利である。** Shiny Blender は学習 100 枚（おそらく 800×800）、Glossy は 128 枚・800×800、NeRF-Casting は 230〜350 枚・876×584 相当で、いずれもユーザーの 90 枚・256 px より画素あたりの形状が細かく、視点も多い。
- 「滑らかで凸」という弱点を明言した一次資料はない。ただし、Ref-NeRF の ball 47.46 dB と toaster 25.70 dB の差、NeRO の angel で Ref-NeRF の Chamfer が悪化することは、凸で滑らかな物体ほど反射方向のパラメータ化が効く、という実証になっている。

### Gaps
- 「Shiny Blender は単一物体で形状が単純、滑らかで丸い」という趣旨の文が検索要約に出たが、出典論文を特定できなかったので採用しない。
- 視点数や解像度を変えたときの光沢手法の頑健性を体系的に評価した研究は見つからなかった。
- coffee シーンが近接光源を含むかどうか（NMF と SpecNeRF の食い違い）は、Blender ソースで確認できるが未確認である。

---

## ユーザーのデータ（Erato 像・α 0.01 の金・近接光源・256 px・学習 90 視点）は既存データと比べてどう位置づくか

### Takeaway
ユーザーのデータは、既存の標準データのどれよりも「画素あたりの形状が粗く（256 px）」「材質がほぼ鏡面（α 0.01）」「照明が近い」側にある。形状で最も近いのは Glossy Synthetic の angel・horse・luyu（像型、Cycles なので相互反射を含む）だが、照明は遠方の HDR 環境マップである。しかも、3DGS-DR や Ref-GS の表ではこの 2 物体（angel、horse）が平均から外れている。

### Cited Findings
- Glossy Synthetic は 800×800・128 枚・PolyHaven の屋内 HDR 環境マップ・Cycles 描画で、像型の angel、horse、luyu を含む — [NeRO](https://arxiv.org/html/2305.17398)
- Shiny Blender は 6 物体・学習 100 / テスト 200、環境マップの遠方照明が主 — [Ref-NeRF](https://arxiv.org/html/2112.03907v1), [SpecNeRF](https://arxiv.org/html/2312.13102v3)
- 近接光源や近接反射を含む公開データは実写（NeRF-Casting の 4 シーン、Ref-Real）で、正解形状がない — [NeRF-Casting](https://arxiv.org/html/2405.14871), [EnvGS](https://arxiv.org/html/2412.15215)
- 3DGS-DR と Ref-GS の Glossy の平均は 6 物体（angel、horse なし） — [3DGS-DR](https://arxiv.org/html/2404.18454v2), [Ref-GS](https://arxiv.org/html/2412.00905)
- Shiny Blender の Blender ソースは公開されている — [プロジェクトページ](https://dorverbin.github.io/refnerf/)

### Inferences
- **画素あたりの面積。** 同じ画角と物体の大きさで 800 px と 256 px を比べると、1 画素の面積は約 (800/256)² ≈ 9.8 倍になる。ユーザーの中央値 8°・90% 点 28° という法線のばらつきは、既存データの解像度では小さくなる（縮小すれば大きくなる）と見込まれる。256 px に縮小して既存手法を走らせると、既存データでもユーザーの状況に近づけられる可能性がある。
- **位置づけの軸**は四つ: (1) 画素の中の法線のばらつき（ユーザーは大）、(2) 材質の α（ユーザーは 0.01。既存データの α は未確認）、(3) 照明の近さ（ユーザーは近接。既存の合成は遠方、実写は近接だが正解なし）、(4) 正解の有無（ユーザーは合成なので法線・材質・照明の正解がすべてある。既存の合成は形状のみか、法線の MAE だけ）。
- **ユーザーの指標を既存データで測る提案。** Shiny Blender の Blender ソースを使えば、ユーザーと同じ指標（1 画素の範囲の法線のばらつきの中央値と 90% 点）を既存データでも計算でき、「既存データにはこの種の微細形状が少ない」ことを数値で示せる。Glossy Synthetic のメッシュが配布されているかは未確認である。
- **既存の表との関係。** ユーザーの失敗（共有照明の当てはめの破綻）は、既存の表の「全画素平均 PSNR」では見えにくい。像型の angel・horse を除いた平均が広く使われていることもあり、既存のベンチマークの値の高さは、像・ほぼ鏡面・近接光源での性能を保証しない。

### Gaps
- Glossy Synthetic の各物体の材質（α）と、メッシュの公開の有無は未確認。
- 既存データでの「1 画素の範囲の法線のばらつき」の実測値は、どの論文にもない。

---

### 会議名・年の確認状況（参考）
- 一次情報で確認したもの: Ref-NeRF = CVPR 2022（口頭発表、Best Student Paper Honorable Mention、[プロジェクトページ](https://dorverbin.github.io/refnerf/)）。3DGS-DR = SIGGRAPH Conference Papers '24（DOI 10.1145/3641519.3657456、[本文](https://arxiv.org/html/2404.18454v2)）。GaussianShader = CVPR 2024（[CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.pdf)）。Ref-GS = CVPR 2025（[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Ref-GS_Directional_Factorization_for_2D_Gaussian_Splatting_CVPR_2025_paper.html)）。EnvGS = CVPR 2025（[CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_EnvGS_Modeling_View-Dependent_Appearance_with_Environment_Gaussian_CVPR_2025_paper.pdf)）。IRGS = CVPR 2025（[CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_IRGS_Inter-Reflective_Gaussian_Splatting_with_2D_Gaussian_Ray_Tracing_CVPR_2025_paper.pdf)）。Ref-Gaussian = ICLR 2025（[論文集 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/abf3682c9cf9245a0294a4bebe4544ff-Paper-Conference.pdf)）。NMF = ICCV 2023（[CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Mai_Neural_Microfacet_Fields_for_Inverse_Rendering_ICCV_2023_paper.html)）。Mip-Splatting = CVPR 2024（[CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html)）。mip-NeRF = ICCV 2021。IBL-NeRF = Pacific Graphics 2023（PDF のヘッダ）。Kaplanyan ら = HPG 2016（PDF のヘッダ）。
- 掲載誌のみ確認: NeRO = ACM TOG（arXiv ページの "Journal: TOG"。SIGGRAPH 2023 での発表かは未確認）。NeRF-Casting = arXiv ページは "Journal: TOG"、ACM DL の DOI は 10.1145/3680528.3687585（SIGGRAPH Asia 2024 の論文集系 DOI と推定、未確認）。
- 未確認: NDE（CVPR 2024 と記憶するが、取得ページに記載なし）、SpecNeRF（同じく CVPR 2024 と記憶、未確認）、Spec-Gaussian（NeurIPS 2024 と記憶、未確認）、ENVIDR（ICCV 2023 と記憶、未確認）、TextureSplat（会議名の記載なし）、3iGS・GS-ROR・RelightGS（未調査）、SSR-GS・Ref-DGS・GS-2M（2025〜2026 年のプレプリント、要約のみ）。
