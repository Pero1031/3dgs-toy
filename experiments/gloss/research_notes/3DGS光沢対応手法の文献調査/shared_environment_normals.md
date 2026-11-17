# 反射方向パラメータ化と共有環境照明による鏡面の扱い（Ref-NeRF から 3DGS 系へ）と法線の得方

調査日 2026-11-17。論文本文は WebFetch（要約付き取得ツール）経由で読んだもので、PDF の逐語読みではない。「本文確認」は arXiv HTML / ar5iv を取得して項目を問い合わせたもの、「スニペット」は検索結果の要約のみを指す。逆レンダリング側（NeRV、NeRFactor、TensoIR、NeRO 等）の整理は既存調査 `D:\GoLD_rayt\out\studies\nerf_survey\reports\NeRF質感再構成の文献調査.md` にあり、ここでは重複を避ける。光線追跡（NeRF-Casting、EnvGS、IRGS）、平面鏡、SH の置き換え、物理ベース逆レンダリング（GS-IR、R3DG）は他の担当範囲なので、境界の説明に必要な範囲だけ触れる。

---

## 1. Ref-NeRF の要点（IDE、拡散・鏡面の分解、法線の予測と正則化）と、3DGS 系に引き継がれたもの

### Takeaway
Ref-NeRF は「視線を法線で折り返した反射方向 ω_r で方向依存の色を引く」「拡散色 + 鏡面の色味 × 鏡面色」「粗さで方向エンコードの高次成分を減衰させる（IDE）」「密度勾配法線を、MLP が予測する滑らかな法線に置き換え、2 つの正則化で縛る」という 4 点を確立した。3DGS 系はこのうち反射方向・拡散 + 色味 × 鏡面の分解・粗さでぼかすという構造をほぼそのまま継ぎ、IDE+MLP は事前フィルタ済みのキューブマップや球面 Mip グリッドに、予測法線は「ガウシアンの最短軸 + 残差」や「2DGS サーフェルの法線 + 深度法線との整合損失」に置き換えた。

### Cited Findings
- **反射方向**: 方向 MLP には視線方向ではなく ω_r = 2(ω_o·n)n − ω_o を入力し、フレネル様の効果のため n·ω_o も渡す — [Ref-NeRF（ar5iv、本文確認）](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **分解**: 空間 MLP が拡散色 c_d、鏡面の色味 s、粗さ ρ（softplus）、ボトルネック b を出し、最終色は c = γ(c_d + s ⊙ c_s)（c_s は方向 MLP の鏡面色、γ は sRGB へのトーンマップ） — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **IDE**: 方向を球面調和 Y_ℓ^m で符号化し、ω_r を中心・集中度 κ = 1/ρ の vMF 分布で期待値をとる。閉形式は E[Y_ℓ^m] = A_ℓ(κ)·Y_ℓ^m(ω_r)、A_ℓ(κ) ≈ exp(−ℓ(ℓ+1)/(2κ)) で、粗いほど高次が減衰する — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **法線**: 密度勾配法線 n̂ = −∇τ/‖∇τ‖ は雑音が多く評価・可視化用。反射方向の計算には空間 MLP が出す予測法線 n̂′ を使う — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **正則化**: 予測法線損失 R_p = Σ w_i‖n̂_i − n̂′_i‖²（重み 合成 3×10⁻⁴、実写 10⁻³）、向き損失 R_o = Σ w_i max(0, n̂′_i·d̂)²（見えているサンプルで裏向きの法線を罰する、合成で 0.1） — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **アブレーション（Shiny Blender、PSNR / 法線 MAE°）**: 完全版 35.96 / 18.38、反射方向なし（視線方向入力）29.47 / 16.19、R_o なし 31.62 / 52.56、予測法線なし 30.91 / 30.67、粗さなし 33.39 / 25.96、拡散色なし 33.32 / 26.13、IDE の代わりに位置エンコード 35.90 / 20.31。反射方向なしのほうが MAE が低い理由は本文に説明がない — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **相互反射への部分的な対処**: 相互反射と自己遮蔽は照明の空間変化を生むが、ボトルネック b を方向 MLP に渡すことで部分的にしか扱わない — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **限界**: 反射方向パラメータ化は相互反射と遠方でない照明を明示的に扱わず、その場合 mip-NeRF に対する利得は縮む。予測法線は密度勾配法線から離れることがあり、面として捉えられない領域では一定の法線を予測して視線方向と反射方向を 1 対 1 に写すだけになる（付録 B）。凹面については論じていない — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **実写での利得は小さい**: sedan 25.53→25.65、toy car 24.00→24.25、garden spheres 23.40→23.46 dB（mip-NeRF→Ref-NeRF）、garden spheres の SSIM は 0.620→0.601 と下がる — [Ref-NeRF 表 S1](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **会議**: CVPR 2022（pp. 5491–5500） — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Verbin_Ref-NeRF_Structured_View-Dependent_Appearance_for_Neural_Radiance_Fields_CVPR_2022_paper.html)
- **3DGS 系への継承（GaussianShader）**: 色 = γ(c_d + s ⊙ L_s(ω_o, n, ρ) + c_r(ω_o)) と Ref-NeRF の分解に残差色 c_r を足した形。IDE ではなく学習可能なキューブマップを採ったのは学習効率のため — [GaussianShader（本文確認）](https://arxiv.org/html/2311.17977)
- **継承（Normal-GS）**: 鏡面の方向依存は IDE で、拡散は法線と「積分方向照明ベクトル（IDIV）」の内積で表す — [Normal-GS（NeurIPS 2024、スニペット）](https://arxiv.org/abs/2410.20593v1)

### Inferences
- Ref-NeRF のアブレーションで最も効いているのは「予測法線」と「向き損失」（外すと 4〜5 dB 落ち、MAE が 30〜52° に悪化）であり、IDE と位置エンコードの差（0.06 dB）は小さい。つまり反射方向系の性能は照明の符号化方式よりも法線の質で決まるという構図が源流からある。ユーザーの実験で真の形状を与えても 0.2〜0.4 dB しか改善しなかったのは、ベンチマークでの大きな法線効果が「推定法線がひどい」ことの裏返しであり、真の法線のもとではボトルネックが相互反射と微細形状に移る、という読みと整合する。
- Ref-NeRF の「実写では利得が消える」は、ユーザーの「像全体では共有照明が点ごとの SH に届かない」と同じ型の現象で、源流の論文自身が限界（相互反射・近接照明）として認めている。

### Gaps
- Ref-NeRF の IDE で使う SH の次数の上限（ℓ の集合）は今回の抽出で取れていない。
- Shiny Blender の物体数について、取得ツールは「66 物体」と返したが、表 1 の 6 シーン（teapot、toaster、car、ball、coffee、helmet）と矛盾するので抽出誤りとみなす（既存調査でも同じ旗を立てた）。実写シーン数も「33」と返したが付録は 3 シーンのみで、同じく抽出誤りとみられる。
- 公式コードは multinerf（後述）にあるが、Shiny Blender 用の設定ファイルの有無は README からは確認できなかった。

---

## 2. 3DGS 系の反射方向・共有環境の手法と、それぞれの工夫

### Takeaway
系譜は「GaussianShader（2023 年 11 月、前方シェーディング、最短軸 + 残差法線、事前フィルタ済みキューブマップ）」→「3DGS-DR（遅延シェーディングで画素単位の法線、法線の伝播、粗さなし）」→「2DGS サーフェル + 遅延シェーディング（Ref-GS: 方向特徴 × 空間特徴の外積と粗さ軸付きの球面 Mip グリッド、Ref-Gaussian: split-sum の PBR + 材質を見た法線伝播 + メッシュ追跡による可視性）」→「SDF 併用（GS-ROR2）」「近傍反射を別のガウシアン集合に分離（Ref-DGS）」「プリミティブ内の法線テクスチャ（TextureSplat）」と進んだ。共通して、遠方照明は全プリミティブで共有するグリッド／キューブマップで持ち、相互反射や近接照明は「プリミティブごとの残差色」か「空間特徴」に押し込むという二層構造をとる。

### Cited Findings

#### 一覧（照明・粗さ・法線・仮定）
| 手法 | 会議 | 拡散と鏡面 | 共有照明の表現 | 粗さ | 法線の得方と正則化 | 相互反射・近接光 |
|---|---|---|---|---|---|---|
| GaussianShader | CVPR 2024 | c_d + s⊙L_s + 残差 c_r（3 次 SH） | 6×64×64 キューブマップ、粗さ別にミップへ事前フィルタ（GGX NDF） | あり（ρ） | 最短軸 v + 残差 Δn1/Δn2（向きで使い分け）、‖Δn‖² 罰則、深度から求めた法線との整合 | 残差色 c_r に押し込む |
| 3DGS-DR | SIGGRAPH 2024 | (1−R)·C + R·E(反射方向)、C は SH 基本色、R は反射強度 | 環境マップ（cubemapencoder、画素ごとに双線形） | なし（粗い反射は SH 基本色へ） | 最短軸（カメラ向きに反転）を画素へ合成、明示的な正則化なし、周期的な「法線の伝播」 | 扱わない（1 画素 1 反射層） |
| Ref-GS | CVPR 2025 | I_d + f_Θ(S, K⊗S) | 経緯度の球面 Mip グリッド（512×1024、16 ch、9 段、粗さが第 3 軸） | あり（ミップ軸） | 2DGS サーフェル法線、深度歪み（λ=100）+ 深度法線整合（λ=0.05） | 空間特徴 K と方向特徴 S の外積で近傍効果を吸収 |
| Ref-Gaussian | ICLR 2025 | 簡易 Disney（アルベド・金属度・粗さ）、拡散は SH | split-sum（BRDF LUT + 粗さ別の事前積分キューブマップ） | あり | 2DGS 接ベクトルの外積、深度法線整合（λ=0.05）+ エッジを見た平滑化（λ=1.0）、材質を見た法線伝播 | TSDF メッシュの BVH 追跡で可視性、遮蔽部はガウシアンごとの SH 間接光 |
| GS-ROR2 | TOG 2025 | Disney（アルベド・粗さ・金属度） | 環境マップ + split-sum | あり | 最短軸 + 残差（GaussianShader 流）、TensoSDF の法線・深度と相互教師 | 直接光のみ、遮蔽も無視 |
| Ref-DGS | SIGGRAPH 2026 | C_diff + C_spec（MLP） | 球面特徴マップ（Sph-Mip、512×1024、d=4、9 段） | あり | 形状用 2DGS ガウシアンのみが法線・深度を出す | 局所反射用の別ガウシアン集合が特徴を出す |
| TextureSplat | 会議未確認（arXiv） | PBR（アルベド・金属度・粗さ） | split-sum + BRDF LUT + SH 間接光 | あり | 各 2D ガウシアンに接空間法線テクスチャ、深度法線整合 | SH 間接光 |

出典は以下の各項目。

#### GaussianShader
- 色は γ(c_d + s ⊙ L_s(ω_o, n, ρ) + c_r(ω_o))、L_s = ∫ L(ω_i) D(r, ρ)(ω_i·n) dω_i（GGX の D）、r = 2(ω_o·n)n − ω_o。環境照明は学習可能な 6×64×64 キューブマップで、粗さと反射方向でミップを補間して引く。残差色 c_r は 3 次 SH で、間接光と複雑な反射を担い、学習の初期は外して材質・照明に色を割り振らせ、後から入れる — [GaussianShader（本文確認）](https://arxiv.org/html/2311.17977)
- 法線: 最適化でガウシアンが平たくなることを観察し、最短軸 v を基本の法線とする。符号の曖昧さのため、ω_o·v > 0 なら n = v + Δn1、そうでなければ n = −(v + Δn2)。‖Δn‖² で残差を小さく保ち、描画した法線マップと、描画した深度から Sobel 様の演算で求めた法線との整合損失を課す（KNN 探索を避けるため）。損失は L_color + 0.01·L_normal + 0.001·L_sparse + 0.001·L_reg — [GaussianShader](https://arxiv.org/html/2311.17977)
- アブレーション（Shiny Blender の縮小解像度、PSNR）: 完全版 32.09、法線整合なし 30.93、残差色なし 31.49、最短軸なし（素朴な法線）31.47、MLP 照明 29.73 — [GaussianShader](https://arxiv.org/html/2311.17977)
- 本文に限界節はなく、凹面・相互反射の失敗は論じていない — [GaussianShader](https://arxiv.org/html/2311.17977)
- 会議は CVPR 2024（pp. 5322–5332） — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.html)

#### 3DGS-DR（Deferred Reflection）
- 各ガウシアンは SH の基本色 c_i(v) とスカラーの反射強度 r_i を持ち、まず基本色マップ C と反射強度マップ R と法線マップ N を画素に合成し、遅延パスで (1 − R)·C + R·E(反射方向) を合成する — [3DGS-DR（本文確認）](https://arxiv.org/html/2404.18454)
- 遅延が効く理由: 色からの勾配が画素ごとに合成された法線を経由して各ガウシアンの法線へ流れるので、重なったガウシアン同士が互いに影響する。前方（ガウシアンごと）のシェーディングでは各ガウシアンの勾配が独立になる。前方シェーディングは ball で 27.64 dB、遅延は 33.66 dB — [3DGS-DR](https://arxiv.org/html/2404.18454)
- 法線は最短軸をカメラ向きに反転したもので、法線への勾配は環境マップの参照からしか来ない（明示的な正則化なし） — [3DGS-DR](https://arxiv.org/html/2404.18454)
- **法線の伝播**: 周期的に不透明度を 0.9 以上、反射強度を 0.001 以上に引き上げ、反射するガウシアンの長い 2 軸を 1.5 倍にして（法線軸は保つ）隣へ法線を広げる。元の 3DGS の不透明度リセット（0.01）とは交互に行う。r_i ≤ 0.1 のガウシアンには伝播のたびに ±10% の色ノイズを加え（「色の妨害」）、過学習した拡散色が反射面の発見を妨げないようにする。r_i > 0.1 の数が増えなくなったら伝播を止め、その後で高次 SH を最適化する — [3DGS-DR](https://arxiv.org/html/2404.18454)
- アブレーション（ball、PSNR）: 完全版 33.66、伝播なし 27.85、色の妨害なし 30.00 — [3DGS-DR](https://arxiv.org/html/2404.18454)
- 粗さのパラメータはなく、粗い反射・光沢の縁・大域照明は SH 基本色に任せる。物理ベースの粗さは将来課題 — [3DGS-DR](https://arxiv.org/html/2404.18454)
- 実写では物体周りの球領域内のガウシアンだけに遅延反射を許し、背景の干渉を抑える — [3DGS-DR](https://arxiv.org/html/2404.18454)
- 会議は SIGGRAPH Conference Papers '24（DOI 10.1145/3641519.3657456） — [3DGS-DR](https://arxiv.org/html/2404.18454)

#### Ref-GS（Directional Factorization for 2D Gaussian Splatting）
- 2DGS サーフェルに拡散色・粗さ・4 次元特徴 f を持たせ、遅延で G バッファ {粗さ M、法線 N、特徴 K} を作る。I = I_d + f_Θ(S, K⊗S) で、S は反射方向 (θ, φ) と粗さ ρ で球面 Mip グリッドを三線形補間した方向特徴、K⊗S は画素ごとの外積を平らにして小 MLP に通す — [Ref-GS（本文確認）](https://arxiv.org/html/2412.00905)
- K⊗S は「大域照明、影、自己遮蔽」を含む空間変化する近傍効果を担うと主張し、K⊗S を外すと相互反射が復元されないと図 7 で示す。遠方照明は Sph-Mip が担う — [Ref-GS](https://arxiv.org/html/2412.00905)
- 法線は 2DGS の損失（深度歪み λ_d = 100、サーフェル法線と深度勾配法線の整合 λ_n = 0.05）で縛る — [Ref-GS](https://arxiv.org/html/2412.00905)
- アブレーション（合成、PSNR）: 完全版 34.00、Sph-Mip なし 29.95、ミップなし 30.12、遅延シェーディングなし 31.79、K⊗S なし 33.37。ミップなしでは粗い面にアーティファクトと形状誤差が出る — [Ref-GS](https://arxiv.org/html/2412.00905)
- 会議は CVPR 2025（pp. 26483–26492） — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Ref-GS_Directional_Factorization_for_2D_Gaussian_Splatting_CVPR_2025_paper.html)

#### Ref-Gaussian（Reflective Gaussian Splatting）
- 2DGS サーフェル（3D ガウシアンは薄い面と相性が悪いため）に簡易 Disney の材質を持たせ、遅延で PBR を当てる。split-sum の第 1 項は 2D LUT、第 2 項は毎反復で粗さ別に事前積分したキューブマップ。拡散は積分拡散照明より SH のほうが当てはまりがよかった — [Ref-Gaussian（本文確認）](https://arxiv.org/html/2412.19282)
- 法線は n = t_u × t_v。深度法線整合（λ_n = 0.05）とエッジを見た法線平滑化（λ_smooth = 1.0）。最初の約 18,000 ステップはガウシアンごとのシェーディング、その後約 40,000 ステップを遅延で行い、第 2 段の前に色と材質をリセットして形状だけ残す。**材質を見た法線伝播**: 金属度 ≥ 0.02 かつ粗さ ≤ 0.1 のガウシアンのスケールを周期的に拡大して法線を隣へ広げる（3DGS-DR の伝播を材質で条件づけたもの） — [Ref-Gaussian](https://arxiv.org/html/2412.19282)
- 相互反射: 3,000 ステップごとに TSDF でメッシュを抽出し、反射方向の BVH 追跡で可視性を判定、遮蔽部ではガウシアンごとの SH 間接色を反射方向で評価して合成する（光線追跡は他担当の範囲なので要点のみ） — [Ref-Gaussian](https://arxiv.org/html/2412.19282)
- アブレーション（表 4、データセット名は未記載だが完全版 30.33 が本文の Glossy Synthetic の値と一致）: 完全版 30.33、PBR なし 29.84、相互反射なし 30.14、遅延なし 28.85、初期段なし 29.94、材質を見た伝播なし 29.68。法線 MAE（表 3）: 完全版 2.15、2DGS なし 4.45、初期段なし 3.53、材質を見た伝播なし 3.86、PBR なし 3.81。著者は相互反射が Glossy Synthetic では少数派の効果だと述べる — [Ref-Gaussian](https://arxiv.org/html/2412.19282)
- 会議は ICLR 2025 — [ICLR 2025 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/abf3682c9cf9245a0294a4bebe4544ff-Abstract-Conference.html)

#### GS-ROR / GS-ROR2（SDF の併用）
- 取得できた arXiv 2406.18544v4 は GS-ROR2（リポジトリ名は GS-ROR）。属性（法線・アルベド・粗さ・金属度）を画素に合成してから遅延でシェーディングし、鏡面ローブを鋭く保つ — [GS-ROR2（本文確認）](https://arxiv.org/html/2406.18544)
- 法線: ガウシアン側は最短軸 + 学習する残差（GaussianShader 流）。SDF 側は TensoSDF から NeuS 型の重みで法線・深度を体積描画する。相互教師は片側に stop-gradient をかけ、L_gs2sdf = |sg[D_gs] − D_sdf| + (1 − ⟨sg[n_gs], n_sdf⟩)、L_sdf2gs = |D_gs − sg[D_sdf]| + (1 − ⟨n_gs, sg[n_sdf]⟩)。SDF 値が閾値を超える中心のガウシアンを刈る。日程は GS 単独 1K → ガウシアン固定で SDF 予熱 3K → 同時 24K → SDF 仕上げ約 15 分 — [GS-ROR2](https://arxiv.org/html/2406.18544)
- 仮定: 直接光のみ、効率のため遮蔽も落とす。遅延パイプラインは不透明面を仮定 — [GS-ROR2](https://arxiv.org/html/2406.18544)
- 会議は ACM TOG 2025（DOI 10.1145/3759248） — [GS-ROR2](https://arxiv.org/html/2406.18544)

#### Ref-DGS（Reflective Dual Gaussian Splatting）
- 形状用ガウシアン（2DGS 形式、拡散色と粗さ、深度と法線を出す）と局所反射用ガウシアン（学習する局所鏡面特徴を持ち、形状を歪めない想定）の二集合に分け、遠方は球面特徴マップを反射方向と粗さで引く。MLP（3 層、幅 64）が大域特徴・局所特徴・粗さ・cos_NV から鏡面色を出し、C = C_diff + C_spec — [Ref-DGS（本文確認）](https://arxiv.org/html/2603.07664)
- 著者は、環境・単一反射モデル（GaussianShader、Spec-Gaussian）は自己反射や相互反射などの近傍鏡面反射を捉えられないと述べ、Ref-GS は近傍効果を空間特徴と単一のガウシアン集合に吸収するので反射由来の形状アーティファクトが出ると批判する — [Ref-DGS](https://arxiv.org/html/2603.07664)
- 会議は SIGGRAPH Conference Papers '26（DOI 10.1145/3799902.3811163） — [Ref-DGS](https://arxiv.org/html/2603.07664)

#### TextureSplat（プリミティブ内の法線テクスチャ）
- 1 プリミティブ 1 値の材質・法線では、Ref-Gaussian や 3DGS-DR の高周波なハイライトや鋭い反射が不足すると仮説を立て、各 2D ガウシアンの局所空間にアルベド・金属度・粗さと接空間法線のテクスチャを持たせる（光線とスプラットの交点の (u, v) で引く）。遅延シェーディング、split-sum 環境照明、BRDF LUT、SH 間接光。深度法線整合で縛る — [TextureSplat（本文確認）](https://arxiv.org/html/2506.13348)

#### その他（スニペットまたは境界）
- **Normal-GS**（NeurIPS 2024）: 法線を 3DGS の描画に組み込み、拡散を法線と IDIV の内積、鏡面を IDE で表す。ガウシアンの離散性・雑音が面推定を妨げると主張 — [Normal-GS（スニペット）](https://arxiv.org/abs/2410.20593v1), [NeurIPS 2024 ポスター](https://nips.cc/virtual/2024/poster/93867)
- **Spec-Gaussian**: SH の代わりに異方性球面ガウス（ASG）の外観場を使う。共有の環境ではなくガウシアンごとの表現なので本系統の外側（会議は未確認） — [Normal-GS 検索結果中の要約（スニペット）](https://liner.com/review/normalgs-3d-gaussian-splatting-with-normalinvolved-rendering)
- **MaterialRefGS**（NeurIPS 2025）: 2DGS で遅延シェーディング時の材質マップを多視点で一致させ、視点間の測光変化から反射強度の事前分布を作る。物体間の遮蔽による間接光は 2DGS の光線追跡で扱う（光線追跡は他担当） — [MaterialRefGS（スニペット）](https://arxiv.org/abs/2510.11387), [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/118850)
- **RTR-GS**（arXiv 2507.07733）: 鏡面成分を学習する反射マップに持たせ、遅延で反射損失が法線の最適化を駆動する — [検索スニペット](https://arxiv.org/html/2507.07733v1)
- **ReflFlow**（arXiv 2512.10095）: 動的シーンで時間変化する環境照明をガウシアンで表し、形状に沿った法線損失で反射方向の精度を上げる — [検索スニペット](https://arxiv.org/pdf/2512.10095)
- **NeRF 側の関連**: ENVIDR（ICCV 2023）は SDF 面モデルと、PBR を模倣するよう事前学習した拡散・鏡面・環境の 3 つの MLP からなるニューラルレンダラーを組み、学習時は SDF と新しい環境 MLP だけを更新する。相互反射の間接光も合成できると主張 — [ENVIDR（CVF）](https://openaccess.thecvf.com/content/ICCV2023/html/Liang_ENVIDR_Implicit_Differentiable_Renderer_with_Neural_Environment_Lighting_ICCV_2023_paper.html)。NDE（CVPR 2024）は角度領域に特徴グリッドの空間エンコードを当て、空間特徴をコーン追跡して空間変化する方向エンコードにすることで相互反射を扱う — [NDE](https://arxiv.org/abs/2405.14847)。NeRO は NeuS 型 SDF + split-sum で直接光と間接光を 2 つの MLP に分け、限界として微細形状の平滑化を認める — [既存調査（NeRO の項）](https://ar5iv.labs.arxiv.org/html/2305.17398)、`D:\GoLD_rayt\out\studies\nerf_survey\reports\NeRF質感再構成の文献調査.md`

### Inferences
- 系統全体が「共有する遠方照明 + プリミティブごと（または画素ごと）の補正項」という二層構造に収束している。GaussianShader の c_r、3DGS-DR の SH 基本色、Ref-Gaussian の SH 間接光、Ref-GS の K⊗S、Ref-DGS の局所反射ガウシアン、Ref-NeRF のボトルネック b、NDE のコーン追跡特徴がすべて「共有照明で説明できない分」の受け皿である。ユーザーの実験（共有照明単独では点ごとの SH に負け、相互反射と微細凹凸の観測を除くと勝つ）は、まさにこの受け皿がない設定であり、文献の設計はユーザーの観察を裏づけている。次の一手としては「共有照明 + 点ごとの低次 SH 残差」や「反射光線が像に当たる点にだけ局所項を許す」が文献の流れに沿う。
- 法線の扱いは「最短軸 + 残差 + 深度整合」（GaussianShader、GS-ROR2）から「2DGS サーフェル + 深度歪み + 深度整合」（Ref-GS、Ref-Gaussian、Ref-DGS、TextureSplat）へ移った。Ref-Gaussian の「2DGS なし」で MAE が 2.15→4.45 に悪化することは、平たいプリミティブそのものが法線の質を支えていることを示す。
- 遅延シェーディングは「画素で合成した法線で 1 回だけ照明を引く」ので、1 画素内の法線のばらつき（ユーザーの像で中央値 8°、90% 点 28°）は平均法線に潰れ、ローブの広がりは失われる（調査側の推論）。粗さを持つ Ref-GS / Ref-Gaussian はこれを学習で粗さを上げて吸収しうるが、粗さのない 3DGS-DR はできない。逆に前方シェーディング（ガウシアンごとに照明を引いてから合成）は画素内で放射輝度を平均するので物理的には足跡積分に近いが、3DGS-DR は勾配の都合で前方が 6 dB 悪いと報告している。この「勾配の通りやすさ」と「画素内の法線分布の再現」のトレードオフを明示的に論じた論文は見つからなかった。
- ユーザーの金（GGX α 0.01）はほぼ鏡面なので、split-sum の低粗さでのぼけ（GS-ROR2 が限界として明記）や、キューブマップ解像度（GaussianShader は 64×64 の面）が直接効く。3DGS-DR が粗さなし（鏡面のみ）で強いのも、Shiny Blender の多くが低粗さだからかもしれない（推論）。

### Gaps
- 3DGS-DR の環境マップがキューブマップかどうかは本文に明記がなく、サブモジュール名 `cubemapencoder` からの推定にとどまる（README も明言しない）。
- Spec-Gaussian、3iGS（Ref-GS の表に登場、Shiny Blender 30.64）、DeferredGS、GlossyGS は本文を確認しておらず、会議名も未確認。
- Ref-GS のアブレーション「Sph-Mip なし」で何に置き換えたか（MLP か IDE か）は抽出できなかった。
- TextureSplat の会議は未確認（arXiv v2、2025 年 12 月改訂）。

---

## 3. 法線推定の失敗（凹面、細かい凹凸、薄い構造）と遠方照明の仮定違反について、各論文と後続研究が述べていること

### Takeaway
遠方照明の仮定違反（相互反射・近接光）はほぼ全手法が限界として認めるか、残差項に押し込むことで回避しているが、その効果を定量した例は少なく、定量した場合も Shiny Blender / Glossy Synthetic では相互反射の寄与が小さい（Ref-Gaussian で 0.19 dB、Ref-GS で 0.63 dB）。凹面については 3DGS-DR（ベルで法線伝播が 72k ステップまで終わらない）と GS-ROR2（凹部で材質分解が不整合）が明示する。細かい凹凸については TextureSplat が「1 プリミティブ 1 法線」の表現力不足を動機に挙げ、NeRO が微細形状の平滑化を認める程度で、画素より細かい法線のばらつきを照明モデル側でどう扱うかを論じた論文は見つからなかった。

### Cited Findings
- **Ref-NeRF**: 相互反射と遠方でない照明は明示的に扱わず、その場合は利得が縮む。面として捉えられない領域では一定の法線を予測してしまう。凹面は論じない — [Ref-NeRF](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **GaussianShader**: 限界節がなく、凹面・相互反射の失敗は論じない。間接光は残差色に入れる設計 — [GaussianShader](https://arxiv.org/html/2311.17977)
- **3DGS-DR**: 1 画素 1 反射層（古典的な遅延シェーディングの制約）。車の窓が不整合（前窓は完全反射、横窓は完全透過、図 7）。凹形状（ベル）は収束が遅く、法線伝播が約 72k ステップまで完了しない。実写は前景マスクに依存し、背景物体が環境マップの当てはめを妨げる — [3DGS-DR](https://arxiv.org/html/2404.18454)
- **3DGS-DR の README**: 既定の設定では法線がうまく伝播しないことがあり、`--opac_lr0_interval` の既定値 200 は場合によって不安定・低品質になりうる — [3DGS-DR GitHub](https://github.com/gapszju/3DGS-DR)
- **GS-ROR2**: split-sum は非常に低い粗さ（鏡面に近い面）でぼける。直接光のみなので、相互反射のある凹部で材質分解が不整合になる。不透明面を仮定するので Coffee の液体で失敗する。ガウシアン系の結果は NeRO や TensoSDF よりぼける場合がある — [GS-ROR2](https://arxiv.org/html/2406.18544)
- **Ref-GS**: K⊗S が相互反射を含む近傍効果を吸収すると主張するが、専用の失敗例節はない。限界はニューラルデコーダによる描画の遅さと CG エンジンへの統合の難しさ（付録 B） — [Ref-GS](https://arxiv.org/html/2412.00905)
- **Ref-DGS による Ref-GS 批判**: 近傍効果を単一のガウシアン集合と空間特徴に吸収すると反射由来の形状アーティファクトが出る。一般に、モデル化されていない鏡面効果が形状に吸収されて、面の縮み、へこみ、細部の消失が起きる — [Ref-DGS](https://arxiv.org/html/2603.07664)
- **Ref-Gaussian**: 3D ガウシアンは薄い面と相性が悪いので 2DGS を選ぶ。可視性を反射方向だけで近似するので材質推定に雑音が入りうる。相互反射は Glossy Synthetic では少数派の効果（外すと 30.33→30.14 dB） — [Ref-Gaussian](https://arxiv.org/html/2412.19282)
- **TextureSplat**: 1 プリミティブ 1 値の法線・材質では高周波なハイライトと鋭い反射が出ないことを動機とする。限界は全プリミティブ一律のテクスチャ解像度と、双線形フィルタのみ（非アンチエイリアスの 2D ガウシアンと三線形ミップの統合が難しい）による縮小時のアーティファクト — [TextureSplat](https://arxiv.org/html/2506.13348)
- **NeRO**（NeRF 側）: 微細形状が平滑化される、異方性不可、静的光環境、カメラ姿勢に敏感 — [既存調査に引用した NeRO](https://ar5iv.labs.arxiv.org/html/2305.17398)
- **Ref-NeRF の実写**: garden spheres（周囲を映す球が並ぶシーン）では SSIM が mip-NeRF より下がる（0.620→0.601） — [Ref-NeRF 表 S1](https://ar5iv.labs.arxiv.org/html/2112.03907)
- **実写ベンチ（Ref-Real / Shiny Real）での共有照明系の伸び悩み**: Ref-GS の表で 3DGS 23.85 dB に対し GaussianShader 23.46、3DGS-DR 23.80、ENVIDR 23.00、Ref-NeRF 23.62 — [Ref-GS 表 1](https://arxiv.org/html/2412.00905)。Ref-DGS の表で Ref-Gaussian 23.95、Ref-GS 23.93、EnvGS（光線追跡）24.55、Ref-DGS（局所反射ガウシアン）24.89 — [Ref-DGS 表](https://arxiv.org/html/2603.07664)

### Inferences
- 実写の Ref-Real では、点ごとの SH（素の 3DGS、23.85）を共有照明系がほとんど上回れず、明確に上回るのは近傍反射の受け皿を明示的に持つ Ref-DGS と光線追跡の EnvGS だけである。これはユーザーの「像全体では共有照明 22〜24.5 dB が点ごとの SH 26.8 dB に届かない」と同じ傾向で、合成ベンチ（Shiny Blender）の滑らかで相互反射の少ない物体では見えにくい弱点が、実写や複雑な形状で表に出ることを示唆する。
- 定量された相互反射の寄与（Ref-Gaussian 0.19 dB、Ref-GS 0.63 dB）は、ユーザーの像（反射光線の 38% が自分に当たる）より相互反射の少ない物体での値であり、Erato のような彫像では寄与がずっと大きいと予想される（推論）。既存ベンチマークは相互反射が強い物体を十分に含んでいない。
- ユーザーの「画素内の法線のばらつき」（中央値 8°、90% 点 28°）は、ベンチの最良法線 MAE（1.4〜2.6°、Shiny Blender）よりはるかに大きく、ベンチ物体（teapot、toaster、ball 等）が滑らかであることを裏から示す。文献は細かい凹凸を「法線テクスチャで表現力を足す」（TextureSplat）方向で扱い、「画素の足跡内の法線分布をローブの広がりとして照明側でフィルタする」（グラフィックスの法線マップのフィルタリングに相当）方向は見当たらない。後者はユーザーの問題に直接効く空白である（推論）。
- Ref-DGS の「モデル化されていない鏡面効果が形状に吸収されて縮み・へこみが出る」は、形状も同時に学習する手法の失敗様式であり、真の形状を固定したユーザーの実験ではこの経路は閉じている。その代わり誤差は照明（共有の SH）か拡散項に押し込まれるので、照明位置が正しく出た一方で全体 PSNR が伸びなかったことと整合する（推論）。

### Gaps
- 凹面・薄い構造での法線 MAE を物体別・部位別に集計した論文は見つからなかった。
- 遠方照明の仮定違反を、物体の大きさと光源距離の比で系統的に変えて評価した研究は見つからなかった。
- GaussianShader の補遺（CVF の supplemental PDF）は読んでおらず、そこに失敗例があるかは未確認。

---

## 4. 光沢データでの数値（PSNR/SSIM/LPIPS、法線の角度誤差）と比較相手

### Takeaway
Shiny Blender の PSNR は、Ref-NeRF 自論文 35.96、3DGS 系の他論文の表では Ref-NeRF 32.32〜33.32 と、同じ手法でも論文によって 1〜3 dB ずれ、法線 MAE も GaussianShader で 7.00°〜22.31°、3DGS-DR で 2.43°〜4.87° と大きく食い違う。順位は概ね Ref-DGS（35.21）≳ Ref-Gaussian（34.92〜35.04）≈ Ref-GS（34.80）> MaterialRefGS（34.16）> 3DGS-DR（33.94）> ENVIDR（32.88）> GaussianShader（30.42〜31.94）> 3DGS（30.37）だが、1 dB 未満の差や論文をまたいだ比較は読みすぎないほうがよい。

### Cited Findings

#### Ref-NeRF 自論文
- Shiny Blender（PSNR / SSIM / LPIPS / MAE°）: Ref-NeRF 35.96 / 0.967 / 0.058 / 18.38、mip-NeRF 29.76 / 0.942 / 0.092 / 60.38、PhySG（マスク必要）26.21 / 0.921 / 0.121 / 8.46 — [Ref-NeRF 表 1](https://ar5iv.labs.arxiv.org/html/2112.03907)
- Blender（NeRF 合成 8 シーン）: Ref-NeRF 33.99 / 0.966 / 0.038 / 23.22、mip-NeRF 33.09 / 0.961 / 0.043 / 38.30 — [Ref-NeRF 表 2](https://ar5iv.labs.arxiv.org/html/2112.03907)

#### GaussianShader 自論文
- Shiny Blender 平均（PSNR / SSIM / LPIPS）: GaussianShader 31.94 / 0.957 / 0.068、3DGS 30.37 / 0.947 / 0.083、ENVIDR 32.88 / 0.969 / 0.072、Ref-NeRF 32.32 / 0.956 / 0.109、NeRO 29.84 / 0.962 / 0.072、NVDiffRec 28.70 / 0.945 / 0.119 — [GaussianShader](https://arxiv.org/html/2311.17977)
- Glossy Synthetic（補遺、3DGS→本手法）: 26.26→27.36 dB。Tanks and Temples: 29.54→29.73 dB。NeRF Synthetic: 33.38（3DGS 33.30） — [GaussianShader](https://arxiv.org/html/2311.17977)
- 学習時間と FPS（Shiny Blender と NeRF Synthetic の平均）: Ref-NeRF 23 h / 0.03 FPS、ENVIDR 6 h / 1.33、3DGS 0.25 h / 274、本手法 0.58 h / 97。法線 MAE は報告せず定性的のみ — [GaussianShader](https://arxiv.org/html/2311.17977)

#### 3DGS-DR 自論文
- ball（Shiny Blender）: 33.66 dB（3DGS 27.65、Ref-NeRF 33.16、ENVIDR 41.02 で ENVIDR が上）。実写は「同程度」で明確な優位はない — [3DGS-DR 表 1](https://arxiv.org/html/2404.18454)
- 法線 MAE（Shiny Blender）: 3DGS-DR 4.871°、ENVIDR 4.618°、NVDiffRec 17.02°、GaussianShader 22.31°。環境マップの LPIPS: 3DGS-DR 0.511、ENVIDR 0.615、GaussianShader 0.621 — [3DGS-DR 表 2](https://arxiv.org/html/2404.18454)
- 時間（RTX 4090、Shiny Blender）: 3DGS-DR 16 分 / 251 FPS、3DGS 6 分 / 277、GaussianShader 60 分 / 51、ENVIDR 3.2 h / 3。実写は 3DGS-DR 47 分 / 80 FPS — [3DGS-DR 表 3](https://arxiv.org/html/2404.18454)

#### Ref-GS 自論文
- Shiny Blender PSNR: Ref-GS 34.80（SSIM 0.973、LPIPS 0.056）、3DGS-DR 33.94、ENVIDR 32.88、Ref-NeRF 32.32、3iGS 30.64、GaussianShader 30.42、3DGS 30.37、NeRO 29.84 — [Ref-GS 表 1](https://arxiv.org/html/2412.00905)
- Shiny Real（PSNR / SSIM / LPIPS）: Ref-GS 24.44 / 0.682 / 0.224、3iGS 24.10 / 0.657 / 0.231、3DGS 23.85 / 0.660 / 0.230、3DGS-DR 23.80 / 0.659 / 0.236、Ref-NeRF 23.62 / 0.646 / 0.239、GaussianShader 23.46 / 0.647 / 0.257、ENVIDR 23.00 / 0.606 / 0.332 — [Ref-GS 表 1](https://arxiv.org/html/2412.00905)
- Glossy Synthetic: Ref-GS 30.59 / 0.957 / 0.058（LPIPS は ENVIDR と 3DGS-DR が 0.057 でわずかに上）。NeRF Synthetic: 33.20（3iGS 33.60、3DGS 33.30 が上） — [Ref-GS 表 2](https://arxiv.org/html/2412.00905)
- 法線 MAE（Shiny Blender）: Ref-GS 2.21°、3DGS-DR 2.43°、ENVIDR 4.61°、Ref-NeRF 18.38° — [Ref-GS 表 2](https://arxiv.org/html/2412.00905)
- 速度（3DGS 比、3DGS は約 14.5 分・125 FPS）: Ref-GS 描画 0.37×・学習 2.63×、GaussianShader 0.17×・11.05×、3iGS 0.44×・2.07×、3DGS-DR 0.93×・3.25× — [Ref-GS 表 4](https://arxiv.org/html/2412.00905)

#### Ref-Gaussian 自論文
- 法線 MAE: Ref-Gaussian 2.15、3DGS-DR 2.62、ENVIDR 2.74、GaussianShader 7.00。環境マップ LPIPS: 0.4431、3DGS-DR 0.5290、ENVIDR 0.5099、GaussianShader 0.6145。学習時間 0.58 h（3DGS-DR 0.35 h、GaussianShader 0.48 h、ENVIDR 5.84 h）、FPS 122（3DGS-DR 251、GaussianShader 28、ENVIDR 1） — [Ref-Gaussian 表 2](https://arxiv.org/html/2412.19282)
- Shiny Blender PSNR: Ref-Gaussian 35.04、NDE 35.48（NDE が上、ただし 66 FPS 対 122 FPS）。Glossy Blender: Ref-Gaussian 30.33、NeRO 29.73、NeuS 27.86。ball: 37.01（3DGS-DR 33.43、ENVIDR 41.02）。Ref-Real（garden / sedan / toycar）: 22.97 / 26.60 / 24.27 — [Ref-Gaussian](https://arxiv.org/html/2412.19282)

#### GS-ROR2 自論文
- NVS（PSNR / SSIM）: Glossy Blender 29.32 / .9537（GaussianShader 27.34）、Shiny Blender 33.01 / .9689（GaussianShader 30.94） — [GS-ROR2](https://arxiv.org/html/2406.18544)
- 再照明（Glossy Blender、PSNR / SSIM）: NeRO 26.63 / .9331、TensoSDF 24.97 / .9349、GS-ROR2 23.39 / .9140、R3DG 17.09、GaussianShader 16.95、GS-IR 16.26。NeRF 系（SDF）が再照明では上 — [GS-ROR2](https://arxiv.org/html/2406.18544)
- 法線 MAE（Shiny Blender）: GS-ROR2 7.23、GaussianShader 10.93、R3DG 19.81、GS-IR 23.99。メッシュ CD（Glossy Blender、×10²）: GS-ROR2 0.62、PGSR 2.44、GaussianShader 2.53、GOF 2.58、2DGS 2.72 — [GS-ROR2](https://arxiv.org/html/2406.18544)
- 学習 1.5 h / 208 FPS（RTX 4090）、NeRO 12 h / 1/4 FPS、TensoSDF 6 h — [GS-ROR2 表 2](https://arxiv.org/html/2406.18544)

#### Ref-DGS（2026）の統一比較表
- Shiny Synthetic（PSNR / SSIM / LPIPS）: Ref-DGS 35.21 / 0.975 / 0.053、Ref-Gaussian 34.92 / 0.973 / 0.056、Ref-GS 34.80 / 0.973 / 0.056、MaterialRefGS 34.16 / 0.972 / 0.058、3DGS-DR 33.94 / 0.971 / 0.059、Ref-NeRF 33.32 / 0.956 / 0.110、EnvGS 33.26 / 0.968 / 0.063、ENVIDR 32.88 / 0.969 / 0.072、GS-ROR2 32.16 / 0.963 / 0.069、GaussianShader 30.42 / 0.951 / 0.082 — [Ref-DGS](https://arxiv.org/html/2603.07664)
- Glossy Synthetic: MaterialRefGS 30.74、Ref-DGS 30.63、Ref-GS 30.37、Ref-Gaussian 29.61、3DGS-DR 29.36、ENVIDR 29.06、GS-ROR2 28.96、EnvGS 28.17、GaussianShader 27.11、Ref-NeRF 25.65 — [Ref-DGS](https://arxiv.org/html/2603.07664)
- RefReal: Ref-DGS 24.89 / 0.697 / 0.224、EnvGS 24.55、PGSR 24.43、Ref-Gaussian 23.95、Ref-GS 23.93、3DGS-DR 23.80、ENVIDR 23.80、Ref-NeRF 23.62、GaussianShader 23.46（GS-ROR2 と MaterialRefGS はコードの都合で実写の結果なし） — [Ref-DGS](https://arxiv.org/html/2603.07664)
- 法線 MAE（Shiny Synthetic）: Ref-DGS 1.43、Ref-Gaussian 2.17、MaterialRefGS 2.20、Ref-GS 2.21、3DGS-DR 2.43、GS-ROR2 2.64、ENVIDR 4.61、UniSDF 8.71、GaussianShader 10.93、MILo 15.02、PGSR 16.13、Ref-NeRF 18.38。Glossy Synthetic の MAE: Ref-DGS 1.88、MaterialRefGS 2.12、Ref-GS 2.32、Ref-Gaussian 2.44、GS-ROR2 2.51、GaussianShader 4.35、3DGS-DR 4.89 — [Ref-DGS](https://arxiv.org/html/2603.07664)
- FPS（Shiny Synthetic）: Ref-DGS 76.34、Ref-GS 67.63、MaterialRefGS 30.07、EnvGS 26.22 — [Ref-DGS](https://arxiv.org/html/2603.07664)

#### TextureSplat 自論文
- 法線 MAE（Shiny Blender）: TextureSplat 1.78、Ref-Gaussian 2.078、3DGS-DR 4.871、ENVIDR 4.618、NVDiffRec 17.02、GaussianShader 22.31（後半 4 つは 3DGS-DR 自論文の値と同一） — [TextureSplat](https://arxiv.org/html/2506.13348)
- PSNR（Ref-Gaussian との比較）: ball 39.27 対 36.07、teapot 48.27 対 47.15、toaster 28.40 対 28.28。実写 garden 23.34 対 23.09、sedan 26.45 対 26.23、toycar 24.95 対 24.74 — [TextureSplat](https://arxiv.org/html/2506.13348)

### Inferences
- **論文間の食い違い**: (a) Ref-NeRF の Shiny Blender PSNR は自論文 35.96、GaussianShader / Ref-GS の表 32.32、Ref-DGS の表 33.32。(b) GaussianShader は自論文 31.94、Ref-GS / Ref-DGS の表 30.42、GS-ROR2 の表 30.94。(c) GaussianShader の法線 MAE は 22.31（3DGS-DR、TextureSplat）、10.93（GS-ROR2、Ref-DGS）、7.00（Ref-Gaussian）。(d) 3DGS-DR の法線 MAE は自論文 4.871、Ref-GS / Ref-DGS 2.43、Ref-Gaussian 2.62。(e) Ref-Gaussian の Shiny Blender PSNR は自論文 35.04、Ref-DGS の表 34.92。(f) Ref-GS の Glossy Synthetic は自論文 30.59、Ref-DGS の表 30.37。再実行か転記か、法線の計算法（最短軸の合成か深度勾配か、マスク・背景の扱い）、解像度の違いとみられるが、どの表も条件の詳細を書いていない。法線 MAE は特に計算法依存が大きく、論文をまたいだ比較は避けるべき。
- 共有照明系の数値上の優位は合成の光沢データに集中しており、実写（Ref-Real）では差がほぼ消える（3DGS 23.85 対 GaussianShader 23.46、3DGS-DR 23.80）。
- 再照明では SDF を持つ NeRF 系（NeRO 26.63、TensoSDF 24.97）が依然としてガウシアン系（GS-ROR2 23.39）を上回る。共有照明が「正しく分解されているか」の指標として、NVS の PSNR より環境マップ LPIPS や再照明 PSNR が有用だが、報告している論文は一部に限られる。

### Gaps
- 3DGS-DR の Shiny Blender 平均値（自論文の表）は取得できず、ball の値のみ。平均 33.94 は Ref-GS / Ref-DGS の表からの値。
- GS-ROR2 の LPIPS は補遺にあり、今回は取得していない。
- Ref-DGS の表で 3DGS-DR の Glossy Synthetic の CD が「–」になっている理由は未確認。

---

## 5. コード公開の有無（GitHub URL）と依存

### Takeaway
主要手法はすべてコードが公開されており、3DGS 系はいずれも公式 3DGS / 2DGS のラスタライザを改造した独自 CUDA サブモジュールを使う（3DGS-DR は `diff-gaussian-rasterization_c3/_c7` と `cubemapencoder`、Ref-GS は `diff-surfel-rasterization` の 3 変種と nvdiffrast）。Ref-NeRF は JAX の multinerf（2025 年 2 月にアーカイブ）、GaussianShader は残差色の学習コードが未公開と README に記されている点に注意。

### Cited Findings
- **Ref-NeRF**: google-research/multinerf に Ref-NeRF の内部実装が統合されている（mip-NeRF 360 のフォーク、JAX）。Ref-NeRF の結果は「わずかに異なる場合がある」と注記、2025 年 2 月 11 日にアーカイブ（読み取り専用） — [multinerf](https://github.com/google-research/multinerf)。プロジェクトページは dorverbin.github.io/refnerf — [検索結果](https://arxiv.org/abs/2112.03907v1)
- **GaussianShader**: [Asparagus15/GaussianShader](https://github.com/Asparagus15/GaussianShader)。`environment.yml` で conda 環境を作り、`submodules/` がある（中身は README から確認できず）。謝辞に gaussian splatting、Ref-NeRF、nvdiffrec、Point-NeRF。README は「まだ構築中」で、引数の説明と**残差色の学習コード**が未チェック項目。学習フラグ `--lambda_predicted_normal 2e-1`。Glossy Synthetic は `nero2blender.py` で変換 — [GaussianShader GitHub](https://github.com/Asparagus15/GaussianShader)
- **3DGS-DR**: [gapszju/3DGS-DR](https://github.com/gapszju/3DGS-DR)。素の 3DGS の環境を作った上で `diff-gaussian-rasterization_c3`、`diff-gaussian-rasterization_c7`、`cubemapencoder`、`simple-knn`（素の 3DGS と同一）を入れる。nvdiffrast への言及なし。実写は `--use_env_scope` と球の中心・半径で環境光の範囲を限定（独自データでは点群を Blender に読み込んで球を置く） — [3DGS-DR GitHub](https://github.com/gapszju/3DGS-DR)
- **Ref-GS**: [YoujiaZhang/Ref-GS](https://github.com/YoujiaZhang/Ref-GS)（MIT ライセンス）。`diff-surfel-rasterization-real`、`diff-surfel-rasterization`、`diff-surfel-2dgs`、`simple-knn` と NVlabs の nvdiffrast を入れる。Python 3.7.16。メッシュ抽出は 2DGS の方法。実写用 `train-real.py` あり — [Ref-GS GitHub](https://github.com/YoujiaZhang/Ref-GS)
- **Ref-Gaussian**: [LogosIntelligenceLab/ref-gaussian](https://github.com/LogosIntelligenceLab/ref-gaussian)（プロジェクトページは fudan-zvg.github.io から logosintelligencelab.github.io/ref-gaussian に移転）。依存は未確認 — [Ref-Gaussian プロジェクトページ](https://logosintelligencelab.github.io/ref-gaussian/)
- **GS-ROR2**: [NK-CS-ZZL/GS-ROR](https://github.com/NK-CS-ZZL/GS-ROR)。依存は未確認（本文は TensoSDF を SDF 側に使う） — [GS-ROR2](https://arxiv.org/html/2406.18544)
- **TextureSplat**: 論文中で「github.com/maeyounes/TextureSplat で公開予定」と記載。公開状況は未確認 — [TextureSplat](https://arxiv.org/html/2506.13348)
- **Ref-DGS**: 論文ページにコード URL なし — [Ref-DGS](https://arxiv.org/html/2603.07664)
- **MaterialRefGS**: プロジェクトページ wen-yuan-zhang.github.io/MaterialRefGS、コード公開は未確認 — [MaterialRefGS（スニペット）](https://arxiv.org/abs/2510.11387)
- **NDE**: プロジェクトページ lwwu2.github.io/nde — [NDE](https://arxiv.org/abs/2405.14847)

### Inferences
- ユーザーが「真の形状を固定して照明だけ当てはめる」実験を 3DGS 系のコードで再現したい場合、遅延シェーディングと環境マップの部分は 3DGS-DR の `cubemapencoder`、または Ref-GS の Sph-Mip（nvdiffrast 依存）を切り出すのが近道だが、どちらもラスタライザ改造と一体なので、点群・メッシュ上の自前実装に移すほうが条件を揃えやすい（推論）。
- 3DGS-DR の「環境光の範囲を球で限定する」実写用オプションは、近接物体が共有照明に混ざる問題への工学的な対処で、遠方照明の仮定違反を前景マスクで回避していることを意味する。

### Gaps
- 各リポジトリの CUDA / PyTorch のバージョン要件は README に明記がないか、今回の抽出で取れなかった。
- Ref-Gaussian と GS-ROR のリポジトリ README は読んでおらず、依存（nvdiffrast、BVH ライブラリ等）は未確認。
- GaussianShader の残差色の学習コードが現時点（2026 年 11 月）で公開済みかは、README の未チェック項目以上の情報がなく未確認。
