# 近接光の映り込み・相互反射・平面鏡を扱う NeRF / 3DGS 手法

調査日: 2026-11-17。各手法の「反射の計算法」「バウンス数」「法線・粗さ」「コスト」「光沢データでの数値」「弱点」「コード・依存」を記録する。
判断の軸はユーザー実験。金の Erato 像（GGX α 0.01）では反射光線の 38% が像自身に当たる。照明は約 2 m 先の球光源 4 つと真上のソフトボックスで、像の大きさは 0.5 m。遮られた観測のうち上位 4 分の 1 が明るく、無限遠でない光源のために方向が数度ずれる。
数値は出典の表から引いた。評価条件（解像度、マスクの有無、再実行か自己申告か）はできるだけ添えた。

---

## Q1. NeRF 系（近接反射の源流）: NeRF-Casting、Neural Directional Encoding（NDE）、UniSDF の要点

### Takeaway
NeRF-Casting と NDE は、反射方向をシーン自体の場の中へ追跡する。前者は Zip-NeRF のハッシュグリッドに円錐を飛ばして特徴を積分し、後者は tri-plane 特徴を粗さ依存の円錐で追跡する。これで近くの物の映り込みや自己反射を出せる。陽に追跡するのは 1 回の反射だけで、2 次以降は特徴を色に変える MLP に暗黙に押し込まれる。学習は数時間〜十数時間かかり、描画も遅い（NDE は実時間版あり）。UniSDF は反射方向パラメータ化を混ぜる手法で、近接光を陽には追跡しない。

### Cited Findings
**NeRF-Casting（Verbin ら）**
- arXiv のメタデータは「Journal: TOG」。プレプリントは 2405.14871v1、2024-05-23 — [arXiv HTML](https://arxiv.org/html/2405.14871)
- 骨格は Zip-NeRF（マルチスケールのハッシュグリッドと小さな MLP）。カメラ光線のサンプルをアルファ合成して終端点 x̄ と法線 n̄ を求め、視線方向を n̄ で鏡映して反射方向 d′ を得る — [arXiv HTML](https://arxiv.org/html/2405.14871)
- 粗さ ρ も光線に沿って合成する。反射円錐の vMF 幅は κ̄⁻¹ = ṙ + ρ̄（ṙ は画素円錐の半径）。円錐の原点は入射光線のフットプリントを保つようにずらす — [arXiv HTML](https://arxiv.org/html/2405.14871)
- 反射光線は unscented directional sampling で K=5 本選ぶ。各光線は 2 段の proposal で 64 サンプルずつ取る。積分するのは色ではなく**特徴**で、5 本の平均 f̄ を小さな MLP で色に変える。最終色は視線依存項と反射項の凸結合（重み β はサンプルごとに予測） — [arXiv HTML](https://arxiv.org/html/2405.14871)
- 法線は密度場の勾配と予測法線ヘッドの 2 系統。予測法線は非対称な損失（λn=10⁻³、λñ=0.3）で学習し、反射方向の計算に使える — [arXiv HTML](https://arxiv.org/html/2405.14871)
- 記述されている反射段は 1 段のみで、多重バウンスへの言及はない。統計像の頭部やコードが映り込む例（Fig. 6）で、近くの物の映り込みを中心的な主張にしている — [arXiv HTML](https://arxiv.org/html/2405.14871)
- コスト: 学習はシーンあたり約 100 分（V100 8 台、JAX）。Zip-NeRF は約 50 分、UniSDF は約 200 分。描画速度は記載なし — [arXiv HTML](https://arxiv.org/html/2405.14871)
- 限界（著者記載）: 反射の起点はカメラ光線の期待終端点 1 点だけなので、半透明面に弱い。撮影者の映り込みをモデル化していない。最適化は Zip-NeRF の約 2 倍かかる — [arXiv HTML](https://arxiv.org/html/2405.14871)
- コード公開は本文に記載がない。プロジェクトページ nerf-casting.github.io は取得できなかった — [arXiv HTML](https://arxiv.org/html/2405.14871)

**Neural Directional Encoding（NDE、Wu ら）**
- CVPR 2024。コードは https://github.com/lwwu2/nde — [プロジェクトページ](https://lwwu2.github.io/nde/)
- 近距離成分: mip 化した tri-plane 特徴を MLP で密度 σn と特徴 hn に変え、反射光線に沿って円錐追跡する。遠距離成分（GGX で前フィルタした学習特徴キューブマップ、split-sum 風）と合成する — [arXiv HTML](https://arxiv.org/html/2405.14847)
- 円錐のフットプリントは ri = √3·ρ²·‖x − x′i‖（ρ は粗さ）で、GGX ローブの約 75% を覆う。mip レベルは λi = log₂(2ri)。ステップ幅は max(0.5·ri, 0.005) で、occupancy grid による空間スキップを使う。法線は VolSDF 風の位置符号化 MLP の SDF 勾配 — [arXiv HTML](https://arxiv.org/html/2405.14847)
- 「一般的な多重バウンス反射」を扱えると主張する。一方で編集の節では、物体の体積を消しても多重バウンス反射は消えないと認めている（2 次以降が特徴に焼き込まれていることを示唆） — [arXiv HTML](https://arxiv.org/html/2405.14847)
- コスト: 400k ステップで 10〜18 時間（RTX 3090、15 GB）。色ネットワーク単体は 800×800 で 3.03 FPS。WebGL の実時間版 NDE-RT（marching cubes と deferred shading）は 66 FPS で、平均 PSNR 35.48 — [arXiv HTML](https://arxiv.org/html/2405.14847)
- 限界: 法線の質に敏感で、ハッシュグリッドの SDF では破綻するため遅い MLP が必要。透明物体に弱い。NDE-RT は物体の縁で誤差が出る — [arXiv HTML](https://arxiv.org/html/2405.14847)
- 第三者評価: EnvGS の Table 5（Ref-Real と NeRF-Casting シーンの合算）では NDE は PSNR 19.399 / FPS 0.083 で、物体単位の手法はシーン単位のデータで崩れる — [EnvGS arXiv](https://arxiv.org/html/2412.15215)

**UniSDF（Wang ら）**
- NeurIPS 2024。カメラ視点の放射場と反射視点の放射場を、学習した連続の重み場で 3D 空間上でブレンドする。骨格はマルチ解像度グリッド — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/05b12f103c9e613efc4c85674cdc9066-Abstract.html)
- Shiny Blender（NeRF-Casting Table 5 の 6 シーン平均）: UniSDF は 36.82 / 0.976 / 0.044 — [NeRF-Casting arXiv](https://arxiv.org/html/2405.14871)

### Inferences
- NeRF-Casting は追跡先がシーン自身の場なので、像の反射光線が像に当たる自己反射は構造上表現できる。ただし当たった点で得るのは「その点の特徴」で、色への変換は MLP 任せになる。金→金→光源という 2 次の鏡面経路の明るさを正しく出すには、MLP がこの組み合わせを学習データから覚える必要がある。陽な再帰ではない。
- NDE の円錐幅は ρ² に比例するので、α 0.01 相当ではほぼ細い光線になり、近接の鏡像を鋭く出せる設計になっている。ただし法線の質に強く依存する。
- UniSDF の「反射視点の放射場」は方向の符号化であり、近接光の視差（光源が 2 m 先にあることによる数度のずれ）は陽には扱わない。比較の基準として位置づけるのが妥当。

### Gaps
- NeRF-Casting の会議名（SIGGRAPH Asia 2024 の TOG 掲載か）は arXiv の「Journal: TOG」以上には確かめられなかった（未確認）。コード公開の有無も未確認。
- NDE が自作の相互反射シーンで定量評価したかは本文から読み取れなかった（評価は Shiny Blender の 6 シーンと Materials）。
- NeRF-Casting のデータセットは、EnvGS の README によれば著者に連絡して入手する形式（下記 Q5）。

---

## Q2. 3DGS 系: EnvGS、Ref-Gaussian、IRGS、3DGRT/3DGUT、GS-ROR、その他の反射計算法と仮定

### Takeaway
3DGS 系の近接反射の扱いは大きく 5 つに分かれる。
- (a) 反射光線を「別のガウシアン集合」に飛ばす: EnvGS、OptiX による 2D ガウシアンの光線追跡
- (b) 抽出メッシュで可視性だけ判定し、遮られた方向は点ごとの学習済み間接項を使う: Ref-Gaussian
- (c) 同じガウシアンに光線追跡し、当たった点の学習済み出射輝度を返す: IRGS（逆レンダリング）、SpecTRe-GS
- (d) G-buffer を教師に、ガウシアン上で最大 3 バウンスの経路追跡をする: Poirier-Ginter ら、SIGGRAPH Asia 2025
- (e) 光線追跡を使わず、局所反射ガウシアンを splat する: Ref-DGS、SIGGRAPH 2026

陽な多重バウンスを持つのは (d) だけで、他は 1 回の反射と焼き込みで済ませている。

### Cited Findings
**EnvGS（Xie ら、Zhejiang Univ.）**
- CVPR 2025（README に 2025-02-27 採択と記載） — [GitHub](https://github.com/zju3dv/EnvGS)
- 2 組の 2D ガウシアンを使う。base ガウシアンをラスタライズして表面位置 x、法線 n、base 色、ブレンド重み β を得る。反射方向を計算して**環境ガウシアン**の中を光線追跡し、c = (1−β)c_base + β c_ref で合成する — [arXiv HTML](https://arxiv.org/html/2412.15215)
- 追跡器: OptiX の raygen / anyhit プログラムを使う独自 CUDA で、RT コア上で動く。3DGRT に倣い、2D ガウシアン 1 個を三角形 2 枚として BVH に入れる。k-buffer（k=16）で近い交差を保持し、チャンクごとに描画する。逆伝播は前から順に光線を再投射して行い、光線の原点と方向への勾配も出す（付録 D） — [arXiv HTML](https://arxiv.org/html/2412.15215)
- 粗さは扱わない（記載なし）。カメラ光線あたりの反射は 1 回で、多重バウンスの記述はない — [arXiv HTML](https://arxiv.org/html/2412.15215)
- 遠方照明を仮定しない。環境ガウシアンは SfM の bounding box（99.5% 分位、32 分割、各 5 個）に格子状に初期化する。平均 30 万個、70 MB で base の約 15%、上限は 63 万個。単眼法線（StableNormal）と VGG の知覚損失で教師する — [arXiv HTML](https://arxiv.org/html/2412.15215)
- 限界（著者記載）: 透明・屈折物体に弱い。扱うのは反射方向だけ — [arXiv HTML](https://arxiv.org/html/2412.15215)
- 第三者評価: Ref-DGS は、光線追跡系（Ref-Gaussian、EnvGS、MaterialRefGS）は計算量が大きく、学習が大幅に遅くなり、ガウシアンスプラッティングの効率という利点を損なうと批判している — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)

**Ref-Gaussian（Reflective Gaussian Splatting、Yao ら、Fudan）**
- ICLR 2025。MIT ライセンス — [GitHub](https://github.com/fudan-zvg/ref-gaussian)
- 2DGS をもとに、物理ベースの deferred rendering と split-sum を使う。BRDF 項は 2D LUT（ω·N と粗さで引く）。光の項は反復ごとに複数の粗さで前積分してキューブマップに置き、反射方向と粗さで三線形補間する。拡散は SH で置き換える — [arXiv HTML](https://arxiv.org/html/2412.19282)
- 相互反射: 3000 ステップごとに TSDF 融合でメッシュを抽出する。反射方向の光線をメッシュに当て（BVH で高速化）、可視性を 2 値で判定する。遮られた場合の間接光は、**各ガウシアンの視線依存色（SH）を反射方向で評価してアルファブレンドしたもの**（式 10）。当たった点の輝度を引くわけではない — [arXiv HTML](https://arxiv.org/html/2412.19282)
- 法線は 2D ガウシアンの接ベクトルの外積。深度から求めた法線との整合損失とエッジ考慮の平滑化をかける。粗さの初期値は 0.1。material-aware normal propagation で、metallic ≥ 0.02 かつ roughness ≤ 0.1 のガウシアンを拡大し、法線を周囲へ伝える — [arXiv HTML](https://arxiv.org/html/2412.19282)
- 限界（著者記載）: 鏡面ローブ全体の可視性を反射方向 1 本で近似するので、材質推定にノイズが乗りうる。非鏡面シーンは得意でない可能性がある（付録 A.1） — [arXiv HTML](https://arxiv.org/html/2412.19282)
- 光線追跡はサブモジュール `raytracing` で行う。README は謝辞で ashawkey の Raytracing プロジェクトを挙げ、OptiX や nvdiffrast には触れていない — [GitHub](https://github.com/fudan-zvg/ref-gaussian)

**IRGS（Gu ら、Fudan）**
- CVPR 2025 — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_IRGS_Inter-Reflective_Gaussian_Splatting_with_2D_Gaussian_Ray_Tracing_CVPR_2025_paper.pdf)、[GitHub](https://github.com/fudan-zvg/IRGS)
- 逆レンダリング（形状・アルベド・粗さ・照明）。入射光は、32×32 の環境キューブマップからの直接光（可視性で重み付け）と、光線追跡による間接光の和。BRDF は簡略化した Disney（誘電体を仮定） — [arXiv HTML](https://arxiv.org/html/2412.15867)
- 2D ガウシアン追跡器: 3DGRT と同様に各ガウシアンを正二十面体のプロキシで包み、OptiX の BVH で交差判定する。k-buffer（k=16）を使い、透過率 0.03 未満で打ち切る。当たった点では、Stage I（README によれば Ref-Gaussian）で学習した**視線依存色**とアルファを合成して返す。間接光は再帰しない 1 回の追跡 — [arXiv HTML](https://arxiv.org/html/2412.15867)、[GitHub](https://github.com/fudan-zvg/IRGS)
- レンダリング方程式は画素あたり N_r = 256 本の層化半球サンプルで評価する。1 反復の上限は 2^18 本で、約 1024 画素ぶん。リライティングでは split-sum と環境マップの重点サンプリングを使う — [arXiv HTML](https://arxiv.org/html/2412.15867)
- 限界（著者記載）: レンダリング方程式の評価が重く、高フレームレートで描画できない。今後の課題として入射輝度のベイクや PRT を挙げる。2DGS とのタイルソートの違いで品質がわずかに落ちる — [arXiv HTML](https://arxiv.org/html/2412.15867)

**SpecTRe-GS（Tang ら、CVPR 2025）**
- 3DGS の中に効率的な光線追跡器を組み込み、二次光線を問い合わせる。高鏡面の面と粗い面を分けて扱い、法線事前分布の誘導と形状の同時最適化で歪みのない映り込みを目指す — [CVPR poster ページ](https://cvpr.thecvf.com/virtual/2025/poster/33748)
- 粗い面の反射は splatting で描く。高鏡面の反射は「拡散シェーディング + 入射輝度 × 鏡面反射率」で、入射輝度と可視性はガウシアン点群への光線追跡で求める。直接成分は最適化する環境マップで表す。比較相手は 3DGS-DR、GShader、3iGS、GOF、3DGS。シーンは Helmet、MarbleTable、Vase、Pot、Toaster、Mirror、RealBowl、RealPot — [プロジェクトページ](https://spectre-gs.github.io)
- 高鏡面の面は鏡面ローブが集中しているので、少数の二次光線で足りるという設計の理屈（検索結果の要約による） — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_SpecTRe-GS_Modeling_Highly_Specular_Surfaces_with_Reflected_Nearby_Objects_by_CVPR_2025_paper.html)
- 本文 PDF と補足 PDF は取得できず（403、またはサイズ超過）、追跡器の実装、コスト、数値は確認できなかった（Gaps 参照）。プロジェクトページにコードへのリンクはない — [プロジェクトページ](https://spectre-gs.github.io)

**Editable Physically-based Reflections in Raytraced Gaussian Radiance Fields（Poirier-Ginter, Hu, Lalonde, Drettakis）**
- arXiv の掲載情報は SIGGRAPH Asia 2025 Conference Papers, Article 201（arXiv 投稿は 2026-06-29）。コードとデータ: https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/ — [arXiv abs](https://arxiv.org/abs/2606.30861)
- 写真から拡散・鏡面のバッファと形状・BRDF のバッファを取り出す。拡散版のシーンを最適化し、その上で経路追跡して物理ベースの鏡面反射を生成する。鏡の奥の「偽の形状」ではなく、反射体と映る物体を正しく再構成するのが狙い — [arXiv abs](https://arxiv.org/abs/2606.30861)
- 追跡器: OptiX のハードウェア光線追跡を使う。ガウシアンを OBB で囲み、2 階層の加速構造に AABB インスタンスとして載せる。BVH は 1 回だけ走査し、交差を per-pixel linked list に貯めてから近い 16 個を整列・積分する。3DGRT（約 16 個ずつ複数回走査、正二十面体）との違いはこの点。FPS は 2×縮小で 49.93 対 3DGRT 26.77、4×で 143.93 対 75.49、8×で 248.75 対 139.09 — [arXiv HTML](https://arxiv.org/html/2606.30861v1)
- バウンス数: 学習時の最大経路長は 3 バウンス（変更可）、1 spp。オフライン描画は 128 spp と OptiX デノイザ。法線・粗さ・F0 はガウシアンごとの属性で、逆レンダリングではなく視点ごとのバッファで教師する。合成シーンは正解バッファ、実シーンは SD2 を微調整した 1 ステップ推定器（Hypersim と InteriorVerse で学習） — [arXiv HTML](https://arxiv.org/html/2606.30861v1)
- 学習時間は、正解バッファで約 20〜24 分、推定バッファで約 24〜49 分。同じシーンで EnvGS は約 2:40〜3:34 かかり、最適化は約 5 倍速い（著者主張） — [arXiv HTML](https://arxiv.org/html/2606.30861v1)
- 限界: 層抽出ネットワークの質に強く依存する。透明物体は非対応。影は拡散層に焼き込まれ、編集しても更新されない。最終 PSNR はベースラインより低く、著者は「偽の形状」を持たないためとしている — [arXiv HTML](https://arxiv.org/html/2606.30861v1)

**Ref-DGS（Reflective Dual Gaussian Splatting）**
- SIGGRAPH Conference Papers '26 — [arXiv HTML](https://arxiv.org/html/2603.07664)
- 光線追跡を使わない。局所反射ガウシアン（学習する局所鏡面特徴を持ち、形状ガウシアンとは別）を splat して画素ごとの特徴マップを作り、自己反射と相互反射を表す。遠方の反射は球面特徴マップ（Sph-Mip、反射方向と粗さで引く）で表す。両特徴を MLP（隠れ層 3 層、幅 64）で鏡面輝度に変える。鏡像は「表面の奥にある局所ガウシアン」として扱う — [arXiv HTML](https://arxiv.org/html/2603.07664)
- 「環境照明や 1 回反射に頼るモデルは、自己反射や形状が生む相互反射といった近接の鏡面反射を捉えられない」と主張する — [arXiv PDF](https://arxiv.org/pdf/2603.07664)

**3DGRT / 3DGUT（NVIDIA）**
- 3DGRT は SIGGRAPH Asia 2024（Journal Track）、3DGUT は CVPR 2025（Oral）。3DGRT は体積ガウシアンを光線追跡し、反射・屈折・影に必要な二次光線を効率よく扱えると述べる。ハイブリッドの 3DGRUT は一次光線をラスタライズ、二次光線を光線追跡する。playground では挿入物体、反射、屈折、被写界深度などを描ける — [GitHub nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut)
- 3DGRT は専用の光線追跡ハードウェアを必要とし、RT コア付きの NVIDIA GPU を推奨する。CUDA 11.8 / 12.4 / 12.6 / 12.8（既定）/ 13.0（実験的）。Linux と Windows に対応し、Apache-2.0 — [GitHub nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut)
- EnvGS と IRGS の追跡器はいずれも 3DGRT の方式（プロキシ三角形 + BVH + k-buffer）を下敷きにしている — [EnvGS arXiv](https://arxiv.org/html/2412.15215)、[IRGS arXiv](https://arxiv.org/html/2412.15867)

**GS-ROR / GS-ROR²**
- GS-ROR（Zhu, Wang, Yang）: ガウシアンと SDF が深度・法線で相互に教師し合う、反射物体のリライティング手法。SDF を考慮した剪定を行う。RTX 4090 で 200 FPS 以上、NeRF 系の 25% 以下の学習時間と主張 — [arXiv HTML v2](https://arxiv.org/html/2406.18544v2)（検索結果の要約）
- GS-ROR² は Ref-DGS の比較表に登場する。学習 1.34 h、ShinySynthetic 32.16 / 0.963 / 0.069、GlossySynthetic 28.96 / 0.952 / 0.056 — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)

**その他（存在だけ確認）**
- RT-Splatting（反射と透過の同時モデル化、arXiv 2605.18263）、TR-Gaussians（平面の透過と反射を実時間で描く、arXiv 2511.13009）、ReflFlow（動的な鏡面のための形状誘導の光線追跡、arXiv 2512.10095）、RaySplats（光線追跡ベースのガウシアンスプラッティング、arXiv 2501.19196） — [検索結果: RT-Splatting](https://arxiv.org/pdf/2605.18263)、[TR-Gaussians](https://arxiv.org/pdf/2511.13009)、[ReflFlow](https://arxiv.org/pdf/2512.10095)、[RaySplats](https://arxiv.org/pdf/2501.19196)
- MaterialRefGS: Ref-DGS は光線追跡系の一つとして挙げる。学習 2.42 h、30.07 FPS — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)

### Inferences
- **Ref-Gaussian の構造は、ユーザーの「共有照明 + 点ごとの SH」とほぼ同型。** 遮られていなければ環境マップ、遮られていれば点ごとの学習済み間接項（反射方向で SH 評価）を使う。違いは可視性を抽出メッシュで陽に判定する点。ユーザーの像では反射光線の 38% が像自身に当たるので、この切り替えだけで効果が大きく出る可能性がある。Ref-Gaussian の Glossy Synthetic での相互反射のアブレーション（+0.19 dB、Q4）が小さいのは、データ側の相互反射が少ないためと著者自身が述べている。
- **IRGS 型（同じガウシアンに追跡し、当たった点の学習済み出射輝度を返す）は「放射輝度キャッシュ」方式。** キャッシュが当たった点から −ω 方向への出射輝度を正しく持っていれば、2 次以降も暗黙に含まれる。ただし金→金の場合、当たった点の出射輝度そのものが鋭い鏡面反射なので、キャッシュ（SH 等）が低次なら表しきれない。「照らされた金が金に映る」明るい遮られた観測を出すには、高周波のキャッシュか、陽な再帰が要る。
- **陽な多重バウンスを持つのは確認した範囲で Poirier-Ginter らだけ**（学習時に最大 3 バウンス）。この手法は G-buffer（法線・粗さ・F0、拡散と鏡面の分離）を教師に取る。ユーザーは GoLD_rayt で正解バッファを出せる合成データの立場にあり、著者が主に評価した条件（合成シーンで正解バッファ）と一致する。
- **光源の有限距離による方向ずれ**は、光源を 3D の原始要素として表す手法（EnvGS の環境ガウシアン、同一シーンへの光線追跡）なら視差として表現できる。ただし EnvGS の環境ガウシアンは SfM の bounding box（99.5% 分位）内に初期化されるので、約 2 m 先の光源が撮影範囲や点群の外にあれば、範囲を広げないと置かれない可能性がある（推測）。
- EnvGS の反射光線は環境ガウシアンだけを追跡し、base ガウシアンは追跡しない（記述による）。したがって像の自己反射は「環境ガウシアンが像の複製（の放射）を別に学ぶ」ことで表されることになる。1 回反射で、2 次の鏡面は環境ガウシアンの色に焼き込まれる。
- SpecTRe-GS の「高鏡面だけ追跡、粗い面は splat」という分け方は、α 0.01 の金像に設計上合っている。ただし数値とコードが確認できず、評価は難しい。

### Gaps
- SpecTRe-GS の追跡器の実装（OptiX か独自か）、バウンス数、学習時間、FPS、定量値は、CVF の PDF と補足が 403 で取れず未確認。表もプロジェクトページでは画像のみ。
- Ref-Gaussian の `raytracing` サブモジュールの実装（ashawkey の CUDA BVH で OptiX 不要という理解）は README の謝辞からの推測で、未確認。
- EnvGS の環境ガウシアンが視線依存色（SH）を持つかは確認していない。
- GS-ROR が間接光や自己遮蔽をどう扱うか、会議名（GS-ROR と GS-ROR²）は未確認。MaterialRefGS の一次情報（会議名、追跡の詳細）も未確認。
- RT-Splatting、TR-Gaussians、ReflFlow、RaySplats は本文未読。

---

## Q3. 平面鏡特化（MirrorGaussian、Mirror-3DGS 等）の考え方と適用範囲

### Takeaway
鏡の平面を推定し、「鏡像 = 鏡の向こうの仮想カメラ（またはガウシアンの鏡映コピー）から見たシーン」として描いて、鏡マスクで合成する。厳密な鏡像幾何を使うので、平面鏡 1 枚では鏡の奥の偽物体が消え、実時間性も保てる。曲面、粗い反射、自己反射、多重の鏡は対象外。学習に鏡マスクが要る。

### Cited Findings
**MirrorGaussian（Liu ら）**
- ECCV 2024。プロジェクトページにコードへのリンクはない — [プロジェクトページ](https://mirror-gaussian.github.io/)
- 実世界のガウシアンと、それを鏡平面で鏡映したコピーを両方描き、鏡マスクで合成する（dual rendering）。平均位置は鏡映関数で写し、回転は主軸を反転する。視線依存色は SH を再計算せず視線方向を鏡映して扱う — [arXiv HTML](https://arxiv.org/html/2405.11921)
- 平面推定: SAM で取った鏡の 2D 縁と SfM の対応点から縁上の 3D 点を作り、RANSAC で粗い平面を当てたあと、ガウシアンと同時に最適化する。各ガウシアンに鏡ラベル k を持たせ、アルファブレンドで任意視点の鏡マスクを作る — [arXiv HTML](https://arxiv.org/html/2405.11921)
- コスト（V100、平均）: 学習約 1 時間 8 分、1.6 MP で約 155 FPS。素の 3DGS は 47 分、271 FPS — [arXiv HTML](https://arxiv.org/html/2405.11921)
- 数値（SSIM / PSNR / LPIPS）。Coffee-House では MirrorGaussian 0.927 / 26.66 / 0.068、3DGS 0.921 / 25.83 / 0.086、Mirror-NeRF 0.847 / 23.92 / 0.201。Meeting-Room と Lounge の PSNR は MS-NeRF のほうが高い（31.09 / 31.20 に対し 30.63 / 31.10）。Mirror-NeRF は幅 480 px で評価しており直接比較はできない — [arXiv HTML](https://arxiv.org/html/2405.11921)
- 限界: 入力画像の鏡セグメンテーションが要る。dual rendering で描画が遅くなる。平面 1 枚のパラメータ化で、曲面や複数の鏡は評価していない — [arXiv HTML](https://arxiv.org/html/2405.11921)

**Mirror-3DGS（Meng ら、Peking Univ.）**
- arXiv 2404.01168（v2 は 2024-12）。会議名は記載なし — [arXiv HTML v2](https://arxiv.org/html/2404.01168v2)
- 各ガウシアンに学習する鏡属性 m∈[0,1] を持たせ、正解の鏡マスクで教師する。鏡は無限平面 π=(n,d) とし、改良 RANSAC と平面整合損失で求める。鏡映行列 T_m で鏡映カメラ P_m = T_m P_o を作り、マスク内は鏡映カメラの描画、マスク外は元の描画を使う。Stage 1（5k ステップ）で鏡の中身を消して鏡を考慮した深度で教師し、Stage 2（65k ステップ）で両者を融合する — [arXiv HTML v2](https://arxiv.org/html/2404.01168v2)
- RTX 3090 で合成 171.98 FPS、実 95.97 FPS（3DGS は 341.13 / 209.62、Mirror-NeRF は 0.75 / 0.07） — [arXiv HTML v2](https://arxiv.org/html/2404.01168v2)
- Mirror-NeRF データセット（合成 3、実 3）での PSNR / SSIM / LPIPS。合成: Mirror-3DGS 37.89 / 0.97 / 0.01、Mirror-NeRF 38.07 / 0.99 / 0.01、3DGS 37.00 / 0.97 / 0.04。実: Mirror-3DGS 25.93 / 0.81 / 0.06、Mirror-NeRF 25.04 / 0.86 / 0.06、3DGS 22.55 / 0.73 / 0.24。鏡領域だけの PSNR は Mirror-3DGS 32.40（合成）/ 24.53（実）、Mirror-NeRF 29.47 / 22.39 — [arXiv HTML v2](https://arxiv.org/html/2404.01168v2)

**関連**
- TR-Gaussians は平面の透過と反射を 3DGS で実時間に描く（arXiv 2511.13009、本文未読） — [arXiv PDF](https://arxiv.org/pdf/2511.13009)
- MirrorGaussian の比較相手は Mirror-NeRF、MS-NeRF、Ref-NeRF — [arXiv HTML](https://arxiv.org/html/2405.11921)

### Inferences
- 平面鏡特化は、反射面が平面で形がわかっていれば、鏡映という「正しい二次光線の閉形式」を使うので最も安く正確。ただし曲面の金の像には一般化しない。ユーザー実験への直接の適用範囲はない。
- 一方で「鏡映した仮想視点から同じシーンを描く」は、相互反射を同じ表現で説明させる（偽の物体を置かない）という発想の原型。EnvGS や Poirier-Ginter らの「偽の形状を作らない」方向とつながる。

### Gaps
- Mirror-3DGS の採択先は未確認。Mirror-NeRF（ACM MM 2023 と記憶しているが）、MS-NeRF、NeRFReN の会議名は今回は確かめていない（未確認）。
- 複数の鏡や鏡同士の相互反射（合わせ鏡）を扱う 3DGS 手法は見つからなかった。

---

## Q4. 光沢データでの数値と比較相手（相互反射の強いシーンを含む）

### Takeaway
物体単位の合成データ（Shiny Blender / Glossy Synthetic）では、光線追跡系の EnvGS は 3DGS-DR や Ref-Gaussian を上回らない（第三者評価では下回る）。EnvGS が強いのは、近くの物が映り込むシーン単位の実データ（Ref-Real、NeRF-Casting のシーン）で、NeRF-Casting に近い値を約 100 倍速く出す。相互反射そのものを切り出した定量評価は少ない。Ref-Gaussian のアブレーションでは相互反射の寄与は +0.19 dB と小さい（データ側に相互反射が少ないため）。

### Cited Findings
**シーン単位の実データ（EnvGS 論文 Table 1、1/4 解像度、PSNR / SSIM / LPIPS）**
- Ref-Real: EnvGS 24.617 / 0.671 / 0.241、NeRF-Casting 24.670 / 0.659 / 0.246、ZipNeRF 23.677 / 0.635 / 0.247、UniSDF 23.700 / 0.635 / 0.266、3DGS-DR 23.522 / 0.640 / 0.274、GaussianShader 22.875 / 0.622 / 0.314、2DGS 23.804 / 0.654 / 0.281 — [EnvGS arXiv](https://arxiv.org/html/2412.15215)
- NeRF-Casting のシーン: EnvGS 30.444 / 0.886 / 0.148、NeRF-Casting 31.023 / 0.889 / 0.128、ZipNeRF 31.740 / 0.904 / 0.105、UniSDF 30.838 / 0.889 / 0.130、3DGS 28.860 / 0.877 / 0.159、3DGS-DR 28.487 / 0.858 / 0.197 — [EnvGS arXiv](https://arxiv.org/html/2412.15215)
- 前景と近距離領域だけの PSNR（Table 2）: EnvGS は前景 33.295、近距離 46.392。次点の 3DGS は 31.681 / 44.161 — [EnvGS arXiv](https://arxiv.org/html/2412.15215)
- 自前撮影: Audi は EnvGS 28.629 / 0.876 / 0.121、3DGS-DR 26.093 / 0.820 / 0.192。Dog は EnvGS 25.599 / 0.885 / 0.151、3DGS-DR 23.943 / 0.850 / 0.210 — [EnvGS arXiv](https://arxiv.org/html/2412.15215)

**NeRF-Casting 論文（マスク付き、Ref-NeRF 実シーン + 新規 4 シーン、Table 1）**
- NeRF-Casting 33.91 / 0.965 / 0.036、UniSDF 33.38 / 0.961 / 0.042、Zip-NeRF 33.09 / 0.960 / 0.041、Ref-NeRF* 32.59 / 0.957 / 0.042 — [NeRF-Casting arXiv](https://arxiv.org/html/2405.14871)
- Shiny Blender の 6 シーン平均（Table 5）: NeRF-Casting 37.39 / 0.977 / 0.041、UniSDF 36.82 / 0.976 / 0.044、Ref-NeRF 35.96 / 0.967 / 0.059、ENVIDR 35.85 / 0.983 / 0.036（SSIM と LPIPS は ENVIDR が最良）、Zip-NeRF 30.33 / 0.949 / 0.086 — [NeRF-Casting arXiv](https://arxiv.org/html/2405.14871)

**NDE 論文（Shiny Blender 6 シーン + Materials、7 シーン平均）**
- NDE 37.19 / 0.982 / 0.022、Ref-NeRF 35.88 / 0.969 / 0.053、ENVIDR 34.95 / 0.982 / 0.035、NeRO 30.61 / 0.936 / 0.109。相互反射を見せる teaser シーンの PSNR は NDE 34.08、ENVIDR 31.29、NeRO 28.75、Ref-NeRF 28.18 — [NDE arXiv](https://arxiv.org/html/2405.14847)

**Ref-Gaussian 論文（自己申告）**
- Shiny Blender 平均は 35.04 / 0.973 / 0.056（Table 6）、Glossy Synthetic 平均は 30.33 / 0.958 / 0.049（Table 5）。シーンごとの PSNR は、Shiny が ball 37.01、car 31.04、coffee 34.63、helmet 32.32、teapot 47.16、toaster 28.05。Glossy が angel 30.38、bell 32.86、cat 33.01、horse 27.05、luyu 30.04、potion 33.07、tbell 29.84、teapot 26.68。Ref-Real は garden 22.97、sedan 26.60、toycar 24.27 — [Ref-Gaussian arXiv](https://arxiv.org/html/2412.19282)
- 相互反射のアブレーション（Table 4、Glossy Synthetic と推定）: 完全版 30.33 / 0.958 / 0.049、相互反射なし 30.14 / 0.957 / 0.050。著者は、相互反射は Glossy の合成データでは少数派の効果なので表の上では差が見えにくく、Tbell と Bell の頭部で最も目立つ（Fig. 10）と述べる — [Ref-Gaussian arXiv](https://arxiv.org/html/2412.19282)

**EnvGS の物体単位データ（自己申告、Table 8 の Shiny Blender シーン別 PSNR）**
- EnvGS は ball 32.567、car 30.598、coffee 34.312、helmet 31.470、teapot 46.582、toaster 27.427。同表の 3DGS-DR は 33.533、30.236、34.580、31.518、47.038、26.823 — [EnvGS arXiv](https://arxiv.org/html/2412.15215)

**第三者評価（Ref-DGS、SIGGRAPH 2026、Table 3、PSNR / SSIM / LPIPS）**
- ShinySynthetic: Ref-DGS 35.21 / 0.975 / 0.053、Ref-GS 34.80、Ref-Gaussian 34.92 / 0.973 / 0.056、MaterialRefGS 34.16、3DGS-DR 33.94 / 0.971 / 0.059、EnvGS 33.26 / 0.968 / 0.063、Ref-NeRF 33.32、ENVIDR 32.88、GS-ROR² 32.16、GaussianShader 30.42 — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)
- GlossySynthetic: MaterialRefGS 30.74 / 0.961 / 0.047、Ref-DGS 30.63 / 0.958 / 0.052、Ref-GS 30.37 / 0.960 / 0.048、Ref-Gaussian 29.61 / 0.954 / 0.054、3DGS-DR 29.36、ENVIDR 29.06、GS-ROR² 28.96、EnvGS 28.17 / 0.938 / 0.067、GaussianShader 27.11、Ref-NeRF 25.65 — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)
- RefReal: Ref-DGS 24.89 / 0.697 / 0.224、EnvGS 24.55 / 0.671 / 0.243、PGSR 24.43 / 0.696 / 0.220、Ref-Gaussian 23.95 / 0.661 / 0.280、3DGS-DR 23.80 / 0.659 / 0.236。GS-ROR² と MaterialRefGS は、コードが不完全なため RefReal を評価していない — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)
- 法線 MAE（ShinySynthetic）: Ref-DGS 1.43、Ref-Gaussian 2.17、MaterialRefGS 2.20、Ref-GS 2.21、3DGS-DR 2.43、GS-ROR² 2.64、ENVIDR 4.61、UniSDF 8.71、GaussianShader 10.93 — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)

**相互反射・近接物体の映り込みに特化した評価**
- Poirier-Ginter らは合成の Shiny Kitchen / Livingroom / Office で、拡散・鏡面・最終の PSNR を出している。Kitchen では本手法（正解バッファ）が 33.20 / 24.30 / 26.96、EnvGS（推定法線）が 14.33 / 14.48 / 32.82。最終画像の PSNR は EnvGS のほうが高く、本手法は層の分解（拡散・鏡面）が正しい — [arXiv HTML](https://arxiv.org/html/2606.30861v1)
- SpecTRe-GS は Helmet、MarbleTable、Vase、Pot、Toaster、Mirror、RealBowl、RealPot で 3DGS-DR、GShader、3iGS、GOF、3DGS と比べている（表は画像で数値は未取得） — [プロジェクトページ](https://spectre-gs.github.io)

### Inferences
- EnvGS の Shiny Blender 6 シーンの単純平均は、自己申告値から計算すると約 33.83 dB。同表の 3DGS-DR は約 33.96 dB で、Ref-DGS の再実行（33.26 と 33.94）と同じく、物体単位では EnvGS が 3DGS-DR を上回らない。光線追跡の利点は、近くの物体が映り込むシーン単位のデータで出る（Table 2 の近距離領域 PSNR）。
- Ref-Gaussian の Glossy Synthetic のシーン別 PSNR の単純平均は約 30.37 で、表の 30.33 とわずかに食い違う（丸めか別集計か不明）。第三者の Ref-DGS の値は 29.61 でさらに低い。
- ユーザーの像のように「反射光線の約 4 割が自身に当たる」データは、既存ベンチマーク（Shiny Blender、Glossy Synthetic）より相互反射がずっと強い。Ref-Gaussian の +0.19 dB をそのまま当てはめるべきではない。Glossy Synthetic の Tbell や Bell の凹部が、比較的近い相互反射の例になる。
- 各表の評価条件はそろっていない（EnvGS の Table 1 は 1/4 解像度、NeRF-Casting の Table 1 はマスク付き、Ref-DGS は再実行）。手法間の 0.3 dB 程度の差は条件差の範囲内とみるのが安全。

### Gaps
- SpecTRe-GS、IRGS（Shiny / Glossy は未評価）、MaterialRefGS の光沢データでの一次の数値は取得できなかった。IRGS の評価は Synthetic4Relight と TensoIR（逆レンダリング指標）だけ。
- 凹んだ物体や複数物体の相互反射に特化した共通ベンチマークは見つからなかった。

---

## Q5. 計算コスト・ハードウェア要件・コード公開と依存

### Takeaway
光線追跡系の 3DGS（EnvGS、IRGS、Poirier-Ginter ら、3DGRT）は、ほぼすべて OptiX と BVH の上に作られ、RT コア付きの NVIDIA GPU を前提にしている。EnvGS は RTX 4090 で学習 2.5 h、約 26 FPS で、NeRF-Casting の約 100 倍速いが、3DGS-DR（1.0 h、134 FPS）より 5 倍遅い。メッシュ BVH で可視性だけ取る Ref-Gaussian は 0.58 h、122 FPS と軽い。光線追跡なしの Ref-DGS は 12.6 分、76 FPS。NeRF 系は数時間〜十数時間かかる。

### Cited Findings
**学習時間・描画速度**
- EnvGS 論文 Table 1（1/4 解像度、RTX 4090）の FPS と学習時間: EnvGS 26.221 / 2.5 h、3DGS-DR 133.593 / 1.0 h、GaussianShader 27.945 / 1.6 h、2DGS 159.188 / 0.7 h、3DGS 182.307 / 0.6 h、Ref-NeRF* <0.1 / 47 h、UniSDF・ZipNeRF・NeRF-Casting <0.1 / >47 h。本文 4.2 節は「2DGS 200 万個を 1292×839 で 30 FPS」とする — [EnvGS arXiv](https://arxiv.org/html/2412.15215)
- NeRF-Casting の自己申告は V100 8 台で約 100 分（Zip-NeRF 約 50 分、UniSDF 約 200 分） — [NeRF-Casting arXiv](https://arxiv.org/html/2405.14871)。EnvGS の表の「>47 h」とは条件（GPU 台数）が違う
- Ref-Gaussian（Shiny Blender、A6000）: 0.58 h、122 FPS。同表で 3DGS-DR 0.35 h / 251 FPS、GaussianShader 0.48 h / 28 FPS、ENVIDR 5.84 h / 1 FPS — [Ref-Gaussian arXiv](https://arxiv.org/html/2412.19282)
- IRGS: 約 40 分（Stage I 15 分 + Stage II 25 分、RTX 3090）。N_r=256 でのフル画像描画は約 11 秒と記載。高フレームレートの描画はできない — [IRGS arXiv](https://arxiv.org/html/2412.15867)
- NDE: 10〜18 時間（RTX 3090）。NDE-RT は 66 FPS（WebGL） — [NDE arXiv](https://arxiv.org/html/2405.14847)
- Poirier-Ginter ら: 学習約 20〜49 分。描画 FPS は 2×縮小で 49.93（3DGRT 26.77） — [arXiv HTML](https://arxiv.org/html/2606.30861v1)
- Ref-DGS（RTX 4090、ShinySynthetic）: 学習 12.6 分（Ref-Gaussian 35.9 分、3DGS-DR 26.5 分、GS-ROR² 1.34 h、MaterialRefGS 2.42 h）。FPS 76.34（EnvGS 26.22、MaterialRefGS 30.07） — [Ref-DGS arXiv](https://arxiv.org/html/2603.07664)
- 平面鏡: MirrorGaussian は V100 で約 1 h 8 m、約 155 FPS — [arXiv](https://arxiv.org/html/2405.11921)。Mirror-3DGS は RTX 3090 で 172 FPS（合成）/ 96 FPS（実） — [arXiv](https://arxiv.org/html/2404.01168v2)

**依存とコード**
- EnvGS: https://github.com/zju3dv/EnvGS（CVPR 2025）。Python 3.11、PyTorch 2.3.1、CUDA 11.8 推奨、EasyVolcap フレームワーク。サブモジュールは 2D Gaussian Tracer（diff-surfel-tracing）、改造した 2D ガウシアンのラスタライザ、StableNormal（独自データ用）。OptiX のバージョンは README に明記がない（リポジトリのトピックに OptiX）。NeRF-Casting のデータは著者に連絡して入手し、前処理済みの Ref-Real と Shiny Blender、自前データセットは Google Drive で配布 — [GitHub](https://github.com/zju3dv/EnvGS)
- Ref-Gaussian: https://github.com/fudan-zvg/ref-gaussian（ICLR 2025、MIT）。Python 3.8、torch 2.0.0（CUDA 11.7 の例）。サブモジュールは cubemapencoder、diff-surfel-rasterization、simple-knn、raytracing。データは Shiny Blender（合成・実）、Glossy Synthetic（nero2blender.py で変換）、NeRF Synthetic — [GitHub](https://github.com/fudan-zvg/ref-gaussian)
- IRGS: https://github.com/fudan-zvg/IRGS（CVPR 2025）。CMake でビルドする surfel_tracer（2D ガウシアン追跡器）、Stage 1 用の raytracing、diff-surfel-rasterization。データは Synthetic4Relight と TensoIR。OptiX / CUDA のバージョンは README に明記なし — [GitHub](https://github.com/fudan-zvg/IRGS)
- 3DGRT / 3DGUT / 3DGRUT: https://github.com/nv-tlabs/3dgrut（Apache-2.0）。3DGRT は RT コアを要する。CUDA 11.8〜12.8、13.0 は実験的。Linux と Windows（VS Build Tools 2019 以上） — [GitHub](https://github.com/nv-tlabs/3dgrut)
- Poirier-Ginter ら: コードとデータは https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/ 。OptiX のハードウェア光線追跡、OptiX デノイザ — [arXiv](https://arxiv.org/html/2606.30861v1)
- NDE: https://github.com/lwwu2/nde — [プロジェクトページ](https://lwwu2.github.io/nde/)
- コードへのリンクが見つからなかったもの: NeRF-Casting（本文に記載なし）、SpecTRe-GS、MirrorGaussian、Mirror-3DGS、Ref-DGS — [NeRF-Casting](https://arxiv.org/html/2405.14871)、[SpecTRe-GS](https://spectre-gs.github.io)、[MirrorGaussian](https://mirror-gaussian.github.io/)、[Mirror-3DGS](https://arxiv.org/html/2404.01168v2)、[Ref-DGS](https://arxiv.org/html/2603.07664)

### Inferences
- ユーザー環境で光線追跡系（EnvGS、IRGS、3DGRT 系、Poirier-Ginter ら）を動かすには、RT コア付きの NVIDIA GPU と、OptiX SDK を含むビルド環境が実質的に必須になる。Windows で動かす場合、確認した範囲で公式に対応を明記しているのは 3dgrut だけ。
- 「可視性はメッシュ BVH、間接項は点ごとの学習量」の Ref-Gaussian 型は OptiX が不要とみられ（推測）、軽量。ユーザーの「共有照明 + 点ごとの SH」実験に最も移植しやすい。
- 陽な 2〜3 バウンスで金→金の明るい遮蔽観測を説明したい場合、Poirier-Ginter らの追跡器（3DGRT より約 2 倍速、学習 20〜49 分）が現実的な出発点になる。ユーザーは正解の G-buffer を出せる合成データなので、彼らの主評価と同じ条件に立てる。

### Gaps
- 各リポジトリが要求する OptiX SDK のバージョン（7.x か 8.x か）は README から確認できなかった。
- SpecTRe-GS と NeRF-Casting の描画速度、SpecTRe-GS の学習時間は未確認。
- Ref-Gaussian の追跡が CPU か GPU か、OptiX を使うかは README からは不明。
