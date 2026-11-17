# 光沢対応 3DGS / NeRF 手法の公開コード依存と旧世代 GPU（Quadro M4000 / sm_52 / 8 GB / Windows 10）での実行見込み、および PyTorch のみで簡略版を自作する難しさ

調査日: 2026-11-17。各リポジトリの README・.gitmodules・setup.py 等は本日取得したもの。ドライバ／CUDA の世代サポートに関する報道は 2025 年 12 月頃のもので、それ以降の変化は追えていない。
判定の表記: 「確認済み」＝sm_52 / Maxwell での動作・失敗報告あり、「依存から推定」＝依存ライブラリの要件から推定、「不明」＝判断材料が不足。

---

## 1. 各手法の公開コード、ライセンス、依存、要求する CUDA / PyTorch / GPU 世代

### Takeaway
調べた候補のうち公開コードを確認できたのは 3DGRT/3DGUT、EnvGS、Ref-Gaussian、IRGS、Spec-Gaussian、GaussianShader、3DGS-DR、Ref-GS、GS-ROR、NeRO、Ref-NeRF（multinerf）。MirrorGaussian と NeRF-Casting は公式コードを見つけられなかった。Gaussian 系はすべて独自の CUDA 拡張（3DGS / 2DGS ラスタライザの改造版、simple-knn など）を pip でソースビルドする前提。このうち OptiX を要するのは EnvGS、IRGS（第 2 段）、3DGRT の 3 つ。tiny-cuda-nn を含むのは 3dgrut だけ。Ref-NeRF は JAX 実装。

### Cited Findings

**手法ごとの依存（README・設定ファイルから取得）**

| 手法 | コード / ライセンス | Python / PyTorch / CUDA | 独自 CUDA 拡張・主な依存 | OptiX | 出典 |
|---|---|---|---|---|---|
| Spec-Gaussian | github.com/ingra14m/Specular-Gaussians / MIT | Py 3.7、torch 1.13.1+cu116、torch-scatter | `depth-diff-gaussian-rasterization`（ingra14m/diff-gaussian-rasterization-extentions の filter-norm ブランチ）、simple-knn。tcnn・nvdiffrast は記載なし。`--asg_degree` 指定あり（ASG による外観）。高速なアンカー版（Scaffold-GS 由来）あり | 不要 | [README](https://github.com/ingra14m/Specular-Gaussians), [.gitmodules](https://raw.githubusercontent.com/ingra14m/Specular-Gaussians/main/.gitmodules) |
| GaussianShader | github.com/Asparagus15/GaussianShader / LICENSE.md あり（種類は未確認） | Py 3.7.13、torch 1.10.1+cu111 | diff-gaussian-rasterization・simple-knn（いずれも pip パッケージ 0.0.0 として列挙）、**nvdiffrast 0.3.1**。environment.yml には libgcc-ng・ld_impl_linux-64 など Linux のパッケージが並ぶ。README は「まだ整備中」 | 不要 | [README](https://github.com/Asparagus15/GaussianShader), [environment.yml](https://raw.githubusercontent.com/Asparagus15/GaussianShader/main/environment.yml) |
| 3DGS-DR | github.com/gapszju/3DGS-DR / ライセンス表記なし | 「公式 3DGS の環境をまず動かせ」とだけ記載 | `cubemapencoder`、`diff-gaussian-rasterization_c3`、`diff-gaussian-rasterization_c7`、simple-knn、ビューア用 dearpygui。抄録には「遅延シェーディングが生む画素単位の反射勾配」で近傍ガウシアンの法線最適化をつなぐとある | 不要 | [README](https://github.com/gapszju/3DGS-DR) |
| Ref-GS | github.com/YoujiaZhang/Ref-GS / MIT | Py 3.7.16、PyTorch/CUDA の版は記載なし | `diff-surfel-rasterization-real`、`diff-surfel-rasterization`、`diff-surfel-2dgs`、simple-knn、**nvdiffrast**（clone して pip install）。CVPR 2025 | 不要 | [README](https://github.com/YoujiaZhang/Ref-GS), [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Ref-GS_Directional_Factorization_for_2D_Gaussian_Splatting_CVPR_2025_paper.html) |
| GS-ROR | github.com/NK-CS-ZZL/GS-ROR / **CC BY-NC 4.0** | Py 3.10、torch 2.2.1（cu118）。「CUDA 11.8 ランタイムで検証」 | **nvdiffrast**、`r3dg-rasterization`、fused-ssim、simple-knn。謝辞に GaussianShader・NeRO・TensoSDF・Ref-NeuS・R3DG | 不要 | [README](https://github.com/NK-CS-ZZL/GS-ROR) |
| Ref-Gaussian | github.com/fudan-zvg/ref-gaussian / MIT | Py 3.8、torch 2.0.0（コメントに「例: cuda 11.7」） | `cubemapencoder`、`diff-surfel-rasterization`、simple-knn、**`raytracing`**（ashawkey/raytracing＝BVH 付き CUDA メッシュレイトレーサ）。nvdiffrast・OptiX・tcnn は記載なし | 不要 | [README](https://github.com/fudan-zvg/ref-gaussian), [ashawkey/raytracing](https://github.com/ashawkey/raytracing) |
| IRGS | github.com/LogosIntelligenceLab/IRGS（fudan-zvg/IRGS から転送） / LICENSE.md（種類は未確認） | environment.yml（中身は未確認） | 改造版 `diff-surfel-rasterization`（「2DGS のものと少し違う」と明記）、simple-knn、`raytracing`（第 1 段の Ref-Gaussian 用）、**`surfel_tracer`**（CMake でビルドしてから pip） | **要**: surfel_tracer の readme は「OptiX ベースの微分可能 2D ガウシアンレイトレーサ」と説明。CMake で OptiX 用の PTX を生成する。3DGRT を元にしたと謝辞 | [README](https://github.com/fudan-zvg/IRGS), [surfel_tracer](https://github.com/LogosIntelligenceLab/IRGS/tree/main/submodules/surfel_tracer) |
| EnvGS | github.com/zju3dv/EnvGS / LICENSE（種類は未確認） | Py 3.11、torch 2.3.1（cu118）。「CUDA 11.8 推奨」 | EasyVolcap フレームワーク、`diff-surfel-tracing`、`diff-surfel-rasterization-wet`（ch05 / ch07 含め 3 種）、任意で StableNormal | **要**: diff-surfel-tracing は 2DGS と OptiX による微分可能レイトレーサで、OptiX SDK ヘッダを `third_party/optix` に同梱（`OPTIX_HOME` で差し替え可）。同梱の版は未記載。TODO に「OptiX 8.0.0 / 8.1.0 との互換性は未テスト」。例示パスは `/usr/local/cuda`・`LD_LIBRARY_PATH` | [README](https://github.com/zju3dv/EnvGS), [.gitmodules](https://raw.githubusercontent.com/zju3dv/EnvGS/main/.gitmodules), [diff-surfel-tracing](https://github.com/xbillowy/diff-surfel-tracing) |
| 3DGRT / 3DGUT | github.com/nv-tlabs/3dgrut / Apache-2.0 | Linux と Windows に対応（uv のインストールスクリプト）。CUDA 11.8 / 12.4 / 12.6 / 12.8（既定）/ 13.0（実験的）。Windows は VS Build Tools 2019 以降が必要 | サブモジュールは **tiny-cuda-nn** と **optix-dev（v7.5.0 ブランチ）** | 3DGRT は要（「RT コアのある GPU を推奨」「専用のレイトレーシングハードウェアを要する」）。3DGUT はラスタライズ法として説明されている | [README](https://github.com/nv-tlabs/3dgrut), [.gitmodules](https://raw.githubusercontent.com/nv-tlabs/3dgrut/main/.gitmodules) |
| NeRO | github.com/liuyuan-pal/NeRO / MIT | requirements.txt（中身は未確認） | **nvdiffrast**、ashawkey/raytracing。OneDrive のリンクは切れていて Google Drive に移行 | 不要 | [README](https://github.com/liuyuan-pal/NeRO) |
| Ref-NeRF | github.com/google-research/multinerf / Apache-2.0 | **JAX** 実装、Python 3.9。README は「GPU/TPU 用に JAX の更新が要るだろう」と書くだけ | Ref-NeRF と RawNeRF を Mip-NeRF 360 の実装に統合したもの。メモリ不足時は batch_size を下げるよう指示 | 不要 | [README](https://github.com/google-research/multinerf) |
| MirrorGaussian | プロジェクトページは arXiv へのリンクだけで、コードへのリンクはない | — | 手法: ガウシアンを鏡面で鏡映した複製をラスタライズし、マスクで合成。鏡面とガウシアンを同時に最適化 | — | [project page](https://mirror-gaussian.github.io) |
| NeRF-Casting | 公式コードは見つからなかった。Radiancefields の記事は「GitHub のコードページはあるが、早期公開は疑わしい」と書いている（記事の日付は不明） | — | — | — | [radiancefields](https://radiancefields.com/reflecting-on-nerf-casting) |

**共通基盤の要件**
- 公式 3DGS（graphdeco-inria）の README は学習側の要件を「Compute Capability 7.0+ の CUDA GPU」「24 GB VRAM（論文品質まで学習する場合）」と書く。ソフト側は「CUDA SDK 11（11.8 を使用、11.6 は既知の問題あり）」「Windows は VS2019 を使用」。FAQ には「7k 反復だけならずっと少ない」「計算上は 8GB 程度でも可能なはず」とある — [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- 3DGS-DR と Ref-Gaussian の `cubemapencoder` について。setup.py は arch 指定（-gencode / TORCH_CUDA_ARCH_LIST）を持たず、`-D__CUDA_NO_HALF_OPERATORS__` などを定義する。カーネル内のアトミックは float の atomicAdd だけで、__half2 の atomicAdd はコメントアウトされている（「遅い」との注記つき）。処理は 6 面キューブマップ [6,C,L,L] からの単一レベル bilinear / nearest 参照で、ミップマップも trilinear 補間もない。「seamless」モードは面の境界で隣の面から値を取る — [setup.py](https://raw.githubusercontent.com/gapszju/3DGS-DR/main/submodules/cubemapencoder/setup.py), [cubemapencoder.cu](https://raw.githubusercontent.com/gapszju/3DGS-DR/main/submodules/cubemapencoder/src/cubemapencoder.cu)
- ashawkey/raytracing は BVH で高速化した CUDA メッシュレイトレーサ（Python バインディング付き）で、OptiX や RT コアへの言及はない。MIT で、2022-12-03 にアーカイブ（読み取り専用）された — [ashawkey/raytracing](https://github.com/ashawkey/raytracing)
- nvdiffrast（現行ドキュメント）の要件は Linux または Windows、Python 3.9+、**PyTorch 2.7 以上**、「ハイエンドの NVIDIA GPU」。最低 compute capability の記載はなく、ビルド例は arch 8.0〜9.0 を対象にしている。`texture()` は `boundary_mode='cube'`（[minibatch,6,H,W,C]、継ぎ目なしの trilinear）とミップマップに対応する。ゼロベクトルを渡すとゼロを返し、勾配は流れない — [nvdiffrast docs](https://nvlabs.github.io/nvdiffrast/)
- tiny-cuda-nn の FullyFusedMLP（既定）は大量の共有メモリを使い、「RTX 3090 / 2080 Ti 以上でないと動かないだろう」とされる。下位の GPU ではニューロン数を減らすか CutlassMLP を使う。古いアーキテクチャ（Pascal / GTX 10 系など）では PyTorch 拡張の FP16 が既定で無効。Windows は CUDA 11.5 以上と VS2019/2022 が必要。最低 compute capability の明記はない — [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

### Inferences
- Gaussian 系の手法はすべて、3DGS / 2DGS のラスタライザを改造した CUDA 拡張を pip でソースビルドする前提である。「Windows で CUDA 拡張をビルドしない」という制約を守る限り、公式コードは一つも動かせない。Windows で動かす道筋（uv スクリプト、VS Build Tools）を README に書いているのは 3dgrut だけだった。
- GaussianShader の environment.yml は Linux のパッケージを固定しているため、Windows では作り直しが要る。EnvGS の diff-surfel-tracing も Linux のパス（`/usr/local/cuda`、`LD_LIBRARY_PATH`）を前提に書かれている。
- Spec-Gaussian は ASG で外観を表すので、色をガウシアンごとに計算してから標準の α ブレンディングに渡す構造だと推定される。ラスタライザは深度出力付きの 3DGS 改造版で、反射用の新しい出力チャンネルは要らないはず（README の `--asg_degree` から推定。論文本文は未確認）。

### Gaps
- GaussianShader、IRGS、EnvGS、3DGS-DR のライセンスの種類（LICENSE ファイルの中身を取得できなかった）。
- IRGS の surfel_tracer と EnvGS の diff-surfel-tracing が同梱している OptiX ヘッダの版。EnvGS の TODO に「8.0 / 8.1 は未テスト」とあるので 7.x 系と推定するが、確認していない。
- Ref-GS、NeRO、IRGS の requirements.txt / environment.yml の中身（PyTorch と CUDA の版）。
- 3dgrut の Python 依存（requirements.txt の URL が 404 だった）。slangtorch / Slang を使うかどうかは確認できなかった。

---

## 2. sm_52 で動かない決定的な要因と、Maxwell での動作・失敗報告

### Takeaway
今回の候補群について、sm_52 であること自体が決定的な障害になるケースは意外に少ない。決定的なのは次の 4 つである。(1) OptiX 9.1 以降は Turing 以降の GPU 限定で、R590 ドライバを要する。R590 で Maxwell は外れている。(2) CUDA 13 と PyTorch の cu128 / cu129 ホイールは Maxwell を落としている。(3) Windows で CUDA 拡張をソースビルドする必要がある。(4) RT コアがないため、OptiX 系はソフトウェアで BVH を走査することになり遅い（推定）。今回の候補が要求する半精度の演算や Tensor Core 前提の処理は、tiny-cuda-nn の FullyFusedMLP を除くと見当たらなかった。Maxwell（sm_52）での動作報告・失敗報告は、どの手法の Issue でも見つけられなかった。

### Cited Findings
- PyTorch 2.8.0 のリリースノートは「CUDA 12.8 / 12.9 ビルドでは sm50〜sm70 のサポートを削除（バイナリサイズの制約のため）」と書く。一方、dev-discuss の告知は「CUDA 12.8 バイナリから Maxwell と Pascal を外し、Volta は戻す」と書いており、Volta の扱いが食い違っている — [PyTorch 2.8.0 notes](https://newreleases.io/project/pypi/torch/release/2.8.0), [dev-discuss](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)
- 別の告知によると、CUDA 12.6 のホイールは PyTorch 2.14 が最後で、2.15 からは公開されない（Maxwell / Pascal / Volta はソースビルドのみになる） — [dev-discuss notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- JAX の CUDA 12 ホイールは SM 5.2（Maxwell）以上に対応し、CUDA 13 ホイールは SM 7.5 以上（「CUDA 13 で NVIDIA が古い GPU を落とした」ため）。Windows ネイティブの GPU は非対応で、WSL2 は実験的扱い — [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- CUDA 12.8 のリリースノートは「Maxwell / Pascal / Volta のアーキテクチャサポートは機能凍結」、12.9 は「次のメジャー版でオフラインコンパイルとライブラリのサポートを削除」と書く。ただし、これは検索結果の要約から得た記述で、リリースノート本文は確認していない — [検索結果要約: NVIDIA forum ほか](https://forums.developer.nvidia.com/t/optix-9-0-release/322842)
- NVIDIA のドライバは R580 系が Maxwell / Pascal / Volta をサポートする最後の系列で、R590 以降はレガシー扱いになる。Windows では 581.80 が最後のフル対応ドライバと報じられている（2025-12 時点） — [Tom's Hardware](https://www.tomshardware.com/pc-components/gpu-drivers/nvidia-to-axe-maxwell-pascal-and-volta-gpus-with-end-of-driver-support-580-series-drivers-will-be-the-last-to-support-gtx-900-and-1000-cards), [KitGuru](https://www.kitguru.net/components/graphic-cards/joao-silva/nvidia-drops-maxwell-and-pascal-support-with-new-linux-590-branch-drivers/)
- 公式 3DGS は「Compute Capability 7.0+」を要件に掲げる — [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- cubemapencoder は float の atomicAdd しか使わず、arch 指定もない — [cubemapencoder.cu](https://raw.githubusercontent.com/gapszju/3DGS-DR/main/submodules/cubemapencoder/src/cubemapencoder.cu)
- tiny-cuda-nn の FullyFusedMLP は 2080 Ti / 3090 級を想定し、古い GPU では CutlassMLP を使う — [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- 3DGRT について、第三者のガイド（Yotta Labs）は「RT コアのある GPU（Turing 以降）が必須」と書く。公式 README の表現は「RT コアを推奨」「専用ハードウェアを要する」で、ガイドとは言い方が違う。必須とするのは公式文書ではない — [Yotta Labs](https://docs.yottalabs.ai/tutorials/image-and-video-generation/3dgrut-on-a100-build-your-own-3d-scene-reconstruction), [3dgrut README](https://github.com/nv-tlabs/3dgrut)
- 3dgrut の Issue で GTX 1080 や Pascal / Maxwell に関する報告は、検索では見つからなかった — [検索結果（該当なし）](https://github.com/nv-tlabs/3dgrut)

### Inferences

**手法ごとの sm_52（M4000、Windows 10）判定**（「確認済み」に当たる手法はなかった）

| 手法 | 判定 | 根拠・主な障害 |
|---|---|---|
| Spec-Gaussian | 依存から推定: 自前ビルドできれば動く可能性が高い | 3DGS 改造版ラスタライザ、simple-knn、torch-scatter だけで、OptiX・tcnn・nvdiffrast は不要。CUDA 11.x は sm_52 向けにコンパイルできる。公式 3DGS の「CC 7.0+」は、検証済みの範囲を示したもので強い制約ではないと推定する（Maxwell での成功報告は見つからず）。8 GB では反復数とガウシアン数を抑える必要がある |
| 3DGS-DR | 依存から推定: 自前ビルドできれば動く可能性が高い | cubemapencoder は float の atomicAdd だけで arch 制約なし（確認済み）。c3 / c7 ラスタライザは 3DGS の改造版 |
| Ref-Gaussian | 依存から推定: 自前ビルドできれば動く可能性あり | OptiX は不要。ashawkey/raytracing は純 CUDA の BVH（アーカイブ済み。Windows 対応は記載なし）。2DGS ラスタライザも必要 |
| Ref-GS | 依存から推定: 自前ビルドに加えて nvdiffrast が必要 | 2DGS 系の改造ラスタライザ 3 種と nvdiffrast。現行の nvdiffrast は PyTorch 2.7 以上を要求する（ユーザーの 2.7.1 なら条件は満たす）。nvdiffrast の最低 arch は不明 |
| GaussianShader | 依存から推定: ビルドの手間は大きいが不可能ではない | nvdiffrast 0.3.1 と torch 1.10（Linux 前提の環境ファイル）。新しい PyTorch に移植する必要がある |
| GS-ROR | 依存から推定 | nvdiffrast、r3dg-rasterization、fused-ssim。ライセンスは非商用 |
| NeRO | 依存から推定 | nvdiffrast と ashawkey/raytracing。SDF 系で学習が重い（具体的な時間は未確認） |
| IRGS | 不明（困難寄り） | 第 2 段に OptiX が必要。OptiX のヘッダが 7.x なら、ドライバ R580 の範囲で Maxwell に対応するはず（第 3 節）。ただし RT コアがないためソフトウェアでの走査になり遅い。Linux 流の CMake + make 手順を Windows で再現する手間もある |
| EnvGS | 不明（困難寄り） | IRGS と同様に OptiX が必要。加えて EasyVolcap、ラスタライザ 3 種、Linux 前提のパス |
| 3DGRT | 依存から推定: 理屈の上ではありうるが、遅さと VRAM の不足で実用は困難 | OptiX 7.5 ヘッダ（ドライバ 515.48 以上。R580 で満たせる）。OptiX 7 系は CC 5.0 以上をサポートする。Windows スクリプトがあり、CUDA 11.8 を選べる。一方で README は RT コア推奨、第三者は必須と書く。tiny-cuda-nn のサブモジュールが sm_52 でビルドできるかは不明 |
| 3DGUT | 不明 | ラスタライズ法なので RT コアは要らないはず。ただし、3dgrut のビルドが OptiX トレーサや tcnn を必ず含むかどうかは未確認 |
| Ref-NeRF（multinerf） | 依存から推定: Windows ネイティブでは不可 | JAX の GPU は Windows ネイティブ非対応。WSL2 は実験的扱いで、WSL2 + Maxwell の可否は不明（第 2 節 Gaps） |
| MirrorGaussian / NeRF-Casting | 判定不能（コード未公開） | — |

- 決定的な要因の優先度は次のとおり（推定）。
  1. **Windows でのソースビルド**: すべての Gaussian 系手法に共通。ユーザーの PyTorch 2.7.1+cu118 で動かすには、CUDA 11.8 ツールキットと MSVC で `TORCH_CUDA_ARCH_LIST=5.2` を指定してビルドすることになる。
  2. **OptiX 系（EnvGS / IRGS / 3DGRT）**: RT コアがないうえ、OptiX 9.1 以降は使えない。
  3. **将来の更新**: PyTorch cu128 以降、CUDA 13、R590 ドライバで Maxwell は切られていく。新しい版を前提にした更新（例: nvdiffrast は PyTorch 2.7 以上が必須）に追従できなくなる。
  4. **8 GB の VRAM**: 公式 3DGS は論文品質に 24 GB を要する。
- 半精度と Tensor Core について。cubemapencoder は __half のアトミックを意図的に外しており、他の候補でも FP16 必須の記述は見当たらない。tiny-cuda-nn を使うのは 3dgrut だけ。したがって、この理由で決定的に詰む候補は見当たらない。
- 実務的な結論として、公式コードを M4000 で動かすことは「学習目的の確認手段」としては割に合わない。PyTorch で自作する方が合理的と判断する（第 4 節）。

### Gaps
- Maxwell / sm_52 で上記の手法を動かした成功・失敗の報告は、検索の範囲では一つも見つからなかった。各リポジトリの Issue を一件ずつは確認していない。
- 公式 3DGS の diff-gaussian-rasterization を sm_52 向けにビルドした報告（成功・失敗とも）は見つからなかった。「CC 7.0+」が技術的な制約なのか、検証範囲の記述なのかは確認できなかった。
- nvdiffrast の最低 compute capability は公式に記載がない。
- WSL2 上の CUDA が Maxwell で動くかどうかは、未調査。

---

## 3. OptiX の版ごとの対応 GPU 世代と、3DGRT/3DGUT の要件

### Takeaway
OptiX 7.x は「Compute Capability 5.0（Maxwell）以上」をサポートする（7.1 のリリースノートの記述）。OptiX 9.1（2025-12）の配布ページは「Turing 以降」と明記し、R590 ドライバを要求する。R590 は Maxwell を外したので、M4000 で OptiX 9.1 は使えない。OptiX 8.0 / 8.1 / 9.0 の対応世代は一次資料で確認できなかった。ドライバの要件（R535 / R555 / R570）だけなら Maxwell 最後の R580 で満たせる。3dgrut は OptiX 7.5 のヘッダを使うので、ドライバと OptiX の版の面では M4000 も対象に入る（推定）。ただし RT コアはない。

### Cited Findings
- OptiX 7.1.0 のリリースノートの記述として、NVIDIA フォーラムの投稿者（droettger）が「All NVIDIA GPUs of Compute Capability 5.0 (Maxwell) or higher are supported」を引用している。SDK サンプルの CMake が SM 6.0 になっていたのは見落としで、SM 5.0 に書き換えれば Maxwell で動くはず、とも述べている — [NVIDIA forum](https://forums.developer.nvidia.com/t/optix-7-1-issue-with-running-samples-on-a-maxwell-card/140118/2)
- OptiX 9.1.0（2025-12）は R590 以降のドライバが必要。対応 GPU は GeForce / RTX・Quadro / Data Center の「Turing and newer generation GPUs」。OptiX 6 アプリのサポートは R590 でドライバから削除された（使い続けるなら R580 以前） — [OptiX download](https://developer.nvidia.com/designworks/optix/download), [OptiX 9.1 Release](https://forums.developer.nvidia.com/t/optix-9-1-release/354119)
- 各版の必要ドライバは次のとおり。9.0.0（2025-02）R570 以上、8.1.0（2024-10）R555 以上、8.0.0（2023-08）R535 以上、7.7 は 530.41 以上、7.6 は 522.25 以上、7.5（2022-06）は 515.48 以上。どの版にも Windows 10 64bit 版がある。レガシーページには対応 GPU 世代の記載がない。7.6 では「Opacity Micromap は Ada でハードウェア加速、それ以前の世代は最適化されたソフトウェア実装で対応」、7.7 では「Displacement Micromap は RT コアのある GPU で対応」と書かれている — [OptiX legacy downloads](https://developer.nvidia.com/designworks/optix/downloads/legacy)
- 「OptiX 8.0.0 のリリースノートで最低の計算能力が Maxwell とされた」という記述は、フォーラム投稿を検索結果が要約したものしか見つからず、リリースノート本体は確認できていない — [検索結果要約: OptiX forum](https://forums.developer.nvidia.com/c/gaming-and-visualization-technologies/visualization/optix/167)
- 3dgrut は optix-dev を v7.5.0 ブランチで、tiny-cuda-nn と一緒にサブモジュールとして含む — [3dgrut .gitmodules](https://raw.githubusercontent.com/nv-tlabs/3dgrut/main/.gitmodules)
- 3dgrut は Linux と Windows に対応し、CUDA 11.8〜13.0 を選べる（12.8 が既定、13.0 は実験的）。README は 3DGRT について「RT コアのある GPU を推奨」「専用のレイトレーシングハードウェアを要し、3DGS より遅い」と書く。3DGRT は反射・屈折・影などの二次光線を効率よく扱える。プレイグラウンドは挿入物体・反射・屈折・被写界深度に対応する。ベンチマークは RTX 5090 と L40 で取られている — [3dgrut README](https://github.com/nv-tlabs/3dgrut)
- 3DGUT（論文題「Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting」）はラスタライズ法として説明されている — [3dgrut README](https://github.com/nv-tlabs/3dgrut)
- 他のソフトウェアの例では、Blender Cycles で GTX 1080（RT コアなし）に OptiX を使ったときのエラー報告がある。多くはドライバと OptiX の版の不一致が原因とされる — [Blender T82435](https://developer.blender.org/T82435), [Blender T81620](https://developer.blender.org/T81620)

### Inferences
- M4000 で OptiX を使うなら、条件は「OptiX 7.x（または 9.0 以下）＋ドライバ R580 系（Windows は 581.80 まで）」となる。OptiX 7 系がソフトウェアで BVH を走査する形で Maxwell をサポートしていることは、7.1 のリリースノートの記述と整合する。3dgrut（OptiX 7.5）の要件のうち、OptiX とドライバの面は満たせると推定する。
- RT コアがないと、3DGRT の BVH 走査と交差判定は SM 上のソフトウェア処理になる。RTX 世代より桁違いに遅くなると見込む（定量的なデータはない）。3DGRT は RT コアがあっても 3DGS より遅いと README に書かれているので、M4000 で学習を回すのは非現実的だと判断する。
- 3DGUT は、二次光線（反射）を扱う部分で結局 3DGRT のトレーサと組み合わせる構成だと推定する（README の「secondary rays」とプレイグラウンドの記述から）。ラスタライズ単独では相互反射は得られない。

### Gaps
- OptiX 8.0 / 8.1 / 9.0 の公式リリースノートにある対応 GPU の記述（PDF はログインが必要で取得できなかった）。Maxwell が正確にどの版で外れたのか（9.1 で「Turing 以降」と明記された時点までは確認できた）。
- 3dgrut が Slang / slangtorch を使うかどうか、tiny-cuda-nn を何に使っているか（MLP なのか共通ユーティリティなのか）、sm_52 向けにビルドできるか。
- RT コアのない GPU で 3DGRT を動かした速度の報告。

---

## 4. PyTorch のみで簡略版を実装するのに最低限必要なもの

### Takeaway
系統ごとの難しさは次のとおり。
- **低**: ガウシアンごとのシェーディング（Spec-Gaussian 型の ASG、反射方向で環境マップを引く処理をガウシアン単位で行う GaussianShader 型）。
- **低〜中**: 遅延シェーディング（3DGS-DR / Ref-Gaussian / Ref-GS 型。法線や反射強度のチャンネルを足し、画素単位で反射方向を求め、学習可能なキューブマップを grid_sample で引く）。
- **中〜高**: 相互反射（二次光線）。平面鏡に限れば、MirrorGaussian 型の鏡映ラスタライズはずっと易しい。

遅延シェーディングについては、「タイル α ブレンディングの出力にチャンネルを足すだけ」で 8〜9 割は済む。残りの手間は、法線の質を上げる正則化（ガウシアンの扁平化と、深度から求めた法線との整合）と、粗さに応じたミップの扱い。キューブマップの微分可能な参照は、6 面テンソルと grid_sample で書いた先行例がある。

### Cited Findings
- PyTorch のみの 3DGS 実装として、hbb1/torch-splatting（MIT）がある。タイル分割で描画するが、Python のループが遅いため 64×64 のタイルを採用している。RTX 2080 Ti で 512×512、30k 反復、ガウシアン 16,384 個の学習に約 2 時間かかる — [torch-splatting](https://github.com/hbb1/torch-splatting)
- 純 PyTorch の別のパイプライン（project → bin → sort → composite、8×8 タイル、torch.sort を使用）の解説記事がある。その記事によると、gsplat（v1.5.3）には PyTorch の参照ラスタライザが付属する（二次情報で、gsplat 本体は未確認） — [Towards AI](https://pub.towardsai.net/gaussian-splatting-for-self-driving-simulation-built-in-plain-pytorch-on-a-laptop-gpu-49ed66a0d7c6)
- キューブマップを学習可能なパラメータとして扱う先行例:
  - PanoLess（arXiv 2607.25362）は環境を 6 枚の学習可能な L×L の RGB グリッドで持つ。方向の支配軸で面を選んで bilinear で補間し、サンプラを通して勾配を流す — [PanoLess](https://arxiv.org/pdf/2607.25362)
  - PanDA の projection_utils は、キューブの特徴を 6 面に並べ替え、前計算したサンプリンググリッドで `F.grid_sample` を呼ぶ（padding_mode="border"、align_corners=True） — [PanDA projection_utils](https://huggingface.co/spaces/ZidongC/PanDA_Panoramic_Depth_Estimation/blob/main/networks/projection_utils.py)
  - pytorch360convert は正距円筒・キューブマップ・透視投影の間を微分可能に相互変換する（CPU / GPU） — [pytorch360convert](https://science.ecosyste.ms/projects/43013)
  - これらのコードの中身は検索結果の要約でしか確認しておらず、詳細は未検証。
- `grid_sample` は二階微分を持たない（SparseNeuS はそのために自前で実装し直している） — [SparseNeuS grid_sampler](https://huggingface.co/One-2-3-45/code/blob/main/SparseNeuS_demo_v1/ops/grid_sampler.py)
- 3DGS-DR / Ref-Gaussian が使う cubemapencoder 自体が、単一レベルの bilinear 参照（seamless オプション付き）にすぎない。逆伝播は bilinear の重みでの atomicAdd と、UV を通した方向勾配だけ — [cubemapencoder.cu](https://raw.githubusercontent.com/gapszju/3DGS-DR/main/submodules/cubemapencoder/src/cubemapencoder.cu)
- 3DGS-DR は「遅延シェーディングが生む画素単位の反射勾配」で、近傍ガウシアンの法線推定を互いに改善させる。実装は 3 チャンネル（c3）と 7 チャンネル（c7）の 2 種類のラスタライザに分かれている — [3DGS-DR](https://github.com/gapszju/3DGS-DR)
- nvdiffrast の `texture(boundary_mode='cube')` は継ぎ目なしの trilinear とミップマップを提供する。CUDA 版の参照実装として、PyTorch で再現するときの目標にできる — [nvdiffrast docs](https://nvlabs.github.io/nvdiffrast/)
- Ref-GS は遅延描画した表面に方向エンコーディングを施して、向きと視線角の曖昧さを減らす。粗さに応じたシェーディングには球面 Mip-grid を使う — [CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/32417), [radiancefields](https://radiancefields.com/papers/ref-gs-directional-factorization-for-2d-gaussian-splatting)
- MirrorGaussian は、ガウシアンを鏡面で鏡映した複製をラスタライズして鏡像を作り、元の像とマスクで合成する。鏡面のパラメータも同時に最適化する — [MirrorGaussian](https://mirror-gaussian.github.io)
- IRGS の第 1 段は Ref-Gaussian で、`raytracing`（メッシュ BVH）を使う。第 2 段で OptiX による 2D ガウシアンの光線追跡を行う — [IRGS README](https://github.com/fudan-zvg/IRGS)
- EnvGS の diff-surfel-tracing は、2D ガウシアンのパラメータと光線の原点・方向について微分可能で、多重バウンスのパストレースと任意の出力に対応する — [diff-surfel-tracing](https://github.com/xbillowy/diff-surfel-tracing)

### Inferences（実装設計の評価。出典のない部分は筆者の工学的判断）

**A. ガウシアンごとのシェーディング（Spec-Gaussian / GaussianShader 型）— 難易度: 低**
- 必要な部品は 3 つ。
  - ガウシアンごとの法線: 回転行列のうち最小スケール軸の列を取り、カメラ側に向くよう符号を反転する。
  - 反射方向: r = 2(n·v)n − v。
  - 外観モデル: 方向 r（または視線 v）から色を返す、学習可能な関数。
- ラスタライザには手を入れず、色を計算するところで (N,3) を差し替えるだけで済む。環境マップの参照も N 回で済むので軽い。
- Spec-Gaussian の ASG は数式で書ける（ASG の各ローブを評価して小さな MLP に通す）。純 PyTorch で数十行の規模。
- 弱点が 2 つある。ガウシアンの法線はノイズが多く、反射方向が隣のガウシアンと揃わない。3DGS-DR が遅延シェーディングに移ったのはこのため（3DGS-DR の抄録の主張と整合）。また、鋭い鏡面反射は再現しにくい。

**B. 遅延シェーディング（3DGS-DR / Ref-Gaussian / Ref-GS 型）— 難易度: 低〜中**
- タイル α ブレンディングで合成する特徴を、RGB の 3 チャンネルから、例えば次の構成に増やす。
  - 拡散色 3、法線 3、反射強度 1、粗さ 1、深度 1（深度は法線の整合用）の計 9 チャンネル前後。
  - 合成の式 C = Σ w_i f_i は変わらないので、コードの変更は f_i の次元を広げることにほぼ限られる。この意味で「チャンネルを足すだけ」は概ね正しい。
  - autograd が保持する中間テンソルはチャンネル数にほぼ比例して増えるので、メモリは RGB だけのときの約 3 倍になる（8 GB では、画素数かガウシアン数で調整する）。
- 画素ごとの後処理は次のとおり。
  - 合成した法線を正規化する（|n| は 1 未満になる）。
  - カメラ光線から画素ごとの視線方向 v を求め、反射方向 r = reflect(−v, n) を計算する。
  - L = env(r, 粗さ) を引き、C = (1−s)·c_d + s·L で合成する（3DGS-DR 型）。
- 環境マップは 3 通り考えられる。
  - **(i) キューブマップ [6,3,L,L]**: 支配軸で面と UV を選び、6 面を 1 枚のアトラスに並べれば grid_sample 1 回で引ける。面の境界は clamp で妥協する（cubemapencoder の非 seamless モードと同等）。
  - **(ii) 正距円筒 [3,H,2H]**: atan2 / acos で UV に変換し、grid_sample で引く。横方向の巻き戻しは、端の列を連結しておけば済む。極付近は歪む。
  - **(iii) 低次の SH 環境**: 鋭い反射は出ない。
- 粗さの扱いは次の手順で書ける（GaussianShader / Ref-Gaussian の split-sum の簡略版に当たる）。
  - avg_pool2d でミップの段を作る。
  - 粗さからレベルを決め、隣り合う 2 段の grid_sample の結果を線形に補間する。
  - split-sum の BRDF LUT は勾配不要なので、numpy で前計算しておけばよい。
- 本当の手間は法線の質にある。
  - (a) ガウシアンを扁平化する正則化（最小スケールへのペナルティ）。
  - (b) 合成深度の有限差分から求めた法線と、合成法線との整合損失。
  - (c) 法線の符号を揃える処理。
  - いずれも純 PyTorch で書ける。2DGS のサーフェルを使えば法線が定義上決まるので有利だが、自作のラスタライザを 2DGS 化するには、光線とサーフェルの交点で評価するよう書き換える必要がある（中程度の改修）。
- Ref-NeRF / Ref-GS 型の方向エンコーディングは、反射方向 r と粗さ κ から特徴を作って MLP に入れる方式。
  - Ref-NeRF の IDE: SH 係数に、粗さに応じた減衰 exp(−l(l+1)/(2κ)) を掛ける。
  - Ref-GS 型: 球面上の多解像度グリッド。
  - どちらも純 PyTorch で容易（IDE の式は Ref-NeRF の論文によるもので、本調査では一次確認していない）。
  - 学習可能な照明と違って「照明を共有する」制約は弱まる。ユーザーの実験目的（共有照明の検証）からすると、キューブマップ型の方が解釈しやすい。

**C. 相互反射（二次光線）— 難易度: 中〜高。方式によって差が大きい**
- **C1. 平面鏡の鏡映ラスタライズ（MirrorGaussian 型）**: 平均を鏡映し、共分散を Householder 変換して、同じラスタライザでもう 1 回描画するだけ。最も易しい。ただし平面鏡に限られ、ユーザーの物体の自己相互反射（曲面）には使えない。
- **C2. スクリーン空間反射（SSR）**: 合成した深度と色のバッファの上で、反射光線を画面空間で K ステップ進め、深度との交差を探してその色を grid_sample で取る。純 PyTorch でも、画素数 × K のテンソル演算として書ける。欠点は 3 つある。画面外や遮蔽された部分は取れない。交差判定は離散なので勾配が流れるのは色の参照だけになる。物体が自分自身を映す凹部の相互反射は、正面からの視点では遮蔽されて取れないことが多い。ユーザーが観測した 38% の自己交差光線のうち、取れる割合は限られると推定する。
- **C3. ガウシアンへの総当たりの光線評価**: 二次光線ごとに全ガウシアン（またはボクセルで粗く絞り込んだ候補）との交差・応答を評価し、深度でソートして α 合成する。純 PyTorch でも書けるが、計算量は「光線数 × 候補ガウシアン数」になる。例として、200×200 画素の 38%（約 1.5 万本）× 2 万個のガウシアンで約 3 億組になり、中間テンソル 1 つで約 1.2 GB 食う。チャンクに分けて勾配チェックポイントを使えば M4000 でも回せる可能性はあるが、学習は遅い。BVH の代わりに一様グリッドを PyTorch で組むのは可能だが、可変長の候補リストの扱い（パディングやソート）が面倒。
- **C4. Ref-Gaussian 型（可視性だけを光線追跡で求める）**: 抽出したメッシュ（または深度から作った代理形状）で反射光線の遮蔽を判定し、遮蔽された方向には環境光の代わりに「間接光」の項を使う。
  - 三角形の数が少なければ、純 PyTorch の総当たりの光線・三角形交差（Möller–Trumbore をベクトル化し、チャンクに分ける）で済む。可視性の判定に勾配は要らないので、torch.no_grad で回せる。
  - 間接光の項の具体的な定式化（ガウシアンごとの学習項か、ヒット点のガウシアン色か）は Ref-Gaussian の論文で確認する必要がある（本調査では未確認）。
- **C5. 本格的な微分可能レイトレース（IRGS / EnvGS / 3DGRT 型）**: OptiX と RT コアで高速化する前提の設計なので、純 PyTorch での再現は学習目的の範囲を超える。

**ユーザーの観察（相互反射 38%、画素より細かい凹凸）との関係**
- 共有照明＋遅延シェーディング（B）だけでは、自己交差する 38% の光線は原理的に説明できない。3DGS-DR、GaussianShader、間接光なしの Ref-GS も同じ壁に当たる。
- これに手当てしている系統は、Ref-Gaussian（可視性と間接光）、IRGS、EnvGS（いずれも光線追跡）。
- 学習目的なら、C4（可視性マスクだけを総当たりの no_grad 光線追跡で求め、遮蔽方向は別の学習項に回す）が、最小の追加で効果を確かめやすいと考える。
- 画素より細かい凹凸は、画素内の法線分布を実効的な粗さとして扱う考え方で、部分的に吸収できる可能性がある。遅延シェーディングでは、合成法線の長さ |n|<1 を法線のばらつきの指標として、粗さ（ミップのレベル）に加算するのが考えられる（グラフィックスでは Toksvig 型の手法として知られる。本調査では文献を確認していない）。

### Gaps
- PanoLess、PanDA、pytorch360convert のコードの中身（キューブマップを grid_sample で引く具体的な実装）は、検索結果の要約しか見ていない。
- gsplat の PyTorch 参照ラスタライザの範囲（法線や任意チャンネルを出せるか）は未確認。
- Ref-Gaussian の間接光の項と、3DGS-DR の法線推定（最短軸の法線、正則化）の詳細は、論文本文を確認していない。
- ユーザー自作の実装のメモリ量が不明なので、チャンネル追加による VRAM 増の見積もりは定性的なものにとどまる。M4000 の実測スループットもない。

---

## 5. WebGPU（Brush 等）や他の非 CUDA 実装で光沢に対応したものはあるか

### Takeaway
光沢（環境マップを引く反射、遅延シェーディング、反射用の光線追跡）に対応した WebGPU 実装や非 CUDA 実装は見つからなかった。Brush の README は球面調和以外の視線依存表現や反射に触れておらず、ロードマップにもない。非 CUDA の 3DGS 実装（NVIDIA Warp 版、Metal / MPS 版、純 PyTorch 版）はあるが、基本の 3DGS に限られる。

### Cited Findings
- Brush は WebGPU 互換の技術と Burn フレームワークで書かれている。Windows / macOS / Linux（AMD / NVIDIA / Intel）、Android、ブラウザ（Chrome / Edge 134 以降）で動く。README には反射・環境マップ・視線依存の鏡面・3DGUT・光線追跡への言及がなく、反射関連のロードマップもない — [Brush](https://github.com/ArthurBrussee/brush)
- gswarp は NVIDIA Warp による 3DGS のバックエンドで、NVIDIA の GPU が必要。metalsplat は Apple の MPS 上で動く。純 PyTorch の実装には torch-splatting と、Towards AI 記事のパイプラインがある — [検索結果要約](https://pypi.org/project/gswarp/), [metalsplat](https://radiancefields.com/platforms/metalsplat-pytorch), [torch-splatting](https://github.com/hbb1/torch-splatting)

### Inferences
- 光沢に対応した公式コードを M4000 で CUDA 拡張なしに試す既製の手段は、現状ない。ユーザーの自作 PyTorch 実装に第 4 節の簡略版を足すのが、もっとも現実的な学習経路と判断する。Brush で公式 3DGS を動かした実績は、ベースラインの比較用として引き続き使える。

### Gaps
- gswarp（Warp）の最低 compute capability と、Maxwell で動くかどうかは未確認。
- gsplat、OpenSplat など他の実装に、光沢用の拡張やフォークがあるかは網羅的に調べていない。
