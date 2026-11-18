# 3D Gaussian Splatting（PyTorch、スクラッチ実装）

Kerbl et al. (2023) "3D Gaussian Splatting for Real-Time Radiance Field Rendering" を、学習用に
外部の 3DGS ライブラリを使わずに実装したもの。公式実装では CUDA で手書きしている微分可能
ラスタライザも PyTorch のテンソル演算で書き、逆伝播は基本的に autograd に任せている。
入力は nerf-toy と同じ Blender 合成データ形式（`transforms_*.json` + 画像）。

## NeRF（nerf-toy）との違い

|  | NeRF | 3DGS |
|---|---|---|
| シーンの表現 | MLP（密度と色を返す関数。陰的） | 数万〜数百万個の 3D ガウシアン（陽的） |
| 1 画素の描画 | レイ上の数百点で MLP を評価 | ガウシアンを画面に投影し、手前から α ブレンディング |
| 合成の式 | C = Σ c_i α_i T_i（i はレイ上のサンプル点） | C = Σ c_i α_i T_i（i はその画素に重なるガウシアン） |
| 構造の変化 | なし（ネットワークの形は固定） | 学習中にガウシアンを増やす・減らす（densification） |

合成の式は同じで、「レイ上のサンプル点」が「画面上で重なるガウシアン」に置き換わっている。
MLP を何百回も評価する代わりに、投影と並べ替えで済むのが速さの源。

## 構成

| ファイル | 内容 |
|---|---|
| `gs/gaussians.py` | ガウシアンのパラメータ、初期化、densification（clone / split / prune）と Adam の状態の作り直し、.ply 書き出し |
| `gs/rasterize.py` | 微分可能ラスタライザ: 3D→2D 投影、タイル割り当てとソート、α ブレンディング |
| `gs/render.py` | パラメータの活性化 → 投影 → SH で色 → ラスタライズ をつなぐ |
| `gs/sh.py` | 球面調和関数による視点依存の色 |
| `gs/loss.py` | L1 + SSIM 損失、PSNR |
| `gs/camera.py` | カメラ（Blender の camera-to-world を OpenCV 流儀に変換して保持） |
| `gs/dataset.py` | Blender 形式データの読み込み |
| `train.py` | 学習（val 画像・log.csv・チェックポイント・point_cloud.ply を `logs/<expname>/` に出力） |
| `render_video.py` | 学習済みガウシアンで物体の周りを一周する新規視点画像を描画し、描画速度を表示 |
| `make_toy_dataset.py` | 動作確認用のおもちゃデータ（球 + 立方体。nerf-toy と同じ） |
| `tests/test_rasterize.py` | ラスタライザの検算（素朴な全画素版と画像・勾配を突き合わせる、手書き逆伝播の gradcheck） |
| `experiments/` | このリポジトリの実装を使った実験（下記） |

## アルゴリズムとコードの対応

読む順番の目安。論文の式番号は arXiv 版。

1. **表現**（`gaussians.py`）: 各ガウシアンは中心 μ、スケール s（log で保持）、回転 q（クォータニオン）、
   不透明度 o（logit で保持）、SH 係数を持つ。活性化関数（exp / 正規化 / sigmoid）で値域の制約を外している。
2. **3D 共分散**（`rasterize.covariance_3d`）: Σ = R S Sᵀ Rᵀ（式 6）。Σ を直接最適化すると
   半正定値でなくなりうるので、スケールと回転に分解して持つ。
3. **投影**（`rasterize.project_gaussians`）: 透視投影を中心のまわりで一次近似（ヤコビアン J）すると
   ガウシアンはガウシアンに写る。Σ' = J W Σ Wᵀ Jᵀ（式 5、EWA splatting）。
4. **色**（`sh.py`）: 視線方向で SH を評価。学習開始時は 0 次（一定色）だけ使い、1000 iter ごとに次数を上げる。
5. **タイル割り当てとソート**（`rasterize.bin_tiles`）: 画面をタイルに分け、各ガウシアンが（3σ の範囲で）
   覆うタイルを列挙し、キー =（タイル番号, 深度）で 1 回だけソートする。これで「タイルごとに手前から順」の
   並びが全タイル同時に得られる（論文 Sec. 6 の要点）。
6. **α ブレンディング**（`rasterize.composite`）: α_i = o_i exp(-½ dᵀ Σ'⁻¹ d)、
   C = Σ c_i α_i T_i、T_i = Π_{j<i}(1-α_j)。α < 1/255 は捨て、α は 0.99 で頭打ちにする。
7. **損失**（`loss.py`）: L = 0.8 L1 + 0.2 (1 − SSIM)（式 7）。
8. **densification**（`GaussianModel.densify_and_prune`, 論文 Sec. 5.2）: 100 iter ごとに、画面上の位置勾配の
   平均が大きいガウシアンを、小さければ複製（clone）、大きければ 2 つに分割（split）。ほぼ透明なもの・
   大きすぎるものは削除（prune）。3000 iter ごとに不透明度を一斉に下げ（opacity reset）、不要なものを
   prune されやすくする。パラメータの数が変わるので Adam の 1 次・2 次モーメントも同じ形に作り直す（`_replace`）。

## 公式実装との違い（PyTorch で書くための工夫）

- 公式は前向き・逆向きとも CUDA で手書き。ここは前向きだけ書いて autograd で微分する。
  ただし透過率の累積和（`SegmentedExclusiveCumsum`）だけは逆伝播を手で書いている（下記）。
- 公式はタイル内の各画素がガウシアンを手前から 1 つずつ処理し、T < 10⁻⁴ で打ち切る。
  ここは (タイル内の画素 × 交差) の組を全部まとめてテンソルで計算する（打ち切りなし。残りの寄与は
  10⁻⁴ 未満なので結果はほぼ同じ）。
- T_i（区間ごとにリセットされる累積積）は、log 空間で全体の累積和を 1 回とって区間先頭の値を引いて求める。
  autograd に任せると、この「引く」部分の逆伝播が float64 の atomicAdd になり、それを持たない
  Maxwell 世代の GPU ではとても遅い。そこで随伴（区間ごとの逆向き累積和）を手で書いている。
- タイルの大きさは既定で 8px（公式は 16px）。まとめて計算する方式ではタイル内の無駄な画素も計算するので、
  小さいガウシアンが多いと小さいタイルの方が速い。
- タイル割り当ての半径を、不透明度を考慮して「α が 1/255 を下回る距離」で打ち切っている（3σ を超えない範囲）。
  その外側の画素は結局捨てられるので描画結果は厳密に同じまま、計算量だけ減る（テストで確認）。
- 一度に作る (画素 × 交差) テンソルが `--max-entries` を超えるときは、タイルをチャンクに分け、
  gradient checkpointing（逆伝播時に再計算）でメモリを抑える。
- 公式の `antialiasing` オプションや露出補正など、後から追加された機能は入れていない。

## セットアップ

```bash
cd 3dgs-toy
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
.venv/Scripts/python -m pip install -r requirements.txt
```

Quadro M4000 は Maxwell 世代（sm_52）なので、CUDA 12.8 以降のビルドでは動かない。CUDA 11.8 版を使う。

## 使い方

### 0. ラスタライザの検算

```bash
.venv/Scripts/python tests/test_rasterize.py
```

### 1. おもちゃデータで学習

```bash
.venv/Scripts/python make_toy_dataset.py --out data/toy_sphere
.venv/Scripts/python train.py --datadir data/toy_sphere --expname toy --iters 7000
.venv/Scripts/python render_video.py --ckpt logs/toy/ckpt_007000.pt --gif
```

`logs/toy/val_*.png` は左が正解、右が描画。`logs/toy/point_cloud.ply` は公式と同じ形式なので、
SuperSplat などの 3DGS ビューアにドラッグ＆ドロップすれば中身を回して見られる
（ワールドは Blender と同じ +Z 上なので、ビューアによっては横倒しに見える）。

### 2. nerf_synthetic（lego など）

公式の `nerf_synthetic` を `data/nerf_synthetic/lego` に置いて:

```bash
.venv/Scripts/python train.py --datadir data/nerf_synthetic/lego --expname lego --half-res --init-points 50000
```

## 主なオプション

- `--iters` : 学習回数（公式は 30000。densification は既定で前半 `iters/2` まで）
- `--init-points` / `--init-extent` / `--init-center` : 初期点の数、それを置く立方体の半辺長と中心
- `--bg R G B` : 背景色（既定は白。`--black-bkgd` で黒）。背景が一様な色のデータ用
- `--grad-thresh` : densification の閾値（公式 0.0002）。下げるとガウシアンが増える
- `--max-screen-size 0` : 画面上で大きすぎるものの prune を無効化
- `--densify-until 0` : densification をしない（初期点のまま最適化だけする）
- `--tile` / `--max-entries` : ラスタライザの設定。CUDA OOM なら `--max-entries` を下げる
- `--black-bkgd` : 背景を黒として扱う

## 目安（Quadro M4000 での実測、おもちゃデータ 200px）

`--iters 7000`（その他は既定値）の結果:

| 項目 | 値 |
|---|---|
| 学習時間 | 22.6 分（0.13〜0.19 秒/iter。ガウシアンが増えると遅くなる） |
| test PSNR（8 枚） | 30.2 dB |
| ガウシアン数 | 20000（初期）→ 7900（600 iter、不要な初期点が prune される）→ 28335（最終） |
| VRAM（ピーク） | 2.4 GB |
| 描画 | 68 ms/枚（約 15 fps） |

同じ 200px の画像を nerf-toy の NeRF で描くと約 12 秒なので、PyTorch 実装のままでも 150 倍以上速い。
公式の CUDA 実装は処理を 1 つのカーネルにまとめているので、さらに 1〜2 桁速い。

log.csv で 3000 iter の val PSNR だけ極端に低い（約 8 dB）のは、opacity reset で全ガウシアンの
不透明度を 0.01 に下げた直後の状態を評価しているため。数百 iter で元に戻る。

## 実験

| 実験 | 内容 |
|---|---|
| [光沢面で 3DGS は破綻するか](experiments/gloss/README.md) | GoLD_rayt で描いた金の像と同色の拡散面の像で比較。金は像の領域の PSNR で 16〜17 dB 低く、学習視点でも同じくらい悪い（汎化の失敗ではなく表現力の不足）。1 点ずつ SH を当てはめる検証では、次数を上げると学習視点は覚えられるがテスト視点は 24〜27 dB で頭打ちになり、天井は SH の次数ではなく 1 点あたりの観測の粗さで決まる。反射方向で照明を共有するモデルは、照明を正しく復元するが、この像では相互反射（観測の 38%）と画素より細かい凹凸のため全体では点ごとの SH に届かず、それらを除いた所でだけ上回る。点ごとに最適化した粗さは、画素の中の法線のばらつきとはほとんど相関しない（順位相関 0.14。仮説どおりの合成データなら同じ手順で 0.89 まで取り戻せる）。凹凸を制御した球では順位相関 0.93 で合い、大きさは換算の 0.72〜0.87 倍（斜めの観測で反射のぼけが狭くなることと、等方なぼけで近似することで説明できる）。この異方性を補正した換算は、学習するパラメータなしで、点ごとに最適化した粗さと同等以上。3dgs-toy に遅延シェーディング（法線を合成し、反射方向で共有の照明を引く）を入れると、区画ごとのぼけの大小は正しく学べる。ただし学習したガウシアンは面の内側に半透明の厚い層を作るので、合成した法線の長さは画素の中の平均にならず、画素の大きさを変えると正解と逆向きに変わる。再現度は通常の 3DGS と同等以下 |
| [3DGS の光沢対応手法の文献調査](experiments/gloss/reports/3DGS光沢対応手法の文献調査.md) | 上の実験で見えた 4 つの崩れ（観測の粗さ、法線と遠方照明、相互反射、画素より細かい凹凸）に沿って、SH の置き換え・共有照明・光線追跡などの系統を整理。旧 GPU での実行可能性と、3dgs-toy に足す簡略版の順序の提案つき |

## 発展課題（学習用）

- **composite の逆伝播を全部手で書く**: 公式の `backward.cu` と同じく、各画素で後ろのガウシアンから順に
  T を戻しながら ∂L/∂α・∂L/∂c を累積する。`torch.autograd.Function` で書き、`gradcheck` や
  `tests/test_rasterize.py` の素朴な実装と突き合わせる。
- **nerf-toy と比べる**: 同じ `data/toy_sphere` で、PSNR・学習時間・1 枚の描画時間を比べる。
- **densification の効果を見る**: `--densify-until 0` との比較。log.csv のガウシアン数の推移も見る。
- **SH の次数**: `--sh-degree 0`（視点非依存）と 3 の比較。光沢のあるシーンで差が出る。
- **実写データ**: COLMAP の出力（カメラと疎な点群）を読み、SfM 点で初期化する。
