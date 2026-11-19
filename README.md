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
| `gs/dataset.py` | Blender 形式データの読み込み（D-NeRF 形式の各フレームの時刻 `time` も読む） |
| `gs/deform.py` | 動くシーン用の変形場: 基準の位置と時刻から、位置・回転・大きさのずれを返す MLP |
| `gs/spacetime.py` | 動くシーン用の、時刻を持つガウシアン: 各ガウシアンが時刻・時間幅・速度を持ち、その前後でだけ現れる |
| `gs/trajectory.py` | 動くシーン用の、時刻ごとの位置を持つガウシアン: 数と見た目は固定で、位置だけを時刻の節点ごとに持つ |
| `train.py` | 学習（val 画像・log.csv・チェックポイント・point_cloud.ply を `logs/<expname>/` に出力） |
| `render_video.py` | 学習済みガウシアンで物体の周りを一周する新規視点画像を描画し、描画速度を表示 |
| `make_toy_dataset.py` | 動作確認用のおもちゃデータ（球 + 立方体。nerf-toy と同じ） |
| `train_dynamic.py` | 動くシーンの学習（4DGS の最小版。test 画像・時刻を動かした GIF・eval.json を出力） |
| `make_toy_dynamic.py` | 動くおもちゃデータ（球が弾んで脈打ち、立方体が往復する。D-NeRF 形式） |
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

おもちゃデータの正解画像は、1 画素を 4×4 本のレイで描いて平均している（`--supersample 4`、既定）。
はじめは 1 画素 1 本で描いていて、輪郭と模様の境目がギザギザだった。これは見落としで、
PSNR を 30 dB で頭打ちにしていた（下の「目安」を参照）。

### 2. nerf_synthetic（lego など）

公式の `nerf_synthetic` を `data/nerf_synthetic/lego` に置いて:

```bash
.venv/Scripts/python train.py --datadir data/nerf_synthetic/lego --expname lego --half-res --init-points 50000
```

### 3. 動くシーン（4DGS の最小版）

Deformable 3D Gaussians（Yang et al., CVPR 2024）の簡略版。ガウシアンは基準の形を 1 組だけ持ち、
時刻 t の形は MLP が基準の位置 x と t から (δx, δq, δs) を出してずらす。色と不透明度は時刻によらない。
はじめの `--warm-up` 回は変形なしで学習し、おおまかな形ができてから MLP を動かす。
データは D-NeRF と同じ形式（`transforms_*.json` の各フレームに `"time"`）。学習用は 1 時刻に 1 視点しかない。

```bash
.venv/Scripts/python make_toy_dynamic.py
.venv/Scripts/python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy --iters 7000 --warm-up 1000 --densify-until 5000 --init-extent 2.0
.venv/Scripts/python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy_static --iters 7000 --static --densify-until 5000 --init-extent 2.0
```

結果（256 px、学習 100 枚、テストは学習に無い時刻の 20 枚。Quadro M4000）:

| | test PSNR | SSIM | ガウシアン | 学習時間 |
|---|---|---|---|---|
| 変形あり | 25.7 dB | 0.952 | 39,946 | 35 分（0.30 秒/iter、VRAM 3.3 GB） |
| 変形なし（`--static`） | 14.1 dB | 0.812 | 29,332 | 24 分 |

変形なしでは、動く球と立方体が時刻をまたいでぼやけた 1 つの形になる。

**この表と、以下の動くシーンの数値は、正解画像がギザギザだった頃のデータ（`--supersample 1`）で測ったもの。**
今の既定（`--supersample 4`）で変形ありを学習し直すと 28.3 dB、SSIM 0.966（ガウシアン 41,800 個）になる。
動きの正しさ（[実験](experiments/dynamic/README.md)の ratio）は変わらない。ほかの設定は測り直していないので、
下の数値どうしの比較はできるが、今のデータで出る値とは 2〜3 dB ずれるはず。

苦手な例として、立方体が球のまわりを半周するデータ（`--orbit`、約 5 単位動く）も試した（5000 iter）:

- `--warm-up 1000`: 15.1 dB。変形なしで学習するあいだに、時刻ごとに別の場所にある立方体は
  どの場所でもほとんど見えないので、そのガウシアンが消されてしまう。変形を始めた時には、動かす相手が残っていない
- `--warm-up 0`: 8.8 dB。ランダムな初期点のまま MLP を動かすと発散し、シーン全体が画面の外へ飛ぶ
- 変形なし: 11.8 dB

基準の形を 1 つ持ってずらす方式は、物体が自分の大きさ程度しか動かないシーン（D-NeRF の合成データなど）
を想定している。

もう 1 つの表現として、時刻を持つガウシアン（Spacetime Gaussians, Li et al., CVPR 2024 の簡略版）を
`--spacetime` で選べる。各ガウシアンが時刻 μ_t・時間幅 σ_t・速度 v を持ち、時刻 t では位置 x + v(t − μ_t)、
不透明度 o·exp(−½((t − μ_t)/σ_t)²) で描く。MLP も基準の形も要らない。

```bash
.venv/Scripts/python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy_st --iters 7000 --spacetime --densify-until 5000 --init-extent 2.0
```

小さく動くデータでは 23.7 dB（変形場より 2 dB 低い。ガウシアン 243,811 個、33 分）、`--orbit` では 23.0 dB
（変形場は 15.1 dB）。正解の動きと突き合わせた比較は[実験](experiments/dynamic/README.md)にある。

どちらの表現でも、`--lambda-smooth` で「近くのガウシアンの速度をそろえる」損失を足せる（変形場では 1e-4 で
26.3 dB に上がり、面に沿った滑りも減る）。`--spacetime` では `--min-t-scale` で時間幅に下限を付けられる。

3 つ目の表現として、時刻ごとの位置を持つガウシアン（Dynamic 3D Gaussians, Luiten et al., 3DV 2024 の簡略版）を
`--trajectory` で選べる。数・色・不透明度・大きさは固定で、位置だけを時刻の節点（既定 21 個）ごとに持ち、
節点の間は線形に補間する。速度の正則化なしでは崩れるので、`--lambda-smooth` と一緒に使う。

```bash
.venv/Scripts/python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy_tr_smooth --iters 7000 --trajectory --lambda-smooth 1e-4 --densify-until 5000 --init-extent 2.0
```

小さく動くデータでは 24.3 dB（変形場より 1.4 dB 低い）、`--orbit` では 20.4 dB（7000 iter）。
`--knots-init 3` にすると、元の論文のように時間の順に節点を増やしながら学習する。1 時刻 1 視点のこのデータでは
画質が大きく下がる（19.3 dB / 17.6 dB）が、`--orbit` の立方体の動きはこちらのほうが合う。

`--lambda-track` は、おもちゃデータの正解の動きから作った「誤りのない 2D 追跡」に合わせる損失（実際の手法が
学習済みの点追跡モデルで入れる手がかりの、上限を測るためのもの）。`--orbit` のデータでは `--orbit` も付ける。

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
| 学習時間 | 約 22 分（0.13〜0.19 秒/iter。ガウシアンが増えると遅くなる） |
| test PSNR（8 枚） | 37.9 dB |
| ガウシアン数 | 20000（初期）→ 28659（最終。途中で不要な初期点が prune されていったん減る） |
| VRAM（ピーク） | 2.4 GB |
| 描画 | 約 68 ms/枚（約 15 fps） |

学習時間と描画時間は `--supersample 1` のデータで測った値（ガウシアン数がほぼ同じなので変わらないはず）。

### 正解画像のギザギザ（見落とし）

はじめの `make_toy_dataset.py` / `make_toy_dynamic.py` は 1 画素を 1 本のレイで描いていた。
輪郭と模様の境目は階段状になり、半透明の画素は 1 つもない。一方、ラスタライザは画面上の共分散に
0.3 を足してぼかすので、階段状の境目は原理的に再現できない。この食い違いに気づかず、
test PSNR 30.2 dB を「この実装の実力」として載せていた。

2 通りのデータで学習したモデルを、それぞれ両方の試験画像で採点した結果（7000 iter、seed 0、各 1 回）:

| 学習に使ったデータ | ギザギザの試験画像で採点 | 滑らかな試験画像で採点 |
|---|---|---|
| `--supersample 1`（ギザギザ） | 30.4 dB | 35.0 dB |
| `--supersample 4`（滑らか） | 30.5 dB | 37.9 dB |

- 30 dB の頭打ちは採点のせいだった。同じモデルを滑らかな正解と比べるだけで 35.0 dB になる（+4.7 dB）
- 学習も良くなる。同じ滑らかな試験画像で比べて 35.0 → 37.9 dB（+2.9 dB）
- ギザギザの試験画像で採点すると、どちらで学習しても 30.4〜30.5 dB で区別がつかない

同じ 200px の画像を nerf-toy の NeRF で描くと約 12 秒なので、PyTorch 実装のままでも 150 倍以上速い。
公式の CUDA 実装は処理を 1 つのカーネルにまとめているので、さらに 1〜2 桁速い。

log.csv で 3000 iter の val PSNR だけ極端に低い（約 8 dB）のは、opacity reset で全ガウシアンの
不透明度を 0.01 に下げた直後の状態を評価しているため。数百 iter で元に戻る。

## 実験

| 実験 | 内容 |
|---|---|
| [光沢面で 3DGS は破綻するか](experiments/gloss/README.md) | GoLD_rayt で描いた金の像と同色の拡散面の像で比較。金は像の領域の PSNR で 16〜17 dB 低く、学習視点でも同じくらい悪い（汎化の失敗ではなく表現力の不足）。1 点ずつ SH を当てはめる検証では、次数を上げると学習視点は覚えられるがテスト視点は 24〜27 dB で頭打ちになり、天井は SH の次数ではなく 1 点あたりの観測の粗さで決まる。反射方向で照明を共有するモデルは、照明を正しく復元するが、この像では相互反射（観測の 38%）と画素より細かい凹凸のため全体では点ごとの SH に届かず、それらを除いた所でだけ上回る。点ごとに最適化した粗さは、画素の中の法線のばらつきとはほとんど相関しない（順位相関 0.14。仮説どおりの合成データなら同じ手順で 0.89 まで取り戻せる）。凹凸を制御した球では順位相関 0.93 で合い、大きさは換算の 0.72〜0.87 倍（斜めの観測で反射のぼけが狭くなることと、等方なぼけで近似することで説明できる）。この異方性を補正した換算は、学習するパラメータなしで、点ごとに最適化した粗さと同等以上。3dgs-toy に遅延シェーディング（法線を合成し、反射方向で共有の照明を引く）を入れると、区画ごとのぼけの大小は正しく学べる。ただし学習したガウシアンは面の内側に半透明の厚い層を作るので、合成した法線の長さは画素の中の平均にならず、画素の大きさを変えると正解と逆向きに変わる。再現度は通常の 3DGS と同等以下 |
| [3DGS の光沢対応手法の文献調査](experiments/gloss/reports/3DGS光沢対応手法の文献調査.md) | 上の実験で見えた 4 つの崩れ（観測の粗さ、法線と遠方照明、相互反射、画素より細かい凹凸）に沿って、SH の置き換え・共有照明・光線追跡などの系統を整理。旧 GPU での実行可能性と、3dgs-toy に足す簡略版の順序の提案つき |
| [動くシーン: 3 つの表現を正解の動きと突き合わせる](experiments/dynamic/README.md) | 変形場と、時刻を持つガウシアンを、動きの式が分かっているおもちゃデータで比較。小さく動くデータでは変形場が 2 dB 上、大きく動くデータでは時刻つきが 8 dB 上で、得意・不得意が逆になる。変形場でも球の接線方向の動きは正解の移動量の 0.74 倍ずれている（一様な色の面の上では滑っても画像が変わらない）。時刻つきは時間幅がフレーム約 2 枚ぶんまで縮み、動きをほとんど学ばずに学習視点へ過適合する。近くのガウシアンの速度をそろえる正則化は、変形場では滑りを 0.49 倍まで減らすが、時刻つきでは全員が止まる方へ働いて逆効果。時間幅に下限を付けても動きは学ばれない。時刻ごとの位置を追う 3 つ目の表現は、正則化なしでは変形なしより悪い（13.2 dB）。時間の順に学習すると、少ない視点で作った浮遊物が残って 19.3 dB どまりで、最初から全時刻を使うと 24.3 dB になる。正解の動きから作った誤りのない 2D 追跡を損失に入れても、球の面に沿った滑りは 0.74 → 0.51 倍に減るだけで消えない |

## 発展課題（学習用）

- **composite の逆伝播を全部手で書く**: 公式の `backward.cu` と同じく、各画素で後ろのガウシアンから順に
  T を戻しながら ∂L/∂α・∂L/∂c を累積する。`torch.autograd.Function` で書き、`gradcheck` や
  `tests/test_rasterize.py` の素朴な実装と突き合わせる。
- **nerf-toy と比べる**: 同じ `data/toy_sphere` で、PSNR・学習時間・1 枚の描画時間を比べる。
- **densification の効果を見る**: `--densify-until 0` との比較。log.csv のガウシアン数の推移も見る。
- **SH の次数**: `--sh-degree 0`（視点非依存）と 3 の比較。光沢のあるシーンで差が出る。
- **実写データ**: COLMAP の出力（カメラと疎な点群）を読み、SfM 点で初期化する。
