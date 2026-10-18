# DeepSeek V4.1 Flash 1017 on 2 × H200

DeepSeek-V4.1-Flash-1017を **H200 NVL 2枚**で動かす、TensorFoldベースの推論実装です。H200向けCUDAカーネル、CPU RAM上のEngram、画像入力、1Mコンテキスト、共有KVキャッシュ、起動手順と実測結果を含みます。

Based on [TensorFold](https://github.com/bertholomus/TensorFold/tree/8e0966e2b09250abe2f6d9cc51edc20630e6eb1a), pinned at `8e0966e2b09250abe2f6d9cc51edc20630e6eb1a`. Includes patched source, deployment scripts, and measurements for two H200 GPUs. This is an independent derivative, not an official DeepSeek or NVIDIA release. Original documentation: [README.upstream.md](README.upstream.md).

## 実測結果

**1017版・2.9 bpw・単一ストリーム・出力512 tokens・各3回の中央値**。2026-10-18 JST、thinking off、temperature=0、DSpark投機生成あり、warm-up後の測定です。

この配布の設定は `TF_DS_REPLAY=1` のbounded replayを使います。以前のexact full prefill設定と同一計算ではありません。速度・長文検証はこの設定での結果です。

| 入力 | 生成速度 | 最初のcontent chunkまで |
|---|---:|---:|
| コード | **256.53 tokens/s** | 69.45 ms |
| 英文 | **144.38 tokens/s** | 71.45 ms |
| JSON | **293.58 tokens/s** | 114.54 ms |

生成速度は `(completion_tokens − 1) / (最終content chunk時刻 − 最初のcontent chunk時刻)`。prefillを含みません。投機生成では1 chunkに複数tokenが入るため、chunk数はtoken数ではありません。

| 長文・同時実行の検証 | 結果 |
|---|---|
| 1,046,955-token入力を独立に2本 | 各文書の秘密句を3/3正答。HTTP完了まで266.47 / 530.70秒（後者は待機を含む） |
| 上の入力を再利用 | 各1,046,528 tokensがキャッシュにヒット。24-token回答の**HTTP完了**まで0.886 / 0.833秒 |
| 63会話の同時decode | 全会話の256 token IDsが、各単独・draft無効時と完全一致 |
| 全会話が1Mの場合の保持容量 | **計算上36会話分**。実1M入力を36本同時に検証した値ではない |

1Mは**会話ごとの入力＋出力上限**です。KVプールは全会話で共有します。短い入力の生成速度を、1Mの初回処理や63会話同時実行の速度として読まないでください。

測定条件、各試行、メモリ、品質、旧モデルでのマイクロベンチは [H200-BENCHMARKS.md](docs/H200-BENCHMARKS.md) と [JSONデータ](benchmarks/h200/) にあります。

## H200向けの変更

- **CUDA演算の丸め位置を修正**：fused expert内の積を `__fmul_rn` にし、Hadamard変換とのFMA融合による結果の差を修正。sm90のSASSでもFMUL/FADDへの変化を確認しました。
- **行数に応じたexpertカーネル選択**：1行decodeはcp、複数行の投機検証は修正済みfusedを使います。
- **EngramをCPU RAMから読む**：読み取り専用mmapとprefaultを使い、約189.1 GiBのEngramをホスト側に置きます。
- **同一ホストの通信に合わせる**：Spark間通信向けのNCCL Simple / 4 channels固定を除き、NCCLに選択させます。
- **長文・複数会話のメモリ管理**：GPUごとの31.632 GiBの共有KVプール、CPU pinned snapshot、CUDA graph数の制限を加えています。
- **1017公式encoderに対応**：Jinja templateのない1017 tokenizerで、公式Python encoderを使ってthinkingとツール履歴を構成します。

EXL3、DSpark投機生成、基本的な推論・API実装は上流由来です。通常生成対投機生成の速度差を、今回のH200カーネル変更だけの効果としては扱いません。[出典・ライセンスと変更ファイル](docs/H200-ATTRIBUTION.md)

## 必要な環境

| 項目 | 検証した構成 |
|---|---|
| GPU | NVIDIA H200 NVL × 2、各143,771 MiB |
| 接続 | 同一ホストのPCIe。NVLink接続は検出されていない |
| ホストRAM | 1.1 TiB。Engramだけで約189.1 GiBを使い、ほかにローダー・OS用の余裕が必要。最小RAM構成は未検証 |
| ストレージ | モデル本体約385.4 GiBに加え、rank cache、コンテナ、作業領域が必要 |
| ソフトウェア | Linux x86_64、NVIDIA driver 575.57.08、CUDA 13、PyTorch 2.11.0+cu130、Triton 3.6.0 |
| 実行方式 | Docker＋NVIDIA Container Toolkit。使用イメージをdigestで固定 |

最大容量設定でのGPU使用量ピークは141,941 / 141,849 MiB（2秒間隔のサンプリング）。GPUメモリの余裕は大きくありません。この容量設定を小さいGPUへそのまま適用することはできません。

## 起動

```bash
git clone https://github.com/yukimaru77/deepseek-v41-h200.git
cd deepseek-v41-h200
```

固定した重みの取得、コンテナのビルド、両rankの起動は [h200/README.md](h200/README.md) にまとめています。重みはリポジトリに含めず、明示的な取得コマンドでダウンロードします。起動時のモデル自動更新は無効です。

| 用途 | モデルと固定revision |
|---|---|
| EXL3 2.9 bpw主重み | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)、`afc7aa0aab98ee735c0f54e0a122019af1ac08f1` |
| Engram、公式encoder | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)、`1bae0e1088993b9b82de08f59de5be5a34e42170` |

主重みは公開済み量子化モデルを使用しています。この作業で再学習・再量子化した重みではありません。Engramは1017版の末尾2 shardsを使います。

## APIとクライアント

| 形式 | ローカルbase URL | 主なエンドポイント |
|---|---|---|
| OpenAI Chat Completions | `http://127.0.0.1:18891/v1` | `/v1/chat/completions` |
| Anthropic Messages | `http://127.0.0.1:18891` | `/v1/messages` |

モデルIDは `DeepSeek-V4.1-Flash-1017-H200`。思考強度はOff / Low / High / Maxに対応します。サーバーはloopbackへbindします。外部公開は認証付きリバースプロキシ経由にしてください。現在の上流由来のネイティブ認証には既知の不具合があるため、この配布の起動手順では認証境界として使用しません。

```bash
curl http://127.0.0.1:18891/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017-H200","max_tokens":1024,
       "messages":[{"role":"user","content":"17×19はいくつ？"}],
       "thinking":{"type":"adaptive"},"output_config":{"effort":"max"}}'
```

DeepSeek Harness / Claude Codeでthinkingと通常のツール往復を確認しています。[クライアント設定例](h200/CLIENTS.md)。Anthropicの全機能への互換性を意味せず、強制tool_choiceのany / named、tool search、暗号化thinkingなどは未対応です。

画像入力は起動時の `--vision` と、クライアント側の `input: [text, image]` 宣言で有効になります。現在のH200起動スクリプトは両方に対応しています。元の重みに画像認識用の重みも含まれており、別モデルの取得は不要です。ユーザー画像とツール結果内のスクリーンショットをbase64で送れます。[画像入力の設定](h200/README.md#image-input)。上記の初回公開時の速度計測は画像機能を無効にした構成の結果です。

## 品質と適用範囲

固定した同じ300問、thinking off、temperature=0で、1017版はMMLU **171/200（85.5%）**、GSM8K **96/100（96%）**。旧版は175/200、97/100で、この標本ではそれぞれ−2.0 / −1.0ポイントでした。モデルと量子化配布元が両方変わっており、原因は分離できません。

検証した日本語・秘密句・ツール往復が正しかったことは、任意の1M入力での精度保証ではありません。visionはこの公開版の検証範囲に含めていません。

## ライセンス

TensorFold由来のコードは [Apache-2.0](LICENSE)。既存の [NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、ベンダーごとのライセンスを保持しています。1017公式encoderはMITです。モデル重みは配布元のライセンスに従って取得してください。
