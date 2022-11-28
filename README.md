# dependencies (maybe)
```
pip install gym==0.21.0
pip install stable-baselines3
pip install pyglet==1.5.27
```

# PlayGround
強化学習を活用することをめざして色々遊ぶ。
アルゴリズムなどはstable-baselines3を活用予定。
実際に強化学習を使う場合のことを考えると、自分の解きたい問題をどのようにenvに落とし込むかの方が重要。
(アルゴリズムはライブラリが多く存在する)
envをどのようにいじるか、報酬設計をどのように考えてやるか、色々やってみる

* 強化学習全体の枠組みの説明
* envの構成のために作成するクラス、関数の説明

## Chapter1: stable-baselines3のチュートリアルを動かす
(こちらのリンク)[https://github.com/DLR-RM/stable-baselines3]にあるExampleのコードを `tutorial_code.py` にコピーして動かしてみる

## Chapter2: チュートリアルを変更してベースとなる学習環境を作る
* 環境をPendulumに変更する
  + Pendulumの説明
* モデルを保存できるようにする
* 学習成果を保存できるようにする
  + Monitor, env Wrapperの説明

`base_learn.py` に学習してログ保存及びモデル保存するコード
`vis_log.py` に学習したログを可視化するコード
`base_evaluate.py` に学習したモデルを評価してrenderするコード

## Chapter3: 環境への入力を離散化してみる
`discrete_learn.py` に学習コード
`discrete_evaluate.py` に評価コード

* n = 2でも結構学習する
* n = 2の時の分布

## Capter3: PendulumのReward
* 速度制限を取っ払う
* action制限を取っ払う
* 制限の係数を強くしてみる

## Chapte4: Pendulumの環境
* mを大きくしてみる
* 重力を大きくしてみる
