# 5-5 制御ループに組み込む：推定でボールを追いかける

## ねらい

- 推定した角度・距離から「どっちに動くか」を決める **回りこみ** の動きを作れる
- オムニホイール 4 輪の速さを計算し、モータ側のマイコンに送れる
- シミュレータで、**推定の良さが試合の動きにどう効くか** を確かめられる

---

## 理論 1：ボールにまっすぐ向かうとだめな理由

ボールの方向にまっすぐ進むと、ボールが横や後ろにあるとき、ボールに **横からぶつかって** はじいてしまいます。キャッチャーは正面にしかないので、まず **ボールの後ろに回りこみ**、正面に来てから近づく必要があります。

教材の回りこみは、こう決めます。

| ボールの位置 | 進む向き |
|---|---|
| 保持している | 相手ゴールの方向 |
| 正面（±15° 以内） | そのままボールの方向 |
| それ以外 | ボールの方向より **外側** にずらす。近いほど大きくずらす |

ずらす量は「ボールの角度の大きさ（最大 90°）× 近さ」です。

$$
\text{進む向き} = a + \operatorname{sign}(a) \times \min(|a|, 90^\circ) \times \operatorname{clip}\!\left(\frac{40}{\text{距離}},\ 0.3,\ 1\right)
$$

## 理論 2：オムニホイールの速さ

オムニホイールの 4 輪の機体は、向きを変えずにどの方向にも動けます。タイヤ $i$ が、正面から反時計回りに $p_i$ の位置にあるとき、向き $\theta$ に速さ $v$ で進むには、

$$
m_i = v \sin(\theta - p_i)
$$

の速さでタイヤを回します（タイヤの「正の向き」を、上から見て反時計回りにしたとき）。4 輪が 45°, 135°, 225°, 315° にあるなら、正面（$\theta = 0$）に進むときは $(-0.71, -0.71, +0.71, +0.71) \times v$ です。教材で計算して、4 輪の力を足した向きが $\theta$ になることを確かめています。

## 理論 3：止まる仕組みを先に作る

モータを動かすプログラムは、**止まる仕組み** を先に作ります。

- センサからのデータが 100 ms 来なければ止まる（`ball_chase`）
- モータへの指令が 200 ms 来なければ止まる（`motor_receiver`）
- 推定の「自信」が小さいとき（ボールが見えないかもしれない）は動かない
- 画面をタッチするまで動き出さない

---

## 手を動かす 1：シミュレータで動きを見る

教材の 2 次元コートのシミュレータ（`rcjml.sim.field`）で、まっすぐ向かう動きと、回りこむ動きを比べます。ここでは、ロボットは本当のボールの位置を知っている（推定のまちがいがない）とします。

```python
import numpy as np
import matplotlib.pyplot as plt
from rcjml.sim.field import (chase_rule, success_rate, run_episode, MAX_SPEED,
                             FIELD_W, FIELD_H, GOAL_W)

def chase(obs):
    return chase_rule(obs["angle_deg"], obs["distance_cm"], obs["holding"], obs["goal_angle_deg"])

def straight(obs):
    if obs["holding"]:
        return obs["goal_angle_deg"], MAX_SPEED
    return obs["angle_deg"], MAX_SPEED

for name, policy in [("straight", straight), ("chase (go around)", chase)]:
    r = success_rate(policy, n=300)
    print(f"{name:18s} ゴール {r['goal_rate']:.0%}  外に出した {r['out_rate']:.0%}  "
          f"ゴールまで {r['mean_steps_to_goal'] * 0.05:.2f} 秒")
```

| 動き方 | ゴールできた | 外に出した |
|---|---|---|
| まっすぐ向かう | 31% | 69% |
| 回りこむ | **100%** | 0% |

1 本の動きを描いてみます。

```python
def draw(ep, ax, title):
    path = np.array([p[0] for p in ep.path])
    balls = np.array([p[1] for p in ep.path])
    ax.plot(path[:, 0], path[:, 1], color="tab:blue", label="robot")
    ax.plot(balls[:, 0], balls[:, 1], color="tab:orange", label="ball")
    ax.scatter(*path[0], color="tab:blue")
    ax.scatter(*balls[0], color="tab:orange")
    ax.plot([-GOAL_W / 2, GOAL_W / 2], [FIELD_H / 2] * 2, color="gold", lw=6)
    ax.set_xlim(-FIELD_W / 2, FIELD_W / 2)
    ax.set_ylim(-FIELD_H / 2, FIELD_H / 2 + 5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
for ax, (name, policy) in zip(axes, [("straight", straight), ("chase", chase)]):
    ep = run_episode(policy, np.random.default_rng(3))
    draw(ep, ax, f"{name}: {'GOAL' if ep.goal else 'out' if ep.out else 'time up'}")
axes[0].legend(fontsize=8)
plt.show()
```

## 手を動かす 2：推定の良さで、動きはどう変わるか

次に、ロボットが **IR センサの値から推定した** 角度・距離で動くようにします。比べるのは、

- まちがいのない情報（さっきと同じ）
- 従来の方法（重み付きベクトル和、距離はわからないので 40 cm と仮定）
- BallNet（5-3 で学習したもの。距離と保持も推定）

です。

```python
from rcjml.sim import simulate, sample_session, baseline_angle
from rcjml.sim.field import rel_ball, goal_angle
from rcjml.export import layers_from_keras, forward
from rcjml.nn import decode

layers = layers_from_keras(model)

def make_observe(kind, difficulty="normal", seed=1):
    rng = np.random.default_rng(seed)
    session = sample_session(difficulty, rng)
    def observe(ep):
        a, d = rel_ball(ep.robot, ep.ball)
        v = simulate(cfg, a, d, difficulty=difficulty, session=session, rng=rng)
        if kind == "vector sum":
            est = {"angle_deg": float(baseline_angle(cfg, v)[0]), "distance_cm": 40.0, "holding": ep.holding}
        else:
            r = decode(forward(layers, v / cfg.ir.max_reading))
            est = {"angle_deg": float(r["angle_deg"][0]), "distance_cm": float(r["distance_cm"][0]),
                   "holding": bool(r["p_hold"][0] > 0.5)}
        est["goal_angle_deg"] = goal_angle(ep.robot)      # ゴールの方向はわかるとする（第6部でカメラから）
        return est
    return observe

for name, obs in [("perfect", None), ("vector sum", make_observe("vector sum")), ("BallNet", make_observe("BallNet"))]:
    r = success_rate(chase, n=300, observe=obs)
    print(f"{name:11s} ゴール {r['goal_rate']:.0%}  ゴールまで {r['mean_steps_to_goal'] * 0.05:.2f} 秒")
```

| ロボットが見ているもの | ゴールできた | ゴールまでの時間（平均） |
|---|---|---|
| まちがいのない情報 | 100% | 2.29 秒 |
| 従来の方法（角度だけ） | 100% | 2.42 秒 |
| BallNet（角度・距離・保持） | 100% | 2.32 秒 |

どれもゴールはできますが、**推定が良いほど早く** ゴールできます。BallNet は、まちがいのない情報との差が 0.03 秒しかありません。試合では、この差が「相手より先にボールに触れるか」を決めます。

!!! example "やってみよう"
    1. `make_observe(..., difficulty="hard")` にすると、時間の差はどう変わるでしょう。
    2. `chase_rule` の `k_around`（回りこみの強さ）を 0.5 や 1.5 にすると、どうなるでしょう。

## 手を動かす 3：実機で動かす

1. モータ側のマイコンに `firmware/motor-arduino/motor_receiver/motor_receiver.ino` を書き込みます。ピンとモータの向き（`SIGN`）を自分の機体に合わせます
2. CoreS3 に `firmware/cores3/ball_chase/ball_chase.ino` を書き込みます（5-3 と同じく、モデルとヘッダを書き出してから）。タイヤの位置（`WHEEL_DEG`）を自分の機体に合わせます
3. 配線：センサ側 → CoreS3 の Port C、CoreS3 の Port B → モータ側

!!! warning "安全に試す"
    1. **タイヤを床から浮かせて** 電源を入れます
    2. 画面をタッチして動かし、ボールを正面に置いたとき、**4 輪が正面に進む向きに回る** か確かめます。逆に回るタイヤは、`motor_receiver` の `SIGN` を -1 にします
    3. ボールを左に置いて、左に進む向きに回るか確かめます
    4. 最初は `MAX_SPEED` を小さく（600 → 300 など）して、床に置きます
    5. いつでも止められるように、画面をタッチすれば止まること、電源をすぐ切れることを確かめておきます

---

## チェック問題

1. ボールが真後ろ（180°）、20 cm にあります。まっすぐボールに向かうと、何が起きそうですか。
2. 正面に進みたいとき、4 輪（45°, 135°, 225°, 315°）の速さはどうなりますか（$v = 1$ として）。
3. センサからのデータが来なくなったら止まる仕組みは、なぜ必要ですか。

??? note "答え"
    1. 後ろ向きに進んで、ボールにロボットの後ろ側からぶつかり、はじいてしまいます。キャッチャーは正面にしかないので、捕まえられません。
    2. $\sin(0 - 45°), \sin(0 - 135°), \sin(0 - 225°), \sin(0 - 315°)$ ＝ $(-0.71, -0.71, +0.71, +0.71)$ です。
    3. 配線が外れたり、センサ側のマイコンが止まったりしたとき、最後の指令のまま走り続けてしまうと、壁にぶつかったり、人にけがをさせたりするからです。

## つまずきポイント

??? failure "ボールと反対に走っていく"
    角度の向き（左が ＋）が、どこかで逆になっています。5-3 の画面で、ボールを左に置いて矢印が左を向くか確かめます。矢印が正しいなら、タイヤの位置（`WHEEL_DEG`）かモータの向き（`SIGN`）が逆です。

??? failure "その場でくるくる回ってしまう"
    4 輪の速さの符号がばらばらです。1 輪ずつ、正の値を送ったときに「上から見て反時計回り」に回るように、`SIGN` をそろえます。

## もっと深く

!!! deeper "機体の向きを保つ、ボールを予測する"
    このシミュレータのロボットは、いつも相手ゴールの方を向いていました。実機では、ぶつかったりして向きがずれるので、**ジャイロ**（CoreS3 に入っています）で向きを測り、ずれた分だけ回転（`omni4` の `rotation`）を足して、向きを保ちます（**PID 制御** がよく使われます）。
    また、ボールは転がるので、「今の位置」ではなく「少し先の位置」に向かうと、より速く捕まえられます。推定した角度・距離を時間の順に並べて、ボールの速さを求める方法は、第7部（7-4）の「追跡」と同じ考え方です。
