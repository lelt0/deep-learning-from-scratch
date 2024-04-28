# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import cv2
np.random.seed(123)
rng = np.random.default_rng()

from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer

def _change_one_hot_label(X, classes=10):
    T = np.zeros((X.size, classes))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

class CircleTrainer(Trainer):
    def show_plot_image(self, show_ms=0):
        y_test = self.network.predict(self.x_test)
        sc = plt.scatter(self.x_test[:,0], self.x_test[:,1], vmin=0, vmax=1, cmap=cm.bwr, s=25, c=np.argmax(self.t_test, 1))
        sc = plt.scatter(self.x_test[:,0], self.x_test[:,1], vmin=0, vmax=1, cmap=cm.bwr, s=15, c=np.argmax(y_test, 1))
        plt.colorbar(sc)

        # plotの画像化
        buf = io.BytesIO() # bufferを用意
        plt.savefig(buf, format='png') # bufferに保持
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
        dst = cv2.imdecode(enc, 1) # デコード
        dst = dst[:,:,::-1] # BGR->RGB
        plt.clf()

        cv2.imshow('plot', dst)
        if cv2.waitKey(show_ms) == ord('q'):
            exit()

    def train(self, epochs_per_show_plot=None):
        print()
        for i in range(self.max_iter):
            self.train_step()
            print(f'\riter: {i+1}/{self.max_iter} (loss:{self.train_loss_list[-1]:.3f})', end='', flush=True)

            if epochs_per_show_plot is not None:
                epoch = self.current_iter // self.iter_per_epoch
                iter_in_epoch = self.current_iter % self.iter_per_epoch
                if epoch % epochs_per_show_plot == 0 and iter_in_epoch == 0:
                    self.show_plot_image(1)

        print()
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        print("==== Final Test Accuracy ====")
        print(f"test acc: {test_acc:.4f}")

if __name__ == '__main__':
    teacher_func = lambda xy: 1 if (xy[0]**2 + xy[1]**2 <= 1) else 0
    # teacher_func = lambda xy: 1 if (xy[0] - xy[1] <= 0 and xy[0] + xy[1] <= 0) else 0

    # 学習用データ
    x_train = rng.uniform(-2.0, 2.0, (1000,2))
    t_train = np.array([teacher_func(xy) for xy in x_train])
    t_train = _change_one_hot_label(t_train, 2)

    # 評価用データ
    x_test = np.array([(x, y) for x in np.arange(-2.0, 2.1, 0.1) for y in np.arange(-2.0, 2.1, 0.1)])
    t_test = np.array([teacher_func(xy) for xy in x_test])
    t_test = _change_one_hot_label(t_test, 2)

    # 学習
    #network = MultiLayerNet(2, [10,10], 2, activation='pass') # 活性化関数がpassなら、層がいくつあっても線形解にしかならない 
    # network = MultiLayerNet(2, [2], 2) # これもほぼ線形解
    # network = MultiLayerNet(2, [2, 2], 2) # 層を深くしてもほぼ線形解（今回のような問題は抽象化が必要がないから？）
    # network = MultiLayerNet(2, [3], 2) # これは2つの直線で分割するような感じ
    # network = MultiLayerNet(2, [3], 2, activation='sigmoid') # sigmoidにしたら曲線で分割するようになって、400epochくらいからまあまあ正解しだした
    network = MultiLayerNet(2, [4], 2) # これはまあまあ正解
    trainer = CircleTrainer(network, x_train, t_train, x_test, t_test,
                    epochs=400, mini_batch_size=100,
                    optimizer='Adam', optimizer_param={'lr':0.001},
                    evaluate_sample_num_per_epoch=None, verbose=False)
    trainer.train(10)

    # 最終結果
    trainer.show_plot_image(1)

    # 正解率推移
    plt.plot(trainer.train_acc_list, label='train_acc')
    plt.plot(trainer.test_acc_list, label='test_acc')
    plt.legend()
    plt.show()
