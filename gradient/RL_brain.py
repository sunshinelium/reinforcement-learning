import numpy as np
import tensorflow as tf

# 如果一个回合的动作获得的 reward 越大，那么 v 的作用就可以保证

class PolicyGradient:

    def __init__(self, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        pass

    # 选择行为
    def choose_action(self, observation):
        pass

    # 存储回合 transition
    def store_transition(self, s, a, r):
        pass

    # 学习参数更新
    def learn(self, s, a, r, s_):
        pass

    # 衰减回合的 reward
    def _discount_and_norm_rewards(self):
        pass
