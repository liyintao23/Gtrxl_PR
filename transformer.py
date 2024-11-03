import numpy as np
import tensorflow as tf
"""transformer.py
本文件实现了 trabsformer的encoder和decoder。
"""

def sequence_mask(X, valid_len, value=0):
    """对输入序列做mask
    例子：
        X = [[1,2,3,4,5,6,7],
            [8,9,10,11,12,13,14]]
        valid_len = [5,5]

        Xmask = sequence_mask(X,valid_len)
        Xmask = [[1,2,3,4,5,0,0],
                [8,9,10,11,12,0,0]]

    """
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen,
                    dtype=tf.float32)[None, :] < tf.cast(
                        valid_len[:, None], dtype=tf.float32)

    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:
        return tf.where(mask, X, value)

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上mask某些无效元素来执行softmax操作
    """
    #如果不需要给到valid_lens ，则是None，从而使用正常的softmax
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])

        else:
            validimport numpy as np
import tensorflow as tf
"""transformer.py
本文件实现了 trabsformer的encoder和decoder。
"""

def sequence_mask(X, valid_len, value=0):
    """对输入序列做mask
    例子：
        X = [[1,2,3,4,5,6,7],
            [8,9,10,11,12,13,14]]
        valid_len = [5,5]

        Xmask = sequence_mask(X,valid_len)
        Xmask = [[1,2,3,4,5,0,0],
                [8,9,10,11,12,0,0]]

    """
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen,
                    dtype=tf.float32)[None, :] < tf.cast(
                        valid_len[:, None], dtype=tf.float32)

    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:
        return tf.where(mask, X, value)

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上mask某些无效元素来执行softmax操作
    """
    #如果不需要给到valid_lens ，则是None，从而使用正常的softmax
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])

        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # 最后一轴上被maks的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(tf.reshape(X, shape=(-1, shape[-1])),
                              valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)

class DotProductAttention(tf.keras.layers.Layer):
    """点积注意力，Self Attention 文献提到的
    """
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
    def call(self, queries, keys, values, valid_lens,**kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状
    是MultiHeadAttention的辅助函数
    """
    X = tf.reshape(X, shape=(tf.shape(X)[0], tf.shape(X)[1], num_heads, -1))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(-1, tf.shape(X)[2], tf.shape(X)[3]))


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作
    是MultiHeadAttention的辅助函数
    """
    X = tf.reshape(X, shape=(-1, num_heads, tf.shape(X)[1], tf.shape(X)[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.reshape(X, shape=(tf.shape(X)[0], tf.shape(X)[1], -1))
    return X

class MultiHeadAttention(tf.keras.layers.Layer):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)

    def call(self, queries, keys, values, valid_lens ,**kwargs):

        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

        output = self.attention(queries, keys, values, valid_lens ,**kwargs)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class PositionalEncoding(tf.keras.layers.Layer):
    """位置编码
    trabsformer encoder/decoder 需要考虑到输入tensor的位置信息
    故，需要用不同的位置编码来区分开。
    用于transformer encoder/decoder block
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 创建一个足够长的P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)

class PositionWiseFFN(tf.keras.layers.Layer):
    """基于位置的前馈网络
    用于transformer encoder/decoder block
    """
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)
    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(tf.keras.layers.Layer):
    """残差连接后进行层规范化
    用于transformer encoder/decoder block
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(normalized_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)

class EncoderBlock(tf.keras.layers.Layer):
    """transformer编码器块
    transformer encdoer block是一个简单的结构，流程如下
    1. 首先对输入的embedding 进行多头注意力计算
    2. 使用skip结构的add norm 对输出进行归一化
    3. 利用ffn 经过一次线性层
    4. 如同 2。

    这样就保持了和输入一样的形状
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                                num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X,valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X,valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)

class TransformerEncoder(tf.keras.layers.Layer):
    """transformer编码器
    1.首先对输入进行位置编码
    2.利用多层encoderblock 进行计算encoder的结果

    Tips:
        self.blks是堆叠多层EncoderBlock （毕竟EncoderBlock对输入保持不变，
        所以可以“无限”堆叠）

    """
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [EncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_layers)]

    def call(self, X, valid_lens,**kwargs):
        X = self.pos_encoding(X * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens,**kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(tf.keras.layers.Layer):
    """transformer decoder中第i个块
    解码器和编码器的区别除了结构上之外，forward计算流程也不同。
    每个DecoderBlock结构的差异在于会经过两次MultiHeadAttention
    计算流程的不同在于，计算过程不但要考虑解码器的输入，还要考虑编码器的输出。

    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        dec_valid_lens = None


        # 第一个层的多头注意力，首先计算decoder的输入做自注意力
        X2 = self.attention1(X, key_values, key_values,dec_valid_lens, **kwargs)
        # 通过add norm 得到第一层的输出
        Y = self.addnorm1(X, X2, **kwargs)
        # 第二层的多头注意力，会联合 第一层的输出Y和 编码器的输出做编码器-解码器联合注意力
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens,**kwargs)
        # 再经过一层addnorm
        Z = self.addnorm2(Y, Y2, **kwargs)
        # 同上
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state


class TransformerDecoder(tf.keras.layers.Layer):
    """TransformerDecoder
    到这里和TransformerEncoder什么区别
    1.首先对输入进行位置编码
    2.然后经过多层的DecoderBlock
    Tips:
        如TransformerEncoder一样，多层的DecoderBlock会保持输入的大小。
    """
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hidens, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                  num_hiddens, num_heads, dropout, i) for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(X* tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]  # 解码器中2个注意力层
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


# 这两个是import到meta_seq2seq_policy.py 使用
def transformer_encoder(encoder_embeddings,encoder_valid_lens=None,qkv_size=128,num_head=8,num_layer=2,dropout=0.5):
    """encoder_embeddings shape same as tf.placeholder(dtype=tf.float32,shape=[100,20,128])
    """
    encoder = TransformerEncoder(qkv_size, qkv_size, qkv_size, qkv_size, [1, 2], qkv_size, num_head, num_layer, dropout)
    encoder_outputs=encoder(encoder_embeddings,encoder_valid_lens)
    encoder_state = tf.math.reduce_max(encoder_outputs, axis=1)
    return encoder_outputs,encoder_state

def transformer_decoder(decoder_embedding,enc_outputs,encoder_valid_lens=None,vocab_size=2,qkv_size=128,num_head=8,num_layer=2,dropout=0.5):
    """decoder_embedding shape same as tf.placeholder(dtype=tf.float32,shape=[100,20,128])
    """
    transformer_decoder = TransformerDecoder(vocab_size,qkv_size,qkv_size,qkv_size,qkv_size,[1,2],qkv_size,num_head,num_layer,dropout)
    state = transformer_decoder.init_state(enc_outputs,encoder_valid_lens)
    decoder_output , state = transformer_decoder(decoder_embedding,state)
    return decoder_output , state_lens = tf.reshape(valid_lens, shape=-1)
        # 最后一轴上被maks的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(tf.reshape(X, shape=(-1, shape[-1])),
                              valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)

class DotProductAttention(tf.keras.layers.Layer):
    """点积注意力，Self Attention 文献提到的
    """
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
    def call(self, queries, keys, values, valid_lens,**kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状
    是MultiHeadAttention的辅助函数
    """
    X = tf.reshape(X, shape=(tf.shape(X)[0], tf.shape(X)[1], num_heads, -1))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(-1, tf.shape(X)[2], tf.shape(X)[3]))


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作
    是MultiHeadAttention的辅助函数
    """
    X = tf.reshape(X, shape=(-1, num_heads, tf.shape(X)[1], tf.shape(X)[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.reshape(X, shape=(tf.shape(X)[0], tf.shape(X)[1], -1))
    return X

class MultiHeadAttention(tf.keras.layers.Layer):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)

    def call(self, queries, keys, values, valid_lens ,**kwargs):

        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

        output = self.attention(queries, keys, values, valid_lens ,**kwargs)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class PositionalEncoding(tf.keras.layers.Layer):
    """位置编码
    trabsformer encoder/decoder 需要考虑到输入tensor的位置信息
    故，需要用不同的位置编码来区分开。
    用于transformer encoder/decoder block
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 创建一个足够长的P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)

class PositionWiseFFN(tf.keras.layers.Layer):
    """基于位置的前馈网络
    用于transformer encoder/decoder block
    """
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)
    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(tf.keras.layers.Layer):
    """残差连接后进行层规范化
    用于transformer encoder/decoder block
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(normalized_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)

class EncoderBlock(tf.keras.layers.Layer):
    """transformer编码器块
    transformer encdoer block是一个简单的结构，流程如下
    1. 首先对输入的embedding 进行多头注意力计算
    2. 使用skip结构的add norm 对输出进行归一化
    3. 利用ffn 经过一次线性层
    4. 如同 2。

    这样就保持了和输入一样的形状
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                                num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X,valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X,valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)

class TransformerEncoder(tf.keras.layers.Layer):
    """transformer编码器
    1.首先对输入进行位置编码
    2.利用多层encoderblock 进行计算encoder的结果
    
    Tips:
        self.blks是堆叠多层EncoderBlock （毕竟EncoderBlock对输入保持不变，
        所以可以“无限”堆叠）

    """
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [EncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_layers)]

    def call(self, X, valid_lens,**kwargs):
        X = self.pos_encoding(X * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens,**kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(tf.keras.layers.Layer):
    """transformer decoder中第i个块
    解码器和编码器的区别除了结构上之外，forward计算流程也不同。
    每个DecoderBlock结构的差异在于会经过两次MultiHeadAttention
    计算流程的不同在于，计算过程不但要考虑解码器的输入，还要考虑编码器的输出。

    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        
        dec_valid_lens = None
        

        # 第一个层的多头注意力，首先计算decoder的输入做自注意力
        X2 = self.attention1(X, key_values, key_values,dec_valid_lens, **kwargs)
        # 通过add norm 得到第一层的输出
        Y = self.addnorm1(X, X2, **kwargs)
        # 第二层的多头注意力，会联合 第一层的输出Y和 编码器的输出做编码器-解码器联合注意力
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens,**kwargs)
        # 再经过一层addnorm
        Z = self.addnorm2(Y, Y2, **kwargs)
        # 同上
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state


class TransformerDecoder(tf.keras.layers.Layer):
    """TransformerDecoder
    到这里和TransformerEncoder什么区别
    1.首先对输入进行位置编码
    2.然后经过多层的DecoderBlock
    Tips:
        如TransformerEncoder一样，多层的DecoderBlock会保持输入的大小。
    """
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hidens, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                  num_hiddens, num_heads, dropout, i) for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(X* tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]  # 解码器中2个注意力层
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


# 这两个是import到meta_seq2seq_policy.py 使用
def transformer_encoder(encoder_embeddings,encoder_valid_lens=None,qkv_size=128,num_head=8,num_layer=2,dropout=0.5):
    """encoder_embeddings shape same as tf.placeholder(dtype=tf.float32,shape=[100,20,128])
    """
    encoder = TransformerEncoder(qkv_size, qkv_size, qkv_size, qkv_size, [1, 2], qkv_size, num_head, num_layer, dropout)
    encoder_outputs=encoder(encoder_embeddings,encoder_valid_lens)
    encoder_state = tf.math.reduce_max(encoder_outputs, axis=1)
    return encoder_outputs,encoder_state

def transformer_decoder(decoder_embedding,enc_outputs,encoder_valid_lens=None,vocab_size=2,qkv_size=128,num_head=8,num_layer=2,dropout=0.5):
    """decoder_embedding shape same as tf.placeholder(dtype=tf.float32,shape=[100,20,128])
    """
    transformer_decoder = TransformerDecoder(vocab_size,qkv_size,qkv_size,qkv_size,qkv_size,[1,2],qkv_size,num_head,num_layer,dropout)
    state = transformer_decoder.init_state(enc_outputs,encoder_valid_lens)
    decoder_output , state = transformer_decoder(decoder_embedding,state)
    return decoder_output , state