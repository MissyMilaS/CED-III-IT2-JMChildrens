"""Definición de arquitecturas para clasificación de sentimiento (PyTorch).

Contiene: DenseNet, RecurrentNet (RNN vanilla, LSTM y LSTM bidireccional) y build_model.
Todas reciben (x, lengths) y devuelven UN logit por comentario (un número real sin acotar).
La sigmoide se aplica fuera: en la pérdida BCEWithLogitsLoss (entrenamiento) y en
predict_proba (predicción). logit > 0  <=>  probabilidad > 0.5  <=>  clase positiva.

NOTA PARA JORGE: la extensión Transformer (DistilBERT/BERT) puede agregarse al
final de este archivo, o en un archivo aparte, sin tocar lo de arriba.
"""
import torch  # librería principal de deep learning
from torch import nn  # módulos de red neuronal (Embedding, Linear, LSTM, ...)
from torch.nn.utils.rnn import pack_padded_sequence  # para ignorar el padding en RNN/LSTM

PAD, UNK = 0, 1  # índices reservados: 0 = relleno (padding), 1 = palabra desconocida


class DenseNet(nn.Module):  # clase que define la red Dense; hereda de nn.Module
    """Embedding -> promedio de embeddings (ignora padding) -> Dense(ReLU) -> Dropout -> salida.

    No tiene memoria ni orden: es una "bolsa de palabras" con embeddings.
    """

    def __init__(self, vocab_size, emb_dim=64, hidden=64, dropout=0.5):  # hiperparámetros de la red
        super().__init__()  # inicializa la clase padre nn.Module (obligatorio)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)  # tabla: id de palabra -> vector de emb_dim números; el vector de PAD se queda en ceros y no se entrena
        self.net = nn.Sequential(  # encadena capas: la salida de una es la entrada de la siguiente
            nn.Linear(emb_dim, hidden),  # capa densa: emb_dim entradas -> `hidden` neuronas
            nn.ReLU(),  # activación no lineal: max(0, x)
            nn.Dropout(dropout),  # apaga al azar una fracción `dropout` de neuronas al entrenar (reduce sobreajuste)
            nn.Linear(hidden, 1),  # capa de salida: `hidden` neuronas -> 1 logit
        )

    def forward(self, x, lengths=None):  # paso hacia adelante; `lengths` se acepta solo para tener la misma firma que las RNN
        e = self.emb(x)  # ids (B, T) -> vectores (B, T, E); B = comentarios por lote, T = largo máximo
        mask = (x != PAD).unsqueeze(-1).float()  # (B, T, 1): 1.0 donde hay palabra real, 0.0 donde hay padding
        pooled = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # promedio de los vectores solo de palabras reales; clamp evita dividir por 0
        return self.net(pooled).squeeze(-1)  # pasa el promedio por las capas densas y quita la dimensión final: (B, 1) -> (B,)


class RecurrentNet(nn.Module):  # una sola clase sirve para RNN, LSTM y BiLSTM
    """Embedding -> RNN vanilla (tanh) o LSTM (uni o bidireccional) -> estado final -> Dropout -> salida.

    Usa pack_padded_sequence para que el padding no afecte el estado final.
    """

    def __init__(self, cell, vocab_size, emb_dim=64, hidden=64, dropout=0.5,
                 bidirectional=False):  # cell = tipo de celda; bidirectional = leer también de derecha a izquierda
        super().__init__()  # inicializa nn.Module
        if cell not in ("rnn", "lstm"):  # valida el tipo de celda pedido
            raise ValueError("cell debe ser 'rnn' o 'lstm'")  # error claro si se pide otra cosa
        self.cell = cell  # guarda el tipo para usarlo en forward
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)  # igual que en DenseNet: id -> vector
        rnn_cls = nn.RNN if cell == "rnn" else nn.LSTM  # elige la clase: RNN vanilla (tanh) o LSTM (con compuertas)
        self.rnn = rnn_cls(emb_dim, hidden, batch_first=True,
                           bidirectional=bidirectional)  # batch_first: los tensores son (B, T, E); bidirectional=True crea dos pasadas: izq->der y der->izq
        self.drop = nn.Dropout(dropout)  # dropout aplicado al estado final antes de clasificar
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), 1)  # entrada = hidden, o 2*hidden si hay dos direcciones concatenadas; salida = 1 logit

    def forward(self, x, lengths):  # x: ids (B, T); lengths: largo real de cada comentario (B,)
        e = self.emb(x)  # ids -> vectores: (B, T, E)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)  # "empaqueta" la secuencia para que la RNN se detenga en el largo real de cada comentario (lengths debe estar en CPU); enforce_sorted=False permite lotes sin ordenar por largo
        _, h = self.rnn(packed)  # ejecuta la RNN/LSTM; descartamos las salidas por paso (_) y nos quedamos con el estado final h
        if self.cell == "lstm":  # la LSTM devuelve una tupla (h_n, c_n): estado oculto y estado de celda
            h = h[0]  # nos quedamos solo con h_n (el estado oculto final)
        h = h.transpose(0, 1).reshape(h.size(1), -1)  # h es (direcciones, B, hidden) -> (B, direcciones*hidden); en BiLSTM une el estado final de ambas direcciones
        return self.out(self.drop(h)).squeeze(-1)  # dropout -> capa lineal -> logit; squeeze: (B, 1) -> (B,)


def build_model(name, vocab_size, **kw):  # fábrica: crea el modelo a partir de su nombre
    """name: 'dense' | 'vanilla_rnn' | 'lstm' | 'bilstm'. kw: emb_dim, hidden, dropout."""
    if name == "dense":  # red densa sobre promedio de embeddings
        kw.pop("bidirectional", None)  # DenseNet no tiene este parámetro; se descarta si llegó
        return DenseNet(vocab_size, **kw)  # construye la Dense con los hiperparámetros recibidos
    if name == "vanilla_rnn":  # RNN simple
        return RecurrentNet("rnn", vocab_size, **kw)  # celda tanh, una dirección
    if name == "lstm":  # LSTM unidireccional (izquierda -> derecha)
        return RecurrentNet("lstm", vocab_size, **kw)  # celda LSTM, una dirección
    if name == "bilstm":  # LSTM bidireccional: modelo extra del equipo
        return RecurrentNet("lstm", vocab_size, bidirectional=True, **kw)  # celda LSTM, lee en ambos sentidos
    raise ValueError(f"Modelo desconocido: {name}")  # nombre no reconocido


def count_params(model):  # cuenta cuántos números entrenables tiene el modelo (complejidad)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # suma el tamaño de cada tensor entrenable