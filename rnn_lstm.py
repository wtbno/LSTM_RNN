import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

st.set_page_config(
    page_title="Laboratório RNN vs LSTM",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Laboratório visual: RNN vs. LSTM")
st.markdown("""
Este aplicativo treina uma **RNN simples** e uma **LSTM** para prever o próximo valor
de uma sequência. Você pode acompanhar o treinamento em tempo real e inspecionar
como cada modelo usa a memória.

**Dataset:** 25.000 exemplos gerados artificialmente, com padrão temporal e ruído.
""")

# -----------------------------
# Configurações
# -----------------------------
st.sidebar.header("⚙️ Configurações")

n_exemplos = st.sidebar.slider(
    "Quantidade de exemplos",
    min_value=1000,
    max_value=25000,
    value=25000,
    step=1000,
)

tamanho_sequencia = st.sidebar.slider(
    "Tamanho da sequência de entrada",
    min_value=5,
    max_value=40,
    value=15,
)

epocas = st.sidebar.slider(
    "Épocas",
    min_value=1,
    max_value=30,
    value=8,
)

batch_size = st.sidebar.selectbox(
    "Tamanho do batch",
    options=[32, 64, 128, 256],
    index=2,
)

hidden_size = st.sidebar.slider(
    "Neurônios na camada oculta",
    min_value=8,
    max_value=128,
    value=32,
)

learning_rate = st.sidebar.selectbox(
    "Taxa de aprendizado",
    options=[0.1, 0.05, 0.01, 0.005, 0.001],
    index=2,
)

mostrar_animacao = st.sidebar.checkbox(
    "Atualizar gráficos durante o treino",
    value=True,
)

seed = st.sidebar.number_input(
    "Semente aleatória",
    min_value=0,
    value=42,
)

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Dispositivo: **{device}**")

# -----------------------------
# Geração do dataset
# -----------------------------
st.header("1️⃣ Dataset: sequência temporal")

st.markdown("""
Cada exemplo é uma sequência de números. O modelo recebe os primeiros
valores e tenta prever o **próximo valor**.

A série tem:
- uma onda senoidal,
- uma tendência suave,
- ruído aleatório.
""")

def criar_dataset(n_exemplos, tamanho_sequencia):
    X = []
    y = []

    for i in range(n_exemplos):
        inicio = np.random.uniform(0, 20)
        frequencia = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(0.5, 1.5)
        tendencia = np.random.uniform(-0.02, 0.02)

        t = np.arange(tamanho_sequencia + 1)
        serie = (
            amplitude * np.sin(frequencia * t + inicio)
            + tendencia * t
            + np.random.normal(0, 0.05, size=len(t))
        )

        X.append(serie[:-1])
        y.append(serie[-1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Normalização simples
    media = X.mean()
    desvio = X.std()
    X = (X - media) / desvio
    y = (y - media) / desvio

    return X, y, media, desvio

X, y, media, desvio = criar_dataset(n_exemplos, tamanho_sequencia)

X_tensor = torch.tensor(X).unsqueeze(-1)  # [batch, seq_len, 1]
y_tensor = torch.tensor(y).unsqueeze(-1)  # [batch, 1]

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

st.write(f"**Formato de X:** {X.shape} — 25.000 exemplos, {tamanho_sequencia} passos e 1 variável por passo.")
st.write(f"**Formato de y:** {y.shape} — o próximo valor que o modelo deve prever.")

exemplo_idx = st.slider(
    "Escolha um exemplo para visualizar",
    min_value=0,
    max_value=min(1000, n_exemplos - 1),
    value=0,
)

entrada_exemplo = X[exemplo_idx]
alvo_exemplo = y[exemplo_idx]

fig_dataset = go.Figure()
fig_dataset.add_trace(
    go.Scatter(
        y=entrada_exemplo,
        mode="lines+markers",
        name="Entrada da rede",
        line=dict(color="#1f77b4"),
    )
)
fig_dataset.add_trace(
    go.Scatter(
        x=[len(entrada_exemplo)],
        y=[alvo_exemplo],
        mode="markers+text",
        name="Valor a prever",
        text=["próximo valor"],
        textposition="top center",
        marker=dict(color="red", size=12),
    )
)
fig_dataset.update_layout(
    title="Exemplo de sequência: a rede lê da esquerda para a direita",
    xaxis_title="Passo de tempo",
    yaxis_title="Valor normalizado",
    height=400,
)
st.plotly_chart(fig_dataset, use_container_width=True)

# -----------------------------
# Modelos
# -----------------------------
class RNNSimples(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        saida, h_n = self.rnn(x)
        ultimo_passo = saida[:, -1, :]
        return self.fc(ultimo_passo), saida


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        saida, (h_n, c_n) = self.lstm(x)
        ultimo_passo = saida[:, -1, :]
        return self.fc(ultimo_passo), saida

rnn_model = RNNSimples(hidden_size=hidden_size).to(device)
lstm_model = LSTMModel(hidden_size=hidden_size).to(device)

criterion = nn.MSELoss()

optimizer_rnn = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)
optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

# -----------------------------
# Treinamento
# -----------------------------
st.header("2️⃣ Treinamento em tempo real")

col1, col2, col3 = st.columns(3)

with col1:
    botao_treinar = st.button("🚀 Treinar RNN e LSTM", type="primary")

with col2:
    botao_limpar = st.button("🧹 Limpar histórico")

if botao_limpar:
    st.session_state["historico"] = {
        "epoca": [],
        "loss_rnn": [],
        "loss_lstm": [],
    }
    st.session_state["treinou"] = False

if "historico" not in st.session_state:
    st.session_state["historico"] = {
        "epoca": [],
        "loss_rnn": [],
        "loss_lstm": [],
    }
    st.session_state["treinou"] = False

placeholder_loss = st.empty()
placeholder_status = st.empty()

if botao_treinar:
    st.session_state["historico"] = {
        "epoca": [],
        "loss_rnn": [],
        "loss_lstm": [],
    }
    st.session_state["treinou"] = True

    progresso = st.progress(0, text="Iniciando treinamento...")

    for epoca in range(epocas):
        rnn_model.train()
        lstm_model.train()

        loss_total_rnn = 0.0
        loss_total_lstm = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # RNN
            optimizer_rnn.zero_grad()
            pred_rnn, _ = rnn_model(X_batch)
            loss_rnn = criterion(pred_rnn, y_batch)
            loss_rnn.backward()
            optimizer_rnn.step()

            # LSTM
            optimizer_lstm.zero_grad()
            pred_lstm, _ = lstm_model(X_batch)
            loss_lstm = criterion(pred_lstm, y_batch)
            loss_lstm.backward()
            optimizer_lstm.step()

            loss_total_rnn += loss_rnn.item()
            loss_total_lstm += loss_lstm.item()
            n_batches += 1

        loss_media_rnn = loss_total_rnn / n_batches
        loss_media_lstm = loss_total_lstm / n_batches

        st.session_state["historico"]["epoca"].append(epoca + 1)
        st.session_state["historico"]["loss_rnn"].append(loss_media_rnn)
        st.session_state["historico"]["loss_lstm"].append(loss_media_lstm)

        progresso.progress(
            (epoca + 1) / epocas,
            text=f"Época {epoca + 1}/{epocas} — Loss RNN: {loss_media_rnn:.4f} | Loss LSTM: {loss_media_lstm:.4f}",
        )

        if mostrar_animacao:
            with placeholder_loss.container():
                fig_loss = go.Figure()

                fig_loss.add_trace(
                    go.Scatter(
                        x=st.session_state["historico"]["epoca"],
                        y=st.session_state["historico"]["loss_rnn"],
                        mode="lines+markers",
                        name="RNN simples",
                        line=dict(color="#ff7f0e"),
                    )
                )

                fig_loss.add_trace(
                    go.Scatter(
                        x=st.session_state["historico"]["epoca"],
                        y=st.session_state["historico"]["loss_lstm"],
                        mode="lines+markers",
                        name="LSTM",
                        line=dict(color="#2ca02c"),
                    )
                )

                fig_loss.update_layout(
                    title="Loss durante o treinamento",
                    xaxis_title="Época",
                    yaxis_title="Erro quadrático médio",
                    height=420,
                )

                st.plotly_chart(
    fig_loss,
    use_container_width=True,
    key=f"grafico_loss_animacao_{epoca}",
)

    placeholder_status.success("Treinamento concluído!")

# Mostrar histórico mesmo sem animação
if st.session_state["treinou"]:
    with placeholder_loss.container():
        fig_loss = go.Figure()

        fig_loss.add_trace(
            go.Scatter(
                x=st.session_state["historico"]["epoca"],
                y=st.session_state["historico"]["loss_rnn"],
                mode="lines+markers",
                name="RNN simples",
                line=dict(color="#ff7f0e"),
            )
        )

        fig_loss.add_trace(
            go.Scatter(
                x=st.session_state["historico"]["epoca"],
                y=st.session_state["historico"]["loss_lstm"],
                mode="lines+markers",
                name="LSTM",
                line=dict(color="#2ca02c"),
            )
        )

        fig_loss.update_layout(
            title="Loss durante o treinamento",
            xaxis_title="Época",
            yaxis_title="Erro quadrático médio",
            height=420,
        )

        st.plotly_chart(fig_loss, use_container_width=True, key="grafico_loss_final")

# -----------------------------
# Comparação de previsões
# -----------------------------
st.header("3️⃣ RNN vs. LSTM: previsões")

if st.session_state["treinou"]:
    rnn_model.eval()
    lstm_model.eval()

    with torch.no_grad():
        X_exemplo = torch.tensor(X[exemplo_idx]).unsqueeze(0).unsqueeze(-1).to(device)

        pred_rnn, saida_rnn = rnn_model(X_exemplo)
        pred_lstm, saida_lstm = lstm_model(X_exemplo)

    pred_rnn = pred_rnn.item()
    pred_lstm = pred_lstm.item()
    alvo = y[exemplo_idx]

    col_rnn, col_lstm, col_real = st.columns(3)

    with col_rnn:
        st.metric("Previsão da RNN", f"{pred_rnn:.3f}")

    with col_lstm:
        st.metric("Previsão da LSTM", f"{pred_lstm:.3f}")

    with col_real:
        st.metric("Valor real", f"{alvo:.3f}")

    erro_rnn = abs(pred_rnn - alvo)
    erro_lstm = abs(pred_lstm - alvo)

    st.write(f"**Erro absoluto da RNN:** {erro_rnn:.4f}")
    st.write(f"**Erro absoluto da LSTM:** {erro_lstm:.4f}")

    fig_pred = go.Figure()

    fig_pred.add_trace(
        go.Scatter(
            y=X[exemplo_idx],
            mode="lines+markers",
            name="Entrada",
            line=dict(color="#1f77b4"),
        )
    )

    fig_pred.add_trace(
        go.Scatter(
            x=[len(X[exemplo_idx])],
            y=[alvo],
            mode="markers",
            name="Valor real",
            marker=dict(color="black", size=12, symbol="star"),
        )
    )

    fig_pred.add_trace(
        go.Scatter(
            x=[len(X[exemplo_idx])],
            y=[pred_rnn],
            mode="markers",
            name="Previsão RNN",
            marker=dict(color="#ff7f0e", size=12),
        )
    )

    fig_pred.add_trace(
        go.Scatter(
            x=[len(X[exemplo_idx])],
            y=[pred_lstm],
            mode="markers",
            name="Previsão LSTM",
            marker=dict(color="#2ca02c", size=12),
        )
    )

    fig_pred.update_layout(
        title="Comparação das previsões no próximo passo",
        xaxis_title="Passo de tempo",
        yaxis_title="Valor normalizado",
        height=450,
    )

    st.plotly_chart(fig_pred, use_container_width=True)

# -----------------------------
# Visualização da memória
# -----------------------------
st.header("4️⃣ Como a rede “lembra” a sequência")

if st.session_state["treinou"]:
    st.markdown("""
    Abaixo estão os **estados ocultos** ao longo do tempo.

    - Na **RNN**, o estado oculto carrega a informação de forma mais direta.
    - Na **LSTM**, a célula tem mecanismos extras, as portas, que ajudam a decidir
      o que guardar, atualizar e esquecer.
    """)

    st.subheader("Estado oculto da RNN")

    fig_h_rnn = go.Figure(
        data=go.Heatmap(
            z=saida_rnn[0].cpu().numpy().T,
            colorscale="Viridis",
        )
    )
    fig_h_rnn.update_layout(
        title="Ativações ocultas da RNN",
        xaxis_title="Passo de tempo",
        yaxis_title="Neurônio oculto",
        height=450,
    )
    st.plotly_chart(fig_h_rnn, use_container_width=True)

    st.subheader("Estado oculto da LSTM")

    fig_h_lstm = go.Figure(
        data=go.Heatmap(
            z=saida_lstm[0].cpu().numpy().T,
            colorscale="Viridis",
        )
    )
    fig_h_lstm.update_layout(
        title="Ativações ocultas da LSTM",
        xaxis_title="Passo de tempo",
        yaxis_title="Neurônio oculto",
        height=450,
    )
    st.plotly_chart(fig_h_lstm, use_container_width=True)

# -----------------------------
# Explicação visual
# -----------------------------
st.header("5️⃣ Entendendo a diferença")

col_rnn_exp, col_lstm_exp = st.columns(2)

with col_rnn_exp:
    st.markdown("""
    ### 🔁 RNN simples

    A RNN lê a sequência passo a passo:

    \\[
    h_t = \\tanh(W_x x_t + W_h h_{t-1} + b)
    \\]

    Em palavras:

    - \(x_t\) é o valor atual.
    - \(h_{t-1}\) é a memória do passo anterior.
    - \(h_t\) é a nova memória.

    O problema é que, em sequências longas, as informações antigas podem
    ficar cada vez mais fracas. Isso é o chamado **vanishing gradient**.
    """)

with col_lstm_exp:
    st.markdown("""
    ### 🚪 LSTM

    A LSTM adiciona uma **célula de memória** \(c_t\) e três portas:

    - **Forget gate:** decide o que esquecer.
    - **Input gate:** decide o que guardar agora.
    - **Output gate:** decide o que revelar como estado oculto.

    Ela é mais capaz de manter informações relevantes por mais tempo.
    """)

st.info("""
**Dica de estudo:** rode primeiro com poucas épocas e observe se a LSTM reduz o erro mais rapidamente.
Depois aumente o tamanho da sequência. Quanto maior a sequência, mais fácil enxergar a vantagem da LSTM.
""")

st.success("""
Agora você pode abrir o navegador em:

**http://localhost:8501**
""")