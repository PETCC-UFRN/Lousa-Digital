import cv2
import numpy as np
import tkinter as tk
from collections import OrderedDict
import numpy as np


def desenhar_quadrado_contorno(img):
    """
    Desenha um CONTORNO de quadrado de 300x300 pixels no centro da imagem.
    Retorna os limites da Bounding Box [x1, y1, x2, y2].
    """
    altura, largura, _ = img.shape
    TAMANHO_QUADRADO = 300
    ESPESSURA_CONTORNO = 8
    COR_CONTORNO = (255, 255, 255)  # Branco

    if largura < TAMANHO_QUADRADO or altura < TAMANHO_QUADRADO:
        return [0, 0, 0, 0]

    centro_x = largura // 2
    centro_y = altura // 2

    x1 = centro_x - (TAMANHO_QUADRADO // 2)
    y1 = centro_y - (TAMANHO_QUADRADO // 2)
    x2 = centro_x + (TAMANHO_QUADRADO // 2)
    y2 = centro_y + (TAMANHO_QUADRADO // 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), COR_CONTORNO, ESPESSURA_CONTORNO)

    return [x1, y1, x2, y2]


def calcular_similaridade(mascara_alvo, mascara_desenho, limite_minimo=0.75):
    """
    Calcula a similaridade usando a Máscara de Interseção (AND Bitwise).
    Mede o quão bem o desenho do jogador coincide com o contorno da forma.

    :param mascara_alvo: Máscara binária (255) do quadrado alvo.
    :param mascara_desenho: Máscara binária (255) da linha desenhada pelo jogador.
    :param limite_minimo: Hiperparâmetro que define o limite mínimo para se ganhar o jogo.
    """

    if np.sum(mascara_desenho) == 0:
        return 0.0, False

    # Área de acerto = máscara de intersecção entre o desenho e a forma plotada
    mascara_intersecao = cv2.bitwise_and(mascara_desenho, mascara_alvo)

    pixels_intersecao = np.sum(mascara_intersecao) / 255

    pixels_desenho_total = np.sum(mascara_desenho) / 255

    if pixels_desenho_total == 0:
        return 0.0, False

    # Similaridade = razão entre os pixels da área de acerto e os pixels do desenho do jogador
    similaridade = pixels_intersecao / pixels_desenho_total

    atingiu_limite = similaridade >= limite_minimo

    return similaridade, atingiu_limite


def salvar_nome_jogador():
    def submit():
        nonlocal entered_text
        entered_text = entry.get()
        root.destroy()

    entered_text = None
    root = tk.Tk()
    root.title("Salvar pontuação")

    tk.Label(root, text="Insira o nome do jogador:").pack()
    entry = tk.Entry(root, width=40)
    entry.pack()
    tk.Button(root, text="Salvar", command=submit).pack()

    root.mainloop()
    return entered_text


def salvar_pontuacao(nome_jogador, pontuacao):
    filename = "ranking.txt"
    ranking = carrega_ranking()
    if nome_jogador in ranking:
        if pontuacao > ranking[nome_jogador]:
            ranking[nome_jogador] = pontuacao
    else:
        ranking[nome_jogador] = pontuacao
    ranking_ordenado = dict(
        sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    )

    try:
        with open(filename, "w") as f:
            for jogador, score in ranking_ordenado.items():
                f.write(f"{jogador}:={score:.4f}\n")
    except Exception as e:
        print(f"Erro ao salvar a pontuação: {e}")


def carrega_ranking():
    filename = "ranking.txt"
    ranking = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                if ":=" in line:
                    nome_jogador = line[: line.find(":=")].strip()
                    pontuacao_str = line[line.find(":=") + 2 :].strip()

                    try:
                        pontuacao_float = float(pontuacao_str)
                        ranking[nome_jogador] = pontuacao_float
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Erro ao carregar o ranking: {e}")

    ranking_ordenado = dict(
        sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    )

    return ranking_ordenado
