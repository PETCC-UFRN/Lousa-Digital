import cv2
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


def salvar_pontuacao(similaridade):
    filename = "ranking.txt"
    ranking = carrega_ranking()
    ranking.append(similaridade)
    ranking = sorted(ranking, reverse=True)
    try:
        with open(filename, "w") as f:
            for item in ranking:
                f.write(f"{item}\n")
    except Exception as e:
        print(f"Erro ao salvar a pontuação: {e}")


def carrega_ranking():
    filename = "ranking.txt"
    ranking = []
    try:
        with open(filename, "r") as f:
            for line in f:
                ranking.append(float(line.strip()))
    except Exception as e:
        print(f"Erro ao cerregar o ranking: {e}")
    return ranking
