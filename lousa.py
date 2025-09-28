import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import json
import os
from jogo_similaridade import desenhar_quadrado_contorno, calcular_similaridade
from datetime import datetime


class LousaDigital:
    def __init__(self):
        # Configurações da câmera
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 1280)
        self.video.set(4, 720)
        self.largura = 1280
        self.altura = 720

        # Hand tracking
        self.detector = HandDetector(detectionCon=0.8)
        self.desenho = []

        # Máscara separada para o desenho do jogador (para o cálculo de similaridade)
        self.imgCanvas = np.zeros((self.altura, self.largura, 3), np.uint8)

        # Configurações iniciais
        self.cor = (0, 0, 255)
        self.espessura = 20
        self.modo_atual = "desenho"  # desenho, apresentacao
        self.button_cooldown = 0
        self.last_position = None
        self.jogo_ativo = False

        # Cores disponíveis
        self.cores = {
            "Vermelho": (0, 0, 255),
            "Verde": (0, 255, 0),
            "Azul": (255, 0, 0),
            "Preto": (0, 0, 0),
            "Branco": (255, 255, 255),
            "Amarelo": (0, 255, 255),
            "Rosa": (255, 0, 255),
            "Ciano": (255, 255, 0),
        }

        # Botões organizados em abas
        self.setup_buttons()

    def setup_buttons(self):
        # Botões de cores (primeira linha)
        self.botoes_cores = []
        x_start = 50
        for i, (nome, cor) in enumerate(
            list(self.cores.items())[:6]
        ):  # Primeiras 6 cores
            x = x_start + i * 120
            self.botoes_cores.append([x, 20, x + 100, 60, cor, nome])

        # Botões de ferramentas (segunda linha)
        self.botoes_ferramentas = [
            [40, 150, 130, 190, (128, 128, 128), "Limpar"],
            [40, 200, 130, 240, (128, 128, 128), "Salvar"],
            [40, 250, 130, 290, (128, 128, 128), "Carregar"],
            [40, 300, 130, 340, (128, 128, 128), "+"],
            [40, 350, 130, 390, (128, 128, 128), "-"],
            [40, 400, 130, 440, (128, 128, 128), "Desfazer"],
            [20, 450, 150, 490, (0, 200, 0), "Iniciar Jogo"],
        ]

        # Combinando todos os botões
        self.botoes = self.botoes_cores + self.botoes_ferramentas

    def smooth_drawing(self, current_pos):
        if self.last_position is None:
            self.last_position = current_pos
            return [current_pos]

        points = []
        x1, y1 = self.last_position
        x2, y2 = current_pos

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distance > 5:
            steps = int(distance / 3)
            for i in range(steps + 1):
                t = i / max(steps, 1)
                x = int(x1 + (x2 - x1) * t)
                y = int(y1 + (y2 - y1) * t)
                points.append((x, y))
        else:
            points.append(current_pos)

        self.last_position = current_pos
        return points

    def salvar_desenho(self):
        if not self.desenho:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"desenho_{timestamp}.json"

        desenho_serializado = []
        for ponto in self.desenho:
            x, y, cor, espessura = ponto
            desenho_serializado.append(
                {"x": x, "y": y, "cor": cor, "espessura": espessura}
            )

        try:
            with open(filename, "w") as f:
                json.dump(desenho_serializado, f)
            print(f"Desenho salvo como {filename}")
        except Exception as e:
            print(f"Erro ao salvar: {e}")

    def carregar_desenho(self):
        arquivos = [
            f
            for f in os.listdir(".")
            if f.startswith("desenho_") and f.endswith(".json")
        ]
        if not arquivos:
            print("Nenhum desenho salvo encontrado")
            return

        arquivo_recente = max(arquivos)
        try:
            with open(arquivo_recente, "r") as f:
                desenho_carregado = json.load(f)

            self.desenho = []
            for ponto in desenho_carregado:
                self.desenho.append(
                    (ponto["x"], ponto["y"], tuple(ponto["cor"]), ponto["espessura"])
                )

            print(f"Desenho carregado: {arquivo_recente}")
        except Exception as e:
            print(f"Erro ao carregar: {e}")

    def desfazer_ultimo(self):
        if not self.desenho:
            return

        while self.desenho and self.desenho[-1][0] != 0:
            self.desenho.pop()

        if self.desenho and self.desenho[-1][0] == 0:
            self.desenho.pop()

    def processar_botoes(self, x_flip, y):
        """Processa cliques nos botões com cooldown"""
        if self.button_cooldown > 0:
            return

        for bx1, by1, bx2, by2, bcor, texto in self.botoes:
            if bx1 < x_flip < bx2 and by1 < y < by2:
                self.button_cooldown = 15

                if texto in self.cores:
                    self.cor = self.cores[texto]

                elif texto == "+":
                    self.espessura = min(self.espessura + 5, 50)
                elif texto == "-":
                    self.espessura = max(self.espessura - 5, 5)
                elif texto == "Limpar":
                    self.desenho = []
                    self.imgCanvas = np.zeros((self.altura, self.largura, 3), np.uint8)
                elif texto == "Salvar":
                    self.salvar_desenho()
                elif texto == "Carregar":
                    self.carregar_desenho()
                elif texto == "Desfazer":
                    self.desfazer_ultimo()
                    if self.jogo_ativo:
                        self.desenho = []
                        self.imgCanvas = np.zeros(
                            (self.altura, self.largura, 3), np.uint8
                        )
                elif texto == "Iniciar Jogo":
                    self.jogo_ativo = not self.jogo_ativo
                    self.desenho = []
                    self.imgCanvas = np.zeros((self.altura, self.largura, 3), np.uint8)
                    print(f"Jogo: {'ATIVO' if self.jogo_ativo else 'INATIVO'}")
                break

    def desenhar_interface(self, img):
        # Botões de cores
        for bx1, by1, bx2, by2, bcor, texto in self.botoes_cores:
            if texto in self.cores and self.cores[texto] == self.cor:
                cv2.rectangle(
                    img, (bx1 - 3, by1 - 3), (bx2 + 3, by2 + 3), (0, 255, 0), 3
                )

            cv2.rectangle(img, (bx1, by1), (bx2, by2), bcor, cv2.FILLED)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

            text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = bx1 + (bx2 - bx1 - text_size[0]) // 2
            text_y = by1 + (by2 - by1 + text_size[1]) // 2
            text_color = (255, 255, 255)
            if texto == "Branco":
                text_color = (0, 0, 0)
            cv2.putText(
                img,
                texto,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
            )

        # Botões de ferramentas
        for bx1, by1, bx2, by2, bcor, texto in self.botoes_ferramentas:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), bcor, cv2.FILLED)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (50, 50, 50), 2)

            if texto == "Iniciar Jogo" and self.jogo_ativo:
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 4)

            text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = bx1 + (bx2 - bx1 - text_size[0]) // 2
            text_y = by1 + (by2 - by1 + text_size[1]) // 2
            cv2.putText(
                img,
                texto,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Informações na tela
        info_y = 150
        cv2.putText(
            img,
            f"Espessura: {self.espessura}",
            (50, info_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            img,
            f"Pontos: {len(self.desenho)}",
            (50, info_y - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Instruções
        instrucoes = ["1 dedo: Desenhar", "3 dedos: Limpar tudo", "ESC: Sair"]
        for i, instrucao in enumerate(instrucoes):
            cv2.putText(
                img,
                instrucao,
                (50, img.shape[0] - 80 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

    def renderizar_desenho(self, img, draw_on_canvas=False):
        """Renderiza o desenho e, opcionalmente, o armazena no imgCanvas para cálculo"""

        # Cria uma cópia temporária do canvas para não modificar o original
        if draw_on_canvas:
            self.imgCanvas = np.zeros((self.altura, self.largura, 3), np.uint8)
            cor_canvas = (255, 255, 255)

        for id, ponto in enumerate(self.desenho):
            x, y, cor_p, esp_p = ponto

            if draw_on_canvas:
                target_img = self.imgCanvas
                p_cor = cor_canvas
            else:
                target_img = img
                p_cor = cor_p

            if x != 0:
                cv2.circle(target_img, (x, y), esp_p // 2, p_cor, cv2.FILLED)

            if id >= 1:
                ax, ay, _, _ = self.desenho[id - 1]
                if x != 0 and ax != 0:
                    cv2.line(target_img, (x, y), (ax, ay), p_cor, esp_p)

    def executar(self):
        """Loop principal da aplicação"""

        similaridade_texto_display = None

        while True:
            check, img = self.video.read()
            if not check:
                break

            self.altura, self.largura, _ = img.shape

            # Reduz cooldown
            if self.button_cooldown > 0:
                self.button_cooldown -= 1

            # Hand tracking (Feito no frame original)
            resultado = self.detector.findHands(img, draw=True)
            hands = resultado[0]

            dedosLev = 0

            if hands:
                hand = hands[0]
                lmlist = hand["lmList"]
                dedos = self.detector.fingersUp(hand)
                dedosLev = dedos.count(1)

                x, y = lmlist[8][0], lmlist[8][1]
                x_flip = img.shape[1] - x

                self.processar_botoes(x_flip, y)

                # Desenho. A cor original é usada para armazenar o ponto, mas o render será corrigido.
                if dedosLev == 1:
                    pontos_suavizados = self.smooth_drawing((x, y))
                    for px, py in pontos_suavizados:
                        self.desenho.append((px, py, self.cor, self.espessura))
                    cv2.circle(img, (x, y), self.espessura // 2, self.cor, 2)

                elif dedosLev != 1 and dedosLev != 3:
                    if self.desenho and self.desenho[-1][0] != 0:
                        self.desenho.append((0, 0, self.cor, self.espessura))
                    self.last_position = None

                elif dedosLev == 3:
                    self.desenho = []
                    self.imgCanvas = np.zeros((self.altura, self.largura, 3), np.uint8)
                    self.last_position = None

            limites_forma = [0, 0, 0, 0]
            similaridade_texto_display = None

            # 1. Desenha o Contorno ALVO (na imagem real, para visualização)
            if self.jogo_ativo:
                limites_forma = desenhar_quadrado_contorno(img)

                # 2. Cria a MÁSCARA ALVO para o cálculo
                mascara_alvo_bgr = np.zeros((self.altura, self.largura, 3), np.uint8)
                x1, y1, x2, y2 = limites_forma
                margem_erro = self.espessura + 5
                cv2.rectangle(
                    mascara_alvo_bgr, (x1, y1), (x2, y2), (255, 255, 255), margem_erro
                )

                mascara_alvo = cv2.cvtColor(mascara_alvo_bgr, cv2.COLOR_BGR2GRAY)

                # 3. Desenha a linha do jogador na imgCanvas (Máscara Desenho)
                self.renderizar_desenho(img, draw_on_canvas=True)

                mascara_desenho = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)

                # 4. Calcula e ARMAZENA O TEXTO apenas quando o desenho estiver finalizado
                if (
                    dedosLev == 2
                    and len(self.desenho) > 0
                    and self.last_position is None
                ):
                    similaridade, similar = calcular_similaridade(
                        mascara_alvo, mascara_desenho, limite_minimo=0.75
                    )
                    cor_texto = (0, 255, 0) if similar else (0, 0, 255)

                    similaridade_texto_display = {
                        "texto": f"Similaridade: {similaridade*100:.2f}% ({'OK' if similar else 'TENTE NOVAMENTE'})",
                        "cor": cor_texto,
                    }

            # Renderiza desenho
            self.renderizar_desenho(img, draw_on_canvas=False)

            img = cv2.flip(img, 1)

            # Desenha interface (Botões e texto de informações)
            self.desenhar_interface(img)

            # Desenha o texto do jogo de simlaridade (se tivesse sido plotado antes, ia ficar espelhado)
            if similaridade_texto_display:
                cv2.putText(
                    img,
                    similaridade_texto_display["texto"],
                    (200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    similaridade_texto_display["cor"],
                    2,
                )
            # ----------------------------------------------------------

            cv2.imshow("Lousa Digital", img)

            # Teclas de atalho
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("s"):
                self.salvar_desenho()
            elif key == ord("l"):
                self.carregar_desenho()
            elif key == ord("c"):
                self.desenho = []
                self.imgCanvas = np.zeros((self.altura, self.largura, 3), np.uint8)
            elif key == ord("z"):
                self.desfazer_ultimo()

        self.video.release()
        cv2.destroyAllWindows()
