import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime


class LousaDigital:
    def __init__(self):
        # Configurações da câmera
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 1280)
        self.video.set(4, 720)

        # Hand tracking
        self.detector = HandDetector(detectionCon=0.8)
        self.desenho = []

        # Configurações iniciais
        self.cor = (0, 0, 255)
        self.espessura = 20
        self.modo_atual = "desenho"  # desenho, apresentacao
        self.button_cooldown = 0
        self.last_position = None

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
        ]

        # Combinando todos os botões
        self.botoes = self.botoes_cores + self.botoes_ferramentas

    def smooth_drawing(self, current_pos):
        """Suaviza o desenho usando interpolação"""
        if self.last_position is None:
            self.last_position = current_pos
            return [current_pos]

        # Interpola pontos entre a posição anterior e atual
        points = []
        x1, y1 = self.last_position
        x2, y2 = current_pos

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distance > 5:  # Se a distância for grande, interpola
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
        """Salva o desenho em arquivo JSON"""
        if not self.desenho:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"desenho_{timestamp}.json"

        # Converte o desenho para formato serializável
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
        """Carrega o último desenho salvo"""
        # Procura pelo arquivo mais recente
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

            # Converte de volta para o formato interno
            self.desenho = []
            for ponto in desenho_carregado:
                self.desenho.append(
                    (ponto["x"], ponto["y"], tuple(ponto["cor"]), ponto["espessura"])
                )

            print(f"Desenho carregado: {arquivo_recente}")
        except Exception as e:
            print(f"Erro ao carregar: {e}")

    def desfazer_ultimo(self):
        """Desfaz a última linha desenhada"""
        if not self.desenho:
            return

        # Remove pontos até encontrar um ponto de quebra (0,0)
        while self.desenho and self.desenho[-1][0] != 0:
            self.desenho.pop()

        # Remove o ponto de quebra também
        if self.desenho and self.desenho[-1][0] == 0:
            self.desenho.pop()

    def processar_botoes(self, x_flip, y):
        """Processa cliques nos botões com cooldown"""
        if self.button_cooldown > 0:
            return

        for bx1, by1, bx2, by2, bcor, texto in self.botoes:
            if bx1 < x_flip < bx2 and by1 < y < by2:
                self.button_cooldown = 15  # Cooldown de 15 frames

                # Botões de cor
                if texto in self.cores:
                    self.cor = self.cores[texto]

                # Botões de ferramenta
                elif texto == "+":
                    self.espessura = min(self.espessura + 5, 50)
                elif texto == "-":
                    self.espessura = max(self.espessura - 5, 5)
                elif texto == "Limpar":
                    self.desenho = []
                elif texto == "Salvar":
                    self.salvar_desenho()
                elif texto == "Carregar":
                    self.carregar_desenho()
                elif texto == "Desfazer":
                    self.desfazer_ultimo()
                break

    def desenhar_interface(self, img):
        """Desenha a interface com informações adicionais"""
        # Botões de cores
        for bx1, by1, bx2, by2, bcor, texto in self.botoes_cores:
            # Destaca o botão da cor atual
            if texto in self.cores and self.cores[texto] == self.cor:
                cv2.rectangle(
                    img, (bx1 - 3, by1 - 3), (bx2 + 3, by2 + 3), (0, 255, 0), 3
                )

            cv2.rectangle(img, (bx1, by1), (bx2, by2), bcor, cv2.FILLED)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

            # Texto do botão
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

            # Texto do botão
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

    def renderizar_desenho(self, img):
        """Renderiza o desenho com melhor qualidade"""
        for id, ponto in enumerate(self.desenho):  # Corrigido: self.desenho
            x, y, cor_p, esp_p = ponto
            if x != 0:
                cv2.circle(img, (x, y), esp_p // 2, cor_p, cv2.FILLED)

            if id >= 1:
                ax, ay, _, _ = self.desenho[id - 1]
                if x != 0 and ax != 0:
                    cv2.line(img, (x, y), (ax, ay), cor_p, esp_p)

    def executar(self):
        """Loop principal da aplicação"""
        while True:
            check, img = self.video.read()
            if not check:
                break

            # Reduz cooldown
            if self.button_cooldown > 0:
                self.button_cooldown -= 1

            # Hand tracking
            resultado = self.detector.findHands(img, draw=True)
            hands = resultado[0]

            if hands:
                hand = hands[0]
                lmlist = hand["lmList"]
                dedos = self.detector.fingersUp(hand)
                dedosLev = dedos.count(1)

                x, y = lmlist[8][0], lmlist[8][1]
                x_flip = img.shape[1] - x

                # Processa botões
                self.processar_botoes(x_flip, y)

                # Desenho
                if dedosLev == 1:
                    # Desenho suavizado
                    pontos_suavizados = self.smooth_drawing((x, y))
                    for px, py in pontos_suavizados:
                        self.desenho.append((px, py, self.cor, self.espessura))

                    # Mostra preview do pincel
                    cv2.circle(img, (x, y), self.espessura // 2, self.cor, 2)

                elif dedosLev != 1 and dedosLev != 3:
                    # Adiciona ponto de quebra
                    if self.desenho and self.desenho[-1][0] != 0:
                        self.desenho.append((0, 0, self.cor, self.espessura))
                    self.last_position = None

                elif dedosLev == 3:
                    # Limpa tudo
                    self.desenho = []
                    self.last_position = None

            # Renderiza desenho
            self.renderizar_desenho(img)

            # Espelha imagem
            img = cv2.flip(img, 1)

            # Desenha interface
            self.desenhar_interface(img)

            cv2.imshow("Lousa Digital", img)

            # Teclas de atalho
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("s"):  # S para salvar
                self.salvar_desenho()
            elif key == ord("l"):  # L para carregar
                self.carregar_desenho()
            elif key == ord("c"):  # C para limpar
                self.desenho = []
            elif key == ord("z"):  # Z para desfazer
                self.desfazer_ultimo()

        self.video.release()
        cv2.destroyAllWindows()


# Executa o programa
if __name__ == "__main__":
    lousa = LousaDigital()
    lousa.executar()
