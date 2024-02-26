import pygame
import threading


def keep_alive():
    while True:
        pygame.event.pump()


WHITE = (255, 255, 255)
RED = (255, 0, 0)


class Visualizer:
    def __init__(self, samplerate):
        pygame.init()
        # Constants
        WIDTH, HEIGHT = 800, 600
        FPS = 50
        sound_freq = samplerate
        self.sound_frame_size = int(sound_freq / FPS)

        # Colors
        # Create the game window
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Visualizer")
        screen.fill(WHITE)
        pygame.draw.circle(screen, RED, (400, 300), 250, width=5)
        pygame.display.flip()
        self.screen = screen
        self.FPS = FPS
        # keep alive
        # pygame_thread = threading.Thread(target=keep_alive)
        # pygame_thread.start()

    def visualize(self, audio, text=None):
        clock = pygame.time.Clock()
        font = pygame.font.Font('freesansbold.ttf', 12)

        running = True
        idx = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update game logic here

            # Clear the screen
            self.screen.fill(WHITE)

            # Draw game elements here
            # draw a circle
            if idx + self.sound_frame_size > len(audio):
                break
            c = (audio[idx:idx + self.sound_frame_size].max() + 1) / 2
            idx += self.sound_frame_size
            color = (int(255 * c), 0, int(255 * (1 - c)))
            pygame.draw.circle(self.screen, color, (400, 300), 250)
            text_render = font.render(text, True, (0, 0, 0), WHITE)
            textRect = text_render.get_rect()
            textRect.center = (800 // 2, 600 // 2)

            self.screen.blit(text_render, textRect)

            # Update the display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(self.FPS)
        self.screen.fill(WHITE)
        pygame.draw.circle(self.screen, RED, (400, 300), 250, width=5)
        pygame.display.flip()

    def close(self):
        pygame.quit()
