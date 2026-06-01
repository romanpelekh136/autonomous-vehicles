import time
import os
import argparse
import math
import gymnasium as gym
import pygame
from custom_env import CarRacingCustom

def configure_sdl_video_driver():
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


configure_sdl_video_driver()


gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=100000
)


def load_ppo_agent():
    """Завантажує PPO-модель і повертає функцію передбачення."""
    from stable_baselines3 import PPO
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.1,
    }
    model = PPO.load("models/ppo_best/best_model.zip", device="cpu", custom_objects=custom_objects)
    def predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return predict


def load_neat_agent():
    """Завантажує NEAT-геном і повертає функцію передбачення."""
    import neat
    import pickle
    with open("models/neat_best/best_model.pkl", "rb") as f:
        winner = pickle.load(f)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.txt")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    def predict(obs):
        output = net.activate(obs)
        return [output[0], max(0.0, output[1]), max(0.0, output[2])]
    return predict


class RaceViewer:
    """Вікно гонки"""

    DIVIDER_WIDTH = 4
    HEADER_HEIGHT = 40
    LABEL_COLORS = {
        "ppo": (255, 255, 255),   
        "neat": (255, 255, 255), 
    }

    def __init__(self, env_ppo, env_neat, laps_target):
        self.env_ppo = env_ppo
        self.env_neat = env_neat
        self.laps_target = laps_target

        # Розмір панелі відповідає розміру рендеру середовища
        panel_w = min(800, env_ppo.width)
        panel_h = min(600, env_ppo.track_height) + env_ppo.panel_height

        self.panel_w = panel_w
        self.panel_h = panel_h
        self.win_w = panel_w * 2 + self.DIVIDER_WIDTH
        self.win_h = panel_h + self.HEADER_HEIGHT

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h),
            pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        pygame.display.set_caption("PPO vs NEAT ")
        self.clock = pygame.time.Clock()
        self.font_header = pygame.font.SysFont('Arial', 22, bold=True)
        self.font_result = pygame.font.SysFont('Arial', 28, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 18)

        self.surf_ppo = pygame.Surface((panel_w, panel_h))
        self.surf_neat = pygame.Surface((panel_w, panel_h))

    def draw_header(self, ppo_done, neat_done, ppo_time, neat_time, elapsed):
        """Малює хедер з назвами агентів та таймерами."""
        header_rect = pygame.Rect(0, 0, self.win_w, self.HEADER_HEIGHT)
        pygame.draw.rect(self.screen, (25, 25, 30), header_rect)

        # PPO (ліва панель)
        ppo_color = self.LABEL_COLORS["ppo"]
        ppo_label = self.font_header.render("PPO", True, ppo_color)
        self.screen.blit(ppo_label, (20, 8))

        ppo_status = f"{ppo_time:.2f}s" if ppo_done else f"{elapsed:.1f}s"
        ppo_status_surf = self.font_small.render(ppo_status, True, (200, 200, 200))
        self.screen.blit(ppo_status_surf, (80, 11))

        if ppo_done:
            check = self.font_header.render("✓", True, (0, 255, 100))
            self.screen.blit(check, (160, 8))

        # NEAT (права панель)
        neat_color = self.LABEL_COLORS["neat"]
        right_x = self.panel_w + self.DIVIDER_WIDTH
        neat_label = self.font_header.render("NEAT", True, neat_color)
        self.screen.blit(neat_label, (right_x + 20, 8))

        neat_status = f"{neat_time:.2f}s" if neat_done else f"{elapsed:.1f}s"
        neat_status_surf = self.font_small.render(neat_status, True, (200, 200, 200))
        self.screen.blit(neat_status_surf, (right_x + 90, 11))

        if neat_done:
            check = self.font_header.render("✓", True, (0, 255, 100))
            self.screen.blit(check, (right_x + 170, 8))


    def draw_divider(self):
        """Малює вертикальний роздільник між панелями."""
        x = self.panel_w
        pygame.draw.rect(self.screen, (80, 80, 90),
                         (x, self.HEADER_HEIGHT, self.DIVIDER_WIDTH, self.panel_h))

    def draw_results_overlay(self, ppo_time, neat_time, ppo_progress, neat_progress):
        """Малює екран результатів після завершення гонки."""
        overlay = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        cx, cy = self.win_w // 2, self.win_h // 2

        title = self.font_result.render("Результати", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(cx, cy - 80)))

        # Визначення переможця
        ppo_finished = isinstance(ppo_time, float)
        neat_finished = isinstance(neat_time, float)

        if ppo_finished and neat_finished:
            if ppo_time < neat_time:
                winner = "PPO Виграв!"
                win_color = self.LABEL_COLORS["ppo"]
            elif neat_time < ppo_time:
                winner = "NEAT Виграв!"
                win_color = self.LABEL_COLORS["neat"]
            else:
                winner = "НІЧІЯ!"
                win_color = (255, 255, 255)
        elif ppo_finished:
            winner = "PPO Виграв!"
            win_color = self.LABEL_COLORS["ppo"]
        elif neat_finished:
            winner = "NEAT Виграв!"
            win_color = self.LABEL_COLORS["neat"]
        else:
            winner = "Ніхто не фінішував"
            win_color = (200, 80, 80)

        winner_surf = self.font_result.render(winner, True, win_color)
        self.screen.blit(winner_surf, winner_surf.get_rect(center=(cx, cy - 40)))

        ppo_str = f"{ppo_time:.2f}s" if ppo_finished else "Не фінішував"
        neat_str = f"{neat_time:.2f}s" if neat_finished else "Не фінішував"

        ppo_line = self.font_small.render(
            f"PPO:  {ppo_str}  |  progress: {ppo_progress:.2f}", True, self.LABEL_COLORS["ppo"])
        neat_line = self.font_small.render(
            f"NEAT: {neat_str}  |  progress: {neat_progress:.2f}", True, self.LABEL_COLORS["neat"])

        self.screen.blit(ppo_line, ppo_line.get_rect(center=(cx, cy + 10)))
        self.screen.blit(neat_line, neat_line.get_rect(center=(cx, cy + 40)))

        hint = self.font_small.render("", True, (150, 150, 150))
        self.screen.blit(hint, hint.get_rect(center=(cx, cy + 90)))

    def render_frame(self, ppo_done, neat_done, ppo_time, neat_time, elapsed):
        self.screen.fill((0, 0, 0))

        # Рендер кожного середовища на свою поверхню
        if not ppo_done:
            self.env_ppo.unwrapped.render_to_surface(self.surf_ppo)
        if not neat_done:
            self.env_neat.unwrapped.render_to_surface(self.surf_neat)

        # Панелі під хедером
        self.screen.blit(self.surf_ppo, (0, self.HEADER_HEIGHT))
        self.screen.blit(self.surf_neat, (self.panel_w + self.DIVIDER_WIDTH, self.HEADER_HEIGHT))

        self.draw_divider()
        self.draw_header(ppo_done, neat_done, ppo_time, neat_time, elapsed)

        pygame.display.flip()
        self.clock.tick(60)

    def handle_events(self):
        """Обробляє події pygame. Повертає False для виходу."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def wait_for_exit(self):
        """Чекає на натискання клавіші або закриття вікна."""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False
            self.clock.tick(30)


def main():
    parser = argparse.ArgumentParser(description="PPO vs NEAT split-screen race")
    parser.add_argument("--track", default="track_03")
    parser.add_argument("--laps", type=int, default=5)
    args = parser.parse_args()

    print(f"Запуск гонки на трасі '{args.track}', {args.laps} кіл...")

    # Завантаження обох агентів
    ppo_predict = load_ppo_agent()
    neat_predict = load_neat_agent()

    env_ppo = gym.make('CarRacingCustom-v0', render_mode=None, track_name=args.track)
    env_neat = gym.make('CarRacingCustom-v0', render_mode=None, track_name=args.track)

    obs_ppo, _ = env_ppo.reset()
    obs_neat, _ = env_neat.reset()

    viewer = RaceViewer(env_ppo.unwrapped, env_neat.unwrapped, args.laps)

    ppo_done = False
    neat_done = False
    ppo_time = "Не фінішував"
    neat_time = "Не фінішував"
    ppo_progress = 0.0
    neat_progress = 0.0

    start = time.time()
    running = True


    while running:
        if not viewer.handle_events():
            break

        elapsed = time.time() - start

        # Крок PPO
        if not ppo_done:
            action_ppo = ppo_predict(obs_ppo)
            obs_ppo, _, terminated, truncated, info = env_ppo.step(action_ppo)
            ppo_progress = info.get("progress", 0)
            if env_ppo.unwrapped.laps >= args.laps or terminated or truncated:
                ppo_done = True
                ppo_time = time.time() - start

        # Крок NEAT
        if not neat_done:
            action_neat = neat_predict(obs_neat)
            obs_neat, _, terminated, truncated, info = env_neat.step(action_neat)
            neat_progress = info.get("progress", 0)
            if env_neat.unwrapped.laps >= args.laps or terminated or truncated:
                neat_done = True
                neat_time = time.time() - start

        viewer.render_frame(ppo_done, neat_done, ppo_time, neat_time, elapsed)

        # Обидва фінішували
        if ppo_done and neat_done:
            viewer.draw_results_overlay(ppo_time, neat_time, ppo_progress, neat_progress)
            pygame.display.flip()
            viewer.wait_for_exit()
            running = False

    env_ppo.close()
    env_neat.close()
    pygame.quit()


if __name__ == "__main__":
    main()