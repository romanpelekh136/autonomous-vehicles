import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import math

class CarRacingCustom(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.num_rays = 11 
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(self.num_rays,), dtype=np.float32
        )

        # Додаємо місце для панелі приладів
        self.width = 800
        self.track_height = 600
        self.panel_height = 120
        self.height = self.track_height + self.panel_height 

        self.screen = None
        self.clock = None
        self.font = None 

        self.outer_rect = pygame.Rect(50, 50, 700, 500)
        self.inner_rect = pygame.Rect(150, 150, 500, 300)

        self.checkpoints = [
            pygame.Rect(50, 150, 100, 10),
            pygame.Rect(400, 50, 10, 100),
            pygame.Rect(650, 150, 100, 10),
            pygame.Rect(650, 450, 100, 10),
            pygame.Rect(400, 450, 10, 100),
            pygame.Rect(50, 450, 100, 10)
        ]
        self.current_checkpoint = 0
        self.laps = 0 
        self.current_action = [0.0, 0.0, 0.0] 

        self.L = 20.0
        self.max_speed = 10.0
        self.max_ray_length = 200

        self.reset()

    def _get_lidar_data(self):
        angles = np.linspace(-math.pi/2, math.pi/2, self.num_rays)
        distances = []

        for a in angles:
            ray_angle = self.angle + a
            dist = 0
            ray_x = self.car_x
            ray_y = self.car_y
            
            for step in range(self.max_ray_length):
                ray_x += math.cos(ray_angle)
                ray_y -= math.sin(ray_angle)
                dist += 1
                
                pos = (int(ray_x), int(ray_y))
                if not self.outer_rect.collidepoint(pos) or self.inner_rect.collidepoint(pos):
                    break
            distances.append(dist)
            
        return np.array(distances, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_x = 100
        self.car_y = 300
        self.angle = math.pi / 2
        self.speed = 0.0
        self.current_checkpoint = 0
        self.laps = 0
        self.current_action = [0.0, 0.0, 0.0]
        
        observation = self._get_lidar_data()
        return observation, {}

    def step(self, action):
        self.current_action = action 
        steering = action[0] * (math.pi / 4)
        acceleration = action[1]
        brake = action[2]

        self.speed += acceleration * 0.5
        self.speed -= brake * 1.0
        self.speed = np.clip(self.speed, 0, self.max_speed)

        if self.speed > 0:
            self.angle += (self.speed / self.L) * math.tan(steering)
        
        self.car_x += self.speed * math.cos(self.angle)
        self.car_y -= self.speed * math.sin(self.angle)

        terminated = False
        reward = -0.1 
        
        car_pos = (int(self.car_x), int(self.car_y))
        if not self.outer_rect.collidepoint(car_pos) or self.inner_rect.collidepoint(car_pos):
            terminated = True
            reward = -100.0 

        car_rect = pygame.Rect(int(self.car_x - 10), int(self.car_y - 10), 20, 20)
        target_cp = self.checkpoints[self.current_checkpoint]
        
        if car_rect.colliderect(target_cp):
            reward += 10.0 
            if self.current_checkpoint == len(self.checkpoints) - 1:
                self.laps += 1
            self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)

        observation = self._get_lidar_data()
        truncated = False

        return observation, reward, terminated, truncated, {}

    # Допоміжна функція для малювання шкал
    def _draw_bar(self, label, value, min_val, max_val, x, y, color):
        text = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(text, (x, y))
        
        bar_x = int(x + 80)
        bar_y = int(y + 5)
        bar_w = 120
        bar_h = 15
        
        # Рамка
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_w, bar_h), 2)
        
        # Конвертуємо numpy-тип у звичайне число
        val = float(value)
        
        # Заповнення
        if min_val < 0: # Для керма
            center_x = int(bar_x + bar_w / 2)
            fill_w = int(abs(val) * (bar_w / 2))
            if val < 0:
                pygame.draw.rect(self.screen, color, (center_x - fill_w, bar_y, fill_w, bar_h))
            else:
                pygame.draw.rect(self.screen, color, (center_x, bar_y, fill_w, bar_h))
            pygame.draw.line(self.screen, (255, 255, 255), (center_x, bar_y), (center_x, bar_y + bar_h))
        else: # Для газу і гальма
            fill_w = int(val * bar_w)
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_w, bar_h))

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20) 

        # Заливаємо екран чорним, малюємо траву тільки у верхній частині
        self.screen.fill((0, 0, 0)) 
        pygame.draw.rect(self.screen, (30, 100, 30), (0, 0, self.width, self.track_height))
        
        # Трек
        pygame.draw.rect(self.screen, (100, 100, 100), self.outer_rect) 
        pygame.draw.rect(self.screen, (30, 100, 30), self.inner_rect)   

        target_cp = self.checkpoints[self.current_checkpoint]
        pygame.draw.rect(self.screen, (0, 0, 255), target_cp)

        angles = np.linspace(-math.pi/2, math.pi/2, self.num_rays)
        obs = self._get_lidar_data()
        for i, a in enumerate(angles):
            ray_angle = self.angle + a
            end_x = self.car_x + obs[i] * math.cos(ray_angle)
            end_y = self.car_y - obs[i] * math.sin(ray_angle)
            pygame.draw.line(self.screen, (0, 255, 0), (int(self.car_x), int(self.car_y)), (int(end_x), int(end_y)), 2)

        car_length = 40
        car_width = 20
        car_surface = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        car_surface.fill((200, 0, 0)) 
        
        rotated_car = pygame.transform.rotate(car_surface, math.degrees(self.angle))
        rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, rect.topleft)

        # --- ПАНЕЛЬ ПРИЛАДІВ ---
        panel_y = self.track_height
        pygame.draw.rect(self.screen, (40, 40, 40), (0, panel_y, self.width, self.panel_height))
        pygame.draw.line(self.screen, (200, 200, 200), (0, panel_y), (self.width, panel_y), 3)

        # Текстова інформація
        lap_text = self.font.render(f"Laps: {self.laps}", True, (255, 255, 255))
        self.screen.blit(lap_text, (30, panel_y + 20))
        
        speed_text = self.font.render(f"Speed: {self.speed:.1f} / {self.max_speed}", True, (255, 255, 255))
        self.screen.blit(speed_text, (30, panel_y + 60))

        # Шкали дій
        self._draw_bar("Steer", self.current_action[0], -1.0, 1.0, 250, panel_y + 40, (0, 200, 255))
        self._draw_bar("Gas", self.current_action[1], 0.0, 1.0, 500, panel_y + 20, (0, 255, 0))
        self._draw_bar("Brake", self.current_action[2], 0.0, 1.0, 500, panel_y + 60, (255, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None